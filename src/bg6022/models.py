"""The seven durable runtime objects used by the v3 execution foundation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    field_validator,
    model_validator,
)

RunStatus = Literal[
    "planned",
    "running",
    "waiting",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
ResultStatus = Literal["succeeded", "failed", "cancelled", "interrupted", "needs_input"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResultTarget(StrictModel):
    """A value or output port requested from one logical step.

    This is a nested value contract, not another persisted runtime object.  The
    optional ``step_id`` keeps old M0 plans readable while M1 plans identify the
    producer explicitly.
    """

    step_id: str | None = None
    field: str | None = None
    port: str | None = None

    @model_validator(mode="after")
    def _one_target_kind(self) -> ResultTarget:
        if (self.field is None) == (self.port is None):
            raise ValueError("result target must contain exactly one field or port")
        return self


class Request(StrictModel):
    id: str
    description: str
    requested_results: list[ResultTarget] = Field(default_factory=list)
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["cli", "chat"] = "cli"
    operation: Literal["SP", "Opt"] | None = None
    original_text: str | None = None
    user_modifications: dict[str, Any] = Field(default_factory=dict)
    structure_input: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("requested_results", mode="before")
    @classmethod
    def _load_legacy_result_targets(cls, value: Any) -> Any:
        if value is None:
            return []
        return [{"field": item} if isinstance(item, str) else item for item in value]


class InputReference(StrictModel):
    """A logical artifact or an earlier step port, never an arbitrary path."""

    artifact_id: str | None = None
    step_id: str | None = None
    port: str | None = None

    @model_validator(mode="after")
    def _exactly_one_reference(self) -> InputReference:
        artifact = self.artifact_id is not None
        step = self.step_id is not None or self.port is not None
        if artifact == step:
            raise ValueError("input reference must be an artifact_id or a step_id/port pair")
        if step and (not self.step_id or not self.port):
            raise ValueError("step input references require both step_id and port")
        return self


class Step(StrictModel):
    id: str
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputReference] = Field(default_factory=dict)
    goal_checks: list[str] = Field(default_factory=list)
    origin_step_id: str | None = None


class Plan(StrictModel):
    id: str
    revision: int = 1
    request_id: str
    steps: list[Step]
    requested_results: list[ResultTarget] = Field(default_factory=list)

    @field_validator("requested_results", mode="before")
    @classmethod
    def _load_legacy_result_targets(cls, value: Any) -> Any:
        if value is None:
            return []
        return [{"field": item} if isinstance(item, str) else item for item in value]

    @field_validator("revision")
    @classmethod
    def _positive_revision(cls, value: int) -> int:
        if value < 1:
            raise ValueError("plan revision must be positive")
        return value


class Artifact(StrictModel):
    id: str
    artifact_type: str
    role: str
    relative_path: str
    size_bytes: int
    sha256: str
    source: str
    run_id: str
    step_id: str | None = None
    attempt: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("size_bytes")
    @classmethod
    def _nonnegative_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("artifact size cannot be negative")
        return value


class Result(StrictModel):
    run_id: str
    step_id: str
    attempt: int
    status: ResultStatus
    values: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    output_ports: dict[str, str] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    attempt_relative_path: str = ""
    clarification: dict[str, Any] = Field(default_factory=dict)
    parameter_sources: dict[str, str] = Field(default_factory=dict)
    step_fingerprint: str | None = None
    input_bindings: dict[str, str] = Field(default_factory=dict)


class Run(StrictModel):
    id: str
    request: Request
    plan: Plan
    resources: dict[str, Any]
    execution_permission: bool = False
    accepted_execution_sha256: str | None = None
    status: RunStatus = "planned"
    step_status: dict[str, str] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_index: list[Artifact] = Field(default_factory=list)
    result_index: list[str] = Field(default_factory=list)
    current_results: dict[str, str] = Field(default_factory=dict)
    waiting_for: Literal["confirmation", "clarification", "repair", None] = None
    pending_data: dict[str, Any] = Field(default_factory=dict)
    accepted_snapshot: dict[str, Any] = Field(default_factory=dict)
    repair_records: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    attempt_counts: dict[str, int] = Field(default_factory=dict)
    extra_orca_executions: int = 0
    plan_revisions: int = 0
    origin_step_map: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None
    active_seconds: float = 0.0
    created_at: str
    updated_at: str

    _active_interval_started_at: float | None = PrivateAttr(default=None)

    def start_active_interval(self, *, now: float | None = None) -> None:
        """Start the one active execution interval, without changing its total."""

        if self._active_interval_started_at is None:
            self._active_interval_started_at = time.monotonic() if now is None else now

    @property
    def active_interval_open(self) -> bool:
        return self._active_interval_started_at is not None

    def current_active_seconds(self, *, now: float | None = None) -> float:
        """Return persisted active time plus the currently open interval."""

        elapsed = max(0.0, float(self.active_seconds))
        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            elapsed += max(0.0, current - self._active_interval_started_at)
        return elapsed

    def checkpoint_active(self, *, now: float | None = None) -> float:
        """Accumulate the open interval exactly once and leave it open."""

        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            self.active_seconds += max(0.0, current - self._active_interval_started_at)
            self._active_interval_started_at = current
        return self.active_seconds

    def finish_active_interval(self, *, now: float | None = None) -> float:
        """Accumulate and close the current active interval."""

        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            self.active_seconds += max(0.0, current - self._active_interval_started_at)
            self._active_interval_started_at = None
        return self.active_seconds


ExecuteFunction = Callable[[Step, Run, Any], Result]


class Tool(StrictModel):
    name: str
    description: str
    parameter_model: str = "none"
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    parameter_type: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)
    input_ports: dict[str, str] = Field(default_factory=dict)
    output_ports: dict[str, str] = Field(default_factory=dict)
    results: dict[str, str] = Field(default_factory=dict)
    result_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)
    success_conditions: list[str] = Field(default_factory=list)
    repair_capabilities: list[str] = Field(default_factory=list)
    requires_compute_permission: bool = True
    parameter_preparation: Literal["none", "orca_electronic_state"] = "none"
    execution_budget: Literal["none", "orca"] = "none"
    deferred_parameters: list[str] = Field(default_factory=list)
    available: bool = True
    execute_function: ExecuteFunction | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(
        extra="forbid", strict=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    @model_validator(mode="after")
    def _validate_result_metadata(self) -> Tool:
        """Keep presentation metadata attached to the Tool result contract.

        Result metadata is descriptive only.  The executable result contract
        remains ``results``/``output_ports``; metadata can neither add a new
        result nor silently describe a field that the Tool cannot produce.
        """

        declared = set(self.results) | set(self.output_ports)
        unknown = sorted(set(self.result_metadata) - declared)
        if unknown:
            raise ValueError(f"result metadata has undeclared keys: {unknown}")
        allowed = {"label", "description", "caveat"}
        for name, metadata in self.result_metadata.items():
            extra = sorted(set(metadata) - allowed)
            if extra:
                raise ValueError(f"result metadata for {name!r} has unsupported keys: {extra}")
        return self

    def execute(self, step: Step, run: Run, *, cancel: Any) -> Result:
        if self.execute_function is None:
            raise RuntimeError(f"tool {self.name!r} has no executable implementation")
        return self.execute_function(step, run, cancel)

    def validate_parameters(
        self, parameters: dict[str, Any], *, allow_deferred: bool = False
    ) -> dict[str, Any]:
        if self.parameter_type is None:
            if self.parameter_schema:
                raise ValueError(f"tool {self.name!r} has no parameter model")
            if parameters:
                raise ValueError(f"tool {self.name!r} does not accept parameters")
            return {}
        if allow_deferred:
            missing = {
                name
                for name, field in self.parameter_type.model_fields.items()
                if field.is_required() and name not in parameters
            }
            undeclared = missing - set(self.deferred_parameters)
            if undeclared:
                raise ValueError(f"missing required parameters: {sorted(undeclared)}")
            if missing:
                return _validate_partial_model(self.parameter_type, parameters)
        return self.parameter_type.model_validate(parameters, strict=True).model_dump(
            mode="python", exclude_none=True
        )

    def description_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"execute_function"})


__all__ = [
    "Artifact",
    "InputReference",
    "Plan",
    "Request",
    "Result",
    "ResultTarget",
    "Run",
    "RunStatus",
    "Step",
    "Tool",
]


def _validate_partial_model(
    model_type: type[BaseModel], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Validate supplied fields without inventing missing scientific values."""

    known = set(model_type.model_fields)
    extra = set(parameters) - known
    if extra:
        raise ValueError(f"extra inputs are not permitted: {sorted(extra)}")

    # ``FieldInfo.annotation`` does not include constraints kept in
    # ``FieldInfo.metadata`` (for example ge/le on iteration limits), and a
    # TypeAdapter would also skip model field validators.  Build a temporary
    # all-optional view that preserves both so deferred q/m values remain
    # absent while every supplied value is validated exactly as it would be in
    # the complete model.
    partial_fields: dict[str, tuple[Any, None]] = {}
    for name, field in model_type.model_fields.items():
        field_data = field.asdict()
        attributes = dict(field_data["attributes"])
        attributes.pop("default", None)
        attributes.pop("default_factory", None)
        annotation = Annotated[
            field_data["annotation"],
            *field_data["metadata"],
            Field(**attributes),
        ]
        partial_fields[name] = (annotation, None)
    partial_model = create_model(
        f"{model_type.__name__}PartialValidation",
        __base__=model_type,
        **partial_fields,
    )
    try:
        validated = partial_model.model_validate(parameters, strict=True)
    except Exception as error:
        raise ValueError(str(error)) from error
    return validated.model_dump(mode="python", exclude_none=True)
