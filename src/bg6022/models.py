"""The seven durable runtime objects used by the v3 execution foundation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictStr,
    TypeAdapter,
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
Operation = Literal["SP", "Opt", "Freq"]
ScientificCheckStatus = Literal["passed", "not_met", "unverified"]

_OUTPUT_PREFERENCE_DEFAULTS = {
    "layout": "auto",
    "file_content": "auto",
    "detail": "normal",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def validate_output_preferences(value: Any) -> dict[str, str]:
    """Validate the small non-scientific presentation preference contract."""

    if value is None:
        return dict(_OUTPUT_PREFERENCE_DEFAULTS)
    if not isinstance(value, dict):
        raise TypeError("output_preferences must be an object")
    allowed = {
        "layout": {"auto", "plain", "table"},
        "file_content": {"auto", "show", "link_only"},
        "detail": {"brief", "normal", "full"},
    }
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown output preference(s): {unknown}")
    normalized = dict(_OUTPUT_PREFERENCE_DEFAULTS)
    for name, options in allowed.items():
        if name not in value:
            continue
        item = value[name]
        if type(item) is not str or item not in options:
            raise ValueError(f"invalid output preference {name!r}: {item!r}")
        normalized[name] = item
    return normalized


class ResultTarget(StrictModel):
    """A value or output port requested from one logical step.

    This is a nested value contract, not another persisted runtime object.  The
    optional ``step_id`` keeps old M0 plans readable while M1 plans identify the
    producer explicitly.
    """

    step_id: str | None = None
    field: str | None = None
    port: str | None = None
    check: str | None = None

    @model_validator(mode="after")
    def _one_target_kind(self) -> ResultTarget:
        if sum(value is not None for value in (self.field, self.port, self.check)) != 1:
            raise ValueError("result target must contain exactly one field, port, or check")
        return self


class RequiredGeometryBinding(StrictModel):
    """A user-required source for one calculation's geometry input.

    This is a nested Request value contract, not a durable runtime object.
    ``source_operation=None`` with ``source_port='initial_geometry'`` means
    retain the original input geometry; otherwise the named operation and
    output port identify the required upstream source.
    """

    consumer_operation: Operation | None = None
    consumer_tool: str | None = None
    input_port: str
    source_operation: Operation | None
    source_port: str

    @model_validator(mode="after")
    def _one_consumer_selector(self) -> RequiredGeometryBinding:
        if (self.consumer_operation is None) == (self.consumer_tool is None):
            raise ValueError(
                "geometry binding must identify exactly one consumer_operation or consumer_tool"
            )
        return self


_REQUIRED_GEOMETRY_BINDINGS = TypeAdapter(list[RequiredGeometryBinding])


class Request(StrictModel):
    id: str
    description: str
    operations: list[Operation] = Field(default_factory=list)
    requested_results: list[ResultTarget] = Field(default_factory=list)
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["cli", "chat"] = "cli"
    original_text: str | None = None
    user_modifications: dict[str, Any] = Field(default_factory=dict)
    structure_input: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    output_preferences: dict[str, str] = Field(
        default_factory=lambda: dict(_OUTPUT_PREFERENCE_DEFAULTS)
    )

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_operation(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "operation" not in value:
            return value
        data = dict(value)
        legacy = data.pop("operation")
        if legacy is None:
            return data
        if "operations" in data and data["operations"] != [legacy]:
            raise ValueError("legacy operation conflicts with operations")
        data["operations"] = [legacy]
        return data

    @field_validator("operations")
    @classmethod
    def _unique_operations(cls, value: list[Operation]) -> list[Operation]:
        if len(set(value)) != len(value):
            raise ValueError("requested operations must be unique")
        return value

    @field_validator("requested_results", mode="before")
    @classmethod
    def _load_legacy_result_targets(cls, value: Any) -> Any:
        if value is None:
            return []
        return [{"field": item} if isinstance(item, str) else item for item in value]

    @field_validator("structure_input")
    @classmethod
    def _validate_required_geometry_bindings(cls, value: dict[str, Any]) -> dict[str, Any]:
        raw_bindings = value.get("required_bindings")
        if raw_bindings is None:
            return value
        bindings = _REQUIRED_GEOMETRY_BINDINGS.validate_python(raw_bindings, strict=True)
        identities = [
            (
                "operation",
                item.consumer_operation,
                item.input_port,
            )
            if item.consumer_operation is not None
            else ("tool", item.consumer_tool, item.input_port)
            for item in bindings
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("required geometry bindings must be unique per calculation input")
        return value

    @field_validator("output_preferences", mode="before")
    @classmethod
    def _validate_output_preferences(cls, value: Any) -> dict[str, str]:
        return validate_output_preferences(value)

    @property
    def operation(self) -> Operation | None:
        """Read-only compatibility for old single-operation callers."""

        return self.operations[0] if len(self.operations) == 1 else None


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


class GoalCheckRequirement(StrictModel):
    """A source Tool check that must have a declared status before this Step runs."""

    source_step_id: str
    check: str
    required_status: ScientificCheckStatus = "passed"


class ScientificCheckResult(StrictModel):
    """Program-computed evidence for a named scientific goal check."""

    status: ScientificCheckStatus
    input_geometry_sha256: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class Step(StrictModel):
    id: str
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputReference] = Field(default_factory=dict)
    goal_checks: list[GoalCheckRequirement] = Field(default_factory=list)
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
    scientific_checks: dict[str, ScientificCheckResult] = Field(default_factory=dict)
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
    parameter_sources_by_step: dict[str, dict[str, str]] = Field(default_factory=dict)
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
ParameterValidationFunction = Callable[[dict[str, Any], Mapping[str, Any]], None]
ResultProperty = StrictStr


class Tool(StrictModel):
    name: str
    description: str
    operations: list[Operation] = Field(default_factory=list)
    parameter_model: str = "none"
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    parameter_type: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)
    input_ports: dict[str, str] = Field(default_factory=dict)
    output_ports: dict[str, str] = Field(default_factory=dict)
    results: dict[str, str] = Field(default_factory=dict)
    result_properties: dict[str, ResultProperty] = Field(default_factory=dict)
    result_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)
    success_conditions: list[str] = Field(default_factory=list)
    scientific_checks: dict[str, str] = Field(default_factory=dict)
    repair_capabilities: list[str] = Field(default_factory=list)
    requires_compute_permission: bool = True
    parameter_preparation: Literal["none", "orca_electronic_state"] = "none"
    execution_budget: Literal["none", "orca"] = "none"
    deferred_parameters: list[str] = Field(default_factory=list)
    request_parameters: list[str] = Field(default_factory=list)
    geometry_output_input_ports: dict[str, str] = Field(default_factory=dict)
    available: bool = True
    execute_function: ExecuteFunction | None = Field(default=None, exclude=True, repr=False)
    parameter_validation_function: ParameterValidationFunction | None = Field(
        default=None, exclude=True, repr=False
    )
    repair_capabilities_function: Callable[[dict[str, Any]], list[str]] | None = Field(
        default=None, exclude=True, repr=False
    )

    model_config = ConfigDict(
        extra="forbid", strict=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    @model_validator(mode="after")
    def _validate_result_contract(self) -> Tool:
        """Keep presentation and semantic metadata attached to declared outputs.

        Presentation metadata is descriptive only. Result-property identifiers
        are machine-readable; neither mapping can add an undeclared result.
        """

        declared = set(self.results) | set(self.output_ports) | set(self.scientific_checks)
        undeclared_properties = sorted(set(self.result_properties) - declared)
        if undeclared_properties:
            raise ValueError(f"result properties have undeclared keys: {undeclared_properties}")
        unknown = sorted(set(self.result_metadata) - declared)
        if unknown:
            raise ValueError(f"result metadata has undeclared keys: {unknown}")
        from bg6022.output_contracts import validate_declared_output, validate_declared_type

        for kind, outputs in (("field", self.results), ("port", self.output_ports)):
            for name, expected_type in outputs.items():
                validate_declared_type(name, expected_type, kind=kind)
        for name in self.scientific_checks:
            validate_declared_type(name, "scientific_check", kind="check")

        for name, property_name in self.result_properties.items():
            if name in self.scientific_checks:
                expected_type = "scientific_check"
                kind = "check"
            elif name in self.output_ports:
                expected_type = self.output_ports[name]
                kind = "port"
            else:
                expected_type = self.results.get(name)
                kind = "field"
            if expected_type is None:
                continue
            validate_declared_output(name, expected_type, property_name, kind=kind)
        allowed = {"label", "description", "caveat"}
        for name, metadata in self.result_metadata.items():
            extra = sorted(set(metadata) - allowed)
            if extra:
                raise ValueError(f"result metadata for {name!r} has unsupported keys: {extra}")
        unknown_geometry_outputs = sorted(
            set(self.geometry_output_input_ports) - set(self.output_ports)
        )
        if unknown_geometry_outputs:
            raise ValueError(
                f"geometry output declarations have undeclared ports: {unknown_geometry_outputs}"
            )
        unknown_geometry_inputs = sorted(
            set(self.geometry_output_input_ports.values()) - set(self.input_ports)
        )
        if unknown_geometry_inputs:
            raise ValueError(
                f"geometry output declarations have undeclared inputs: {unknown_geometry_inputs}"
            )
        incompatible_geometry_bindings = sorted(
            output
            for output, input_name in self.geometry_output_input_ports.items()
            if self.output_ports[output] != "molecular_geometry"
            or self.input_ports[input_name] != "molecular_geometry"
        )
        if incompatible_geometry_bindings:
            raise ValueError(
                "geometry output declarations must connect molecular-geometry ports: "
                f"{incompatible_geometry_bindings}"
            )
        if len(self.request_parameters) != len(set(self.request_parameters)):
            raise ValueError("request_parameters must not contain duplicates")
        if self.request_parameters and self.parameter_type is None:
            raise ValueError("request_parameters require a parameter model")
        if self.parameter_type is not None:
            unknown_request_parameters = sorted(
                set(self.request_parameters) - set(self.parameter_type.model_fields)
            )
            if unknown_request_parameters:
                raise ValueError(
                    "request_parameters are not fields of the parameter model: "
                    f"{unknown_request_parameters}"
                )
        # Resolve the canonical public directory during registration.  This
        # catches property collisions (including a collision with a check)
        # before a Tool can be exposed to Intake or query handling.
        from bg6022.output_contracts import canonical_public_outputs

        canonical_public_outputs(self)
        return self

    def execute(self, step: Step, run: Run, *, cancel: Any) -> Result:
        if self.execute_function is None:
            raise RuntimeError(f"tool {self.name!r} has no executable implementation")
        return self.execute_function(step, run, cancel)

    def validate_parameters(
        self,
        parameters: dict[str, Any],
        *,
        allow_deferred: bool = False,
        context: Mapping[str, Any] | None = None,
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
        validated = self.parameter_type.model_validate(parameters, strict=True).model_dump(
            mode="python", exclude_none=True
        )
        if self.parameter_validation_function is not None:
            self.parameter_validation_function(validated, context or {})
        return validated

    def applicable_repair_capabilities(self, parameters: dict[str, Any]) -> list[str]:
        """Return repair actions permitted for this Tool and its parameters."""

        if self.repair_capabilities_function is None:
            selected = list(self.repair_capabilities)
        else:
            selected = list(self.repair_capabilities_function(dict(parameters)))
        if not set(selected) <= set(self.repair_capabilities):
            raise ValueError("parameter adapter cannot expand the Tool repair capabilities")
        return selected

    def description_json(self) -> dict[str, Any]:
        description = self.model_dump(mode="json", exclude={"execute_function"})
        description["public_outputs"] = self.public_outputs()
        return description

    def public_outputs(self) -> list[dict[str, Any]]:
        """Return the Tool's single canonical public output directory."""

        from bg6022.output_contracts import canonical_public_outputs

        return canonical_public_outputs(self)


__all__ = [
    "Artifact",
    "InputReference",
    "Plan",
    "Request",
    "Result",
    "ResultTarget",
    "ResultProperty",
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
