"""The seven durable runtime objects used by the v3 execution foundation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
    requirement_id: str | None = None
    field: str | None = None
    port: str | None = None
    check: str | None = None

    @model_validator(mode="after")
    def _one_target_kind(self) -> ResultTarget:
        if sum(value is not None for value in (self.field, self.port, self.check)) != 1:
            raise ValueError("result target must contain exactly one field, port, or check")
        return self


class Requirement(StrictModel):
    """One user-requested capability instance, nested inside Request."""

    id: str
    subject_id: str = "subject_1"
    capability: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "subject_id", "capability")
    @classmethod
    def _nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("requirement identity fields must not be blank")
        return value

    @field_validator("outputs")
    @classmethod
    def _unique_outputs(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("requirement outputs must be unique")
        return value


class RequiredGeometryBinding(StrictModel):
    """A user-required source for one calculation's geometry input.

    This is a nested Request value contract, not a durable runtime object.
    ``source_operation=None`` with ``source_port='initial_geometry'`` means
    retain the original input geometry; otherwise the named operation and
    output port identify the required upstream source.
    """

    consumer_operation: Operation | None = None
    consumer_tool: str | None = None
    consumer_requirement_id: str | None = None
    consumer_requirement_key: str | None = None
    input_port: str
    source_operation: Operation | None = None
    source_requirement_id: str | None = None
    source_requirement_key: str | None = None
    source_port: str

    @model_validator(mode="after")
    def _one_consumer_selector(self) -> RequiredGeometryBinding:
        if (
            sum(
                value is not None
                for value in (
                    self.consumer_operation,
                    self.consumer_tool,
                    self.consumer_requirement_id,
                    self.consumer_requirement_key,
                )
            )
            != 1
        ):
            raise ValueError(
                "geometry binding must identify one consumer operation, Tool, or requirement"
            )
        if (
            sum(
                value is not None
                for value in (
                    self.source_operation,
                    self.source_requirement_id,
                    self.source_requirement_key,
                )
            )
            > 1
        ):
            raise ValueError(
                "geometry binding must select at most one source operation or requirement"
            )
        if (
            self.source_operation is None
            and self.source_requirement_id is None
            and self.source_requirement_key is None
            and self.source_port != "initial_geometry"
        ):
            raise ValueError("an unselected geometry source must use initial_geometry")
        return self


_REQUIRED_GEOMETRY_BINDINGS = TypeAdapter(list[RequiredGeometryBinding])


class Request(StrictModel):
    id: str
    description: str
    operations: list[Operation] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    subjects: dict[str, dict[str, Any]] = Field(default_factory=dict)
    requested_results: list[ResultTarget] = Field(default_factory=list)
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["cli", "chat"] = "cli"
    original_text: str | None = None
    user_modifications: dict[str, Any] = Field(default_factory=dict)
    user_modifications_by_requirement: dict[str, dict[str, Any]] = Field(default_factory=dict)
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

    @field_validator("requirements")
    @classmethod
    def _unique_requirement_ids(cls, value: list[Requirement]) -> list[Requirement]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement ids must be unique")
        return value

    @model_validator(mode="after")
    def _requirement_subjects_are_known(self) -> Request:
        known_subjects = set(self.subjects)
        missing = sorted(
            {
                item.subject_id
                for item in self.requirements
                if known_subjects and item.subject_id not in known_subjects
            }
        )
        if missing:
            raise ValueError(f"requirements refer to unknown subject ids: {missing}")
        unknown_modification_scopes = sorted(
            set(self.user_modifications_by_requirement) - {item.id for item in self.requirements}
        )
        if unknown_modification_scopes:
            raise ValueError(
                "user modifications refer to unknown requirement ids: "
                f"{unknown_modification_scopes}"
            )
        unknown_result_scopes = sorted(
            {
                target.requirement_id
                for target in self.requested_results
                if target.requirement_id is not None
            }
            - {item.id for item in self.requirements}
        )
        if unknown_result_scopes:
            raise ValueError(f"results refer to unknown requirement ids: {unknown_result_scopes}")
        return self

    @field_validator("requested_results", mode="before")
    @classmethod
    def _load_legacy_result_targets(cls, value: Any) -> Any:
        if value is None:
            return []
        return [{"field": item} if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def _validate_required_geometry_bindings(self) -> Request:
        value = self.structure_input
        identity = value.get("molecule_identity")
        if identity is not None:
            from bg6022.molecule_identity import validate_identity_constraint

            validate_identity_constraint(identity)
        raw_bindings = value.get("required_bindings")
        if raw_bindings is None:
            return self
        bindings = _REQUIRED_GEOMETRY_BINDINGS.validate_python(raw_bindings, strict=True)
        requirement_ids = {item.id for item in self.requirements}
        for binding in bindings:
            if (
                binding.consumer_requirement_key is not None
                or binding.source_requirement_key is not None
            ):
                raise ValueError("Request geometry bindings must use program requirement ids")
            for requirement_id in (
                binding.consumer_requirement_id,
                binding.source_requirement_id,
            ):
                if requirement_id is not None and requirement_id not in requirement_ids:
                    raise ValueError(
                        f"geometry binding refers to unknown requirement id {requirement_id!r}"
                    )
        identities = [
            (
                "operation",
                item.consumer_operation,
                item.input_port,
            )
            if item.consumer_operation is not None
            else ("tool", item.consumer_tool, item.input_port)
            if item.consumer_tool is not None
            else (
                "requirement",
                item.consumer_requirement_id,
                item.input_port,
            )
            for item in bindings
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("required geometry bindings must be unique per calculation input")
        return self

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
    input_artifact_sha256_by_port: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class Step(StrictModel):
    id: str
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputReference] = Field(default_factory=dict)
    goal_checks: list[GoalCheckRequirement] = Field(default_factory=list)
    requirement_id: str | None = None
    subject_id: str | None = None
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
    extra_executions_by_category: dict[str, int] = Field(default_factory=dict)
    plan_revisions: int = 0
    origin_step_map: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None
    active_seconds: float = 0.0
    created_at: str
    updated_at: str

    _active_interval_started_at: float | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_execution_budget(cls, value: Any) -> Any:
        """Read pre-category Run budgets into the generic Tool category contract."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        budget = dict(data.get("budget", {}))
        category_limits = dict(budget.get("max_extra_executions_by_category", {}))
        legacy_limit = budget.get("max_extra_orca_executions")
        legacy_limit = category_limits.pop("orca", legacy_limit)
        if legacy_limit is not None:
            category_limits.setdefault("electronic_structure", legacy_limit)
        if category_limits:
            budget["max_extra_executions_by_category"] = category_limits
        budget.pop("max_extra_orca_executions", None)
        data["budget"] = budget
        execution_counts = dict(data.get("extra_executions_by_category", {}))
        legacy_count = execution_counts.pop("orca", data.get("extra_orca_executions", 0))
        if legacy_count:
            execution_counts.setdefault("electronic_structure", legacy_count)
        data["extra_executions_by_category"] = execution_counts
        return data

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
ResultValidationFunction = Callable[[Run, Step, Result], bool]
ResultProperty = StrictStr


@dataclass(frozen=True)
class ToolPreparation:
    """Transient result from a Tool's optional parameter preparation hook."""

    step: Step | None
    missing_fields: tuple[str, ...] = ()
    parameter_sources: Mapping[str, str] = dataclass_field(default_factory=dict)
    question: str | None = None


@dataclass(frozen=True)
class RepairOption:
    """A bounded, evidence-backed action offered by one Tool adapter."""

    action: str
    failed_step_id: str
    candidate_artifact_id: str | None
    parameter_patch: dict[str, Any]
    evidence_refs: tuple[str, ...]
    reason: str

    @property
    def option_id(self) -> str:
        return self.action

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.parameter_patch

    @property
    def input_aliases(self) -> Mapping[str, str]:
        if self.candidate_artifact_id is None:
            return {}
        return {"last_complete_geometry": self.candidate_artifact_id}

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "parameters": dict(self.parameter_patch),
            "input_aliases": list(self.input_aliases),
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


class Tool(StrictModel):
    name: str
    description: str
    display_name: str | None = None
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
    scientific_check_input_ports: dict[str, str] = Field(default_factory=dict)
    result_check_prerequisites: dict[str, list[str]] = Field(default_factory=dict)
    repair_capabilities: list[str] = Field(default_factory=list)
    repair_parameter_fields: dict[str, list[str]] = Field(default_factory=dict)
    repair_parameter_limits: dict[str, dict[str, int]] = Field(default_factory=dict)
    repair_input_aliases: dict[str, list[str]] = Field(default_factory=dict)
    requires_compute_permission: bool = True
    execution_budget: str = "none"
    deferred_parameters: list[str] = Field(default_factory=list)
    request_parameters: list[str] = Field(default_factory=list)
    geometry_output_input_ports: dict[str, str] = Field(default_factory=dict)
    available: bool = True
    execute_function: ExecuteFunction | None = Field(default=None, exclude=True, repr=False)
    preflight_function: Callable[[], None] | None = Field(default=None, exclude=True, repr=False)
    preparation_function: Callable[[Step, Mapping[str, Any]], ToolPreparation] | None = Field(
        default=None, exclude=True, repr=False
    )
    repair_options_function: Callable[[Run, Step, Result], list[RepairOption]] | None = Field(
        default=None, exclude=True, repr=False
    )
    apply_repair_function: (
        Callable[[RepairOption, Run, Step, Result, Mapping[str, Any]], tuple[Step, dict[str, Any]]]
        | None
    ) = Field(default=None, exclude=True, repr=False)
    parameter_validation_function: ParameterValidationFunction | None = Field(
        default=None, exclude=True, repr=False
    )
    result_validation_function: ResultValidationFunction | None = Field(
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
        if self.display_name is not None and not self.display_name.strip():
            raise ValueError("Tool display_name must be nonempty when supplied")
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
        undeclared_prerequisite_outputs = sorted(
            set(self.result_check_prerequisites) - (set(self.results) | set(self.output_ports))
        )
        if undeclared_prerequisite_outputs:
            raise ValueError(
                "result check prerequisites refer to undeclared outputs: "
                f"{undeclared_prerequisite_outputs}"
            )
        for output, checks in self.result_check_prerequisites.items():
            if not checks or len(checks) != len(set(checks)):
                raise ValueError(
                    f"result check prerequisites for {output!r} must be nonempty and unique"
                )

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
        unknown_check_bindings = sorted(
            set(self.scientific_check_input_ports) - set(self.scientific_checks)
        )
        if unknown_check_bindings:
            raise ValueError(
                "scientific check input bindings reference undeclared checks: "
                f"{unknown_check_bindings}"
            )
        unknown_check_inputs = sorted(
            set(self.scientific_check_input_ports.values()) - set(self.input_ports)
        )
        if unknown_check_inputs:
            raise ValueError(
                "scientific check input bindings reference undeclared inputs: "
                f"{unknown_check_inputs}"
            )
        if len(self.request_parameters) != len(set(self.request_parameters)):
            raise ValueError("request_parameters must not contain duplicates")
        if self.request_parameters and self.parameter_type is None:
            raise ValueError("request_parameters require a parameter model")
        if self.execution_budget != self.execution_budget.strip():
            raise ValueError("execution budget category must not contain surrounding whitespace")
        if self.parameter_type is not None:
            unknown_request_parameters = sorted(
                set(self.request_parameters) - set(self.parameter_type.model_fields)
            )
            if unknown_request_parameters:
                raise ValueError(
                    "request_parameters are not fields of the parameter model: "
                    f"{unknown_request_parameters}"
                )
        if len(self.repair_capabilities) != len(set(self.repair_capabilities)):
            raise ValueError("repair_capabilities must not contain duplicates")
        if set(self.repair_parameter_fields) - set(self.repair_capabilities):
            raise ValueError("repair parameter fields must belong to a declared capability")
        if set(self.repair_parameter_limits) - set(self.repair_capabilities):
            raise ValueError("repair parameter limits must belong to a declared capability")
        if set(self.repair_input_aliases) - set(self.repair_capabilities):
            raise ValueError("repair input aliases must belong to a declared capability")
        if self.parameter_type is not None:
            declared_fields = set(self.parameter_type.model_fields)
            for action, fields in self.repair_parameter_fields.items():
                if len(fields) != len(set(fields)) or set(fields) - declared_fields:
                    raise ValueError(
                        f"repair capability {action!r} refers to invalid parameter fields"
                    )
                limits = self.repair_parameter_limits.get(action, {})
                invalid_limit = any(
                    type(limit) is not int or limit < 1 for limit in limits.values()
                )
                if set(limits) - set(fields) or invalid_limit:
                    raise ValueError(f"repair capability {action!r} has invalid parameter limits")
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

    def preflight(self) -> None:
        if self.preflight_function is not None:
            self.preflight_function()

    def prepare(self, step: Step, context: Mapping[str, Any]) -> ToolPreparation:
        if self.preparation_function is None:
            return ToolPreparation(step=step)
        prepared = self.preparation_function(step, context)
        if not isinstance(prepared, ToolPreparation):
            raise TypeError(f"tool {self.name!r} returned an invalid preparation result")
        return prepared

    def repair_options(self, run: Run, step: Step, result: Result) -> list[RepairOption]:
        if self.repair_options_function is None:
            return []
        return list(self.repair_options_function(run, step, result))

    def apply_repair(
        self,
        option: RepairOption,
        run: Run,
        step: Step,
        result: Result,
        proposal: Mapping[str, Any],
    ) -> tuple[Step, dict[str, Any]]:
        if self.apply_repair_function is None:
            raise ValueError(f"tool {self.name!r} does not support repairs")
        return self.apply_repair_function(option, run, step, result, proposal)

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

    def validate_parameter_patch(
        self,
        parameters: dict[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate supplied values without requiring the rest of a Tool's schema."""

        if self.parameter_type is None:
            if parameters:
                raise ValueError(f"tool {self.name!r} does not accept parameters")
            return {}
        validated = _validate_partial_model(self.parameter_type, parameters)
        if self.parameter_validation_function is not None and set(self.request_parameters) <= set(
            validated
        ):
            self.parameter_validation_function(validated, context or {})
        return validated

    def validate_result(self, run: Run, step: Step, result: Result) -> bool:
        """Apply a Tool-local verified-result check when the Tool declares one."""

        if self.result_validation_function is None:
            return True
        return self.result_validation_function(run, step, result) is True

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
