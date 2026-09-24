"""Small, strict data contracts for benchmark inputs and observations."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

BenchmarkCategory = Literal[
    "intake",
    "planning",
    "execution",
    "scientific",
    "query",
    "boundary",
    "repair",
    "extensibility",
]
BenchmarkSupport = Literal["supported", "boundary", "growth"]
BenchmarkMode = Literal["offline", "replay", "live_llm", "live_orca", "live_e2e"]
ObservationStatus = Literal["completed", "failed", "blocked", "exception"]
ObservationStage = Literal[
    "intake",
    "request_normalization",
    "planner",
    "plan_validation",
    "confirmation",
    "execution",
    "repair",
    "result_publish",
    "answer",
    "query",
    "complete",
]
AssertionType = Literal[
    "intent_equals",
    "subject_count",
    "requirement_count",
    "requirement_capabilities",
    "requirement_method_profiles",
    "requirement_outputs",
    "answer_goal",
    "tool_present",
    "forbid_tool",
    "step_count",
    "same_input_geometry",
    "input_from_requirement",
    "input_from_port",
    "input_from_tool_port",
    "result_target_present",
    "result_status",
    "scientific_check",
    "artifact_type_present",
    "no_new_attempt",
    "max_orca_attempts",
    "repair_attempt_count",
    "repair_action",
    "error_category",
    "boundary_blocked",
    "schema_corrections_at_most",
]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class BenchmarkAssertion(BenchmarkModel):
    """A bounded assertion. It deliberately has no expression or code field."""

    type: AssertionType
    value: Any = None
    step_id: StrictStr | None = None
    other_step_id: StrictStr | None = None
    requirement_id: StrictStr | None = None
    tool: StrictStr | None = None
    other_tool: StrictStr | None = None
    critical: bool = False

    @field_validator("critical", mode="before")
    @classmethod
    def _strict_critical(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("critical must be a boolean")
        return value

    @model_validator(mode="after")
    def _semantic_tool_selector(self) -> BenchmarkAssertion:
        if self.type == "input_from_tool_port" and (
            not self.tool or not self.other_tool or not isinstance(self.value, str)
        ):
            raise ValueError(
                "input_from_tool_port needs tool, other_tool, and a port name in value"
            )
        return self


class BenchmarkCase(BenchmarkModel):
    id: StrictStr
    category: BenchmarkCategory
    support: BenchmarkSupport
    mode: BenchmarkMode
    prompt: StrictStr
    repeat: StrictInt = Field(default=1, ge=1, le=5)
    fixture: StrictStr | None = None
    requires_llm: bool = False
    requires_orca: bool = False
    requires_pubchem: bool = False
    assertions: list[BenchmarkAssertion] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is None:
            raise ValueError("case id must be a simple identifier")
        return value

    @field_validator("prompt")
    @classmethod
    def _nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("case prompt must not be blank")
        return value

    @field_validator("requires_llm", "requires_orca", "requires_pubchem", mode="before")
    @classmethod
    def _strict_requirements(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("requires_* values must be booleans")
        return value


class LiveSetup(BenchmarkModel):
    """Finite, data-only initial state for a production Agent live scenario."""

    active_run_fixture: StrictStr | None = None
    active_run_status: Literal["waiting", "succeeded"] | None = None
    waiting_for: Literal["clarification", "confirmation"] | None = None
    pending_data: dict[str, Any] = Field(default_factory=dict)
    recent_messages: list[dict[StrictStr, StrictStr]] = Field(default_factory=list)
    recent_results: list[dict[str, Any]] = Field(default_factory=list)
    last_delivery: list[dict[str, Any]] = Field(default_factory=list)
    published_result_fixture: StrictStr | None = None

    @model_validator(mode="after")
    def _consistent_seed_state(self) -> LiveSetup:
        if self.active_run_status is None:
            if any(
                value is not None
                for value in (
                    self.active_run_fixture,
                    self.waiting_for,
                    self.published_result_fixture,
                )
            ):
                raise ValueError("active Run fields require active_run_status")
            return self
        if not self.active_run_fixture:
            raise ValueError("an active Run state requires active_run_fixture")
        if self.active_run_status == "waiting":
            if self.waiting_for is None or self.published_result_fixture is not None:
                raise ValueError(
                    "a waiting Run needs waiting_for and cannot seed a published Result"
                )
        elif self.waiting_for is not None or not self.published_result_fixture:
            raise ValueError("a succeeded Run needs published_result_fixture and cannot be waiting")
        return self


class LiveResultFixture(BenchmarkModel):
    """Bounded, previously verified Result evidence used to seed a query case."""

    evidence_class: Literal["real_orca_fixture"]
    tool: StrictStr
    output_geometry: StrictStr
    values: dict[str, Any]
    checks: dict[StrictStr, StrictBool]


class CaseObservation(BenchmarkModel):
    case_id: StrictStr
    run_index: StrictInt = Field(ge=1)
    status: ObservationStatus
    stage: ObservationStage
    intake: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    run: dict[str, Any] | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    response_text: StrictStr | None = None
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    llm_structured_outputs: list[dict[str, Any]] = Field(default_factory=list)
    orca_attempts: StrictInt = Field(default=0, ge=0)
    orca_successes: StrictInt = Field(default=0, ge=0)
    orca_failures: StrictInt = Field(default=0, ge=0)
    repair_attempts: StrictInt = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    error_category: StrictStr | None = None
    error_message: StrictStr | None = None
    error_diagnostics: list[dict[StrictStr, StrictStr]] = Field(default_factory=list)
    critical_violations: list[StrictStr] = Field(default_factory=list)
    unrequested_compute: bool = False
    unexpected_recomputation: bool = False


class AssertionResult(BenchmarkModel):
    assertion_type: StrictStr
    passed: bool
    expected: Any = None
    observed: Any = None
    detail: StrictStr | None = None
    critical: bool = False


class CaseResult(BenchmarkModel):
    case_id: StrictStr
    category: BenchmarkCategory
    support: BenchmarkSupport
    mode: BenchmarkMode
    run_index: StrictInt
    passed: bool
    dimensions: dict[str, bool | None]
    failed_stage: ObservationStage | None = None
    assertions: list[AssertionResult]
    observation: CaseObservation


class Benchmark12TurnObservation(BenchmarkModel):
    """One public message/response pair observed by the full pipeline driver."""

    index: StrictInt = Field(ge=1)
    user_message: StrictStr
    response_text: StrictStr
    run_id: StrictStr | None = None
    run_status: StrictStr | None = None
    waiting_for: StrictStr | None = None
    llm_purposes: list[StrictStr] = Field(default_factory=list)
    orca_attempts_after_turn: StrictInt = Field(default=0, ge=0)
    delivery_status: StrictStr | None = None
    delivery_properties: list[StrictStr] = Field(default_factory=list)


class Benchmark12Observation(BenchmarkModel):
    """Read-only production evidence collected after a Benchmark 1.2 dialogue."""

    item_id: StrictStr
    run_index: StrictInt = Field(ge=1)
    turns: list[Benchmark12TurnObservation] = Field(default_factory=list)
    request: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    final_response_text: StrictStr | None = None
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    orca_attempts: StrictInt = Field(default=0, ge=0)
    orca_successes: StrictInt = Field(default=0, ge=0)
    orca_failures: StrictInt = Field(default=0, ge=0)
    repair_attempts: StrictInt = Field(default=0, ge=0)
    final_status: StrictStr
    failed_stage: StrictStr | None = None
    error_category: StrictStr | None = None
    pre_confirmation_orca_attempts: StrictInt = Field(default=0, ge=0)
    post_confirmation_orca_attempts: StrictInt = Field(default=0, ge=0)
    confirmation_turn_index: StrictInt | None = Field(default=None, ge=1)
    confirmation_required: bool = False
    public_delivery_properties: list[StrictStr] = Field(default_factory=list)
    delivery_status: StrictStr | None = None
    legacy_control_plane_used: bool = False
    external_calls: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0.0, ge=0)


class Benchmark12Result(BenchmarkModel):
    """Deterministic grade for one expanded Benchmark 1.2 item."""

    item_id: StrictStr
    task_id: StrictStr
    object_id: StrictStr
    variant_id: StrictStr
    run_index: StrictInt = Field(ge=1)
    passed: StrictBool
    dimensions: dict[StrictStr, bool | None]
    failed_stage: StrictStr | None = None
    error_category: StrictStr | None = None
    failures: list[StrictStr] = Field(default_factory=list)
    observation: Benchmark12Observation


__all__ = [
    "AssertionResult",
    "AssertionType",
    "BenchmarkAssertion",
    "BenchmarkCase",
    "Benchmark12Observation",
    "Benchmark12Result",
    "Benchmark12TurnObservation",
    "CaseObservation",
    "CaseResult",
    "LiveResultFixture",
    "LiveSetup",
]
