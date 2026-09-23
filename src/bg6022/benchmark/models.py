"""Small, strict data contracts for benchmark inputs and observations."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

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
    critical: bool = False

    @field_validator("critical", mode="before")
    @classmethod
    def _strict_critical(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("critical must be a boolean")
        return value


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
    orca_attempts: StrictInt = Field(default=0, ge=0)
    orca_successes: StrictInt = Field(default=0, ge=0)
    orca_failures: StrictInt = Field(default=0, ge=0)
    repair_attempts: StrictInt = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    error_category: StrictStr | None = None
    error_message: StrictStr | None = None
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


__all__ = [
    "AssertionResult",
    "AssertionType",
    "BenchmarkAssertion",
    "BenchmarkCase",
    "CaseObservation",
    "CaseResult",
]
