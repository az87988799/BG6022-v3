"""The seven durable runtime objects used by the v3 execution foundation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

RunStatus = Literal["planned", "running", "succeeded", "failed", "cancelled", "interrupted"]
ResultStatus = Literal["succeeded", "failed", "cancelled", "interrupted"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Request(StrictModel):
    id: str
    description: str
    requested_results: list[str] = Field(default_factory=list)
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["cli"] = "cli"


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


class Plan(StrictModel):
    id: str
    revision: int = 1
    request_id: str
    steps: list[Step]
    requested_results: list[str] = Field(default_factory=list)

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
    attempt_relative_path: str


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
    parameter_model: str
    parameter_schema: dict[str, Any]
    parameter_type: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)
    input_ports: dict[str, str] = Field(default_factory=dict)
    output_ports: dict[str, str] = Field(default_factory=dict)
    results: dict[str, str] = Field(default_factory=dict)
    success_conditions: list[str] = Field(default_factory=list)
    repair_capabilities: list[str] = Field(default_factory=list)
    execute_function: ExecuteFunction | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(
        extra="forbid", strict=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    def execute(self, step: Step, run: Run, *, cancel: Any) -> Result:
        if self.execute_function is None:
            raise RuntimeError(f"tool {self.name!r} has no executable implementation")
        return self.execute_function(step, run, cancel)

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if self.parameter_type is None:
            raise ValueError(f"tool {self.name!r} has no parameter model")
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
    "Run",
    "RunStatus",
    "Step",
    "Tool",
]
