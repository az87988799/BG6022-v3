"""Credential-free snapshots of existing runtime objects."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from bg6022.models import Result, Run
from bg6022.tools.registry import ToolRegistry, build_registry

from .models import CaseObservation


def observation_from_runtime(
    *,
    case_id: str,
    run_index: int,
    status: str,
    stage: str,
    intake: Any = None,
    request: Any = None,
    plan: Any = None,
    run: Run | None = None,
    results: list[Result] | None = None,
    response_text: str | None = None,
    llm_calls: list[Any] | None = None,
    llm_structured_outputs: list[dict[str, Any]] | None = None,
    elapsed_seconds: float = 0.0,
    error_category: str | None = None,
    error_message: str | None = None,
    error_diagnostics: list[dict[str, str]] | None = None,
    registry: ToolRegistry | None = None,
) -> CaseObservation:
    run_dump = _dump(run) if run is not None else None
    result_dumps = [_dump(item) for item in (results or [])]
    artifacts = run_dump.get("artifact_index", []) if run_dump else []
    count, successes, failures = orca_attempt_outcomes(run, registry=registry)
    repair_attempts = len(run.repair_records) if run is not None else 0
    return CaseObservation(
        case_id=case_id,
        run_index=run_index,
        status=status,
        stage=stage,
        intake=_dump(intake) if intake is not None else None,
        request=_dump(request) if request is not None else None,
        plan=_dump(plan) if plan is not None else None,
        run=run_dump,
        results=result_dumps,
        artifacts=artifacts,
        response_text=response_text,
        llm_calls=[_dump(item) for item in (llm_calls or [])],
        llm_structured_outputs=[dict(item) for item in (llm_structured_outputs or [])],
        orca_attempts=count,
        orca_successes=successes,
        orca_failures=failures,
        repair_attempts=repair_attempts,
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
        error_category=error_category,
        error_message=(error_message[:1000] if error_message else None),
        error_diagnostics=[dict(item) for item in (error_diagnostics or [])],
    )


def count_orca_attempts(run: Run | None, registry: ToolRegistry | None) -> int:
    return orca_attempt_outcomes(run, registry=registry)[0]


def orca_attempt_outcomes(run: Run | None, registry: ToolRegistry | None) -> tuple[int, int, int]:
    if run is None:
        return 0, 0, 0
    if registry is None:
        registry = build_registry()
    steps = {step.id: step for step in run.plan.steps}
    total = 0
    successes = 0
    failures = 0
    for record in run.attempts:
        step = steps.get(str(record.get("step_id", "")))
        if step is None:
            continue
        budget = registry.get(step.tool).execution_budget
        if budget == "electronic_structure":
            total += 1
            if record.get("status") == "succeeded":
                successes += 1
            elif record.get("status") in {"failed", "cancelled", "interrupted"}:
                failures += 1
    return total, successes, failures


_LLM_PURPOSES = ("semantic", "intake", "planner", "repair", "answer")


def summarize_llm_calls(calls: list[Any]) -> dict[str, Any]:
    summary = {
        "calls": len(calls),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "schema_corrections": 0,
        "by_purpose": {purpose: _empty_call_summary() for purpose in _LLM_PURPOSES},
    }
    for call in calls:
        raw = _dump(call)
        usage = raw.get("usage", {})
        input_tokens = _int_token(usage, "prompt_tokens", "input_tokens")
        output_tokens = _int_token(usage, "completion_tokens", "output_tokens")
        total = _int_token(usage, "total_tokens")
        if not total:
            total = input_tokens + output_tokens
        corrections = raw.get("structured_correction_count", 0)
        if type(corrections) is not int or corrections < 0:
            corrections = 0
        summary["input_tokens"] += input_tokens
        summary["output_tokens"] += output_tokens
        summary["total_tokens"] += total
        summary["schema_corrections"] += corrections
        purpose = str(raw.get("purpose") or "other")
        purpose_summary = summary["by_purpose"].setdefault(purpose, _empty_call_summary())
        purpose_summary["calls"] += 1
        purpose_summary["input_tokens"] += input_tokens
        purpose_summary["output_tokens"] += output_tokens
        purpose_summary["total_tokens"] += total
        purpose_summary["schema_corrections"] += corrections
    return summary


def _empty_call_summary() -> dict[str, int]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "schema_corrections": 0,
    }


def _int_token(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if type(value) is int and value > 0:
            return value
    return 0


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value


__all__ = [
    "count_orca_attempts",
    "observation_from_runtime",
    "orca_attempt_outcomes",
    "summarize_llm_calls",
]
