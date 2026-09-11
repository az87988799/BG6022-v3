"""Deterministic scientific result rendering and bounded natural-language help."""

from __future__ import annotations

import json
from typing import Any

from bg6022.llm import LlmClient
from bg6022.models import Result, Run
from bg6022.planner import load_prompt


def render_result(run: Run, result: Result) -> str:
    """Render only values that passed the Tool's success contract."""

    if result.status != "succeeded":
        category = result.diagnostics.get("category", "unknown_failure")
        reason = result.diagnostics.get("reason") or "no verified scientific result"
        return f"Calculation did not complete successfully ({category}): {reason}."
    step = next((item for item in run.plan.steps if item.id == result.step_id), None)
    parameters = {} if step is None else step.parameters
    lines = [f"Run {run.id} succeeded for step {result.step_id}."]
    for name, value in result.values.items():
        if isinstance(value, dict) and "value" in value:
            lines.append(f"{name}: {value['value']} {value.get('unit', '')}".rstrip())
        else:
            lines.append(f"{name}: {value}")
    if parameters:
        lines.append(
            "Method/environment: "
            f"{parameters.get('method_profile', 'unknown')}/"
            f"{parameters.get('environment', 'unknown')}"
        )
        if "charge" in parameters:
            lines.append(
                f"Charge/multiplicity: {parameters.get('charge')}/{parameters.get('multiplicity')}"
            )
    sources = sorted(set(result.parameter_sources.values()))
    if sources:
        lines.append("Parameter sources: " + ", ".join(sources))
    structure_sources = _structure_sources(run, result)
    if structure_sources:
        lines.append("Structure source: " + ", ".join(structure_sources))
    if run.repair_records:
        lines.append(f"Bounded repair attempts: {len(run.repair_records)}.")
    if parameters.get("method_profile") and result.step_id and _is_opt_step(run, result.step_id):
        lines.append(
            "This is a local optimized electronic energy; frequency stability and "
            "global minimum were not verified."
        )
    lines.append(f"Result path: {result.attempt_relative_path}/result.json")
    return "\n".join(lines)


def render_run(run: Run, result: Result | None = None) -> str:
    if run.status == "succeeded" and result is not None:
        return render_result(run, result)
    if run.status == "failed":
        reason = run.pending_data.get("reason") if run.pending_data else None
        suffix = f": {reason}" if reason else "."
        if result is not None and result.status == "succeeded":
            return (
                f"Run {run.id} failed before all requested results were completed{suffix} "
                f"The last valid step was {result.step_id}; it is not the requested Run success."
            )
        return f"Run {run.id} failed{suffix}"
    if result is not None:
        return render_result(run, result)
    if run.status == "waiting":
        if run.waiting_for == "confirmation":
            return "The prepared calculation is waiting for confirmation."
        return "The run is waiting for additional information."
    return f"Run {run.id} is {run.status}."


def explain_result(
    client: LlmClient | None, *, run: Run, result: Result, cancel: Any = None
) -> str | None:
    """Optionally add prose; the deterministic result remains authoritative."""

    if client is None or result.status != "succeeded":
        return None
    try:
        return client.complete_text(
            [
                {"role": "system", "content": load_prompt("answer")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "deterministic_result": result.values,
                            "checks": result.checks,
                            "step": result.step_id,
                            "run_status": run.status,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            purpose="answer",
            cancel=cancel,
        )
    except Exception:
        return None


def context_answer(run: Run, result: Result | None, question: str) -> str:
    if result is None:
        return "No saved calculation result is available for that question."
    return render_run(run, result)


def _is_opt_step(run: Run, step_id: str) -> bool:
    step = next((item for item in run.plan.steps if item.id == step_id), None)
    return step is not None and step.tool == "optimize_geometry"


def _structure_sources(run: Run, result: Result) -> list[str]:
    values: list[str] = []
    for artifact_id in result.input_artifact_ids:
        artifact = next((item for item in run.artifact_index if item.id == artifact_id), None)
        if artifact is None:
            continue
        source = artifact.metadata.get("structure_source") or artifact.source
        if isinstance(source, str) and source and source not in values:
            values.append(source)
    return values


__all__ = ["context_answer", "explain_result", "render_result", "render_run"]
