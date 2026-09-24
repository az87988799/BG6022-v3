"""Persist inspection-ready benchmark cases, costs, environment, and reports."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bg6022.config import AppConfig
from bg6022.planner import load_prompt
from bg6022.tools.registry import build_registry

from . import BENCHMARK_VERSION
from .models import CaseResult
from .observers import summarize_llm_calls

_DIMENSIONS = (
    "intent_correct",
    "plan_correct",
    "scientific_correct",
    "execution_correct",
    "boundary_correct",
    "repair_success",
    "no_extra_compute",
    "answer_correct",
    "cost_within_bound",
)


def build_environment(
    *,
    repository_root: str | Path,
    config: AppConfig | None,
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    registry = build_registry(config)
    prompt_hashes: dict[str, str] = {}
    for name in ("intake", "planner", "repair", "answer"):
        prompt_hashes[name] = hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
    tool_catalog = registry.describe()
    method_catalog = registry.method_capability_catalog()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_dir": str(Path(benchmark_dir).resolve()),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "branch": _git_value(root, "branch", "--show-current"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": config.llm.model if config is not None else None,
        "resources": config.resources if config is not None else None,
        "prompt_sha256": prompt_hashes,
        "tool_catalog_sha256": _stable_hash(tool_catalog),
        "method_catalog_sha256": _stable_hash(method_catalog),
    }


def summarize_results(
    results: list[CaseResult],
    *,
    run_id: str,
    environment: dict[str, Any],
    skipped_cases: list[str] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    dimension_values: dict[str, list[bool]] = defaultdict(list)
    for case_result in results:
        for name, value in case_result.dimensions.items():
            if value is not None:
                dimension_values[name].append(value)
    dimensions = {
        name: {
            "passed": sum(values),
            "total": len(values),
            "rate": (sum(values) / len(values)) if values else None,
        }
        for name, values in dimension_values.items()
    }
    for name in _DIMENSIONS:
        dimensions.setdefault(name, {"passed": 0, "total": 0, "rate": None})

    llm = summarize_llm_calls([call for result in results for call in result.observation.llm_calls])
    total_orca = sum(result.observation.orca_attempts for result in results)
    orca_successes = sum(result.observation.orca_successes for result in results)
    orca_failures = sum(result.observation.orca_failures for result in results)
    repair_attempts = sum(result.observation.repair_attempts for result in results)
    violations = [
        {
            "case_id": result.case_id,
            "run_index": result.run_index,
            "code": code,
        }
        for result in results
        for code in result.observation.critical_violations
    ]
    critical_case_failures = [
        {"case_id": result.case_id, "run_index": result.run_index}
        for result in results
        if any(assertion.critical and not assertion.passed for assertion in result.assertions)
    ]
    case_count = len(results)
    unrequested_compute = sum(result.observation.unrequested_compute for result in results)
    unexpected_recomputation = sum(
        result.observation.unexpected_recomputation for result in results
    )
    by_mode = _group_pass_rates(results, lambda item: item.mode)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "run_id": run_id,
        "commit": environment.get("commit"),
        "branch": environment.get("branch"),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": case_count,
        "passed_cases": sum(result.passed for result in results),
        "failed_cases": sum(not result.passed for result in results),
        "skipped_cases": list(skipped_cases or []),
        "dimensions": dimensions,
        "by_mode": by_mode,
        "replay_pass_rate": by_mode.get("replay", {"passed": 0, "total": 0, "rate": None}),
        "live_llm_pass_rate": by_mode.get("live_llm", {"passed": 0, "total": 0, "rate": None}),
        "live_orca_pass_rate": by_mode.get("live_orca", {"passed": 0, "total": 0, "rate": None}),
        "live_e2e_pass_rate": by_mode.get("live_e2e", {"passed": 0, "total": 0, "rate": None}),
        "supported_task_success": _rate_for_support(results, "supported"),
        "boundary_accuracy": dimensions["boundary_correct"],
        "unrequested_compute": unrequested_compute,
        "unrequested_compute_rate": (unrequested_compute / case_count if case_count else None),
        "unexpected_recomputation": unexpected_recomputation,
        "unexpected_recomputation_rate": (
            unexpected_recomputation / case_count if case_count else None
        ),
        "critical_violations": violations,
        "critical_violation_count": len(violations),
        "critical_violation_count_deprecated": True,
        "critical_assertion_failures": violations,
        "critical_assertion_failure_count": len(violations),
        "critical_case_failures": critical_case_failures,
        "critical_case_failure_count": len(critical_case_failures),
        "execution_safety_violation_count": unrequested_compute + unexpected_recomputation,
        "cost": {
            **llm,
            "orca_attempts": total_orca,
            "orca_successes": orca_successes,
            "orca_failures": orca_failures,
            "repair_attempts": repair_attempts,
            "wall_time_seconds": (
                max(0.0, float(elapsed_seconds))
                if elapsed_seconds is not None
                else sum(result.observation.elapsed_seconds for result in results)
            ),
        },
        "environment": environment,
        "cases": [result.model_dump(mode="json") for result in results],
    }


def write_report(
    output_dir: str | Path,
    results: list[CaseResult],
    *,
    environment: dict[str, Any],
    skipped_cases: list[str] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = summarize_results(
        results,
        run_id=destination.name,
        environment=environment,
        skipped_cases=skipped_cases,
        elapsed_seconds=elapsed_seconds,
    )
    _write_json(destination / "environment.json", environment)
    _write_json(destination / "summary.json", summary)
    (destination / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    with (destination / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in results:
            handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
    failures = [item for item in results if not item.passed]
    if failures:
        failure_dir = destination / "failures"
        failure_dir.mkdir()
        for item in failures:
            _write_json(
                failure_dir / f"{item.case_id}_{item.run_index}.json", item.model_dump(mode="json")
            )
    return summary


def write_matrix_report(
    output_dir: str | Path,
    expanded_cases: list[Any],
    results: list[CaseResult],
) -> dict[str, Any]:
    """Write scientific matrix cell detail alongside the standard benchmark report."""

    destination = Path(output_dir).resolve()
    if not destination.is_dir():
        raise ValueError(f"benchmark report directory does not exist: {destination}")
    results_by_case: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        results_by_case[result.case_id].append(result)

    cells: list[dict[str, Any]] = []
    for expanded in expanded_cases:
        case_results = sorted(
            results_by_case.get(expanded.case.id, []), key=lambda item: item.run_index
        )
        failed = [item for item in case_results if not item.passed]
        cells.append(
            {
                "case_id": expanded.case.id,
                "object_id": expanded.object_id,
                "object_label": expanded.object_label,
                "task_id": expanded.task_id,
                "task_label": expanded.task_label,
                "cost_class": expanded.cost_class,
                "run_count": len(case_results),
                "passed_runs": sum(item.passed for item in case_results),
                "orca_attempts": sum(item.observation.orca_attempts for item in case_results),
                "wall_time_seconds": sum(item.observation.elapsed_seconds for item in case_results),
                "failed_stages": sorted(
                    {item.failed_stage for item in failed if item.failed_stage is not None}
                ),
                "error_categories": sorted(
                    {
                        item.observation.error_category
                        for item in failed
                        if item.observation.error_category is not None
                    }
                ),
            }
        )

    by_task = _aggregate_matrix_cells(cells, "task_id", "task_label")
    by_object = _aggregate_matrix_cells(cells, "object_id", "object_label")
    matrix_summary = {
        "schema": "bg6022.scientific_matrix.v1",
        "cell_count": len(cells),
        "run_count": sum(cell["run_count"] for cell in cells),
        "passed_runs": sum(cell["passed_runs"] for cell in cells),
        "orca_attempts": sum(cell["orca_attempts"] for cell in cells),
        "wall_time_seconds": sum(cell["wall_time_seconds"] for cell in cells),
        "cells": cells,
        "by_task": by_task,
        "by_object": by_object,
    }
    _write_json(destination / "matrix.json", matrix_summary)
    (destination / "matrix.md").write_text(render_matrix_markdown(matrix_summary), encoding="utf-8")
    return matrix_summary


def render_matrix_markdown(summary: dict[str, Any]) -> str:
    cells = summary["cells"]
    objects: list[tuple[str, str]] = []
    tasks: list[tuple[str, str]] = []
    for cell in cells:
        if (cell["object_id"], cell["object_label"]) not in objects:
            objects.append((cell["object_id"], cell["object_label"]))
        if (cell["task_id"], cell["task_label"]) not in tasks:
            tasks.append((cell["task_id"], cell["task_label"]))
    by_coordinate = {(cell["task_id"], cell["object_id"]): cell for cell in cells}
    lines = [
        "# Scientific Matrix v1",
        "",
        f"Enabled cells: {summary['cell_count']}",
        f"Runs passed: {summary['passed_runs']}/{summary['run_count']}",
        f"ORCA attempts: {summary['orca_attempts']}",
        f"Wall time: {summary['wall_time_seconds']:.2f}s",
        "",
        "## Task × Scientific Object",
        "",
        "| Task / Object | " + " | ".join(label for _, label in objects) + " |",
        "|---|" + "---|" * len(objects),
    ]
    for task_id, task_label in tasks:
        values = [
            _format_matrix_cell(by_coordinate.get((task_id, object_id))) for object_id, _ in objects
        ]
        lines.append(f"| {task_id} {task_label} | " + " | ".join(values) + " |")

    for heading, key in (("By Task", "by_task"), ("By Object", "by_object")):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Name | Cells | Passed runs | ORCA attempts | Wall time |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for group in summary[key]:
            name = f"{group['id']} {group['label']}"
            lines.append(
                f"| {name} | {group['cell_count']} | "
                f"{group['passed_runs']}/{group['run_count']} | "
                f"{group['orca_attempts']} | {group['wall_time_seconds']:.2f}s |"
            )

    failures = [cell for cell in cells if cell["passed_runs"] < cell["run_count"]]
    lines.extend(["", "## Failed cells", ""])
    if failures:
        lines.extend(
            [
                "| Cell | Passed runs | Failed stage | Error category |",
                "|---|---:|---|---|",
            ]
        )
        for cell in failures:
            stage = ", ".join(cell["failed_stages"]) or "unknown"
            category = ", ".join(cell["error_categories"]) or "unknown"
            lines.append(
                f"| {cell['case_id']} | {cell['passed_runs']}/{cell['run_count']} | "
                f"{stage} | {category} |"
            )
    else:
        lines.append("No failed cells.")
    return "\n".join(lines) + "\n"


def _aggregate_matrix_cells(
    cells: list[dict[str, Any]], id_key: str, label_key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(cell[id_key], []).append(cell)
    return [
        {
            "id": group_id,
            "label": grouped[0][label_key],
            "cell_count": len(grouped),
            "run_count": sum(cell["run_count"] for cell in grouped),
            "passed_runs": sum(cell["passed_runs"] for cell in grouped),
            "orca_attempts": sum(cell["orca_attempts"] for cell in grouped),
            "wall_time_seconds": sum(cell["wall_time_seconds"] for cell in grouped),
        }
        for group_id, grouped in groups.items()
    ]


def _format_matrix_cell(cell: dict[str, Any] | None) -> str:
    if cell is None:
        return "—"
    rendered = f"{cell['passed_runs']}/{cell['run_count']}"
    if cell["passed_runs"] < cell["run_count"]:
        stages = ", ".join(cell["failed_stages"]) or "unknown stage"
        categories = ", ".join(cell["error_categories"]) or "unknown error"
        rendered += f" ({stages}; {categories})"
    return rendered


def render_summary_markdown(summary: dict[str, Any]) -> str:
    dimensions = summary["dimensions"]
    cost = summary["cost"]
    lines = [
        f"# BG6022 Benchmark v{BENCHMARK_VERSION}",
        "",
        f"Commit: {summary.get('commit') or 'unknown'}",
        f"Branch: {summary.get('branch') or 'unknown'}",
        f"Model: {summary['environment'].get('model') or 'not used'}",
        f"Benchmark version: {summary['benchmark_version']}",
        "",
        "## Capability",
        "",
        _metric_line("Supported task success", summary.get("supported_task_success")),
        _metric_line("Intent accuracy", dimensions["intent_correct"]),
        _metric_line("Plan accuracy", dimensions["plan_correct"]),
        _metric_line("Scientific correctness", dimensions["scientific_correct"]),
        _metric_line("Execution correctness", dimensions["execution_correct"]),
        _metric_line("Boundary accuracy", dimensions["boundary_correct"]),
        _metric_line("Repair success", dimensions["repair_success"]),
        f"Unrequested compute: {summary['unrequested_compute']}",
        f"Unexpected recomputation: {summary['unexpected_recomputation']}",
        f"Critical assertion failures: {summary['critical_assertion_failure_count']}",
        f"Critical case failures: {summary['critical_case_failure_count']}",
        f"Execution safety violations: {summary['execution_safety_violation_count']}",
        "",
        "## Reliability by mode",
        "",
        _metric_line("Replay pass rate", summary.get("replay_pass_rate")),
        _metric_line("Live LLM pass rate", summary.get("live_llm_pass_rate")),
        _metric_line("Live ORCA pass rate", summary.get("live_orca_pass_rate")),
        _metric_line("Live end-to-end pass rate", summary.get("live_e2e_pass_rate")),
        "",
        "## Cost",
        "",
        f"LLM calls: {cost['calls']}",
        f"Input tokens: {cost['input_tokens']}",
        f"Output tokens: {cost['output_tokens']}",
        f"Total tokens: {cost['total_tokens']}",
        f"Schema corrections: {cost['schema_corrections']}",
        f"ORCA attempts: {cost['orca_attempts']}",
        f"ORCA successes: {cost['orca_successes']}",
        f"ORCA failures: {cost['orca_failures']}",
        f"Repair attempts: {cost['repair_attempts']}",
        f"Wall time: {cost['wall_time_seconds']:.2f}s",
        "",
        "### LLM calls by purpose",
        "",
        "| Purpose | Calls | Input tokens | Output tokens | Total tokens | Schema corrections |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for purpose, stats in cost.get("by_purpose", {}).items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                purpose,
                stats["calls"],
                stats["input_tokens"],
                stats["output_tokens"],
                stats["total_tokens"],
                stats["schema_corrections"],
            )
        )
    lines.extend(
        [
            "## Cases",
            "",
            f"Passed: {summary['passed_cases']}/{summary['case_count']}",
        ]
    )
    skipped = summary.get("skipped_cases", [])
    if skipped:
        lines.extend([f"Not selected: {len(skipped)} live/holdout cases", ""])
    failures = [item for item in summary["cases"] if not item["passed"]]
    diagnostic_cases = [
        item for item in summary["cases"] if item.get("observation", {}).get("error_diagnostics")
    ]
    if diagnostic_cases:
        lines.extend(["", "## LLM diagnostics", ""])
        for item in diagnostic_cases:
            observation = item["observation"]
            lines.append(f"### {item['case_id']} (run {item['run_index']})")
            lines.append("")
            for diagnostic in observation.get("error_diagnostics", []):
                path = diagnostic.get("path", "$")
                purpose = diagnostic.get("purpose", "llm")
                category = diagnostic.get("category", "error")
                message = diagnostic.get("message", "")
                lines.append(f"- `{path}` ({purpose}/{category}): {message or 'no detail'}")
            lines.append("")
    if failures:
        lines.extend(["", "## Failed cases", ""])
        for item in failures:
            lines.extend(
                [
                    f"### {item['case_id']} (run {item['run_index']})",
                    "",
                    f"Stage: {item.get('failed_stage') or item['observation']['stage']}",
                    f"Mode: {item['mode']}",
                    "",
                ]
            )
            for assertion in item["assertions"]:
                if not assertion["passed"]:
                    lines.append(
                        f"- `{assertion['assertion_type']}`: {assertion.get('detail') or 'failed'}"
                    )
            if item["observation"].get("error_message"):
                lines.append(f"- Error: {item['observation']['error_message']}")
            for diagnostic in item["observation"].get("error_diagnostics", []):
                lines.append(
                    "- Diagnostic `{}:{}`: {}".format(
                        diagnostic.get("purpose", "llm"),
                        diagnostic.get("path", diagnostic.get("category", "error")),
                        diagnostic.get("message", ""),
                    )
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _rate_for_support(results: list[CaseResult], support: str) -> dict[str, Any]:
    selected = [item for item in results if item.support == support]
    return {
        "passed": sum(item.passed for item in selected),
        "total": len(selected),
        "rate": sum(item.passed for item in selected) / len(selected) if selected else None,
    }


def _group_pass_rates(results: list[CaseResult], key_function) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[str(key_function(result))].append(result)
    return {
        name: {
            "passed": sum(item.passed for item in group),
            "total": len(group),
            "rate": sum(item.passed for item in group) / len(group) if group else None,
        }
        for name, group in sorted(groups.items())
    }


def _metric_line(label: str, metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("total", 0) == 0:
        return f"{label}: n/a"
    return f"{label}: {metric['passed']}/{metric['total']}"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "build_environment",
    "render_summary_markdown",
    "summarize_results",
    "write_report",
]
