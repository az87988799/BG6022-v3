"""Compare two immutable benchmark report directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BenchmarkReportError(ValueError):
    """A previous report cannot be compared safely."""


def compare_reports(baseline_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    baseline = _read_summary(baseline_dir)
    candidate = _read_summary(candidate_dir)
    if baseline.get("benchmark_version") != candidate.get("benchmark_version"):
        raise BenchmarkReportError(
            "benchmark versions differ; capability pass-rate comparison is not valid"
        )
    baseline_suite = baseline.get("suite_identity")
    candidate_suite = candidate.get("suite_identity")
    if (baseline_suite is None) != (candidate_suite is None) or (
        baseline_suite is not None and baseline_suite != candidate_suite
    ):
        raise BenchmarkReportError(
            "benchmark suite identities differ; capability pass-rate comparison is not valid"
        )
    before = _case_map(baseline)
    after = _case_map(candidate)
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        left = before.get(key)
        right = after.get(key)
        change = {"case_id": key[0], "run_index": key[1]}
        if left is None:
            added.append(change | {"candidate_passed": bool(right["passed"])})
        elif right is None:
            removed.append(change | {"baseline_passed": bool(left["passed"])})
        elif not left["passed"] and right["passed"]:
            improved.append(change | {"baseline": "FAIL", "candidate": "PASS"})
        elif left["passed"] and not right["passed"]:
            regressed.append(change | {"baseline": "PASS", "candidate": "FAIL"})

    baseline_cost = baseline.get("cost", {})
    candidate_cost = candidate.get("cost", {})
    cost_delta = {
        key: _numeric(candidate_cost.get(key)) - _numeric(baseline_cost.get(key))
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "calls",
            "schema_corrections",
            "orca_attempts",
            "wall_time_seconds",
        )
    }
    return {
        "baseline": {
            "run_id": baseline.get("run_id"),
            "benchmark_version": baseline.get("benchmark_version"),
            "commit": baseline.get("commit"),
            "passed_cases": baseline.get("passed_cases"),
            "case_count": baseline.get("case_count"),
        },
        "candidate": {
            "run_id": candidate.get("run_id"),
            "benchmark_version": candidate.get("benchmark_version"),
            "commit": candidate.get("commit"),
            "passed_cases": candidate.get("passed_cases"),
            "case_count": candidate.get("case_count"),
        },
        "improved": improved,
        "regressed": regressed,
        "added": added,
        "removed": removed,
        "cost_delta": cost_delta,
        "critical_violation_delta": int(candidate.get("critical_violation_count", 0))
        - int(baseline.get("critical_violation_count", 0)),
        "unrequested_compute_delta": int(candidate.get("unrequested_compute", 0))
        - int(baseline.get("unrequested_compute", 0)),
        "unexpected_recomputation_delta": int(candidate.get("unexpected_recomputation", 0))
        - int(baseline.get("unexpected_recomputation", 0)),
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = [
        "# BG6022 Benchmark Comparison",
        "",
        f"Baseline: {baseline.get('commit') or baseline.get('run_id') or 'unknown'}",
        f"Candidate: {candidate.get('commit') or candidate.get('run_id') or 'unknown'}",
        f"Benchmark version: {baseline.get('benchmark_version') or 'unknown'}",
        "",
        "## Case changes",
        "",
        "Improved:",
    ]
    lines.extend(_case_lines(comparison["improved"]))
    lines.extend(["", "Regressed:"])
    lines.extend(_case_lines(comparison["regressed"]))
    lines.extend(["", "Added:"])
    lines.extend(_case_lines(comparison["added"]))
    lines.extend(["", "Removed:"])
    lines.extend(_case_lines(comparison["removed"]))
    lines.extend(["", "## Cost delta (candidate - baseline)", ""])
    for key, value in comparison["cost_delta"].items():
        lines.append(f"- {key}: {value:+g}")
    lines.extend(
        [
            "",
            f"Critical violations: {comparison['critical_violation_delta']:+d}",
            f"Unrequested compute: {comparison['unrequested_compute_delta']:+d}",
            f"Unexpected recomputation: {comparison['unexpected_recomputation_delta']:+d}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _read_summary(directory: str | Path) -> dict[str, Any]:
    path = Path(directory).resolve() / "summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkReportError(f"cannot read benchmark summary {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise BenchmarkReportError(f"benchmark summary has no cases array: {path}")
    return data


def _case_map(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for item in summary["cases"]:
        key = (str(item.get("case_id", "")), int(item.get("run_index", 0)))
        if key in result:
            raise BenchmarkReportError(f"duplicate case result in summary: {key}")
        result[key] = item
    return result


def _case_lines(items: list[dict[str, Any]]) -> list[str]:
    return [f"- {item['case_id']} run {item['run_index']}" for item in items] or ["- None"]


def _numeric(value: Any) -> float:
    return float(value) if type(value) in {int, float} else 0.0


__all__ = ["BenchmarkReportError", "compare_reports", "render_comparison_markdown"]
