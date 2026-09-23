from __future__ import annotations

import json

import pytest

from bg6022.benchmark.compare import (
    BenchmarkReportError,
    compare_reports,
    render_comparison_markdown,
)


def _report(path, *, passed, token_count):
    path.mkdir()
    summary = {
        "run_id": path.name,
        "commit": path.name,
        "passed_cases": int(passed),
        "case_count": 1,
        "critical_violation_count": 0,
        "unrequested_compute": 0,
        "unexpected_recomputation": 0,
        "cost": {"total_tokens": token_count, "calls": 1, "orca_attempts": 0},
        "cases": [{"case_id": "B009", "run_index": 1, "passed": passed}],
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_compare_identifies_regression_and_cost_delta(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _report(baseline, passed=True, token_count=100)
    _report(candidate, passed=False, token_count=80)
    result = compare_reports(baseline, candidate)
    assert result["regressed"] == [
        {"case_id": "B009", "run_index": 1, "baseline": "PASS", "candidate": "FAIL"}
    ]
    assert result["cost_delta"]["total_tokens"] == -20
    markdown = render_comparison_markdown(result)
    assert "B009 run 1" in markdown
    assert markdown.isascii()


def test_compare_rejects_capability_scores_across_versions(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _report(baseline, passed=True, token_count=100)
    _report(candidate, passed=True, token_count=80)
    for path, version in ((baseline, "1.0"), (candidate, "1.1")):
        summary_path = path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["benchmark_version"] = version
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BenchmarkReportError, match="benchmark versions differ"):
        compare_reports(baseline, candidate)
