from __future__ import annotations

import json

from bg6022.benchmark.compare import compare_reports, render_comparison_markdown


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
