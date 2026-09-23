from __future__ import annotations

import json

from bg6022.benchmark.grader import grade_case
from bg6022.benchmark.models import BenchmarkCase, CaseObservation
from bg6022.benchmark.report import write_report


def test_summary_case_count_and_critical_gate_are_not_averaged(tmp_path) -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "critical_report_case",
            "category": "planning",
            "support": "supported",
            "mode": "offline",
            "prompt": "unrequested step",
            "assertions": [{"type": "forbid_tool", "value": "optimize_geometry", "critical": True}],
        },
        strict=True,
    )
    observation = CaseObservation.model_validate(
        {
            "case_id": case.id,
            "run_index": 1,
            "status": "completed",
            "stage": "complete",
            "plan": {"steps": [{"tool": "optimize_geometry"}]},
        },
        strict=True,
    )
    summary = write_report(
        tmp_path / "report",
        [grade_case(case, observation)],
        environment={"commit": "abc", "branch": "test", "model": None},
    )
    stored = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    rows = (tmp_path / "report" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["case_count"] == len(rows) == len(stored["cases"]) == 1
    assert summary["critical_violation_count"] == 1
    assert summary["cost"]["calls"] == 0
    assert summary["cost"]["orca_attempts"] == 0
