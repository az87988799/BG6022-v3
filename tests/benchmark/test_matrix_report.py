from __future__ import annotations

import json
from dataclasses import replace

from bg6022.benchmark.matrix import expand_matrix
from bg6022.benchmark.models import CaseObservation, CaseResult
from bg6022.benchmark.report import write_matrix_report


def _result(case_id: str, *, passed: bool, stage: str | None, category: str | None):
    return CaseResult(
        case_id=case_id,
        category="scientific",
        support="supported",
        mode="live_orca",
        run_index=1,
        passed=passed,
        dimensions={},
        failed_stage=stage,
        assertions=[],
        observation=CaseObservation(
            case_id=case_id,
            run_index=1,
            status="completed" if passed else "failed",
            stage="complete" if passed else "execution",
            orca_attempts=1,
            elapsed_seconds=12.5,
            error_category=category,
        ),
    )


def test_matrix_report_has_task_object_breakdowns_and_failure_diagnostics(tmp_path) -> None:
    expanded = expand_matrix("benchmarks/scientific_v1")[:2]
    expanded = [replace(item, case=item.case.model_copy(update={"repeat": 1})) for item in expanded]
    results = [
        _result(expanded[0].case.id, passed=True, stage=None, category=None),
        _result(expanded[1].case.id, passed=False, stage="execution", category="timeout"),
    ]
    output = tmp_path / "report"
    output.mkdir()

    summary = write_matrix_report(output, expanded, results)

    assert summary["cell_count"] == 2
    assert summary["run_count"] == 2
    assert summary["passed_runs"] == 1
    assert summary["orca_attempts"] == 2
    stored = json.loads((output / "matrix.json").read_text(encoding="utf-8"))
    assert stored["by_task"][0]["id"] == "T001"
    assert stored["by_object"][1]["id"] == "water"
    markdown = (output / "matrix.md").read_text(encoding="utf-8")
    assert "Task × Scientific Object" in markdown
    assert "By Task" in markdown and "By Object" in markdown
    assert "execution" in markdown and "timeout" in markdown
