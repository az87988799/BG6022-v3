from __future__ import annotations

from pathlib import Path

from live_helpers import install_json_handler, make_live_config

from bg6022.benchmark.grader import grade_case
from bg6022.benchmark.loader import load_cases
from bg6022.benchmark.report import write_report
from bg6022.benchmark.runner import run_case
from bg6022.llm import LlmError

_WATER_XYZ = (
    "3\nwater geometry fixture; coordinates in angstrom\n"
    "O  0.000000  0.000000  0.000000\n"
    "H  0.758602  0.000000  0.504284\n"
    "H -0.758602  0.000000  0.504284\n"
)


def test_llm_schema_diagnostics_and_structured_outputs_reach_report(
    monkeypatch, tmp_path: Path
) -> None:
    case = next(item for item in load_cases("benchmarks/v1") if item.id == "B001_single_opt")

    def scripted(purpose, _messages):
        if purpose == "planner":
            raise LlmError(
                "model JSON failed local schema",
                category="schema_error",
                purpose="planner",
                diagnostics=(
                    {
                        "path": "answer_goals.0.output",
                        "message": "expected a declared result field",
                    },
                ),
            )
        return {
            "intent": "chemistry_compute",
            "subjects": {"water": {"key": "water", "inline_xyz": _WATER_XYZ}},
            "requirements": [
                {
                    "key": "opt_water",
                    "subject_key": "water",
                    "capability": "optimize_geometry",
                    "parameters": {"method_request": "r2scan3c"},
                    "outputs": ["opt_final_electronic_energy"],
                    "input_bindings": {},
                }
            ],
        }

    install_json_handler(monkeypatch, scripted)
    observation = run_case(
        case,
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "failed"
    assert observation.error_category == "schema_error"
    assert observation.error_diagnostics == [
        {
            "purpose": "planner",
            "category": "schema_error",
            "path": "answer_goals.0.output",
            "message": "expected a declared result field",
        }
    ]
    assert observation.llm_structured_outputs[0]["purpose"] == "intake"

    report_dir = tmp_path / "report"
    write_report(
        report_dir,
        [grade_case(case, observation)],
        environment={"commit": "test", "branch": "test", "model": "fake"},
    )
    report = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "answer_goals.0.output" in report
    assert "expected a declared result field" in report
    failure = (report_dir / "failures" / "B001_single_opt_1.json").read_text(encoding="utf-8")
    assert "llm_structured_outputs" in failure
    assert "error_diagnostics" in failure
