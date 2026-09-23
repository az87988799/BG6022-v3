from __future__ import annotations

from bg6022.benchmark.grader import grade_case
from bg6022.benchmark.models import BenchmarkCase, CaseObservation


def test_grader_uses_runtime_facts_not_answer_wording() -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "intent_case",
            "category": "intake",
            "support": "supported",
            "mode": "offline",
            "prompt": "single point",
            "assertions": [
                {"type": "intent_equals", "value": "chemistry_compute"},
                {"type": "requirement_count", "value": 1},
            ],
        },
        strict=True,
    )
    base = {
        "case_id": case.id,
        "run_index": 1,
        "status": "completed",
        "stage": "complete",
        "intake": {"intent": "chemistry_compute"},
        "request": {"requirements": [{"capability": "single_point"}]},
    }
    first = grade_case(
        case, CaseObservation.model_validate(base | {"response_text": "A."}, strict=True)
    )
    second = grade_case(
        case, CaseObservation.model_validate(base | {"response_text": "B."}, strict=True)
    )
    assert first.passed is True
    assert second.passed is True


def test_critical_failure_remains_visible_as_a_violation() -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "critical_case",
            "category": "planning",
            "support": "supported",
            "mode": "offline",
            "prompt": "avoid an unrequested tool",
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
    result = grade_case(case, observation)
    assert result.passed is False
    assert result.observation.critical_violations == ["assertion:forbid_tool"]


def test_result_status_uses_latest_attempt_per_step() -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "latest_result_case",
            "category": "execution",
            "support": "supported",
            "mode": "replay",
            "prompt": "check the published result",
            "assertions": [{"type": "result_status", "value": "succeeded"}],
        },
        strict=True,
    )
    observation = CaseObservation.model_validate(
        {
            "case_id": case.id,
            "run_index": 1,
            "status": "completed",
            "stage": "complete",
            "results": [
                {"step_id": "s01_opt", "attempt": 1, "status": "failed"},
                {"step_id": "s01_opt", "attempt": 2, "status": "succeeded"},
            ],
        },
        strict=True,
    )
    assert grade_case(case, observation).passed is True


def test_method_profile_assertion_checks_resolution_status() -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "method_resolution_case",
            "category": "intake",
            "support": "supported",
            "mode": "offline",
            "prompt": "use a proposed profile",
            "assertions": [
                {
                    "type": "requirement_method_profiles",
                    "value": {
                        "profiles": ["pbe0_d3bj_def2svp"],
                        "resolution_statuses": ["proposed"],
                    },
                }
            ],
        },
        strict=True,
    )
    observation = CaseObservation.model_validate(
        {
            "case_id": case.id,
            "run_index": 1,
            "status": "completed",
            "stage": "complete",
            "request": {
                "requirements": [
                    {
                        "parameters": {"method_profile": "pbe0_d3bj_def2svp"},
                        "constraints": {"method_resolution": {"status": "proposed"}},
                    }
                ]
            },
        },
        strict=True,
    )
    assert grade_case(case, observation).passed is True
