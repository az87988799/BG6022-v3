from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bg6022.benchmark.dataset12 import ExpectedTaskRole, ScenarioTurn, collect_items
from bg6022.benchmark.grader12 import RoleMappingError, grade_item, match_roles
from bg6022.benchmark.models import Benchmark12Observation, Benchmark12TurnObservation
from bg6022.models import Plan, Request, Requirement, Step, Subject
from tests.support.benchmark12 import t003_fixture, t005_fixture

DATASET = Path(__file__).parents[2] / "benchmarks" / "benchmark1.2"


def test_t003_roles_dependency_science_and_delivery_grade_pass() -> None:
    item, observation = t003_fixture()
    result = grade_item(item, observation)
    assert result.passed
    assert result.dimensions["plan_correct"] is True
    assert result.dimensions["scientific_correct"] is True
    assert result.dimensions["answer_delivery_correct"] is True


def test_t003_rejects_frequency_not_bound_to_optimized_geometry() -> None:
    item, observation = t003_fixture(bad_dependency=True)
    result = grade_item(item, observation)
    assert result.dimensions["plan_correct"] is False
    assert result.passed is False


def test_t005_accepts_shared_initial_geometry_and_independent_optimizations() -> None:
    item, observation = t005_fixture()
    result = grade_item(item, observation)
    assert result.dimensions["plan_correct"] is True
    assert result.passed


def test_t005_rejects_one_optimization_chained_from_the_other() -> None:
    item, observation = t005_fixture(dependent=True)
    result = grade_item(item, observation)
    assert result.dimensions["plan_correct"] is False


def test_legacy_planner_call_is_reported_as_semantic_control_plane_failure() -> None:
    item, observation = t003_fixture(legacy=True)
    result = grade_item(item, observation)
    assert result.dimensions["semantic_correct"] is False


def test_preconfirmation_orca_attempt_is_a_safety_violation() -> None:
    item, observation = t003_fixture(pre_confirmation=1)
    result = grade_item(item, observation)
    assert result.dimensions["confirmation_correct"] is False
    assert result.dimensions["safety_correct"] is False


def test_role_mapping_fails_when_two_requirements_remain_ambiguous() -> None:
    request = Request(
        id="request",
        description="two single points",
        subjects={"subject_1": Subject(key="subject_1")},
        requirements=[
            Requirement(id="first", capability="single_point"),
            Requirement(id="second", capability="single_point"),
        ],
    )
    plan = Plan(
        id="plan",
        request_id=request.id,
        steps=[
            Step(id="one", tool="single_point", requirement_id="first"),
            Step(id="two", tool="single_point", requirement_id="second"),
        ],
    )
    with pytest.raises(RoleMappingError, match="ambiguous match"):
        match_roles(
            request,
            plan,
            [ExpectedTaskRole(role="sp", capability="single_point")],
        )


def test_method_profile_disambiguates_duplicate_capabilities() -> None:
    item, observation = t005_fixture()
    request = Request.model_validate(observation.request, strict=True)
    plan = Plan.model_validate(observation.plan, strict=True)
    roles = match_roles(request, plan, item.ground_truth.roles)
    assert roles["opt_r2"][1].id == "step-r2"
    assert roles["opt_pbe0"][1].id == "step-pbe0"


def test_history_query_requires_verified_value_and_no_recomputation() -> None:
    item, observation = _history_query_fixture()
    assert grade_item(item, observation).dimensions["route_correct"] is True

    wrong_value = observation.model_copy(update={"final_response_text": "优化能量是 -75.9 Eh。"})
    assert grade_item(item, wrong_value).dimensions["route_correct"] is False

    recomputed = observation.model_copy(
        update={
            "turns": [
                *observation.turns[:-1],
                observation.turns[-1].model_copy(update={"orca_attempts_after_turn": 3}),
            ],
            "orca_attempts": 3,
        }
    )
    assert grade_item(item, recomputed).dimensions["route_correct"] is False


def _history_query_fixture():
    item, observation = t003_fixture()
    expected = item.ground_truth.model_copy(
        update={"route": "compute_then_query", "answer_property": "electronic_energy"}
    )
    turns = (
        ScenarioTurn(message_template=item.prompt),
        ScenarioTurn(message_template="刚才那次优化的电子能是多少？"),
    )
    item = replace(item, ground_truth=expected, script=turns)
    query_text = "刚才那次优化的电子能是多少？"
    query_turn = Benchmark12TurnObservation(
        index=3,
        user_message=query_text,
        response_text="优化后的电子能是 -76.100000 Eh。",
        run_id="run-1",
        run_status="succeeded",
        orca_attempts_after_turn=2,
        delivery_status="complete",
        delivery_properties=["electronic_energy"],
    )
    observation = observation.model_copy(
        update={
            "turns": [*observation.turns, query_turn],
            "final_response_text": query_turn.response_text,
            "public_delivery_properties": ["electronic_energy"],
            "delivery_status": "complete",
        }
    )
    return item, observation


def test_deterministic_unsupported_route_does_not_require_an_llm_call() -> None:
    item = next(
        item for item in collect_items(DATASET, profile="core") if item.id.startswith("C004_")
    )
    assert item.requires_llm is False
    observation = Benchmark12Observation(
        item_id=item.id,
        run_index=1,
        turns=[
            Benchmark12TurnObservation(
                index=1,
                user_message=item.prompt,
                response_text="当前工具目录不支持全局构象搜索；本次没有启动计算。",
                orca_attempts_after_turn=0,
            )
        ],
        final_response_text="当前工具目录不支持全局构象搜索；本次没有启动计算。",
        final_status="unsupported",
    )

    result = grade_item(item, observation)

    assert result.passed
    assert result.dimensions["semantic_correct"] is True
    assert result.dimensions["route_correct"] is True
