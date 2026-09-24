from __future__ import annotations

from collections import Counter

import pytest

from bg6022.canonicalize import canonicalize_semantic_request
from bg6022.models import Request, Requirement, RequirementInputBinding, Subject
from bg6022.plan_builder import PlanBuildError, build_plan
from bg6022.semantic import SemanticProposal
from bg6022.tools.registry import build_registry

REGISTRY = build_registry()


def _subject():
    return {
        "key": "subject_1",
        "query": "water",
        "input_kind": "name",
        "evidence": "water",
    }


def _task(key, capability, *, method=None, properties=()):
    return {
        "key": key,
        "subject_key": "subject_1",
        "capability": capability,
        "method_request": method,
        "parameters": {},
        "requested_properties": list(properties),
    }


def _request(tasks, relations=(), *, name="plan_case"):
    proposal = SemanticProposal.model_validate(
        {
            "mode": "compute",
            "subjects": [_subject()],
            "tasks": tasks,
            "relations": list(relations),
        },
        strict=True,
    )
    return canonicalize_semantic_request(
        "Calculate water using the requested operations.",
        proposal,
        request_id=name,
        registry=REGISTRY,
    )


def _build(request):
    return build_plan(request, registry=REGISTRY, plan_id=f"plan_{request.id}")


def _requirement_step(plan, requirement_id):
    return next(step for step in plan.steps if step.requirement_id == requirement_id)


def test_single_point_and_opt_have_deterministic_preparation_and_targets():
    sp_request = _request(
        [_task("t1", "single_point", method="r²SCAN-3c", properties=["energy"])],
        name="plan_sp",
    )
    sp_plan = _build(sp_request)
    assert [step.tool for step in sp_plan.steps] == [
        "resolve_molecule",
        "generate_geometry",
        "single_point",
    ]
    assert sp_plan.requested_results[0].field == "sp_electronic_energy"

    opt_request = _request(
        [_task("t1", "optimize_geometry", method="r²SCAN-3c", properties=["energy"])],
        name="plan_opt",
    )
    opt_plan = _build(opt_request)
    assert [step.tool for step in opt_plan.steps] == [
        "resolve_molecule",
        "generate_geometry",
        "optimize_geometry",
    ]
    opt = _requirement_step(opt_plan, opt_request.requirements[0].id)
    assert opt.inputs["geometry"].step_id == opt_plan.steps[1].id
    assert opt.inputs["geometry"].port == "geometry"


def test_opt_to_freq_uses_optimized_geometry_and_no_extra_opt_delivery():
    request = _request(
        [
            _task("t1", "optimize_geometry", method="r²SCAN-3c"),
            _task("t2", "frequency", method="r²SCAN-3c", properties=["frequencies"]),
        ],
        [
            {
                "type": "use_output",
                "source_task": "t1",
                "target_task": "t2",
                "property": "geometry",
            }
        ],
        name="plan_opt_freq",
    )
    plan = _build(request)
    opt = _requirement_step(plan, request.requirements[0].id)
    freq = _requirement_step(plan, request.requirements[1].id)

    assert freq.inputs["geometry"].step_id == opt.id
    assert freq.inputs["geometry"].port == "optimized_geometry"
    assert [
        (item.step_id, item.field or item.port or item.check)
        for item in plan.requested_results
    ] == [(freq.id, "vibrational_frequencies")]


def test_dual_sp_difference_consumes_one_shared_initial_geometry():
    request = _request(
        [
            _task("t1", "single_point", method="r²SCAN-3c"),
            _task("t2", "single_point", method="PBE0"),
        ],
        [{"type": "difference", "tasks": ["t1", "t2"]}],
        name="plan_difference",
    )
    plan = _build(request)
    first, second, difference = request.requirements
    first_step = _requirement_step(plan, first.id)
    second_step = _requirement_step(plan, second.id)
    difference_step = _requirement_step(plan, difference.id)

    assert first_step.inputs["geometry"] == second_step.inputs["geometry"]
    assert first_step.inputs["geometry"].step_id == plan.steps[1].id
    assert difference_step.inputs["energy_a"].step_id == first_step.id
    assert difference_step.inputs["energy_b"].step_id == second_step.id
    assert plan.requested_results[0].field == "method_energy_difference"


def test_dual_opt_compare_is_independent_and_shares_initial_geometry():
    request = _request(
        [
            _task("t1", "optimize_geometry", method="r²SCAN-3c"),
            _task("t2", "optimize_geometry", method="PBE0"),
        ],
        [{"type": "compare", "tasks": ["t1", "t2"], "property": "energy"}],
        name="plan_dual_opt",
    )
    plan = _build(request)
    first_step = _requirement_step(plan, request.requirements[0].id)
    second_step = _requirement_step(plan, request.requirements[1].id)

    assert first_step.inputs["geometry"] == second_step.inputs["geometry"]
    assert first_step.inputs["geometry"].step_id not in {first_step.id, second_step.id}
    assert Counter(target.field for target in plan.requested_results) == Counter(
        ["opt_final_electronic_energy", "opt_final_electronic_energy"]
    )
    assert request.answer_goals[0].output == "opt_final_electronic_energy"


def test_requirement_cycle_stops_without_planner_fallback():
    request = Request(
        id="cycle_request",
        description="cycle",
        source="chat",
        subjects={"subject_1": Subject(key="subject_1")},
        requirements=[
            Requirement(
                id="req_a",
                subject_id="subject_1",
                capability="optimize_geometry",
                input_bindings={
                    "geometry": RequirementInputBinding(
                        source_requirement_id="req_b", source_port="optimized_geometry"
                    )
                },
            ),
            Requirement(
                id="req_b",
                subject_id="subject_1",
                capability="optimize_geometry",
                input_bindings={
                    "geometry": RequirementInputBinding(
                        source_requirement_id="req_a", source_port="optimized_geometry"
                    )
                },
            ),
        ],
    )

    with pytest.raises(PlanBuildError, match="contains a cycle"):
        _build(request)
