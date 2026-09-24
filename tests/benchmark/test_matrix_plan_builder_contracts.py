from __future__ import annotations

from collections import Counter
from pathlib import Path

from bg6022.benchmark.matrix import expand_matrix
from bg6022.benchmark.runner import _build_contract
from bg6022.plan_builder import build_plan
from bg6022.tools.registry import build_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = build_registry()


def test_plan_builder_preserves_scientific_matrix_t001_through_t005_contracts():
    expanded = expand_matrix(ROOT / "benchmarks" / "scientific_v1")
    selected = {
        item.task_id: item
        for item in expanded
        if item.object_id == "water" and item.task_id in {"T001", "T002", "T003", "T004", "T005"}
    }

    assert set(selected) == {"T001", "T002", "T003", "T004", "T005"}
    for task_id, item in selected.items():
        _intake, request, canonical_plan = _build_contract(
            item.fixture_override,
            prompt=item.case.prompt,
            case_id=item.case.id,
            run_index=1,
            registry=REGISTRY,
        )
        deterministic = build_plan(
            request,
            registry=REGISTRY,
            plan_id=f"deterministic_{task_id}",
        )

        assert Counter(step.tool for step in deterministic.steps) == Counter(
            step.tool for step in canonical_plan.steps
        ), task_id
        assert _dependency_edges(deterministic) == _dependency_edges(canonical_plan), task_id
        assert _result_identities(deterministic, request) == _result_identities(
            canonical_plan, request
        ), task_id
        assert _geometry_lineages(deterministic) == _geometry_lineages(canonical_plan), task_id
        if task_id == "T004":
            assert _answer_goal_semantics(request) == [
                ("compare", "numeric_difference", "sp_electronic_energy")
            ]
        elif task_id == "T005":
            assert _answer_goal_semantics(request) == [
                ("compare", "side_by_side", "opt_final_electronic_energy")
            ]
        else:
            assert _answer_goal_semantics(request) == []


def _dependency_edges(plan):
    steps = {step.id: step for step in plan.steps}
    return Counter(
        (
            steps[reference.step_id].tool,
            reference.port,
            step.tool,
            input_name,
        )
        for step in plan.steps
        for input_name, reference in step.inputs.items()
        if reference.step_id is not None
    )


def _result_identities(plan, request):
    steps = {step.id: step for step in plan.steps}
    requirements = {item.id: item for item in request.requirements}
    return Counter(
        (
            steps[target.step_id].tool,
            target.field or target.port or target.check,
            requirements[target.requirement_id or steps[target.step_id].requirement_id].capability,
        )
        for target in plan.requested_results
        if target.step_id is not None
        and (target.requirement_id is not None or steps[target.step_id].requirement_id is not None)
    )


def _geometry_lineages(plan):
    steps = {step.id: step for step in plan.steps}

    def lineage(step_id, seen=()):
        if step_id in seen:
            return "cycle"
        step = steps[step_id]
        geometry = step.inputs.get("geometry")
        if geometry is None:
            return None
        if geometry.artifact_id is not None:
            return "initial_geometry"
        source = steps[geometry.step_id]
        if source.tool == "generate_geometry":
            return "initial_geometry"
        if source.tool == "optimize_geometry":
            return f"optimized:{source.requirement_id}"
        return lineage(source.id, (*seen, step_id))

    return Counter(
        (step.requirement_id, step.tool, lineage(step.id))
        for step in plan.steps
        if step.requirement_id is not None and "geometry" in step.inputs
    )


def _answer_goal_semantics(request):
    return [(goal.kind, goal.mode, goal.output) for goal in request.answer_goals]
