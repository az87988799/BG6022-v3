from __future__ import annotations

import math
from pathlib import Path

import pytest

from bg6022.agent import (
    Agent,
    _looks_like_parameter_only_change,
    _normalize_explicit_plan,
    _query_value_is_compatible,
)
from bg6022.answer import render_selected_facts
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Requirement, ResultTarget, Run, Step
from bg6022.orca.input import OrcaInputSpec, render_input
from bg6022.orca.profiles import B3LYP_D3BJ_DEF2SVP, PBE0_D3BJ_DEF2SVP, get_profile
from bg6022.planner import (
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    _intake_schema,
    _query_property_evidence_matches,
    normalize_user_explicit_parameters,
    proposal_to_plan,
    request_from_intake,
    validate_request_plan,
)
from bg6022.session import utc_now
from bg6022.tools.geometry_distance import (
    GeometryDistanceParameters,
    measure_distance,
)
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.registry import build_registry, merge_explicit_step_parameters

M3_XYZ = b"""3
M3 exact geometry fixture; not an optimized molecule
O 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 2.0 0.0
"""


def _config(tmp_path: Path, *, default_method: str = "r2scan3c"):
    config_path = tmp_path / "config.toml"
    executable = tmp_path / "missing-orca.exe"
    config_path.write_text(
        f"""[orca]
executable = '{executable.as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false

[defaults]
method_profile = '{default_method}'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def test_independent_method_optimizations_keep_answer_goal_out_of_plan(tmp_path: Path) -> None:
    registry = build_registry()
    intake = IntakeOutput.model_validate(
        {
            "intent": "chemistry_compute",
            "subjects": {
                "water": {
                    "key": "water",
                    "molecule_query": "water",
                    "molecule_input_kind": "name",
                    "molecule_name_evidence": "水分子",
                }
            },
            "requirements": [
                {
                    "key": "opt_r2scan",
                    "subject_key": "water",
                    "capability": "optimize_geometry",
                    "parameters": {
                        "method_request": "r²SCAN-3c",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    "outputs": ["opt_final_electronic_energy"],
                },
                {
                    "key": "opt_pbe0",
                    "subject_key": "water",
                    "capability": "optimize_geometry",
                    "parameters": {
                        "method_request": "PBE0",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    "outputs": ["opt_final_electronic_energy"],
                },
            ],
            "answer_goals": [
                {
                    "kind": "compare",
                    "requirement_keys": ["opt_r2scan", "opt_pbe0"],
                    "output": "opt_final_electronic_energy",
                    "mode": "side_by_side",
                }
            ],
        },
        strict=True,
    )
    request = request_from_intake(
        "分别用 r²SCAN-3c 与 PBE0 方法优化水分子，并比较它们的能量",
        intake,
        request_id="request_method_opt_comparison",
        registry=registry,
    )
    requirements = request.requirements
    plan_proposal = PlanProposal(
        steps=[
            PlanStepProposal(
                key="resolve_water",
                tool="resolve_molecule",
                subject_id=next(iter(request.subjects)),
                parameters={"query": "water", "input_kind": "name"},
            ),
            PlanStepProposal(
                key="generate_water_geometry",
                tool="generate_geometry",
                subject_id=next(iter(request.subjects)),
                inputs={
                    "molecule": InputBindingProposal(step_key="resolve_water", port="molecule")
                },
            ),
            *[
                PlanStepProposal(
                    key=f"step_{item.id}",
                    tool=item.capability,
                    requirement_id=item.id,
                    subject_id=item.subject_id,
                    parameters=dict(item.parameters),
                    inputs={
                        "geometry": InputBindingProposal(
                            step_key="generate_water_geometry", port="geometry"
                        )
                    },
                )
                for item in requirements
            ],
        ],
        requested_results=[
            PlanTargetProposal(step_key=f"step_{item.id}", field="opt_final_electronic_energy")
            for item in requirements
        ],
    )
    plan = proposal_to_plan(
        request,
        plan_proposal,
        registry,
        plan_id="plan_method_opt_comparison",
        artifact_aliases={},
    )

    assert len(request.subjects) == 1
    assert len(requirements) == 2
    assert [item.capability for item in requirements] == [
        "optimize_geometry",
        "optimize_geometry",
    ]
    assert len(request.answer_goals) == 1
    assert request.answer_goals[0].mode == "side_by_side"
    assert all(item.capability != "same_geometry_method_energy_difference" for item in requirements)
    optimization_steps = [step for step in plan.steps if step.tool == "optimize_geometry"]
    assert len({step.requirement_id for step in optimization_steps}) == 2
    assert optimization_steps[0].inputs["geometry"] == optimization_steps[1].inputs["geometry"]
    assert [step.parameters["method_profile"] for step in optimization_steps] == [
        "r2scan3c",
        "pbe0_d3bj_def2svp",
    ]
    assert requirements[1].constraints["method_resolution"]["status"] == "proposed"

    registry = build_registry()
    with pytest.raises(ValueError, match="must provide requirement_id"):
        proposal_to_plan(
            request,
            PlanProposal(
                steps=[
                    PlanStepProposal(
                        key="first",
                        tool="optimize_geometry",
                        parameters=dict(requirements[0].parameters),
                        inputs={
                            "geometry": InputBindingProposal(
                                step_key="generate_water_geometry", port="geometry"
                            )
                        },
                    ),
                    PlanStepProposal(
                        key="second",
                        tool="optimize_geometry",
                        parameters=dict(requirements[1].parameters),
                        inputs={
                            "geometry": InputBindingProposal(
                                step_key="generate_water_geometry", port="geometry"
                            )
                        },
                    ),
                ],
                requested_results=plan_proposal.requested_results,
            ),
            registry,
            plan_id="plan_missing_requirement_ids",
            artifact_aliases={},
        )

    config = _config(tmp_path)
    config.runtime.confirm_before_compute = False
    agent = Agent(config, build_registry(config))
    run = agent._create_chat_run(request, plan)
    assert run.execution_permission is False
    answer_facts = [
        {
            "requirement_id": requirement.id,
            "name": "opt_final_electronic_energy",
            "result_property": "electronic_energy",
            "output_ref": f"out_{index}",
            "method_profile": plan_step.parameters["method_profile"],
        }
        for index, (requirement, plan_step) in enumerate(
            zip(requirements, optimization_steps, strict=True), 1
        )
    ]
    answer_goals, comparison_note = Agent._answer_goal_context(run, answer_facts)
    assert answer_goals[0]["mode"] == "side_by_side"
    assert [item["output_ref"] for item in answer_goals[0]["calculations"]] == [
        "out_1",
        "out_2",
    ]
    assert comparison_note is not None and "绝对能量差" in comparison_note


def test_distance_parameters_and_exact_measurement() -> None:
    geometry = parse_xyz_bytes(M3_XYZ)
    assert measure_distance(geometry, GeometryDistanceParameters(atom_i=1, atom_j=2)) == 1.0
    with pytest.raises(ValueError, match="outside"):
        measure_distance(geometry, GeometryDistanceParameters(atom_i=1, atom_j=4))
    assert math.isclose(
        measure_distance(geometry, GeometryDistanceParameters(atom_i=2, atom_j=3)),
        math.sqrt(5),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        measure_distance(geometry, GeometryDistanceParameters(atom_i=3, atom_j=2)),
        math.sqrt(5),
        rel_tol=0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"atom_i": 0, "atom_j": 2},
        {"atom_i": -1, "atom_j": 2},
        {"atom_i": 1, "atom_j": 1},
        {"atom_i": 1.0, "atom_j": 2},
        {"atom_i": True, "atom_j": 2},
        {"atom_i": "1", "atom_j": 2},
        {"atom_i": 1, "atom_j": 2, "extra": 3},
    ],
)
def test_distance_parameters_reject_unsafe_indices(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GeometryDistanceParameters.model_validate(payload, strict=True)


def test_distance_tool_is_operation_free_and_declares_only_indices() -> None:
    registry = build_registry()
    tool = registry.get("geometry_distance")
    assert tool.operations == []
    assert tool.request_parameters == ["atom_i", "atom_j"]
    assert tool.requires_compute_permission is False
    assert tool.execution_budget == "none"
    assert registry.request_parameter_fields([], ["interatomic_distance"]) == {"atom_i", "atom_j"}
    plan = Plan(
        id="distance-index-scope",
        request_id="distance-index-scope-request",
        steps=[Step(id="distance", tool="geometry_distance")],
    )
    assert registry.request_index_parameter_fields_for_plan(plan) == {"atom_i", "atom_j"}
    distance_capability = next(
        item for item in registry.result_capabilities() if item["name"] == "interatomic_distance"
    )
    assert distance_capability["property"] == "distance"


def test_operation_free_chat_intake_and_plan_are_supported() -> None:
    registry = build_registry()
    schema = _intake_schema((), registry.result_capabilities(), registry=registry)
    intake = schema.model_validate(
        {
            "intent": "chemistry_compute",
            "operations": [],
            "requested_results": ["interatomic_distance"],
            "explicit_parameters": {"atom_i": 1, "atom_j": 2},
        },
        strict=True,
    )
    request = request_from_intake(
        "measure atoms 1 and 2",
        intake,
        request_id="request_distance",
        registry=registry,
    )
    proposal = PlanProposal(
        steps=[
            PlanStepProposal(
                key="distance",
                tool="geometry_distance",
                parameters={},
                inputs={"geometry": InputBindingProposal(artifact_alias="request_geometry")},
            )
        ],
        requested_results=[PlanTargetProposal(step_key="distance", field="interatomic_distance")],
    )
    plan = proposal_to_plan(
        request,
        proposal,
        registry,
        plan_id="plan_distance",
        artifact_aliases={"request_geometry": "request_geometry"},
    )
    assert request.operations == []
    assert plan.steps[0].tool == "geometry_distance"
    assert plan.steps[0].parameters == {"atom_i": 1, "atom_j": 2}


def test_one_chat_request_can_contain_two_scoped_distance_requirements() -> None:
    registry = build_registry()
    subject_id = "subject_distance"
    requirements = [
        Requirement(
            id="distance_a",
            subject_id=subject_id,
            capability="geometry_distance",
            parameters={"atom_i": 1, "atom_j": 2},
            outputs=["interatomic_distance"],
        ),
        Requirement(
            id="distance_b",
            subject_id=subject_id,
            capability="geometry_distance",
            parameters={"atom_i": 2, "atom_j": 3},
            outputs=["interatomic_distance"],
        ),
    ]
    targets = [
        ResultTarget(requirement_id=item.id, field="interatomic_distance") for item in requirements
    ]
    request = Request(
        id="request_distance_twice",
        description="measure two separate atom pairs",
        source="chat",
        requirements=requirements,
        subjects={subject_id: {"key": "water", "structure_input": {}}},
        requested_results=targets,
    )
    inputs = {"geometry": InputReference(artifact_id="geometry")}
    plan = Plan(
        id="plan_distance_twice",
        request_id=request.id,
        steps=[
            Step(
                id="distance_a",
                tool="geometry_distance",
                parameters=requirements[0].parameters,
                inputs=inputs,
                requirement_id=requirements[0].id,
                subject_id=subject_id,
            ),
            Step(
                id="distance_b",
                tool="geometry_distance",
                parameters=requirements[1].parameters,
                inputs=inputs,
                requirement_id=requirements[1].id,
                subject_id=subject_id,
            ),
        ],
        requested_results=targets,
    )

    validated = validate_request_plan(request, plan, registry)

    assert len(validated.steps) == 2
    assert validated.steps[0].parameters == {"atom_i": 1, "atom_j": 2}
    assert validated.steps[1].parameters == {"atom_i": 2, "atom_j": 3}


def test_distance_execution_consumes_exact_artifact_without_orca(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "m3.xyz"
    source.write_bytes(M3_XYZ)
    request = Request(
        id="request_distance_execute",
        description="measure two atoms",
        requested_results=[ResultTarget(field="interatomic_distance")],
    )
    step = Step(
        id="distance",
        tool="geometry_distance",
        parameters={"atom_i": 1, "atom_j": 2},
        inputs={"geometry": InputReference(artifact_id="__input_geometry__")},
    )
    plan = Plan(
        id="plan_distance_execute",
        request_id=request.id,
        steps=[step],
        requested_results=[ResultTarget(step_id=step.id, field="interatomic_distance")],
    )
    before = source.read_bytes()
    run, result = Agent(config, build_registry(config)).execute_plan(
        request, plan, xyz_path=source, execute=True
    )
    assert run.status == "succeeded"
    assert result.status == "succeeded"
    value = result.values["interatomic_distance"]
    assert value["value"] == 1.0
    assert value["unit"] == "angstrom"
    assert value["atom_indices"] == [1, 2]
    assert value["atom_symbols"] == ["O", "H"]
    assert value["geometry_sha256"] == run.artifact_index[0].sha256
    assert source.read_bytes() == before
    assert run.extra_orca_executions == 0
    assert run.attempts[0]["status"] == "succeeded"


def test_distance_execution_rejects_out_of_range_without_success_value(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "m3.xyz"
    source.write_bytes(M3_XYZ)
    request = Request(
        id="request_distance_bad_index",
        description="measure an out-of-range atom pair",
        requested_results=[ResultTarget(field="interatomic_distance")],
    )
    step = Step(
        id="distance",
        tool="geometry_distance",
        parameters={"atom_i": 1, "atom_j": 4},
        inputs={"geometry": InputReference(artifact_id="__input_geometry__")},
    )
    plan = Plan(
        id="plan_distance_bad_index",
        request_id=request.id,
        steps=[step],
        requested_results=[ResultTarget(step_id=step.id, field="interatomic_distance")],
    )
    run, result = Agent(config, build_registry(config)).execute_plan(
        request, plan, xyz_path=source, execute=True
    )
    assert run.status == "failed"
    assert result.status == "failed"
    assert result.values == {}
    assert result.diagnostics["category"] == "distance_measurement_failed"
    assert source.read_bytes() == M3_XYZ


def test_request_parameter_merge_preserves_user_precedence() -> None:
    registry = build_registry()
    tool = registry.get("geometry_distance")
    step = Step(
        id="distance",
        tool=tool.name,
        parameters={"atom_i": 1, "atom_j": 2},
        inputs={"geometry": InputReference(artifact_id="geometry")},
    )
    request = Request(
        id="request",
        description="distance",
        explicit_parameters={"atom_j": 3},
        user_modifications={"atom_j": 2},
    )
    merged = merge_explicit_step_parameters(tool, step, request)
    assert merged.parameters == {"atom_i": 1, "atom_j": 2}


def test_opt_to_distance_binding_requires_tool_selector() -> None:
    registry = build_registry()
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt"],
        requested_results=["interatomic_distance"],
        explicit_parameters={"atom_i": 1, "atom_j": 2},
        structure_input={
            "required_bindings": [
                {
                    "consumer_tool": "geometry_distance",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ]
        },
    )
    request = request_from_intake(
        "optimize then measure atoms 1 and 2",
        intake,
        request_id="request_opt_distance",
        registry=registry,
    )
    assert request.structure_input["required_bindings"][0]["consumer_tool"] == ("geometry_distance")


def test_b3lyp_profile_is_complete_and_repair_scope_is_empty() -> None:
    profile = get_profile("B3LYP-D3(BJ)/def2-SVP")
    assert profile == B3LYP_D3BJ_DEF2SVP
    rendered = render_input(
        OrcaInputSpec(
            operation="SP",
            method_profile=profile.name,
            environment="gas",
            charge=0,
            multiplicity=1,
            cores=4,
            maxcore_mb=192,
        )
    ).decode("ascii")
    assert "! B3LYP D3BJ def2-SVP def2/J RIJCOSX TightSCF SP" in rendered
    assert "r2SCAN-3c" not in rendered
    registry = build_registry()
    for tool_name in ("single_point", "optimize_geometry", "frequency"):
        tool = registry.get(tool_name)
        assert tool.applicable_repair_capabilities({"method_profile": profile.name}) == []


def test_pbe0_profile_is_complete_across_existing_orca_operations() -> None:
    profile = get_profile("PBE0-D3(BJ)/def2-SVP")
    assert profile == PBE0_D3BJ_DEF2SVP
    assert profile.supported_operations == frozenset({"SP", "Opt", "Freq"})
    for operation in ("SP", "Opt", "Freq"):
        rendered = render_input(
            OrcaInputSpec(
                operation=operation,
                method_profile=profile.name,
                environment="gas",
                charge=0,
                multiplicity=1,
                cores=4,
                maxcore_mb=192,
            )
        ).decode("ascii")
        assert f"! PBE0 D3BJ def2-SVP def2/J RIJCOSX TightSCF {operation}" in rendered
        assert "%maxcore 192" in rendered
    assert any(
        item["name"] == profile.name for item in build_registry().method_capability_catalog()
    )


def test_b3lyp_default_is_selected_when_step_omits_method(tmp_path: Path) -> None:
    config = _config(tmp_path, default_method="b3lyp_d3bj_def2svp")
    registry = build_registry(config)
    request = Request(
        id="request",
        description="use configured method",
        operations=["SP"],
        explicit_parameters={"charge": 0, "multiplicity": 1},
    )
    step = Step(
        id="sp",
        tool="single_point",
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(artifact_id="geometry")},
    )
    run = Run(
        id="run",
        request=request,
        plan=Plan(id="plan", request_id=request.id, steps=[step]),
        resources=config.resources,
        budget={},
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    prepared = Agent(config, registry)._prepare_tool_step(run, step, registry.get(step.tool))
    assert prepared is not None
    assert prepared.parameters["method_profile"] == "b3lyp_d3bj_def2svp"
    assert run.parameter_sources_by_step[step.id]["method_profile"] == "default_policy"


@pytest.mark.parametrize(
    ("configured_method", "step_method", "expected_method"),
    [
        ("b3lyp_d3bj_def2svp", None, "b3lyp_d3bj_def2svp"),
        ("b3lyp_d3bj_def2svp", "r2scan3c", "r2scan3c"),
        ("r2scan3c", "b3lyp_d3bj_def2svp", "b3lyp_d3bj_def2svp"),
    ],
)
def test_explicit_plan_normalization_preserves_method_precedence(
    tmp_path: Path,
    configured_method: str,
    step_method: str | None,
    expected_method: str,
) -> None:
    config = _config(tmp_path, default_method=configured_method)
    registry = build_registry(config)
    parameters: dict[str, object] = {"charge": 0, "multiplicity": 1}
    if step_method is not None:
        parameters["method_profile"] = step_method
    plan = Plan(
        id="explicit-default-plan",
        request_id="explicit-default-request",
        steps=[
            Step(
                id="sp",
                tool="single_point",
                parameters=parameters,
                inputs={"geometry": InputReference(artifact_id="geometry")},
            )
        ],
    )

    normalized = _normalize_explicit_plan(registry, plan, defaults=config.defaults)

    assert normalized.steps[0].parameters["method_profile"] == expected_method


def _opt_to_distance_chat_run(tmp_path: Path, *, atom_j: int, session_id: str) -> tuple[Agent, Run]:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(
        id=f"request-{session_id}",
        description="optimize the fixture, then measure two atoms",
        source="chat",
        operations=["Opt"],
        requested_results=[ResultTarget(field="interatomic_distance")],
        explicit_parameters={
            "atom_i": 1,
            "atom_j": atom_j,
            "charge": 0,
            "multiplicity": 1,
        },
        structure_input={
            "xyz_text": M3_XYZ.decode("utf-8"),
            "required_bindings": [
                {
                    "consumer_tool": "geometry_distance",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ],
        },
    )
    plan = Plan(
        id=f"plan-{session_id}",
        request_id=request.id,
        steps=[
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={"geometry": InputReference(artifact_id="__input_geometry__")},
            ),
            Step(
                id="distance",
                tool="geometry_distance",
                inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
            ),
        ],
        requested_results=[ResultTarget(step_id="distance", field="interatomic_distance")],
    )
    plan = validate_request_plan(request, plan, registry)
    agent = Agent(config, registry, session_id=session_id)
    run = agent._create_chat_run(request, plan)
    return agent, run


def test_distance_index_update_checks_known_geometry_before_writing_run(
    tmp_path: Path,
) -> None:
    agent, run = _opt_to_distance_chat_run(tmp_path, atom_j=2, session_id="distance-index-update")
    agent.advance(run)
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    assert run.attempts == []

    accepted = agent._apply_parameter_update(run, {"atom_j": 3})

    assert accepted.run is run
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    assert next(step for step in run.plan.steps if step.id == "distance").parameters == {
        "atom_i": 1,
        "atom_j": 3,
    }
    before_rejected_update = run.model_dump(mode="json")

    rejected = agent._apply_parameter_update(run, {"atom_j": 99})

    assert "rejected" in rejected.text
    assert run.model_dump(mode="json") == before_rejected_update
    assert run.attempts == []


def test_invalid_initial_distance_index_never_reaches_confirmation(tmp_path: Path) -> None:
    agent, run = _opt_to_distance_chat_run(tmp_path, atom_j=99, session_id="distance-index-initial")

    agent.advance(run)

    assert run.status == "failed"
    assert run.waiting_for is None
    assert run.pending_data["category"] == "parameter_validation"
    assert "outside the known" in run.pending_data["reason"]
    assert run.attempts == []


def test_atom_index_update_is_classified_as_parameter_only_change() -> None:
    assert _looks_like_parameter_only_change("只把 atom_j 改为99，不改变当前分子或任务目标。")


def test_user_atom_index_assignments_override_model_intake_values() -> None:
    normalized = normalize_user_explicit_parameters(
        "保持 atom_i=1，只把 atom_j 改为99。",
        {"atom_i": 2, "atom_j": 3},
    )

    assert normalized.explicit_parameters == {"atom_i": 1, "atom_j": 99}


def test_waiting_parameter_update_precedes_intake_blocking(monkeypatch, tmp_path: Path) -> None:
    agent, run = _opt_to_distance_chat_run(
        tmp_path, atom_j=3, session_id="distance-index-blocking-update"
    )
    agent.advance(run)
    agent._session["active_run_id"] = run.id
    agent._save_session()
    before = run.model_dump(mode="json")

    def blocked_intake(*_args, **_kwargs) -> IntakeOutput:
        return IntakeOutput(
            intent="chemistry_compute",
            explicit_parameters={"atom_j": 99},
            missing_fields=["atom_i", "atom_j"],
        )

    monkeypatch.setattr("bg6022.agent.intake_message", blocked_intake)
    response = agent.handle_message("只把 atom_j 改为99。")

    assert response.run is not None
    assert response.run.id == run.id
    assert "rejected" in response.text
    assert response.run.model_dump(mode="json") == before
    assert run.model_dump(mode="json") == before
    assert run.attempts == []


def test_new_distance_request_uses_new_index_scope_after_waiting_opt(
    monkeypatch, tmp_path: Path
) -> None:
    agent, run = _opt_to_distance_chat_run(
        tmp_path, atom_j=3, session_id="distance-index-new-request"
    )
    agent.advance(run)
    assert run.status == "waiting"
    agent._session["active_run_id"] = run.id
    agent._save_session()
    captured: dict[str, object] = {}
    original_normalize = normalize_user_explicit_parameters

    def spy_normalize(message, parameters, candidates=(), *, parameter_names=None):
        captured["parameter_names"] = tuple(parameter_names or ())
        return original_normalize(
            message,
            parameters,
            candidates,
            parameter_names=parameter_names,
        )

    monkeypatch.setattr("bg6022.agent.normalize_user_explicit_parameters", spy_normalize)

    def new_distance_intake(*_args, **_kwargs) -> IntakeOutput:
        return IntakeOutput(
            intent="chemistry_compute",
            requested_results=["interatomic_distance"],
            explicit_parameters={"atom_i": 2, "atom_j": 3},
            missing_fields=["a fresh distance request"],
        )

    monkeypatch.setattr("bg6022.agent.intake_message", new_distance_intake)
    response = agent.handle_message("测量新距离，atom_i=2.5，atom_j=3")

    assert set(captured["parameter_names"]) == {"atom_i", "atom_j"}
    assert "没有更新" in response.text or "不是受支持" in response.text
    assert run.status == "waiting"
    assert run.attempts == []


@pytest.mark.parametrize(
    "message", ["atom_i=2.5", "atom_i=2e0", "atom_i=2/3", "不要 atom_i=2", "atom_i=2, atom_i=3"]
)
def test_waiting_parameter_scope_rejects_non_integer_user_text(
    monkeypatch, tmp_path: Path, message: str
) -> None:
    agent, run = _opt_to_distance_chat_run(
        tmp_path, atom_j=3, session_id=f"distance-index-reject-{len(message)}"
    )
    agent.advance(run)
    assert run.status == "waiting"
    agent._session["active_run_id"] = run.id
    agent._save_session()
    before = run.model_dump(mode="json")

    def blocked_intake(*_args, **_kwargs) -> IntakeOutput:
        return IntakeOutput(
            intent="chemistry_compute",
            explicit_parameters={"atom_i": 2},
            missing_fields=["atom_i", "atom_j"],
        )

    monkeypatch.setattr("bg6022.agent.intake_message", blocked_intake)
    response = agent.handle_message(message)

    assert response.run is not None
    assert "没有更新" in response.text or "rejected" in response.text
    assert run.model_dump(mode="json") == before
    assert run.attempts == []


@pytest.mark.parametrize(
    "value",
    [
        {"value": True, "unit": "angstrom"},
        {"value": "1.0", "unit": "angstrom"},
        {"value": 1.0, "unit": "bohr"},
        {"value": float("nan"), "unit": "angstrom"},
        {"value": float("inf"), "unit": "angstrom"},
        None,
    ],
)
def test_distance_query_compatibility_rejects_bad_persisted_values(value: object) -> None:
    assert not _query_value_is_compatible(value, "angstrom")


def test_distance_query_evidence_and_rendering() -> None:
    message = "刚才算出的原子间距是多少？"
    assert _query_property_evidence_matches("distance", "原子间距", message)
    fact = {
        "kind": "field",
        "name": "interatomic_distance",
        "expected_type": "angstrom",
        "value": {
            "value": math.sqrt(5),
            "unit": "angstrom",
            "atom_indices": [2, 3],
            "atom_symbols": ["H", "H"],
        },
        "metadata": {"label": "原子间距离"},
    }
    rendered = render_selected_facts([fact])
    assert "Å" in rendered
    assert "第 2 号 H" in rendered
    assert "来源：已验证几何" in rendered
