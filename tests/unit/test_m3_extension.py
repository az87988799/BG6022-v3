from __future__ import annotations

import math
from pathlib import Path

import pytest

from bg6022.agent import Agent, _query_value_is_compatible
from bg6022.answer import render_selected_facts
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, ResultTarget, Run, Step
from bg6022.orca.input import OrcaInputSpec, render_input
from bg6022.orca.profiles import B3LYP_D3BJ_DEF2SVP, get_profile
from bg6022.planner import (
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    _intake_schema,
    _query_property_evidence_matches,
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

[defaults]
method_profile = '{default_method}'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(config_path)


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


def test_one_chat_request_cannot_contain_two_distance_steps() -> None:
    registry = build_registry()
    request = Request(
        id="request_distance_twice",
        description="measure one pair",
        source="chat",
        requested_results=[ResultTarget(field="interatomic_distance")],
        explicit_parameters={"atom_i": 1, "atom_j": 2},
    )
    inputs = {"geometry": InputReference(artifact_id="geometry")}
    plan = Plan(
        id="plan_distance_twice",
        request_id=request.id,
        steps=[
            Step(id="distance_a", tool="geometry_distance", inputs=inputs),
            Step(id="distance_b", tool="geometry_distance", inputs=inputs),
        ],
        requested_results=[ResultTarget(step_id="distance_a", field="interatomic_distance")],
    )
    with pytest.raises(ValueError, match="only one distance"):
        validate_request_plan(request, plan, registry)


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
    prepared = Agent(config, registry)._prepare_orca_step(run, step)
    assert prepared is not None
    assert prepared.parameters["method_profile"] == "b3lyp_d3bj_def2svp"
    assert run.parameter_sources_by_step[step.id]["method_profile"] == "default_policy"


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
