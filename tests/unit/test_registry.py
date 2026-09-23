from __future__ import annotations

import pytest

from bg6022.models import GoalCheckRequirement, InputReference, Plan, Request, Step, Tool
from bg6022.tools.registry import build_registry, describe_tools


def test_registered_tools_are_public() -> None:
    names = {item["name"] for item in describe_tools()}
    assert names == {
        "resolve_molecule",
        "generate_geometry",
        "single_point",
        "optimize_geometry",
        "frequency",
        "geometry_distance",
        "geometry_angle",
        "same_geometry_method_energy_difference",
    }
    registry = build_registry()
    assert "energy_difference" not in registry.names()
    assert registry.get("energy_difference").name == "same_geometry_method_energy_difference"


def test_orca_result_properties_are_declared_as_machine_readable_contracts() -> None:
    registry = build_registry()

    assert registry.get("single_point").result_properties == {
        "sp_electronic_energy": "electronic_energy",
        "energy_data": "energy_data",
    }
    assert registry.get("optimize_geometry").result_properties == {
        "opt_final_electronic_energy": "electronic_energy",
        "optimized_geometry": "molecular_geometry",
        "energy_data": "energy_data",
    }


def test_result_property_cannot_describe_an_undeclared_output() -> None:
    with pytest.raises(ValueError, match="undeclared keys"):
        Tool(
            name="measurement",
            description="Test measurement tool.",
            results={"energy": "Eh"},
            result_properties={"free_energy": "electronic_energy"},
        )


def test_input_reference_rejects_arbitrary_path_shape() -> None:
    with pytest.raises(ValueError):
        InputReference(step_id="compute")


def test_plan_keeps_step_reference_logical() -> None:
    request = Request(id="request_1", description="test")
    plan = Plan(
        id="plan_1",
        request_id=request.id,
        steps=[
            Step(
                id="compute",
                tool="single_point",
                inputs={"geometry": InputReference(artifact_id="artifact_water")},
            )
        ],
    )
    assert plan.steps[0].inputs["geometry"].artifact_id == "artifact_water"


def test_validate_plan_rejects_invalid_following_step_before_execution() -> None:
    request = Request(id="request_1", description="test")
    plan = Plan(
        id="plan_1",
        request_id=request.id,
        steps=[
            Step(
                id="first",
                tool="single_point",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputReference(artifact_id="artifact_water")},
            ),
            Step(
                id="second",
                tool="optimize_geometry",
                parameters={"charge": "not-an-int", "multiplicity": 1},
                inputs={"geometry": InputReference(step_id="first", port="missing")},
            ),
        ],
    )
    with pytest.raises(ValueError, match="invalid parameters"):
        build_registry().validate_plan(plan)


def test_validate_plan_rejects_self_reference_without_key_error() -> None:
    request = Request(id="request_1", description="test")
    plan = Plan(
        id="plan_1",
        request_id=request.id,
        steps=[
            Step(
                id="self",
                tool="optimize_geometry",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputReference(step_id="self", port="optimized_geometry")},
            )
        ],
    )
    with pytest.raises(ValueError, match="earlier step"):
        build_registry().validate_plan(plan)


def test_local_minimum_gate_requires_same_geometry_reference_as_frequency_check() -> None:
    request = Request(id="request_gate_binding", description="test gate binding")
    plan = Plan(
        id="plan_gate_binding",
        request_id=request.id,
        steps=[
            Step(
                id="frequency",
                tool="frequency",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputReference(artifact_id="checked_geometry")},
            ),
            Step(
                id="single_point",
                tool="single_point",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputReference(artifact_id="different_geometry")},
                goal_checks=[
                    GoalCheckRequirement(
                        source_step_id="frequency", check="local_minimum_supported"
                    )
                ],
            ),
        ],
    )

    with pytest.raises(ValueError, match="input source checked"):
        build_registry().validate_plan(plan)
