from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event

import pytest

from bg6022.agent import _requested_results_satisfied
from bg6022.config import load_config
from bg6022.models import (
    Artifact,
    InputReference,
    Plan,
    Request,
    Requirement,
    Result,
    ResultTarget,
    Run,
    Step,
)
from bg6022.orca.profiles import get_profile
from bg6022.planner import validate_request_plan
from bg6022.session import (
    create_run,
    publish_step_result,
    register_bytes_artifact,
    run_directory,
    save_result,
    utc_now,
)
from bg6022.tools.energy_difference import _validate_pair
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
""",
        encoding="utf-8",
    )
    return load_config(path)


def _energy_data(
    *, method: str, value: float, geometry: Artifact, step: Step
) -> dict[str, object]:
    profile = get_profile(method)
    observation = {"value": value, "unit": "Eh", "token": f"{value:.8f}"}
    return {
        "schema": "bg6022.energy_data.v1",
        "property": "electronic_energy",
        "value": value,
        "unit": "Eh",
        "observation": observation,
        "method_profile": profile.name,
        "method_keyword": profile.orca_keyword,
        "operation": "SP",
        "charge": 0,
        "multiplicity": 1,
        "geometry": {"artifact_id": geometry.id, "sha256": geometry.sha256},
        "source": {"step_id": step.id, "attempt": 1},
    }


def _fingerprint(step: Step) -> str:
    payload = json.dumps(
        step.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _comparison_run(config, *, geometry_b_bytes: bytes | None = None):
    subject_id = "subject_water"
    requirements = [
        Requirement(
            id="req_sp_a",
            subject_id=subject_id,
            capability="single_point",
            parameters={
                "method_profile": "r2scan3c",
                "environment": "gas",
                "charge": 0,
                "multiplicity": 1,
            },
            outputs=["sp_electronic_energy", "energy_data"],
        ),
        Requirement(
            id="req_sp_b",
            subject_id=subject_id,
            capability="single_point",
            parameters={
                "method_profile": "b3lyp_d3bj_def2svp",
                "environment": "gas",
                "charge": 0,
                "multiplicity": 1,
            },
            outputs=["sp_electronic_energy", "energy_data"],
        ),
        Requirement(
            id="req_delta",
            subject_id=subject_id,
            capability="energy_difference",
            outputs=["method_energy_difference"],
        ),
    ]
    targets = [
        ResultTarget(requirement_id="req_sp_a", field="sp_electronic_energy"),
        ResultTarget(requirement_id="req_sp_a", port="energy_data"),
        ResultTarget(requirement_id="req_sp_b", field="sp_electronic_energy"),
        ResultTarget(requirement_id="req_sp_b", port="energy_data"),
        ResultTarget(requirement_id="req_delta", field="method_energy_difference"),
    ]
    request = Request(
        id="request_delta",
        description="compare two methods",
        operations=["SP", "SP"],
        requirements=requirements,
        subjects={subject_id: {"key": "water", "structure_input": {}}},
        requested_results=targets,
        source="chat",
    )
    initial_xyz = b"3\nwater\nO 0 0 0\nH 0 0.7 0.6\nH 0 -0.7 0.6\n"
    run = Run(
        id="run_delta",
        request=request,
        plan=Plan(id="plan_delta", request_id=request.id, steps=[]),
        resources=config.resources,
        status="planned",
        session_id="session_delta",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    geometry_a = register_bytes_artifact(
        config.data_root_path,
        run,
        initial_xyz,
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test/geometry-a.xyz",
        extension=".xyz",
    )
    geometry_b = (
        register_bytes_artifact(
            config.data_root_path,
            run,
            geometry_b_bytes,
            artifact_type="molecular_geometry",
            role="initial_geometry",
            source="test/geometry-b.xyz",
            extension=".xyz",
        )
        if geometry_b_bytes is not None
        else geometry_a
    )
    source_steps = [
        Step(
            id=step_id,
            tool="single_point",
            parameters=dict(requirement.parameters),
            inputs={"geometry": InputReference(artifact_id=geometry.id)},
            requirement_id=requirement.id,
            subject_id=subject_id,
        )
        for step_id, requirement, geometry in (
            ("sp_a", requirements[0], geometry_a),
            ("sp_b", requirements[1], geometry_b),
        )
    ]
    difference = Step(
        id="difference",
        tool="energy_difference",
        inputs={
            "energy_a": InputReference(step_id="sp_a", port="energy_data"),
            "energy_b": InputReference(step_id="sp_b", port="energy_data"),
        },
        requirement_id="req_delta",
        subject_id=subject_id,
    )
    run.plan = Plan(
        id="plan_delta",
        request_id=request.id,
        steps=[*source_steps, difference],
        requested_results=targets,
    )

    energies = []
    for step, geometry, value in zip(
        source_steps, (geometry_a, geometry_b), (-10.125, -10.25), strict=True
    ):
        payload = _energy_data(
            method=step.parameters["method_profile"], value=value, geometry=geometry, step=step
        )
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        energy_artifact = register_bytes_artifact(
            config.data_root_path,
            run,
            content,
            artifact_type="energy_data",
            role="verified_energy_data",
            source=f"{step.id}/attempt-01/energy_data.json",
            extension=".json",
            step_id=step.id,
            attempt=1,
            metadata={
                "property": "electronic_energy",
                "unit": "Eh",
                "method_profile": payload["method_profile"],
                "operation": "SP",
                "charge": 0,
                "multiplicity": 1,
                "geometry_sha256": geometry.sha256,
            },
        )
        result = Result(
            run_id=run.id,
            step_id=step.id,
            attempt=1,
            status="succeeded",
            values={
                "sp_electronic_energy": {
                    "value": value,
                    "unit": "Eh",
                    "token": f"{value:.8f}",
                }
            },
            artifact_ids=[energy_artifact.id],
            output_ports={"energy_data": energy_artifact.id},
            input_artifact_ids=[geometry.id],
            input_bindings={"geometry": geometry.id},
            attempt_relative_path=f"{step.id}/attempt-01",
            step_fingerprint=_fingerprint(step),
        )
        path = save_result(config.data_root_path, run, result)
        run.current_results[step.id] = path.relative_to(
            run_directory(config.data_root_path, run.id)
        ).as_posix()
        run.step_status[step.id] = "succeeded"
        energies.append(energy_artifact)
    return run, difference, energies


def test_method_difference_consumes_two_verified_energy_ports(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run, step, energy_artifacts = _comparison_run(config)

    result = build_registry(config).get("energy_difference").execute(
        step, run, cancel=Event()
    )

    assert result.status == "succeeded"
    value = result.values["method_energy_difference"]
    assert value["value"] == pytest.approx(-0.125)
    assert value["unit"] == "Eh"
    assert value["direction"] == "B - A"
    assert value["method_a"] == "r2scan3c"
    assert value["method_b"] == "b3lyp_d3bj_def2svp"
    assert result.input_bindings == {
        "energy_a": energy_artifacts[0].id,
        "energy_b": energy_artifacts[1].id,
    }


def test_requirement_scoped_outputs_complete_for_repeated_tool_instances(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    run, step, energy_artifacts = _comparison_run(config)
    result = registry.get("energy_difference").execute(step, run, cancel=Event())
    assert result.status == "succeeded"

    publish_step_result(
        config.data_root_path,
        run,
        step,
        registry.get("energy_difference"),
        result,
        expected_input_bindings=result.input_bindings,
        expected_input_hashes={
            artifact.id: artifact.sha256 for artifact in energy_artifacts
        },
    )

    assert _requested_results_satisfied(config.data_root_path, run, registry)


def test_method_difference_rejects_mismatched_geometry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run, step, _ = _comparison_run(
        config, geometry_b_bytes=b"3\nother water geometry\nO 0 0 0\nH 0 0.8 0.6\nH 0 -0.7 0.6\n"
    )

    result = build_registry(config).get("energy_difference").execute(
        step, run, cancel=Event()
    )

    assert result.status == "failed"
    assert "geometry_sha256" in result.diagnostics["reason"]


def test_method_difference_rejects_artifact_id_instead_of_current_output_port(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run, step, energy_artifacts = _comparison_run(config)
    direct_reference_step = step.model_copy(
        update={
            "inputs": {
                "energy_a": InputReference(artifact_id=energy_artifacts[0].id),
                "energy_b": InputReference(step_id="sp_b", port="energy_data"),
            }
        }
    )

    result = build_registry(config).get("energy_difference").execute(
        direct_reference_step, run, cancel=Event()
    )

    assert result.status == "failed"
    assert "current successful energy_data output port" in result.diagnostics["reason"]


def test_method_difference_requires_distinct_method_profiles() -> None:
    with pytest.raises(ValueError, match="two different method profiles"):
        _validate_pair(
            {
                "method_profile": "r2scan3c",
                "geometry_sha256": "a" * 64,
                "charge": 0,
                "multiplicity": 1,
            },
            {
                "method_profile": "r2scan3c",
                "geometry_sha256": "a" * 64,
                "charge": 0,
                "multiplicity": 1,
            },
        )


def test_plan_wires_energy_data_ports_without_copying_scalar_parameters() -> None:
    registry = build_registry()
    subject_id = "subject_1"
    requirements = [
        Requirement(
            id="req_sp_a",
            subject_id=subject_id,
            capability="single_point",
            parameters={"method_profile": "r2scan3c", "charge": 0, "multiplicity": 1},
            outputs=["sp_electronic_energy", "energy_data"],
        ),
        Requirement(
            id="req_sp_b",
            subject_id=subject_id,
            capability="single_point",
            parameters={
                "method_profile": "b3lyp_d3bj_def2svp",
                "charge": 0,
                "multiplicity": 1,
            },
            outputs=["sp_electronic_energy", "energy_data"],
        ),
        Requirement(
            id="req_delta",
            subject_id=subject_id,
            capability="energy_difference",
            outputs=["method_energy_difference"],
        ),
    ]
    targets = [
        ResultTarget(requirement_id="req_sp_a", field="sp_electronic_energy"),
        ResultTarget(requirement_id="req_sp_a", port="energy_data"),
        ResultTarget(requirement_id="req_sp_b", field="sp_electronic_energy"),
        ResultTarget(requirement_id="req_sp_b", port="energy_data"),
        ResultTarget(requirement_id="req_delta", field="method_energy_difference"),
    ]
    request = Request(
        id="request_method_compare",
        description="compare r2scan3c and B3LYP on the same structure",
        operations=["SP", "SP"],
        requirements=requirements,
        subjects={subject_id: {"key": "subject_1", "structure_input": {}}},
        requested_results=targets,
        source="chat",
    )
    shared_geometry = InputReference(artifact_id="same_geometry")
    plan = Plan(
        id="plan_method_compare",
        request_id=request.id,
        steps=[
            Step(
                id="sp_a",
                tool="single_point",
                parameters=requirements[0].parameters,
                inputs={"geometry": shared_geometry},
                requirement_id="req_sp_a",
                subject_id=subject_id,
            ),
            Step(
                id="sp_b",
                tool="single_point",
                parameters=requirements[1].parameters,
                inputs={"geometry": shared_geometry},
                requirement_id="req_sp_b",
                subject_id=subject_id,
            ),
            Step(
                id="delta",
                tool="energy_difference",
                inputs={
                    "energy_a": InputReference(step_id="sp_a", port="energy_data"),
                    "energy_b": InputReference(step_id="sp_b", port="energy_data"),
                },
                requirement_id="req_delta",
                subject_id=subject_id,
            ),
        ],
        requested_results=targets,
    )

    validated = validate_request_plan(request, plan, registry)

    assert validated.steps[0].inputs["geometry"] == validated.steps[1].inputs["geometry"]
    assert validated.steps[2].inputs["energy_a"].step_id == "sp_a"
    assert validated.steps[2].inputs["energy_b"].step_id == "sp_b"
    assert not validated.steps[2].parameters
