from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bg6022.agent import INPUT_GEOMETRY_PLACEHOLDER, Agent
from bg6022.config import load_config
from bg6022.llm import LlmClient
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Step
from bg6022.orca.profiles import B3LYP_D3BJ_DEF2SVP
from bg6022.planner import intake_message, plan_message, proposal_to_plan, request_from_intake
from bg6022.session import artifact_path, run_directory
from bg6022.tools.geometry_distance import measure_distance
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.registry import build_registry

ROOT = Path(__file__).parents[2]
WATER_XYZ = ROOT / "examples" / "water.xyz"
ETHANOL_XYZ = ROOT / "tests" / "fixtures" / "m3_ethanol.xyz"
AMMONIA_XYZ = ROOT / "tests" / "fixtures" / "m3_ammonia.xyz"


def _live_config(pytestconfig):
    config_path = pytestconfig.getoption("--orca-config")
    if not config_path:
        pytest.fail("--orca-config is required for M3 live tests")
    return load_config(config_path)


def _b3_parameters() -> dict[str, object]:
    return {
        "method_profile": B3LYP_D3BJ_DEF2SVP.name,
        "environment": "gas",
        "charge": 0,
        "multiplicity": 1,
    }


def _execute_single_point(config, geometry: Path):
    registry = build_registry(config)
    request = Request(
        id="m3_live_sp_request",
        description="M3 live B3LYP single point",
        operations=["SP"],
        requested_results=[ResultTarget(field="sp_electronic_energy")],
        explicit_parameters=_b3_parameters() | {"method_profile": B3LYP_D3BJ_DEF2SVP.name},
    )
    step = Step(
        id="sp",
        tool="single_point",
        parameters=dict(request.explicit_parameters),
        inputs={"geometry": InputReference(artifact_id=INPUT_GEOMETRY_PLACEHOLDER)},
    )
    plan = Plan(
        id="m3_live_sp_plan",
        request_id=request.id,
        steps=[step],
        requested_results=[ResultTarget(step_id=step.id, field="sp_electronic_energy")],
    )
    run, result = Agent(config, registry).execute_plan(
        request, plan, xyz_path=geometry, execute=True
    )
    return run, result


def _assert_b3_run(run, result) -> None:
    assert result.status == "succeeded"
    assert run.resources["cores"] == 4
    assert run.resources["memory_mb"] == 1024
    assert run.resources["maxcore_mb"] == 192
    assert run.resources["max_concurrent_jobs"] == 1
    facts = result.diagnostics["facts"]
    assert facts["orca_version"] == "6.1.1"
    assert facts["input_hashes_match"] is True
    assert result.checks["normal_termination"] is True
    assert result.checks["scf_converged"] is True
    assert any(
        item.get("step_id") == result.step_id and item.get("attempt") == result.attempt
        for item in run.attempts
    )


def configured_attempt_path(run, result) -> Path:
    raw = result.diagnostics["raw_paths"]["input"]
    return Path(raw)


def _inline_message(phrase: str) -> str:
    return f"{phrase}\n\n{WATER_XYZ.read_text(encoding='utf-8')}"


@pytest.mark.live_llm
def test_live_llm_distance_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    phrases = (
        "请使用下面的 XYZ，只测量第 1 和第 2 号原子之间的距离。",
        "只求这份结构里 1 和 2 两个原子的间距，不做优化。",
        "Use this XYZ to measure the distance between atoms 1 and 2; do not optimize it.",
    )
    for expression_index, phrase in enumerate(phrases, start=1):
        for repetition in (1, 2):
            message = _inline_message(phrase)
            client = LlmClient(config)
            intake = intake_message(
                client,
                message,
                capability_catalog=registry.result_capabilities(),
                registry=registry,
            )
            assert intake.intent == "chemistry_compute"
            assert intake.operations == []
            assert intake.requested_results == ["interatomic_distance"]
            assert intake.explicit_parameters == {"atom_i": 1, "atom_j": 2}
            request = request_from_intake(
                message,
                intake,
                request_id=f"m3_live_llm_distance_{expression_index}_{repetition}",
                registry=registry,
            )
            proposal = plan_message(client, request, registry=registry)
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"m3_live_llm_distance_plan_{expression_index}_{repetition}",
                artifact_aliases={
                    "request_geometry": "request_geometry",
                    INPUT_GEOMETRY_PLACEHOLDER: INPUT_GEOMETRY_PLACEHOLDER,
                },
            )
            distance_steps = [step for step in plan.steps if step.tool == "geometry_distance"]
            assert len(distance_steps) == 1
            assert distance_steps[0].parameters["atom_i"] == 1
            assert distance_steps[0].parameters["atom_j"] == 2
            assert [
                operation for step in plan.steps for operation in registry.get(step.tool).operations
            ] == []
            print(
                json.dumps(
                    {
                        "case": "LLM-distance",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_b3lyp_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    phrases = (
        (
            "请用完整的 B3LYP-D3(BJ)/def2-SVP 气相组合优化下面的水分子，"
            "输出优化后的电子能。总电荷 0，自旋多重度 1。"
        ),
        (
            "Use the registered B3LYP-D3(BJ)/def2-SVP gas profile to optimize this XYZ "
            "and return the optimized electronic energy; charge 0, multiplicity 1."
        ),
        (
            "Run an Opt with B3LYP-D3BJ/def2-SVP in the gas phase on this XYZ, "
            "q=0 and multiplicity=1, and report the optimized energy."
        ),
    )
    for expression_index, phrase in enumerate(phrases, start=1):
        for repetition in (1, 2):
            message = _inline_message(phrase)
            client = LlmClient(config)
            intake = intake_message(
                client,
                message,
                capability_catalog=registry.result_capabilities(),
                registry=registry,
            )
            assert intake.intent == "chemistry_compute"
            assert intake.operations == ["Opt"]
            assert intake.requested_results == ["opt_final_electronic_energy"]
            assert intake.explicit_parameters["method_profile"] == B3LYP_D3BJ_DEF2SVP.name
            request = request_from_intake(
                message,
                intake,
                request_id=f"m3_live_llm_b3_{expression_index}_{repetition}",
                registry=registry,
            )
            proposal = plan_message(client, request, registry=registry)
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"m3_live_llm_b3_plan_{expression_index}_{repetition}",
                artifact_aliases={
                    "request_geometry": "request_geometry",
                    INPUT_GEOMETRY_PLACEHOLDER: INPUT_GEOMETRY_PLACEHOLDER,
                },
            )
            opt_steps = [step for step in plan.steps if step.tool == "optimize_geometry"]
            assert len(opt_steps) == 1
            assert opt_steps[0].parameters["method_profile"] == B3LYP_D3BJ_DEF2SVP.name
            print(
                json.dumps(
                    {
                        "case": "LLM-b3lyp",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_orca
@pytest.mark.skipif(os.name != "nt", reason="real ORCA execution is Windows-only")
def test_live_b3lyp_water_sp(pytestconfig) -> None:
    config = _live_config(pytestconfig)
    run, result = _execute_single_point(config, WATER_XYZ)
    _assert_b3_run(run, result)
    input_text = configured_attempt_path(run, result).read_text(encoding="ascii")
    assert "! B3LYP D3BJ def2-SVP def2/J RIJCOSX TightSCF SP" in input_text
    assert "r2SCAN-3c" not in input_text
    assert "nprocs 4" in input_text
    assert "%maxcore 192" in input_text
    assert "sp_electronic_energy" in result.values
    print(json.dumps({"case": "L1", "run_id": run.id, "result": result.values}))


@pytest.mark.live_orca
@pytest.mark.skipif(os.name != "nt", reason="real ORCA execution is Windows-only")
def test_live_b3lyp_water_opt_freq_independent_sp(pytestconfig) -> None:
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    parameters = _b3_parameters()
    request = Request(
        id="m3_live_chain_request",
        description="M3 live B3LYP water Opt then Freq and independent SP",
        operations=["Opt", "Freq", "SP"],
        requested_results=[ResultTarget(field="sp_electronic_energy")],
        explicit_parameters=dict(parameters),
    )
    opt = Step(
        id="opt",
        tool="optimize_geometry",
        parameters=dict(parameters),
        inputs={"geometry": InputReference(artifact_id=INPUT_GEOMETRY_PLACEHOLDER)},
    )
    freq = Step(
        id="freq",
        tool="frequency",
        parameters=dict(parameters),
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
    )
    sp = Step(
        id="sp",
        tool="single_point",
        parameters=dict(parameters),
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
    )
    plan = Plan(
        id="m3_live_chain_plan",
        request_id=request.id,
        steps=[opt, freq, sp],
        requested_results=[ResultTarget(step_id=sp.id, field="sp_electronic_energy")],
    )
    run, result = Agent(config, registry).execute_plan(
        request, plan, xyz_path=WATER_XYZ, execute=True
    )
    assert result.status == "succeeded"
    assert result.values["sp_electronic_energy"]["unit"] == "Eh"
    for step_id in ("opt", "freq", "sp"):
        assert run.step_status[step_id] == "succeeded"
    frequency_result = _load_result(config, run, "freq")
    assert frequency_result.scientific_checks["frequency_complete"].status == "passed"
    assert frequency_result.values["vibrational_frequencies"]["unit"] == "cm^-1"
    assert frequency_result.scientific_checks["local_minimum_supported"].status in {
        "passed",
        "not_met",
        "unverified",
    }
    assert len([item for item in run.attempts if item.get("phase") == "finished"]) == 3
    print(json.dumps({"case": "L2", "run_id": run.id, "result": result.values}))


@pytest.mark.live_orca
@pytest.mark.skipif(os.name != "nt", reason="real ORCA execution is Windows-only")
def test_live_b3lyp_ethanol_opt_distance(pytestconfig) -> None:
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    parameters = _b3_parameters()
    request = Request(
        id="m3_live_ethanol_request",
        description="M3 live B3LYP ethanol Opt then distance between atoms 1 and 3",
        operations=["Opt"],
        requested_results=[ResultTarget(field="interatomic_distance")],
        explicit_parameters={**parameters, "atom_i": 1, "atom_j": 3},
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
    opt = Step(
        id="opt",
        tool="optimize_geometry",
        parameters=dict(parameters),
        inputs={"geometry": InputReference(artifact_id=INPUT_GEOMETRY_PLACEHOLDER)},
    )
    distance = Step(
        id="distance",
        tool="geometry_distance",
        parameters={},
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
    )
    plan = Plan(
        id="m3_live_ethanol_plan",
        request_id=request.id,
        steps=[opt, distance],
        requested_results=[ResultTarget(step_id=distance.id, field="interatomic_distance")],
    )
    before = ETHANOL_XYZ.read_bytes()
    run, result = Agent(config, registry).execute_plan(
        request, plan, xyz_path=ETHANOL_XYZ, execute=True
    )
    assert result.status == "succeeded"
    assert result.values["interatomic_distance"]["unit"] == "angstrom"
    optimized_artifact_id = _load_result(config, run, "opt").output_ports["optimized_geometry"]
    optimized_artifact = next(
        item for item in run.artifact_index if item.id == optimized_artifact_id
    )
    optimized_geometry = parse_xyz_bytes(
        artifact_path(config.data_root_path, run, optimized_artifact).read_bytes()
    )
    expected = measure_distance(
        optimized_geometry,
        registry.get("geometry_distance").parameter_type.model_validate(
            {"atom_i": 1, "atom_j": 3}, strict=True
        ),
    )
    assert abs(result.values["interatomic_distance"]["value"] - expected) <= 1e-10
    assert ETHANOL_XYZ.read_bytes() == before
    assert run.extra_orca_executions == 0
    print(json.dumps({"case": "L3", "run_id": run.id, "result": result.values}))


@pytest.mark.live_orca
@pytest.mark.skipif(os.name != "nt", reason="real ORCA execution is Windows-only")
def test_live_b3lyp_ammonia_sp(pytestconfig) -> None:
    config = _live_config(pytestconfig)
    run, result = _execute_single_point(config, AMMONIA_XYZ)
    _assert_b3_run(run, result)
    assert result.values["sp_electronic_energy"]["unit"] == "Eh"
    print(json.dumps({"case": "L4", "run_id": run.id, "result": result.values}))


def _load_result(config, run, step_id: str) -> Result:
    relative = run.current_results[step_id]
    path = run_directory(config.data_root_path, run.id) / relative
    return Result.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)
