from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from bg6022.agent import INPUT_GEOMETRY_PLACEHOLDER, Agent
from bg6022.config import load_config
from bg6022.llm import LlmClient
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Step
from bg6022.orca.profiles import B3LYP_D3BJ_DEF2SVP
from bg6022.planner import (
    intake_blocking_requirements,
    intake_message,
    plan_message,
    proposal_to_plan,
    request_from_intake,
)
from bg6022.session import artifact_path, run_directory
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


def _record_llm_evidence(config, case_id: str, payload: dict[str, Any]) -> None:
    """Persist credential-free request/plan/call metadata outside the repository."""

    evidence_root = Path(config.data_root_path) / "m3-llm-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in case_id
    )
    path = evidence_root / f"{safe_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _call_metadata(client: LlmClient) -> list[dict[str, Any]]:
    return [asdict(call) for call in client.calls]


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
            _record_llm_evidence(
                config,
                f"distance-{expression_index}-{repetition}",
                {
                    "case": "LLM-distance",
                    "expression": expression_index,
                    "repetition": repetition,
                    "message": message,
                    "intake": _model_dump(intake),
                    "request": _model_dump(request),
                    "proposal": _model_dump(proposal),
                    "plan": _model_dump(plan),
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
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
            "and return the optimized electronic energy; charge 0, multiplicity 1, please."
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
            _record_llm_evidence(
                config,
                f"b3lyp-{expression_index}-{repetition}",
                {
                    "case": "LLM-b3lyp",
                    "expression": expression_index,
                    "repetition": repetition,
                    "message": message,
                    "intake": _model_dump(intake),
                    "request": _model_dump(request),
                    "proposal": _model_dump(proposal),
                    "plan": _model_dump(plan),
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
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


@pytest.mark.live_llm
def test_live_llm_m2_composed_opt_result_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    cases = (
        (
            "opt-sp",
            (
                "请使用下面的 XYZ 先做 Opt 几何优化，然后对成功优化后的结构执行独立的 SP "
                "单点能计算，只返回 SP 电子能；总电荷 0，多重度 1。"
            ),
            "SP",
            "single_point",
            "sp_electronic_energy",
        ),
        (
            "opt-sp",
            (
                "Run Opt geometry optimization on this XYZ, then run an independent SP "
                "single-point calculation on the successful optimized geometry and return "
                "only the SP electronic energy; charge=0, multiplicity=1, please."
            ),
            "SP",
            "single_point",
            "sp_electronic_energy",
        ),
        (
            "opt-sp",
            (
                "先优化下面的结构，再在优化后的结构上独立做 single point (SP)，报告单点电子能，"
                "不要把优化末态能当作 SP；q=0, multiplicity=1。"
            ),
            "SP",
            "single_point",
            "sp_electronic_energy",
        ),
        (
            "opt-freq",
            (
                "请先对下面的 XYZ 做 Opt 优化，再对成功的优化结构做 Freq 频率计算，返回振动频率；"
                "总电荷 0，多重度 1。"
            ),
            "Freq",
            "frequency",
            "vibrational_frequencies",
        ),
        (
            "opt-freq",
            (
                "Run Opt on this XYZ first, then calculate vibrational frequencies with Freq "
                "on the successful optimized geometry; charge=0, multiplicity=1, please."
            ),
            "Freq",
            "frequency",
            "vibrational_frequencies",
        ),
        (
            "opt-freq",
            (
                "先优化这个输入结构，然后在优化后的结构上计算振动频率，不要直接对初始结构做频率；"
                "q=0，spin multiplicity=1。"
            ),
            "Freq",
            "frequency",
            "vibrational_frequencies",
        ),
    )
    for case_index, (
        case_name,
        phrase,
        consumer_operation,
        consumer_tool,
        result_name,
    ) in enumerate(cases, start=1):
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
            assert intake.operations == ["Opt", consumer_operation]
            assert result_name in intake.requested_results
            request = request_from_intake(
                message,
                intake,
                request_id=f"m3_live_llm_{case_name}_{case_index}_{repetition}",
                registry=registry,
            )
            bindings = request.structure_input.get("required_bindings", [])
            assert {
                (
                    item.get("consumer_operation"),
                    item.get("source_operation"),
                    item.get("source_port"),
                )
                for item in bindings
                if item.get("consumer_operation") == consumer_operation
            } == {(consumer_operation, "Opt", "optimized_geometry")}
            proposal = plan_message(client, request, registry=registry)
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"m3_live_llm_{case_name}_plan_{case_index}_{repetition}",
                artifact_aliases={
                    "request_geometry": "request_geometry",
                    INPUT_GEOMETRY_PLACEHOLDER: INPUT_GEOMETRY_PLACEHOLDER,
                },
            )
            opt_steps = [step for step in plan.steps if step.tool == "optimize_geometry"]
            consumer_steps = [step for step in plan.steps if step.tool == consumer_tool]
            assert len(opt_steps) == len(consumer_steps) == 1
            assert consumer_steps[0].inputs["geometry"] == InputReference(
                step_id=opt_steps[0].id, port="optimized_geometry"
            )
            assert [
                operation for step in plan.steps for operation in registry.get(step.tool).operations
            ] == ["Opt", consumer_operation]
            _record_llm_evidence(
                config,
                f"m2-{case_name}-{case_index}-{repetition}",
                {
                    "case": f"LLM-{case_name}",
                    "expression": case_index,
                    "repetition": repetition,
                    "message": message,
                    "intake": _model_dump(intake),
                    "request": _model_dump(request),
                    "proposal": _model_dump(proposal),
                    "plan": _model_dump(plan),
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": f"LLM-{case_name}",
                        "expression": case_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_opt_distance_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    phrases = (
        (
            "请使用下面的 XYZ，先做几何优化，然后只在成功的优化结构上测量第 1 和第 2 号原子的距离；"
            "电荷为0，多重度为1。"
        ),
        (
            "Optimize this XYZ first, then measure only the distance between atoms 1 and 2 "
            "in the successful optimized geometry; charge=0 and multiplicity=1; "
            "return only distance."
        ),
        (
            "先优化这个结构，再用优化后的结构计算 1、2 号原子间距，只返回距离；"
            "q=0，spin multiplicity=1。"
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
            assert "interatomic_distance" in intake.requested_results
            request = request_from_intake(
                message,
                intake,
                request_id=f"m3_live_llm_opt_distance_{expression_index}_{repetition}",
                registry=registry,
            )
            bindings = request.structure_input.get("required_bindings", [])
            assert {
                (item.get("consumer_tool"), item.get("source_operation"), item.get("source_port"))
                for item in bindings
            } == {("geometry_distance", "Opt", "optimized_geometry")}
            proposal = plan_message(client, request, registry=registry)
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"m3_live_llm_opt_distance_plan_{expression_index}_{repetition}",
                artifact_aliases={
                    "request_geometry": "request_geometry",
                    INPUT_GEOMETRY_PLACEHOLDER: INPUT_GEOMETRY_PLACEHOLDER,
                },
            )
            opt_steps = [step for step in plan.steps if step.tool == "optimize_geometry"]
            distance_steps = [step for step in plan.steps if step.tool == "geometry_distance"]
            assert len(opt_steps) == len(distance_steps) == 1
            assert distance_steps[0].inputs["geometry"] == InputReference(
                step_id=opt_steps[0].id, port="optimized_geometry"
            )
            assert distance_steps[0].parameters == {"atom_i": 1, "atom_j": 2}
            assert [
                operation for step in plan.steps for operation in registry.get(step.tool).operations
            ] == ["Opt"]
            _record_llm_evidence(
                config,
                f"opt-distance-{expression_index}-{repetition}",
                {
                    "case": "LLM-opt-distance",
                    "expression": expression_index,
                    "repetition": repetition,
                    "message": message,
                    "intake": _model_dump(intake),
                    "request": _model_dump(request),
                    "proposal": _model_dump(proposal),
                    "plan": _model_dump(plan),
                    "confirmation": "not started; direct planning evidence",
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": "LLM-opt-distance",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_history_distance_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    geometry_catalog = [
        {
            "alias": "geometry_1",
            "description": "previous successful water optimization",
            "created_at": "2026-09-16T00:00:00Z",
            "run_status": "succeeded",
            "system": "H2O",
            "geometry": {"role": "verified optimized_geometry", "atom_count": 3},
            "calculation": {
                "method_profile": "r2scan3c",
                "environment": "gas",
                "charge": 0,
                "multiplicity": 1,
            },
        }
    ]
    phrases = (
        "请复用刚才成功优化的水结构，执行新的距离测量，只计算第 1 和第 2 号原子之间的距离。",
        (
            "Perform a new distance measurement using the previous successful optimized water "
            "geometry; measure only atoms 1 and 2."
        ),
        "请对最近那个已验证的优化结构执行新的距离计算，输出 1、2 号原子间距，不要重新优化。",
    )
    for expression_index, phrase in enumerate(phrases, start=1):
        for repetition in (1, 2):
            client = LlmClient(config)
            intake = intake_message(
                client,
                phrase,
                context={"geometry_catalog": geometry_catalog},
                geometry_catalog=geometry_catalog,
                capability_catalog=registry.result_capabilities(),
                registry=registry,
            )
            assert intake.intent == "chemistry_compute"
            assert intake.history_geometry_alias == "geometry_1"
            assert intake.operations == []
            assert intake.requested_results == ["interatomic_distance"]
            request = request_from_intake(
                phrase,
                intake,
                request_id=f"m3_live_llm_history_distance_{expression_index}_{repetition}",
                registry=registry,
            )
            proposal = plan_message(
                client,
                request,
                registry=registry,
                context={"geometry_catalog": geometry_catalog},
            )
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"m3_live_llm_history_distance_plan_{expression_index}_{repetition}",
                artifact_aliases={"geometry_1": "geometry_1"},
            )
            assert len(plan.steps) == 1
            assert plan.steps[0].tool == "geometry_distance"
            assert plan.steps[0].inputs["geometry"] == InputReference(artifact_id="geometry_1")
            _record_llm_evidence(
                config,
                f"history-distance-{expression_index}-{repetition}",
                {
                    "case": "LLM-history-distance",
                    "expression": expression_index,
                    "repetition": repetition,
                    "message": phrase,
                    "geometry_catalog": geometry_catalog,
                    "intake": _model_dump(intake),
                    "request": _model_dump(request),
                    "proposal": _model_dump(proposal),
                    "plan": _model_dump(plan),
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": "LLM-history-distance",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_confirmation_update_and_rejection(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    phrases = (
        (
            "请使用下面的 XYZ 先做 Opt 优化，再在成功的优化结构上只测 atom_i=1、atom_j=2 的距离；"
            "电荷=0，多重度=1。"
        ),
        (
            "Start an Opt on this XYZ and then measure atom_i=1 and atom_j=2 in the successful "
            "optimized geometry; "
            "charge=0, multiplicity=1; do not start the calculation yet."
        ),
        (
            "先优化这个输入结构，再测量优化后结构中 atom_i=1、atom_j=2 的距离；只要距离，"
            "q=0，spin multiplicity=1。"
        ),
    )
    for expression_index, phrase in enumerate(phrases, start=1):
        for repetition in (1, 2):
            client = LlmClient(config)
            agent = Agent(
                config,
                registry,
                llm=client,
                session_id=f"m3-live-confirm-v10-{expression_index}-{repetition}",
            )
            initial = agent.handle_message(_inline_message(phrase))
            assert initial.run is not None
            run = initial.run
            assert run.status == "waiting"
            assert run.waiting_for == "confirmation"
            assert run.attempts == []
            distance_step = next(
                step for step in run.plan.steps if step.tool == "geometry_distance"
            )
            assert distance_step.parameters == {"atom_i": 1, "atom_j": 2}

            updated = agent.handle_message(
                "只修改当前待确认任务的距离参数：atom_j=3。重新显示确认，不要开始计算。"
            )
            assert updated.run is not None
            run = updated.run
            assert run.status == "waiting"
            assert run.waiting_for == "confirmation"
            distance_step = next(
                step for step in run.plan.steps if step.tool == "geometry_distance"
            )
            assert distance_step.parameters == {"atom_i": 1, "atom_j": 3}
            assert run.attempts == []
            before_rejected_update = run.model_dump(mode="json")

            rejected = agent.handle_message(
                "只修改当前待确认任务的距离参数：保持 atom_i=1，只把 atom_j 改为99。"
            )

            assert rejected.run is not None
            assert "rejected" in rejected.text
            assert rejected.run.model_dump(mode="json") == before_rejected_update
            assert rejected.run.attempts == []
            _record_llm_evidence(
                config,
                f"confirmation-update-{expression_index}-{repetition}",
                {
                    "case": "LLM-confirmation-update",
                    "expression": expression_index,
                    "repetition": repetition,
                    "message": _inline_message(phrase),
                    "confirmation_before_update": initial.text,
                    "confirmation_after_update": updated.text,
                    "rejected_update_response": rejected.text,
                    "run_after_update": _model_dump(run),
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": "LLM-confirmation-update",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_completed_distance_query_variants_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    query_phrases = (
        "刚才算出的第1和第2号原子距离是多少？",
        "What is the saved distance between atoms 1 and 2 from the calculation just completed?",
        "请查询刚刚保存的原子间距离，不要重新计算。",
    )
    for expression_index, query in enumerate(query_phrases, start=1):
        for repetition in (1, 2):
            client = LlmClient(config)
            agent = Agent(
                config,
                registry,
                llm=client,
                session_id=f"m3-live-query-v7-{expression_index}-{repetition}",
            )
            request = Request(
                id=f"m3_live_query_request_{expression_index}_{repetition}",
                description="deterministic completed distance setup for LLM query",
                source="chat",
                requested_results=[ResultTarget(field="interatomic_distance")],
                explicit_parameters={"atom_i": 1, "atom_j": 2},
            )
            distance = Step(
                id="distance",
                tool="geometry_distance",
                parameters={"atom_i": 1, "atom_j": 2},
                inputs={"geometry": InputReference(artifact_id=INPUT_GEOMETRY_PLACEHOLDER)},
            )
            plan = Plan(
                id=f"m3_live_query_plan_{expression_index}_{repetition}",
                request_id=request.id,
                steps=[distance],
                requested_results=[ResultTarget(step_id=distance.id, field="interatomic_distance")],
            )
            run, initial_result = agent.execute_plan(
                request, plan, xyz_path=WATER_XYZ, execute=True
            )
            assert run.status == "succeeded"
            assert initial_result.status == "succeeded"
            agent._session["active_run_id"] = run.id
            agent._save_session()
            run_id = run.id
            result_count = len(run.result_index)

            queried = agent.handle_message(query)

            queried_run = queried.run or agent._coerce_run(None)
            assert queried_run is not None
            assert queried_run.id == run_id
            assert queried_run.status == "succeeded"
            assert len(queried_run.result_index) == result_count
            if "Å" in queried.text:
                query_outcome = "answered"
            else:
                assert "多个" in queried.text or "multiple" in queried.text.lower()
                query_outcome = "safe_ambiguity_clarification"
            _record_llm_evidence(
                config,
                f"completed-distance-query-{expression_index}-{repetition}",
                {
                    "case": "LLM-completed-distance-query",
                    "expression": expression_index,
                    "repetition": repetition,
                    "initial_setup": "deterministic geometry_distance execution on WATER_XYZ",
                    "initial_result": _model_dump(initial_result),
                    "query": query,
                    "query_response": queried.text,
                    "query_outcome": query_outcome,
                    "run_id": run_id,
                    "result_count_before_and_after": [result_count, len(queried_run.result_index)],
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": "LLM-completed-distance-query",
                        "expression": expression_index,
                        "repetition": repetition,
                        "calls": len(client.calls),
                    }
                )
            )


@pytest.mark.live_llm
def test_live_llm_missing_and_unsupported_requests_twice(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config = _live_config(pytestconfig)
    registry = build_registry(config)
    cases = (
        (
            "missing-distance-indices",
            "请使用下面的 XYZ 计算原子间距离，但不指定原子编号，不要优化。",
            {"atom_i", "atom_j"},
        ),
        (
            "unsupported-free-energy",
            "请用下面的 XYZ 做 Opt 优化并报告 Gibbs free energy；总电荷 0，多重度 1。",
            None,
        ),
        (
            "unsupported-free-energy-en",
            "Use this XYZ to optimize the molecule and report Gibbs free energy; "
            "charge=0, multiplicity=1.",
            None,
        ),
    )
    for case_index, (case_name, phrase, expected_missing) in enumerate(cases, start=1):
        for repetition in (1, 2):
            message = _inline_message(phrase)
            client = LlmClient(config)
            intake = intake_message(
                client,
                message,
                capability_catalog=registry.result_capabilities(),
                registry=registry,
            )
            blocking = intake_blocking_requirements(intake, registry)
            assert blocking
            if expected_missing is not None:
                assert expected_missing <= set(blocking)
            else:
                assert intake.unresolved_results
                assert any(
                    "Gibbs" in item or "free" in item.lower() for item in intake.unresolved_results
                )
            with pytest.raises(ValueError):
                request_from_intake(
                    message,
                    intake,
                    request_id=f"m3_live_llm_{case_name}_{case_index}_{repetition}",
                    registry=registry,
                )
            _record_llm_evidence(
                config,
                f"boundary-{case_name}-{repetition}",
                {
                    "case": f"LLM-{case_name}",
                    "expression": case_index,
                    "repetition": repetition,
                    "message": message,
                    "intake": _model_dump(intake),
                    "blocking_requirements": list(blocking),
                    "request_created": False,
                    "call_metadata": _call_metadata(client),
                    "status": "passed",
                },
            )
            print(
                json.dumps(
                    {
                        "case": f"LLM-{case_name}",
                        "expression": case_index,
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
    expected = math.dist(optimized_geometry.coordinates[0], optimized_geometry.coordinates[2])
    assert abs(result.values["interatomic_distance"]["value"] - expected) <= 1e-10
    assert result.values["interatomic_distance"]["geometry_artifact_id"] == optimized_artifact.id
    assert result.values["interatomic_distance"]["geometry_sha256"] == optimized_artifact.sha256
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
