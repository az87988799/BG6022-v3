from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import bg6022.agent as agent_module
from bg6022.agent import Agent
from bg6022.config import LlmSettings, load_config
from bg6022.llm import LlmClient
from bg6022.models import InputReference, Plan, Request, Step
from bg6022.orca.profiles import resolve_parameters
from bg6022.planner import (
    ElectronicStateCandidate,
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
)
from bg6022.tools.orca import FrequencyParameters
from bg6022.tools.registry import build_registry

WATER_XYZ = (
    "3\nwater\n"
    "O 0.000000 0.000000 0.000000\n"
    "H 0.758602 0.000000 0.504284\n"
    "H -0.758602 0.000000 0.504284\n"
)


def _json_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "offline-intake-test",
            "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


INITIAL_MESSAGE = "电荷设为0，多重度设为1，优化步数设为1，SCF上限设为16"
UPDATE_MESSAGE = (
    "电荷设为+1，多重度设为2，方法设为r2scan-3c，环境设为gas，优化步数设为2，SCF上限设为24"
)
INVALID_UPDATE_MESSAGE = "方法设为b3lyp"


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


def _install_agent_entry_stubs(
    monkeypatch: pytest.MonkeyPatch,
    operations: list[str],
    *,
    initial_parameters: dict[str, Any],
    continuation_parameters: dict[str, Any] | None = None,
) -> None:
    def intake(_client: Any, message: str, **_kwargs: Any) -> IntakeOutput:
        if message == UPDATE_MESSAGE:
            values = continuation_parameters or {}
            evidence = [
                ElectronicStateCandidate(field="charge", raw_value="+1", evidence="电荷设为+1"),
                ElectronicStateCandidate(
                    field="multiplicity", raw_value="2", evidence="多重度设为2"
                ),
            ]
            return IntakeOutput(
                intent="chemistry_compute",
                operations=operations,
                explicit_parameters=values,
                electronic_state_candidates=evidence,
            )
        if message == INVALID_UPDATE_MESSAGE:
            return IntakeOutput(
                intent="chemistry_compute",
                operations=operations,
                explicit_parameters={"method_profile": "b3lyp"},
            )
        return IntakeOutput(
            intent="chemistry_compute",
            operations=operations,
            explicit_parameters=initial_parameters,
            electronic_state_candidates=[
                ElectronicStateCandidate(field="charge", raw_value="0", evidence="电荷设为0"),
                ElectronicStateCandidate(
                    field="multiplicity", raw_value="1", evidence="多重度设为1"
                ),
            ],
            structure_input={
                "xyz_text": WATER_XYZ,
                "required_bindings": [
                    {
                        "consumer_operation": operation,
                        "input_port": "geometry",
                        "source_operation": "Opt",
                        "source_port": "optimized_geometry",
                    }
                    for operation in ("Freq", "SP")
                    if operation in operations and "Opt" in operations
                ],
            },
        )

    def plan(_client: Any, request: Any, *, registry: Any, **_kwargs: Any) -> PlanProposal:
        steps: list[PlanStepProposal] = []
        has_opt = False
        for operation in request.operations:
            key, tool_name = {
                "Opt": ("opt", "optimize_geometry"),
                "Freq": ("freq", "frequency"),
                "SP": ("sp", "single_point"),
            }[operation]
            if operation == "Opt" or not has_opt:
                geometry = InputBindingProposal(artifact_alias="request_geometry")
            else:
                geometry = InputBindingProposal(step_key="opt", port="optimized_geometry")
            steps.append(
                PlanStepProposal(
                    key=key,
                    tool=tool_name,
                    parameters={"method_profile": "r2scan3c", "environment": "gas"},
                    inputs={"geometry": geometry},
                )
            )
            has_opt = has_opt or operation == "Opt"
        return PlanProposal(steps=steps)

    monkeypatch.setattr(agent_module, "intake_message", intake)
    monkeypatch.setattr(agent_module, "plan_message", plan)


@pytest.mark.parametrize(
    "operations",
    [["Opt", "Freq"], ["Opt", "SP"], ["Opt", "Freq", "SP"]],
)
def test_request_geometry_and_scoped_parameters_reach_composite_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operations: list[str]
) -> None:
    config = _config(tmp_path)
    _install_agent_entry_stubs(
        monkeypatch,
        operations,
        initial_parameters={
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 1,
            "scf_maxiter": 16,
        },
    )
    agent = Agent(config, build_registry(config), llm=None, session_id="scoped-entry")

    response = agent.handle_message(INITIAL_MESSAGE)

    assert response.run is not None
    run = response.run
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    assert run.attempts == []
    assert run.attempt_counts == {}
    steps = {step.tool: step for step in run.plan.steps}
    assert steps["optimize_geometry"].parameters["geom_maxiter"] == 1
    assert steps["optimize_geometry"].parameters["scf_maxiter"] == 16
    for tool_name in ("frequency", "single_point"):
        if tool_name not in steps:
            continue
        assert "geom_maxiter" not in steps[tool_name].parameters
        assert steps[tool_name].parameters["scf_maxiter"] == 16
        assert steps[tool_name].parameters["charge"] == 0
        assert steps[tool_name].parameters["multiplicity"] == 1
    assert run.parameter_sources_by_step[steps["optimize_geometry"].id]["geom_maxiter"] == (
        "request_explicit"
    )


def test_confirmation_update_recomputes_each_applicable_step_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_agent_entry_stubs(
        monkeypatch,
        ["Opt", "Freq", "SP"],
        initial_parameters={
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 1,
            "scf_maxiter": 16,
        },
        continuation_parameters={
            "charge": 1,
            "multiplicity": 2,
            "method_profile": "r2scan-3c",
            "environment": "gas",
            "geom_maxiter": 2,
            "scf_maxiter": 24,
        },
    )
    agent = Agent(config, build_registry(config), llm=None, session_id="scoped-update")
    initial = agent.handle_message(INITIAL_MESSAGE)
    assert initial.run is not None and initial.run.waiting_for == "confirmation"

    updated = agent.handle_message(UPDATE_MESSAGE)

    assert updated.run is not None
    run = updated.run
    assert run.status == "waiting" and run.waiting_for == "confirmation"
    assert run.execution_permission is False
    assert run.accepted_snapshot == {}
    assert run.accepted_execution_sha256 is None
    assert run.attempt_counts == {}
    assert run.request.explicit_parameters["charge"] == 1
    assert run.request.explicit_parameters["multiplicity"] == 2
    steps = {step.tool: step for step in run.plan.steps}
    for tool_name in ("optimize_geometry", "frequency", "single_point"):
        parameters = steps[tool_name].parameters
        assert parameters["charge"] == 1
        assert parameters["multiplicity"] == 2
        assert parameters["method_profile"] == "r2scan3c"
        assert parameters["environment"] == "gas"
        assert parameters["scf_maxiter"] == 24
    assert steps["optimize_geometry"].parameters["geom_maxiter"] == 2
    assert "geom_maxiter" not in steps["frequency"].parameters
    assert "geom_maxiter" not in steps["single_point"].parameters

    before = {
        "request": run.request.model_dump(mode="json"),
        "plan": run.plan.model_dump(mode="json"),
        "pending_data": dict(run.pending_data),
        "status": run.status,
        "waiting_for": run.waiting_for,
        "accepted_snapshot": dict(run.accepted_snapshot),
        "accepted_execution_sha256": run.accepted_execution_sha256,
        "execution_permission": run.execution_permission,
    }
    rejected = agent.handle_message(INVALID_UPDATE_MESSAGE)

    assert rejected.run is not None
    assert "rejected" in rejected.text
    assert rejected.run.request.model_dump(mode="json") == before["request"]
    assert rejected.run.plan.model_dump(mode="json") == before["plan"]
    assert rejected.run.pending_data == before["pending_data"]
    assert rejected.run.status == before["status"]
    assert rejected.run.waiting_for == before["waiting_for"]
    assert rejected.run.accepted_snapshot == before["accepted_snapshot"]
    assert rejected.run.accepted_execution_sha256 == before["accepted_execution_sha256"]
    assert rejected.run.execution_permission == before["execution_permission"]


def test_sp_only_request_cannot_silently_drop_geom_maxiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_agent_entry_stubs(
        monkeypatch,
        ["SP"],
        initial_parameters={
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 2,
        },
    )
    agent = Agent(config, build_registry(config), llm=None, session_id="unscoped-request")

    response = agent.handle_message(INITIAL_MESSAGE)

    assert response.run is None
    assert "geom_maxiter" in response.text
    assert "no compatible calculation step" in response.text
    assert agent._session["active_run_id"] is None
    runs = Path(config.data_root_path) / "runs"
    assert not runs.exists() or not any(runs.iterdir())


def test_preparation_error_fails_run_and_closes_active_interval(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    agent = Agent(config, build_registry(config), llm=None, session_id="prepare-failure")
    run = agent._create_chat_run(
        request=Request(
            id="request_bad_freq_parameter",
            description="frequency plan contains an Opt-only parameter",
            operations=["Freq"],
        ),
        plan=Plan(
            id="plan_bad_freq_parameter",
            request_id="request_bad_freq_parameter",
            steps=[
                Step(
                    id="freq",
                    tool="frequency",
                    parameters={
                        "method_profile": "r2scan3c",
                        "environment": "gas",
                        "charge": 0,
                        "multiplicity": 1,
                        "geom_maxiter": 2,
                    },
                    inputs={"geometry": InputReference(artifact_id="missing_geometry")},
                )
            ],
        ),
    )

    agent.advance(run)

    assert run.status == "failed"
    assert run.waiting_for is None
    assert run.pending_data["category"] == "parameter_preparation"
    assert run.pending_data["step_id"] == "freq"
    assert "geom_maxiter" in run.pending_data["reason"]
    assert run.attempts == []
    assert run.attempt_counts == {}
    assert run.step_status.get("freq") != "running"
    assert run.active_interval_open is False


def test_profile_parameter_merge_scopes_request_values_to_tool_fields() -> None:
    resolved = resolve_parameters(
        request_parameters={
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 2,
            "scf_maxiter": 32,
        },
        tool_parameters={"method_profile": "r2scan3c", "environment": "gas"},
        parameter_fields=FrequencyParameters.model_fields,
    )

    assert resolved.effective_parameters == {
        "method_profile": "r2scan3c",
        "environment": "gas",
        "charge": 0,
        "multiplicity": 1,
        "scf_maxiter": 32,
    }
    assert "geom_maxiter" not in resolved.effective_parameters
    assert "geom_maxiter" not in resolved.parameter_sources


@pytest.mark.parametrize(
    "continuation_structure",
    [None, {}, {"required_bindings": []}],
    ids=["omitted", "empty-object", "empty-bindings"],
)
def test_production_intake_path_normalizes_empty_structure_for_parameter_continuation(
    tmp_path: Path,
    continuation_structure: dict[str, Any] | None,
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    initial_message = (
        "用这个水分子结构做 Opt 后，在优化结构上独立做 SP；"
        "初始几何优化步数为1，电荷为0，多重度为1。"
    )
    initial_intake: dict[str, Any] = {
        "intent": "chemistry_compute",
        "operations": ["Opt", "SP"],
        "molecule_query": None,
        "molecule_input_kind": None,
        "history_geometry_alias": None,
        "structure_input": {
            "xyz_text": WATER_XYZ,
            "required_bindings": [
                {
                    "consumer_operation": "SP",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ],
        },
        "explicit_parameters": {"geom_maxiter": 1},
        "electronic_state_candidates": [
            {"field": "charge", "raw_value": "0", "evidence": "电荷为0"},
            {"field": "multiplicity", "raw_value": "1", "evidence": "多重度为1"},
        ],
        "requested_results": ["sp_electronic_energy"],
        "missing_fields": [],
    }
    continuation: dict[str, Any] = {
        "intent": "chemistry_compute",
        "operations": [],
        "molecule_query": None,
        "molecule_input_kind": None,
        "history_geometry_alias": None,
        "explicit_parameters": {"geom_maxiter": 100},
        "electronic_state_candidates": [],
        "requested_results": [],
        "missing_fields": [],
    }
    if continuation_structure is not None:
        continuation["structure_input"] = continuation_structure

    planner = {
        "steps": [
            {
                "key": "opt",
                "tool": "optimize_geometry",
                "parameters": {"method_profile": "r2scan3c", "environment": "gas"},
                "inputs": {"geometry": {"artifact_alias": "request_geometry"}},
            },
            {
                "key": "sp",
                "tool": "single_point",
                "parameters": {"method_profile": "r2scan3c", "environment": "gas"},
                "inputs": {"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
            },
        ],
        "requested_results": [
            {"step_key": "sp", "field": "sp_electronic_energy"},
        ],
    }
    responses = [
        _json_response(initial_intake),
        _json_response(planner),
        _json_response(continuation),
    ]
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return responses.pop(0)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=http_client,
        api_key="offline-test-key",
    )
    agent = Agent(config, registry, llm=llm, session_id="real-intake-continuation")
    try:
        initial = agent.handle_message(initial_message)
        assert initial.run is not None
        run_id = initial.run.id
        assert initial.run.waiting_for == "confirmation"
        assert initial.run.attempts == []
        assert [call.purpose for call in llm.calls] == ["intake", "planner"]
        active_run = agent._coerce_run(None)
        assert active_run is not None and active_run.id == run_id
        assert active_run.status == "waiting"
        assert active_run.waiting_for == "confirmation"
        patch_intake = IntakeOutput(
            intent="chemistry_compute",
            explicit_parameters={"geom_maxiter": 100},
        )
        assert agent_module._is_parameter_continuation(
            active_run,
            patch_intake,
            "把几何优化迭代上限改为100。",
            {"geom_maxiter": 100},
        )
        non_continuations = [
            patch_intake.model_copy(update={"requested_results": ["sp_electronic_energy"]}),
            patch_intake.model_copy(update={"structure_input": {"xyz_text": WATER_XYZ}}),
            patch_intake.model_copy(
                update={
                    "structure_input": {
                        "required_bindings": [
                            {
                                "consumer_operation": "SP",
                                "input_port": "geometry",
                                "source_operation": "Opt",
                                "source_port": "optimized_geometry",
                            }
                        ]
                    }
                }
            ),
            patch_intake.model_copy(update={"history_geometry_alias": "geometry_1"}),
            patch_intake.model_copy(update={"molecule_query": "ethanol"}),
        ]
        assert all(
            not agent_module._is_parameter_continuation(
                active_run,
                intake,
                "把几何优化迭代上限改为100。",
                {"geom_maxiter": 100},
            )
            for intake in non_continuations
        )
        assert not agent_module._is_parameter_continuation(
            active_run,
            patch_intake,
            "将分子改为乙醇，同时把几何优化迭代上限改为100。",
            {"geom_maxiter": 100},
        )

        updated = agent.handle_message("把几何优化迭代上限改为100。")

        assert updated.run is not None
        run = updated.run
        assert run.id == run_id
        assert run.waiting_for == "confirmation"
        assert run.execution_permission is False
        assert run.accepted_snapshot == {}
        assert run.accepted_execution_sha256 is None
        assert run.attempts == []
        steps = {step.tool: step for step in run.plan.steps}
        assert steps["optimize_geometry"].parameters["geom_maxiter"] == 100
        assert "geom_maxiter" not in steps["single_point"].parameters
        assert steps["single_point"].inputs["geometry"].step_id == steps["optimize_geometry"].id
        assert steps["single_point"].inputs["geometry"].port == "optimized_geometry"
        assert [call.purpose for call in llm.calls] == ["intake", "planner", "intake"]
        assert len(request_bodies) == 3
        assert not responses
    finally:
        http_client.close()


@pytest.mark.parametrize(
    ("blocking_field", "blocking_text"),
    [
        ("unresolved_results", ["Gibbs自由能尚未支持"]),
        ("missing_fields", ["未实现 Gibbs 自由能结果"]),
    ],
)
def test_unresolved_goal_blocks_pending_parameter_update_before_planning(
    tmp_path: Path,
    blocking_field: str,
    blocking_text: list[str],
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    initial_message = "用这个水分子结构做 Opt，几何优化步数为1，电荷为0，多重度为1。"
    initial_intake = {
        "intent": "chemistry_compute",
        "operations": ["Opt"],
        "structure_input": {"xyz_text": WATER_XYZ},
        "explicit_parameters": {"geom_maxiter": 1},
        "electronic_state_candidates": [
            {"field": "charge", "raw_value": "0", "evidence": "电荷为0"},
            {"field": "multiplicity", "raw_value": "1", "evidence": "多重度为1"},
        ],
        "requested_results": ["opt_final_electronic_energy"],
    }
    planner = {
        "steps": [
            {
                "key": "opt",
                "tool": "optimize_geometry",
                "parameters": {"method_profile": "r2scan3c", "environment": "gas"},
                "inputs": {"geometry": {"artifact_alias": "request_geometry"}},
            }
        ],
        "requested_results": [
            {"step_key": "opt", "field": "opt_final_electronic_energy"},
        ],
    }
    blocked_intake = {
        "intent": "chemistry_compute",
        "operations": [],
        "explicit_parameters": {"geom_maxiter": 100},
        "requested_results": [],
        blocking_field: blocking_text,
        "structure_input": {"required_bindings": []},
    }
    responses = [
        _json_response(initial_intake),
        _json_response(planner),
        _json_response(blocked_intake),
    ]
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return responses.pop(0)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=http_client,
        api_key="offline-test-key",
    )
    agent = Agent(config, registry, llm=llm, session_id="unresolved-goal-block")
    try:
        initial = agent.handle_message(initial_message)
        assert initial.run is not None
        assert initial.run.waiting_for == "confirmation"
        active_before = agent._coerce_run(None)
        assert active_before is not None
        before = active_before.model_dump(mode="json")
        run_count_before = len(list((Path(config.data_root_path) / "runs").glob("*/run.json")))

        blocked = agent.handle_message(
            "把几何优化迭代上限改为100，同时计算水的 SP 电子能和 Gibbs 自由能。"
        )

        assert "尚未支持或尚未明确" in blocked.text
        assert blocking_text[0] in blocked.text
        assert blocked.run is None
        active_after = agent._coerce_run(None)
        assert active_after is not None and active_after.id == active_before.id
        assert active_after.model_dump(mode="json") == before
        assert (
            len(list((Path(config.data_root_path) / "runs").glob("*/run.json"))) == run_count_before
        )
        assert active_after.attempts == []
        assert [call.purpose for call in llm.calls] == ["intake", "planner", "intake"]
        assert len(request_bodies) == 3
        assert not responses
    finally:
        http_client.close()


def test_repeated_unknown_intake_parameter_stops_before_planner_or_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    invalid_intake = {
        "intent": "chemistry_compute",
        "operations": ["Opt"],
        "molecule_query": "water",
        "molecule_input_kind": "name",
        "explicit_parameters": {"max_iterations": 100},
        "requested_results": ["opt_final_electronic_energy"],
    }
    responses = [_json_response(invalid_intake), _json_response(invalid_intake)]
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return responses.pop(0)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=http_client,
        api_key="offline-test-key",
    )
    agent = Agent(config, registry, llm=llm, session_id="unknown-intake-parameter")
    try:
        response = agent.handle_message("几何优化迭代上限为100")

        assert "请求解析阶段未获得有效模型响应（schema_error）" in response.text
        assert response.run is None
        assert agent._coerce_run(None) is None
        assert [call.purpose for call in llm.calls] == ["intake", "intake"]
        assert [call.category for call in llm.calls] == ["schema_error", "schema_error"]
        assert len(request_bodies) == 2
        assert list((Path(config.data_root_path) / "runs").glob("*/run.json")) == []
        assert not responses
    finally:
        http_client.close()
