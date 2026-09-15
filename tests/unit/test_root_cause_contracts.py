from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bg6022.agent import Agent
from bg6022.config import LlmSettings, load_config
from bg6022.llm import LlmClient, LlmError
from bg6022.models import Plan, Request, ResultTarget, Run
from bg6022.planner import (
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    _intake_schema,
    intake_message,
    proposal_to_plan,
    request_from_intake,
)
from bg6022.session import execution_fingerprint, utc_now
from bg6022.tools.registry import build_registry


def _step(key: str, tool: str, geometry: dict) -> PlanStepProposal:
    parameters = {"charge": 0, "multiplicity": 1}
    return PlanStepProposal(
        key=key,
        tool=tool,
        parameters=parameters,
        inputs={"geometry": geometry},
    )


def _composed_request(*, source_operation: str | None, source_port: str) -> Request:
    return Request(
        id="request_geometry_source",
        description="optimize, then run a single point with a specified geometry source",
        operations=["Opt", "SP"],
        requested_results=[ResultTarget(field="sp_electronic_energy")],
        structure_input={
            "required_bindings": [
                {
                    "consumer_operation": "SP",
                    "input_port": "geometry",
                    "source_operation": source_operation,
                    "source_port": source_port,
                }
            ]
        },
    )


def _plan(source: dict) -> PlanProposal:
    return PlanProposal(
        steps=[
            _step("opt", "optimize_geometry", {"artifact_alias": "initial_geometry"}),
            _step("sp", "single_point", source),
        ],
        requested_results=[PlanTargetProposal(step_key="sp", field="sp_electronic_energy")],
    )


def test_registry_capabilities_are_the_intake_target_vocabulary() -> None:
    registry = build_registry()
    capabilities = registry.result_capabilities()
    names = {item["name"] for item in capabilities}

    assert {
        "sp_electronic_energy",
        "opt_final_electronic_energy",
        "optimized_geometry",
        "vibrational_frequencies",
        "frequency_complete",
    } <= names
    optimized = [item for item in capabilities if item["name"] == "optimized_geometry"]
    assert len(optimized) == 1
    assert optimized[0]["kind"] == "port"
    assert optimized[0]["tool"] == "optimize_geometry"


def test_intake_schema_rejects_free_text_even_without_history_candidates() -> None:
    schema = _intake_schema((), build_registry().result_capabilities())
    base = {
        "intent": "chemistry_compute",
        "operations": ["SP"],
        "requested_results": ["sp_electronic_energy"],
    }
    assert schema.model_validate(base, strict=True).requested_results == ["sp_electronic_energy"]

    for invalid in ("energy", "SP electronic energy", "opt_final_electronic_energy"):
        with pytest.raises(ValueError):
            schema.model_validate(
                {**base, "requested_results": [invalid]},
                strict=True,
            )


def test_legacy_energy_alias_is_scoped_by_operation_and_ambiguous_combinations_stop() -> None:
    registry = build_registry()

    assert registry.resolve_result_target("energy", ["SP"]) == ResultTarget(
        field="sp_electronic_energy"
    )
    assert registry.resolve_result_target("energy", ["Opt"]) == ResultTarget(
        field="opt_final_electronic_energy"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_result_target("energy", ["Opt", "SP"])
    with pytest.raises(ValueError, match="not in the Tool capability catalog"):
        registry.resolve_result_target("SP electronic energy", ["SP"])


def test_intake_normalizes_only_declared_result_targets() -> None:
    registry = build_registry()
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        requested_results=["sp_electronic_energy"],
    )

    request = request_from_intake(
        "calculate the single point energy",
        intake,
        request_id="request_normalized_target",
        registry=registry,
    )

    assert request.requested_results == [ResultTarget(field="sp_electronic_energy")]


def test_intake_corrects_generic_energy_before_request_and_planner() -> None:
    registry = build_registry()
    bodies = [
        {
            "intent": "chemistry_compute",
            "operations": ["Opt", "SP"],
            "molecule_query": "water",
            "molecule_input_kind": "name",
            "requested_results": ["energy"],
            "structure_input": {
                "required_bindings": [
                    {
                        "consumer_operation": "SP",
                        "input_port": "geometry",
                        "source_operation": "Opt",
                        "source_port": "optimized_geometry",
                    }
                ]
            },
        },
        {
            "intent": "chemistry_compute",
            "operations": ["Opt", "SP"],
            "molecule_query": "water",
            "molecule_input_kind": "name",
            "requested_results": ["sp_electronic_energy"],
            "structure_input": {
                "required_bindings": [
                    {
                        "consumer_operation": "SP",
                        "input_port": "geometry",
                        "source_operation": "Opt",
                        "source_port": "optimized_geometry",
                    }
                ]
            },
        },
    ]
    http_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        http_requests.append(request)
        content = json.dumps(bodies.pop(0))
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=client,
        api_key="offline-test-key",
    )
    try:
        intake = intake_message(
            llm,
            "计算水分子优化后的单点能",
            capability_catalog=registry.result_capabilities(),
        )
        request = request_from_intake(
            "计算水分子优化后的单点能",
            intake,
            request_id="request_intake_energy_correction",
            registry=registry,
        )
    finally:
        client.close()

    assert len(http_requests) == 2
    assert request.operations == ["Opt", "SP"]
    assert request.requested_results == [ResultTarget(field="sp_electronic_energy")]
    assert request.structure_input["required_bindings"][0]["source_port"] == ("optimized_geometry")
    assert [call.category for call in llm.calls] == ["ambiguous_result", "success"]
    assert [call.structured_correction_count for call in llm.calls] == [0, 1]


def test_intake_corrects_free_text_sp_energy_before_request_creation() -> None:
    registry = build_registry()
    base = {
        "intent": "chemistry_compute",
        "operations": ["SP"],
        "molecule_query": "water",
        "molecule_input_kind": "name",
    }
    bodies = [
        {**base, "requested_results": ["SP electronic energy"]},
        {**base, "requested_results": ["sp_electronic_energy"]},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [
                    {"message": {"content": json.dumps(bodies.pop(0)), "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=client,
        api_key="offline-test-key",
    )
    try:
        intake = intake_message(
            llm,
            "对水分子做 SP 并返回 SP electronic energy",
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
        request = request_from_intake(
            "对水分子做 SP 并返回 SP electronic energy",
            intake,
            request_id="request_sp_energy_correction",
            registry=registry,
        )
    finally:
        client.close()

    assert request.requested_results == [ResultTarget(field="sp_electronic_energy")]
    assert [call.category for call in llm.calls] == ["schema_error", "success"]
    assert [call.structured_correction_count for call in llm.calls] == [0, 1]


def test_intake_registry_binding_error_uses_bounded_structured_correction() -> None:
    registry = build_registry()
    base = {
        "intent": "chemistry_compute",
        "operations": ["Opt", "SP"],
        "molecule_query": "water",
        "molecule_input_kind": "name",
        "requested_results": ["sp_electronic_energy"],
        "structure_input": {
            "xyz_text": "3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24",
        },
    }
    wrong = {
        **base,
        "structure_input": {
            **base["structure_input"],
            "required_bindings": [
                {
                    "consumer_operation": "SP",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "hessian",
                }
            ],
        },
    }
    corrected = {
        **base,
        "structure_input": {
            **base["structure_input"],
            "required_bindings": [
                {
                    "consumer_operation": "SP",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ],
        },
    }
    bodies = [wrong, corrected]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [
                    {"message": {"content": json.dumps(bodies.pop(0)), "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=client,
        api_key="offline-test-key",
    )
    try:
        intake = intake_message(
            llm,
            "优化水分子后在优化结构上做 SP",
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
        request = request_from_intake(
            "优化水分子后在优化结构上做 SP",
            intake,
            request_id="request_geometry_binding_correction",
            registry=registry,
        )
    finally:
        client.close()

    assert request.structure_input["xyz_text"].startswith("3\nwater")
    assert request.structure_input["required_bindings"][0]["source_port"] == ("optimized_geometry")
    assert [call.category for call in llm.calls] == ["schema_error", "success"]
    assert [call.structured_correction_count for call in llm.calls] == [0, 1]


def test_intake_preserves_explicit_initial_geometry_binding_from_user_semantics() -> None:
    registry = build_registry()
    body = {
        "intent": "chemistry_compute",
        "operations": ["Opt", "SP"],
        "molecule_query": "water",
        "molecule_input_kind": "name",
        "requested_results": ["sp_electronic_energy"],
        "structure_input": {
            "required_bindings": [
                {
                    "consumer_operation": "SP",
                    "input_port": "geometry",
                    "source_operation": None,
                    "source_port": "initial_geometry",
                }
            ]
        },
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [{"message": {"content": json.dumps(body), "role": "assistant"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(api_key_env="TEST_BG6022_KEY", request_timeout_seconds=5),
        client=client,
        api_key="offline-test-key",
    )
    try:
        intake = intake_message(
            llm,
            "优化水分子，然后对原始结构做 SP",
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
        request = request_from_intake(
            "优化水分子，然后对原始结构做 SP",
            intake,
            request_id="request_initial_geometry_from_intake",
            registry=registry,
        )
    finally:
        client.close()

    assert request.structure_input["required_bindings"] == [
        {
            "consumer_operation": "SP",
            "input_port": "geometry",
            "source_operation": None,
            "source_port": "initial_geometry",
        }
    ]


def test_composite_intake_without_geometry_source_stops_for_user_clarification() -> None:
    registry = build_registry()
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt", "SP"],
        requested_results=["sp_electronic_energy"],
    )

    with pytest.raises(ValueError, match="请明确 SP 使用优化后的结构还是初始结构"):
        request_from_intake(
            "优化水分子并做 SP，但未说明 SP 用哪个结构",
            intake,
            request_id="request_missing_geometry_source",
            registry=registry,
        )


def test_legacy_request_and_accepted_snapshot_keep_fingerprint_on_confirmation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    legacy_request = {
        "id": "request_legacy_snapshot",
        "description": "legacy accepted SP request",
        "operation": "SP",
        "requested_results": [{"field": "sp_electronic_energy"}],
        "structure_input": {"xyz_text": "legacy input geometry"},
    }
    legacy_plan = {
        "id": "plan_legacy_snapshot",
        "request_id": "request_legacy_snapshot",
        "steps": [
            {
                "id": "sp",
                "tool": "single_point",
                "parameters": {"charge": 0, "multiplicity": 1},
                "inputs": {"geometry": {"artifact_id": "artifact_original"}},
            }
        ],
        "requested_results": [{"step_id": "sp", "field": "sp_electronic_energy"}],
    }
    resources = {"cores": 4, "memory_mb": 1024, "maxcore_mb": 192}
    accepted_snapshot = {"request": legacy_request, "plan": legacy_plan}
    plan = Plan.model_validate(legacy_plan, strict=True)
    previous_fingerprint = execution_fingerprint(
        plan,
        resources,
        [],
        snapshot=accepted_snapshot,
    )
    run = Run.model_validate(
        {
            "id": "run_legacy_snapshot",
            "request": legacy_request,
            "plan": legacy_plan,
            "resources": resources,
            "status": "waiting",
            "waiting_for": "confirmation",
            "accepted_snapshot": accepted_snapshot,
            "accepted_execution_sha256": previous_fingerprint,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
        strict=True,
    )

    assert run.request.operations == ["SP"]
    assert "required_bindings" not in run.request.structure_input
    assert run.accepted_snapshot == accepted_snapshot
    assert (
        execution_fingerprint(
            run.plan,
            run.resources,
            [],
            snapshot=run.accepted_snapshot,
        )
        == previous_fingerprint
    )

    class NoModelCalls:
        def complete_json(self, *_args, **_kwargs):
            raise AssertionError("confirmation must not ask a model to rewrite the Plan")

    agent = Agent(
        config,
        build_registry(config),
        llm=NoModelCalls(),
        session_id="legacy_snapshot_confirmation",
    )
    original_plan = run.plan.model_copy(deep=True)

    def finish_confirmation(current: Run, *, cancel=None):
        del cancel
        assert current.execution_permission
        assert current.plan == original_plan
        return None

    agent.advance = finish_confirmation
    response = agent.confirm(run)

    assert response.run is run
    assert run.plan == original_plan
    assert run.accepted_execution_sha256 == execution_fingerprint(
        run.plan,
        run.resources,
        [],
        snapshot=run.accepted_snapshot,
    )


def test_ambiguous_energy_after_one_correction_stops_at_intake() -> None:
    registry = build_registry()
    body = json.dumps(
        {
            "intent": "chemistry_compute",
            "operations": ["Opt", "SP"],
            "molecule_query": "water",
            "molecule_input_kind": "name",
            "requested_results": ["energy"],
        }
    )
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlmClient(
        LlmSettings(
            api_key_env="TEST_BG6022_KEY",
            request_timeout_seconds=5,
            structured_output_corrections=1,
        ),
        client=client,
        api_key="offline-test-key",
    )
    try:
        with pytest.raises(LlmError) as raised:
            intake_message(
                llm,
                "同时进行优化和单点计算，但没有明确能量目标",
                capability_catalog=registry.result_capabilities(),
            )
    finally:
        client.close()

    assert raised.value.category == "ambiguous_result"
    assert raised.value.purpose == "intake"
    assert request_count == 2
    assert [call.category for call in llm.calls] == ["ambiguous_result", "ambiguous_result"]


def test_required_optimized_geometry_binding_rejects_initial_geometry_candidate() -> None:
    registry = build_registry()
    request = _composed_request(
        source_operation="Opt",
        source_port="optimized_geometry",
    )
    accepted = proposal_to_plan(
        request,
        _plan({"step_key": "opt", "port": "optimized_geometry"}),
        registry,
        plan_id="plan_opt_to_sp",
        artifact_aliases={"initial_geometry": "initial_geometry"},
    )
    assert accepted.steps[1].inputs["geometry"].step_id == accepted.steps[0].id
    assert accepted.steps[1].inputs["geometry"].port == "optimized_geometry"

    with pytest.raises(ValueError, match="must reference"):
        proposal_to_plan(
            request,
            _plan({"artifact_alias": "initial_geometry"}),
            registry,
            plan_id="plan_opt_to_sp_wrong_source",
            artifact_aliases={"initial_geometry": "initial_geometry"},
        )


def test_required_frequency_geometry_binding_rejects_initial_geometry_candidate() -> None:
    registry = build_registry()
    request = Request(
        id="request_frequency_source",
        description="optimize and calculate frequencies on optimized geometry",
        operations=["Opt", "Freq"],
        requested_results=[ResultTarget(field="vibrational_frequencies")],
        structure_input={
            "required_bindings": [
                {
                    "consumer_operation": "Freq",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ]
        },
    )
    proposal = PlanProposal(
        steps=[
            _step("opt", "optimize_geometry", {"artifact_alias": "initial_geometry"}),
            _step("freq", "frequency", {"step_key": "opt", "port": "optimized_geometry"}),
        ],
        requested_results=[PlanTargetProposal(step_key="freq", field="vibrational_frequencies")],
    )
    accepted = proposal_to_plan(
        request,
        proposal,
        registry,
        plan_id="plan_opt_to_freq",
        artifact_aliases={"initial_geometry": "initial_geometry"},
    )
    assert accepted.steps[1].inputs["geometry"].step_id == accepted.steps[0].id

    wrong_source = proposal.model_copy(
        update={
            "steps": [
                proposal.steps[0],
                proposal.steps[1].model_copy(
                    update={
                        "inputs": {
                            "geometry": InputBindingProposal(artifact_alias="initial_geometry")
                        }
                    }
                ),
            ]
        }
    )
    with pytest.raises(ValueError, match="must reference"):
        proposal_to_plan(
            request,
            wrong_source,
            registry,
            plan_id="plan_opt_to_freq_wrong_source",
            artifact_aliases={"initial_geometry": "initial_geometry"},
        )


def test_required_initial_geometry_binding_survives_an_opt_step() -> None:
    registry = build_registry()
    request = _composed_request(source_operation=None, source_port="initial_geometry")

    accepted = proposal_to_plan(
        request,
        _plan({"artifact_alias": "initial_geometry"}),
        registry,
        plan_id="plan_opt_then_initial_sp",
        artifact_aliases={"initial_geometry": "initial_geometry"},
    )
    assert accepted.steps[0].inputs["geometry"] == accepted.steps[1].inputs["geometry"]

    with pytest.raises(ValueError, match="reuse the original geometry"):
        proposal_to_plan(
            request,
            _plan({"step_key": "opt", "port": "optimized_geometry"}),
            registry,
            plan_id="plan_opt_then_initial_sp_wrong_source",
            artifact_aliases={"initial_geometry": "initial_geometry"},
        )
