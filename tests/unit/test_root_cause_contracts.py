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
    _request_target_matches,
    intake_blocking_requirements,
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
    assert {"geometry", "geometry_atom_count", "molecule_formula"} <= names


def test_intake_geometry_target_and_legacy_geometry_alias_keep_distinct_meanings() -> None:
    registry = build_registry()

    assert registry.resolve_result_target("geometry", ["SP"], canonical_only=True) == (
        ResultTarget(port="geometry")
    )
    assert registry.resolve_result_target("geometry", ["Opt"], canonical_only=True) == (
        ResultTarget(port="geometry")
    )
    assert registry.resolve_result_target("optimized_geometry", ["Opt"], canonical_only=True) == (
        ResultTarget(port="optimized_geometry")
    )
    assert registry.resolve_result_target("geometry", ["Opt"]) == ResultTarget(
        port="optimized_geometry"
    )
    intake_request = request_from_intake(
        "optimize and return the initial geometry",
        IntakeOutput(
            intent="chemistry_compute",
            operations=["Opt"],
            requested_results=["geometry"],
        ),
        request_id="request_new_geometry_target",
        registry=registry,
    )
    assert intake_request.requested_results == [ResultTarget(port="geometry")]
    both_geometries = request_from_intake(
        "return both the initial and optimized geometries",
        IntakeOutput(
            intent="chemistry_compute",
            operations=["Opt"],
            requested_results=["geometry", "optimized_geometry"],
        ),
        request_id="request_both_geometry_targets",
        registry=registry,
    )
    assert both_geometries.requested_results == [
        ResultTarget(port="geometry"),
        ResultTarget(port="optimized_geometry"),
    ]

    legacy_request = Request(
        id="request_legacy_geometry_alias",
        description="legacy geometry target",
        operations=["Opt"],
        requested_results=[ResultTarget(field="geometry")],
    )
    assert _request_target_matches(
        legacy_request.requested_results[0],
        ResultTarget(port="optimized_geometry"),
        legacy_request,
        registry,
    )
    assert not _request_target_matches(
        legacy_request.requested_results[0],
        ResultTarget(port="geometry"),
        legacy_request,
        registry,
    )


def test_single_operation_on_history_geometry_uses_only_the_history_alias() -> None:
    registry = build_registry()
    schema = _intake_schema((), registry.result_capabilities(), registry=registry)
    payload = {
        "intent": "chemistry_compute",
        "operations": ["SP"],
        "requested_results": ["sp_electronic_energy"],
        "molecule_query": None,
        "molecule_input_kind": None,
        "history_geometry_alias": "geometry_1",
        "structure_input": {},
        "electronic_state_candidates": [],
        "explicit_parameters": {},
        "missing_fields": [],
        "unresolved_results": [],
    }

    intake = schema.model_validate(payload, strict=True)
    request = request_from_intake(
        "Reuse the recent optimized structure and run SP.",
        intake,
        request_id="request_history_sp",
        registry=registry,
    )

    assert request.structure_input["history_geometry_alias"] == "geometry_1"
    assert request.structure_input["required_bindings"] == []
    assert request.operations == ["SP"]


def test_agent_preserves_trailing_newline_in_inline_xyz() -> None:
    registry = build_registry()
    config = load_config(Path("config.toml"))
    xyz = "3\nwater\nO 0 0 0\nH 0 1 0\nH 1 0 0\n"
    intake_body = {
        "intent": "chemistry_compute",
        "operations": ["SP"],
        "requested_results": ["sp_electronic_energy"],
        "molecule_query": None,
        "molecule_input_kind": None,
        "history_geometry_alias": None,
        "structure_input": {},
        "explicit_parameters": {"charge": 0, "multiplicity": 1},
        "electronic_state_candidates": [],
        "missing_fields": [],
        "unresolved_results": [],
    }
    planner_body = {
        "steps": [
            {
                "key": "sp",
                "tool": "single_point",
                "parameters": {"charge": 0, "multiplicity": 1},
                "inputs": {"geometry": {"artifact_alias": "request_geometry"}},
            }
        ],
        "requested_results": [{"step_key": "sp", "field": "sp_electronic_energy"}],
    }

    class StubLlm:
        calls = []

        def complete_json(self, _messages, schema, **kwargs):
            model = schema
            payload = intake_body if kwargs.get("purpose") == "intake" else planner_body
            return model.model_validate(payload, strict=True)

    agent = Agent(config, registry, llm=StubLlm(), session_id="inline_xyz_newline")
    response = agent.handle_message("run SP on this geometry\n\n" + xyz)

    assert response.run is not None
    assert response.run.request.structure_input["xyz_text"] == xyz


def test_preparation_results_can_be_requested_with_sp_and_need_real_producers() -> None:
    registry = build_registry()
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        requested_results=["molecule_formula", "geometry_atom_count", "sp_electronic_energy"],
    )
    request = request_from_intake(
        "resolve water, report its formula and atom count, then calculate SP energy",
        intake,
        request_id="request_preparation_results",
        registry=registry,
    )
    proposal = PlanProposal(
        steps=[
            PlanStepProposal(
                key="molecule",
                tool="resolve_molecule",
                parameters={"query": "water", "input_kind": "name"},
            ),
            PlanStepProposal(
                key="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputBindingProposal(step_key="molecule", port="molecule")},
            ),
            PlanStepProposal(
                key="sp",
                tool="single_point",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputBindingProposal(step_key="geometry", port="geometry")},
            ),
        ],
        requested_results=[
            PlanTargetProposal(step_key="molecule", field="molecule_formula"),
            PlanTargetProposal(step_key="geometry", field="geometry_atom_count"),
            PlanTargetProposal(step_key="sp", field="sp_electronic_energy"),
        ],
    )

    plan = proposal_to_plan(request, proposal, registry, plan_id="plan_preparation_results")

    assert plan.requested_results == [
        ResultTarget(step_id=plan.steps[0].id, field="molecule_formula"),
        ResultTarget(step_id=plan.steps[1].id, field="geometry_atom_count"),
        ResultTarget(step_id=plan.steps[2].id, field="sp_electronic_energy"),
    ]
    assert plan.steps[2].inputs["geometry"].step_id == plan.steps[1].id
    with pytest.raises(ValueError, match="no producer"):
        proposal_to_plan(
            request,
            PlanProposal(
                steps=[
                    PlanStepProposal(
                        key="sp",
                        tool="single_point",
                        parameters={"charge": 0, "multiplicity": 1},
                        inputs={
                            "geometry": InputBindingProposal(artifact_alias="initial_geometry")
                        },
                    )
                ],
                requested_results=[
                    PlanTargetProposal(step_key="sp", field="molecule_formula"),
                    PlanTargetProposal(step_key="sp", field="geometry_atom_count"),
                    PlanTargetProposal(step_key="sp", field="sp_electronic_energy"),
                ],
            ),
            registry,
            plan_id="plan_missing_preparation_producers",
            artifact_aliases={"initial_geometry": "initial_geometry"},
        )


def test_unresolved_requirements_block_but_deferred_charge_and_multiplicity_do_not() -> None:
    registry = build_registry()
    deferred = IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt"],
        missing_fields=["charge", "multiplicity"],
    )
    unknown = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        requested_results=["sp_electronic_energy"],
        missing_fields=["Gibbs free energy is not supported"],
    )
    unresolved = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        requested_results=["sp_electronic_energy"],
        unresolved_results=["Gibbs free energy is not supported"],
    )

    assert intake_blocking_requirements(deferred, registry) == ()
    assert intake_blocking_requirements(unknown, registry) == (
        "Gibbs free energy is not supported",
    )
    assert intake_blocking_requirements(unresolved, registry) == (
        "Gibbs free energy is not supported",
    )
    with pytest.raises(ValueError, match="Gibbs free energy is not supported"):
        request_from_intake(
            "compute SP energy and Gibbs free energy",
            unresolved,
            request_id="request_unresolved_target",
            registry=registry,
        )
    qa = IntakeOutput(intent="chemistry_qa", missing_fields=["Gibbs free energy"])
    assert intake_blocking_requirements(qa, registry) == ()


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


def test_intake_parameter_catalog_is_tool_derived_and_operation_scoped() -> None:
    registry = build_registry()
    parameters = registry.request_parameter_capabilities()
    optimize = next(item for item in parameters if item["tool"] == "optimize_geometry")
    assert {"geom_maxiter", "scf_maxiter"} <= set(optimize["schema"]["properties"])
    assert (
        "Geometry optimization iterations"
        in optimize["schema"]["properties"]["geom_maxiter"]["description"]
    )
    assert (
        "SCF electronic iterations"
        in optimize["schema"]["properties"]["scf_maxiter"]["description"]
    )

    schema = _intake_schema((), registry.result_capabilities(), registry=registry)
    valid = schema.model_validate(
        {
            "intent": "chemistry_compute",
            "operations": ["Opt", "Freq"],
            "explicit_parameters": {"geom_maxiter": 100, "scf_maxiter": 80},
        },
        strict=True,
    )
    assert valid.explicit_parameters == {"geom_maxiter": 100, "scf_maxiter": 80}
    with pytest.raises(ValueError, match="outside the Tool catalog"):
        schema.model_validate(
            {
                "intent": "chemistry_compute",
                "operations": ["SP"],
                "explicit_parameters": {"geom_maxiter": 100},
            },
            strict=True,
        )
    with pytest.raises(ValueError, match="outside the Tool catalog"):
        schema.model_validate(
            {
                "intent": "chemistry_compute",
                "operations": ["Opt"],
                "explicit_parameters": {"max_iterations": 100},
            },
            strict=True,
        )


def test_intake_corrects_parameter_name_once_using_real_tool_catalog() -> None:
    registry = build_registry()
    base = {
        "intent": "chemistry_compute",
        "operations": ["Opt"],
        "molecule_query": "water",
        "molecule_input_kind": "name",
        "requested_results": ["opt_final_electronic_energy"],
    }
    bodies = [
        {**base, "explicit_parameters": {"max_iterations": 100}},
        {**base, "explicit_parameters": {"geom_maxiter": 100}},
    ]
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [
                    {
                        "message": {"content": json.dumps(bodies.pop(0))},
                        "finish_reason": "stop",
                    }
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
            "几何优化迭代上限为100",
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
        request = request_from_intake(
            "几何优化迭代上限为100",
            intake,
            request_id="request_parameter_name_correction",
            registry=registry,
        )
    finally:
        client.close()

    assert request.explicit_parameters == {"geom_maxiter": 100}
    assert len(request_bodies) == 2
    first_user_payload = json.loads(request_bodies[0]["messages"][-1]["content"])
    offered = first_user_payload["parameter_capability_catalog"]
    opt_parameters = next(item for item in offered if item["tool"] == "optimize_geometry")
    assert "geom_maxiter" in opt_parameters["schema"]["properties"]
    assert [call.category for call in llm.calls] == ["schema_error", "success"]
    assert [call.structured_correction_count for call in llm.calls] == [0, 1]


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


def test_intake_preserves_inline_xyz_bytes_without_sending_coordinates_for_reasoning() -> None:
    registry = build_registry()
    xyz = (
        "9\n"
        "BG6022 v3 RDKit ETKDGv3 seed 61453\n"
        "C 0.877486959719 0.186878790281 0.048503482205\n"
        "C -0.464154573725 -0.481312634760 -0.045048464337\n"
        "O -1.494024866655 0.355496472209 -0.417791348573\n"
        "H 0.796951895584 1.300046601842 0.043713246997\n"
        "H 1.338309215505 -0.165508583886 1.010736576699\n"
        "H 1.585091788618 -0.141887083072 -0.747137411129\n"
        "H -0.765723813954 -0.995610105095 0.907513662445\n"
        "H -0.397274395559 -1.305459584944 -0.788498759346\n"
        "H -1.476662209533 1.247356127426 -0.011990984960\n"
    )
    message = (
        "Use exactly this ethanol XYZ. Run Opt with geom_maxiter=1, then Freq and an "
        "independent SP on the successful Opt geometry; charge=0, multiplicity=1.\n\n" + xyz
    )
    returned = {
        "intent": "chemistry_compute",
        "operations": ["Opt", "Freq", "SP"],
        "molecule_query": "ethanol",
        "molecule_input_kind": "name",
        "explicit_parameters": {
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 1,
        },
        "requested_results": ["vibrational_frequencies", "sp_electronic_energy"],
        "structure_input": {},
        "missing_fields": [],
        "unresolved_results": [],
    }
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "response-model",
                "choices": [
                    {
                        "message": {"content": json.dumps(returned)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 25, "completion_tokens": 20},
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
            message,
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
    finally:
        client.close()

    assert len(request_bodies) == 1
    assert request_bodies[0]["thinking"] == {"type": "disabled"}
    model_context = json.loads(request_bodies[0]["messages"][-1]["content"])
    assert xyz not in model_context["message"]
    assert "Use exactly this ethanol XYZ" in model_context["message"]
    assert "Valid inline XYZ for 9 atoms" in model_context["message"]
    assert intake.structure_input["xyz_text"] == xyz
    assert intake.molecule_query is None
    assert intake.molecule_input_kind is None


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
