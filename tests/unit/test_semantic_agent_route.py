from __future__ import annotations

from pathlib import Path
from typing import Any

from bg6022.agent import Agent
from bg6022.config import RuntimeSettings, load_config
from bg6022.llm import LlmCall
from bg6022.models import Plan, Request, Requirement, Run, Step, Subject
from bg6022.session import save_run, utc_now
from bg6022.tools.registry import build_registry


def test_semantic_planner_feature_flag_defaults_enabled_after_gates() -> None:
    assert RuntimeSettings(data_root="data").semantic_planner_v1 is True


def _semantic_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = true

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(config_path)


class _SemanticClient:
    def __init__(self) -> None:
        self.calls: list[LlmCall] = []

    def complete_json(self, _messages: Any, schema: Any, **kwargs: Any) -> Any:
        purpose = str(kwargs["purpose"])
        self.calls.append(
            LlmCall(
                purpose=purpose,
                request_model="test-model",
                schema_version="semantic-test",
                elapsed_seconds=0.01,
                usage={"prompt_tokens": 120, "completion_tokens": 24},
            )
        )
        return schema.model_validate(
            {
                "mode": "compute",
                "subjects": [
                    {
                        "key": "subject_1",
                        "query": "water",
                        "input_kind": "name",
                        "evidence": "water",
                    }
                ],
                "tasks": [
                    {
                        "key": "t1",
                        "subject_key": "subject_1",
                        "capability": "optimize_geometry",
                        "method_request": "PBE0",
                        "parameters": {},
                        "requested_properties": [],
                    }
                ],
                "relations": [],
            },
            strict=True,
        )


def test_feature_flag_routes_compute_to_one_semantic_call_and_no_planner(
    monkeypatch, tmp_path: Path
) -> None:
    config = _semantic_config(tmp_path)
    registry = build_registry(config)
    client = _SemanticClient()
    agent = Agent(config, registry, llm=client, session_id="semantic_agent_route")
    planner_calls: list[str] = []

    def forbidden_planner(*_args: Any, **_kwargs: Any) -> Any:
        planner_calls.append("planner")
        raise AssertionError("semantic feature path must not call the Planner LLM")

    monkeypatch.setattr("bg6022.agent.plan_message", forbidden_planner)

    def pause_before_execution(run: Run, *, cancel: Any = None) -> None:
        del cancel
        run.status = "waiting"
        run.waiting_for = "confirmation"
        save_run(config.data_root_path, run)
        return None

    monkeypatch.setattr(agent, "advance", pause_before_execution)
    response = agent.handle_message("Optimize water with PBE0.")

    assert response.run is not None
    assert response.run.status == "waiting"
    assert [call.purpose for call in client.calls] == ["semantic"]
    assert planner_calls == []
    assert [step.tool for step in response.run.plan.steps] == [
        "resolve_molecule",
        "generate_geometry",
        "optimize_geometry",
    ]
    requirement = response.run.request.requirements[0]
    assert requirement.parameters["method_profile"] == "pbe0_d3bj_def2svp"
    assert response.run.plan.requested_results[0].field == "opt_final_electronic_energy"
    assert response.run.plan.requested_results[1].port == "optimized_geometry"


def test_ambiguous_iteration_update_is_clarified_without_mutating_waiting_run(
    monkeypatch, tmp_path: Path
) -> None:
    config = _semantic_config(tmp_path)
    registry = build_registry(config)
    client = _SemanticClient()
    agent = Agent(config, registry, llm=client, session_id="semantic_iteration_guard")
    requirement = Requirement(
        id="req_opt",
        subject_id="subject_1",
        capability="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
    )
    request = Request(
        id="request_pending",
        description="pending water optimization",
        source="chat",
        subjects={"subject_1": Subject(key="subject_1", molecule_query="water")},
        requirements=[requirement],
    )
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters=dict(requirement.parameters),
        requirement_id=requirement.id,
        subject_id=requirement.subject_id,
    )
    run = Run(
        id="run_pending",
        request=request,
        plan=Plan(id="plan_pending", request_id=request.id, steps=[step]),
        resources=config.resources,
        status="waiting",
        waiting_for="confirmation",
        session_id=agent.session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    monkeypatch.setattr(agent, "_coerce_run", lambda _run_id=None: run)

    response = agent.handle_message("迭代上限设为 100")

    assert "几何优化迭代或 SCF 电子迭代" in response.text
    assert response.run is run
    assert run.request.requirements[0].parameters == requirement.parameters
    assert client.calls == []
