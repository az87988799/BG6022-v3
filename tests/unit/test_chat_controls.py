from __future__ import annotations

import json
from pathlib import Path

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import Plan, Request, Run, Step
from bg6022.planner import (
    InputBindingProposal,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
)
from bg6022.session import create_run, load_run, save_run, save_session, utc_now
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
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
    return load_config(path)


class FakePlanner:
    def complete_json(self, _messages, schema, **kwargs):
        if kwargs.get("purpose") == "intake":
            return schema.model_validate(
                {
                    "intent": "chemistry_compute",
                    "operation": "Opt",
                    "molecule_query": "O",
                    "molecule_input_kind": "smiles",
                    "requested_results": ["opt_final_electronic_energy"],
                },
                strict=True,
            )
        return PlanProposal(
            steps=[
                PlanStepProposal(
                    key="molecule",
                    tool="resolve_molecule",
                    parameters={"query": "O", "input_kind": "smiles"},
                ),
                PlanStepProposal(
                    key="geometry",
                    tool="generate_geometry",
                    inputs={"molecule": InputBindingProposal(step_key="molecule", port="molecule")},
                ),
                PlanStepProposal(
                    key="opt",
                    tool="optimize_geometry",
                    parameters={
                        "method_profile": "r2scan3c",
                        "environment": "gas",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    inputs={"geometry": InputBindingProposal(step_key="geometry", port="geometry")},
                ),
            ],
            requested_results=[
                PlanTargetProposal(step_key="opt", field="opt_final_electronic_energy"),
                PlanTargetProposal(step_key="opt", port="optimized_geometry"),
            ],
        )


class RevisingPlanner(FakePlanner):
    def __init__(self) -> None:
        self.plan_contexts: list[dict[str, object]] = []

    def complete_json(self, messages, schema, **kwargs):
        if kwargs.get("purpose") == "intake":
            return super().complete_json(messages, schema, **kwargs)
        context = json.loads(messages[-1]["content"])
        self.plan_contexts.append(context)
        proposal = super().complete_json(messages, schema, **kwargs)
        if len(self.plan_contexts) == 1:
            return proposal.model_copy(
                update={
                    "requested_results": [
                        PlanTargetProposal(
                            step_key="missing_step", field="opt_final_electronic_energy"
                        )
                    ]
                }
            )
        return proposal


def test_chat_prepares_then_waits_for_one_confirmation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent = Agent(config, build_registry(config), llm=FakePlanner())
    response = agent.handle_message("optimize water")
    assert response.run is not None
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert response.run.pending_data["operation"] == "Opt"
    assert response.run.execution_permission is False

    repeated = agent.confirm(response.run)
    assert repeated.run is not None
    assert repeated.run.status == "failed"
    assert "execution_boundary" in repeated.text
    again = agent.confirm(response.run)
    assert again.run is not None
    assert len(again.run.result_index) == 2
    assert "仍然有效的已完成结果" in again.text
    assert "尚未完成：几何优化" in again.text
    assert "停止原因：" in again.text


def test_planner_receives_local_validation_feedback_for_one_bounded_revision(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    planner = RevisingPlanner()
    agent = Agent(config, build_registry(config), llm=planner)

    response = agent.handle_message("optimize water")

    assert response.run is not None
    assert response.run.status == "waiting"
    assert len(planner.plan_contexts) == 2
    feedback = planner.plan_contexts[1]["validation_feedback"]
    assert isinstance(feedback, str)
    assert "unknown step key" in feedback


def test_invalid_parameter_intake_does_not_create_a_run(tmp_path: Path) -> None:
    class InvalidParameterIntake:
        calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "intent": "chemistry_compute",
                "operation": "Opt",
                "molecule_query": "water",
                "molecule_input_kind": "name",
                "explicit_parameters": {"multiplicity": 1.5},
            }

    config = _config(tmp_path)
    client = InvalidParameterIntake()
    agent = Agent(config, build_registry(config), llm=client)

    response = agent.handle_message("优化水，多重度设为1.5")

    assert "不是受支持的整数值" in response.text
    assert response.run is None
    assert agent._session["active_run_id"] is None
    assert client.calls == 1


def test_ambiguous_parameter_cannot_mutate_a_waiting_request(tmp_path: Path) -> None:
    class AmbiguousParameterIntake:
        def complete_json(self, *_args, **_kwargs):
            return {
                "intent": "chemistry_compute",
                "operation": "Opt",
                "explicit_parameters": {"multiplicity": 3},
            }

    config = _config(tmp_path)
    session_id = "session_parameter_guard"
    request = Request(
        id="request_pending",
        description="pending water Opt",
        operation="Opt",
        explicit_parameters={"charge": 0, "multiplicity": 1},
    )
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
    )
    run = Run(
        id="run_pending",
        request=request,
        plan=Plan(id="plan_pending", request_id=request.id, steps=[step]),
        resources=config.resources,
        status="waiting",
        waiting_for="confirmation",
        session_id=session_id,
        pending_data={"step_id": step.id, "parameters": dict(step.parameters)},
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    save_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "pending_prompt": None,
        },
    )
    agent = Agent(
        config,
        build_registry(config),
        llm=AmbiguousParameterIntake(),
        session_id=session_id,
    )

    response = agent.handle_message("多重度为1或3")
    persisted = load_run(config.data_root_path, run.id)

    assert "存在冲突" in response.text
    assert response.run is not None and response.run.status == "waiting"
    assert persisted.status == "waiting"
    assert persisted.waiting_for == "confirmation"
    assert persisted.request == request
    assert persisted.plan == run.plan
    assert persisted.attempt_counts == {}
