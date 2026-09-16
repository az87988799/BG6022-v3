from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from bg6022.agent import Agent, _step_fingerprint
from bg6022.answer import AnswerOutput
from bg6022.config import load_config
from bg6022.models import Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.output_contracts import is_compatible_value
from bg6022.planner import IntakeOutput, PlanProposal, PlanStepProposal, PlanTargetProposal
from bg6022.session import create_run, register_bytes_artifact, save_result, save_run, utc_now
from bg6022.tools.registry import ToolRegistry, build_registry


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


def test_dynamic_output_contract_is_strict_and_public() -> None:
    tool = Tool(
        name="records",
        description="Return a verified record list.",
        results={"records": "record_list"},
        result_properties={"records": "records"},
        result_metadata={"records": {"label": "记录列表", "description": "已验证记录"}},
        requires_compute_permission=False,
    )
    capability = ToolRegistry([tool]).result_capabilities()[0]

    assert capability["type"] == "record_list"
    assert capability["property"] == "records"
    assert is_compatible_value([{"value": 1.5}], "record_list")
    assert not is_compatible_value([{"value": float("nan")}], "record_list")
    with pytest.raises(ValueError, match="unsupported public type"):
        Tool(
            name="unknown_type",
            description="Invalid output.",
            results={"value": "unregistered_numeric_type"},
            requires_compute_permission=False,
        )


def test_verified_initial_geometry_is_delivered_and_reindexed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tool = Tool(
        name="geometry_source",
        description="Return a verified initial geometry.",
        output_ports={"geometry": "molecular_geometry"},
        result_properties={"geometry": "molecular_geometry"},
        result_metadata={
            "geometry": {
                "label": "初始 XYZ 结构文件",
                "description": "已通过 XYZ 解析校验的初始结构",
            }
        },
        requires_compute_permission=False,
    )
    registry = ToolRegistry([tool])
    request = Request(
        id="request_geometry_delivery",
        description="return the initial XYZ file",
        requested_results=[ResultTarget(port="geometry")],
        source="chat",
    )
    step = Step(id="geometry", tool=tool.name)
    plan = Plan(
        id="plan_geometry_delivery",
        request_id=request.id,
        steps=[step],
        requested_results=[ResultTarget(step_id=step.id, port="geometry")],
    )
    run = Run(
        id="run_geometry_delivery",
        request=request,
        plan=plan,
        resources=config.resources,
        status="succeeded",
        session_id="session_geometry_delivery",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    xyz = b"2\ninitial\nH 0 0 0\nH 0 0 0.74\n"
    artifact = register_bytes_artifact(
        config.data_root_path,
        run,
        xyz,
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test:verified_initial_geometry",
        extension=".xyz",
        step_id=step.id,
        attempt=1,
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        artifact_ids=[artifact.id],
        output_ports={"geometry": artifact.id},
        attempt_relative_path="geometry/attempt-01",
        step_fingerprint=_step_fingerprint(step),
    )
    result_path = save_result(config.data_root_path, run, result)
    relative = result_path.relative_to(Path(config.data_root_path) / "runs" / run.id).as_posix()
    run.current_results[step.id] = relative
    run.result_index.append(relative)
    run.step_status[step.id] = "succeeded"
    save_run(config.data_root_path, run)

    agent = Agent(config, registry, llm=object(), session_id=run.session_id)
    response = agent._response_for_run(run, result)

    assert len(response.files) == 1
    delivered = response.files[0]
    assert Path(delivered["path"]).read_bytes() == xyz
    assert delivered["artifact_id"] == artifact.id
    assert delivered["sha256"] == hashlib.sha256(xyz).hexdigest()
    assert "初始 XYZ" in response.text

    agent._session["active_run_id"] = run.id
    catalog, bindings = agent._build_geometry_catalog()
    assert catalog[0]["geometry"]["role"] == "verified initial_geometry"
    assert bindings[catalog[0]["alias"]]["port"] == "geometry"
    verified = agent._verify_history_geometry_binding(bindings[catalog[0]["alias"]])
    assert verified[-1] == "geometry"


def test_answer_needs_tools_allows_one_bounded_intake_review(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    intake_feedback: list[str | None] = []

    def fake_intake(_client: Any, _message: str, **kwargs: Any) -> IntakeOutput:
        intake_feedback.append(kwargs.get("validation_feedback"))
        if len(intake_feedback) == 1:
            return IntakeOutput(intent="daily_qa")
        return IntakeOutput(
            intent="chemistry_compute",
            molecule_query="water",
            molecule_input_kind="name",
            requested_results=["geometry"],
        )

    def fake_plan(_client: Any, request: Request, **_kwargs: Any) -> PlanProposal:
        assert request.requested_results == [ResultTarget(port="geometry")]
        return PlanProposal(
            steps=[
                PlanStepProposal(
                    key="molecule",
                    tool="resolve_molecule",
                    parameters={"query": "water", "input_kind": "name"},
                ),
                PlanStepProposal(
                    key="geometry",
                    tool="generate_geometry",
                    inputs={"molecule": {"step_key": "molecule", "port": "molecule"}},
                ),
            ],
            requested_results=[PlanTargetProposal(step_key="geometry", port="geometry")],
        )

    class AnswerClient:
        calls: list[str] = []

        def complete_json(self, _messages: Any, _schema: Any, **kwargs: Any) -> AnswerOutput:
            self.calls.append(str(kwargs.get("purpose")))
            return AnswerOutput(action="needs_tools", requested_results=["geometry"])

    client = AnswerClient()
    agent = Agent(config, registry, llm=client, session_id="session_route_review")
    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr("bg6022.agent.plan_message", fake_plan)

    def stop_before_tools(run: Run, *, cancel: Any = None) -> None:
        del cancel
        run.status = "waiting"
        run.waiting_for = "confirmation"
        save_run(config.data_root_path, run)
        return None

    monkeypatch.setattr(agent, "advance", stop_before_tools)
    response = agent.handle_message("water 的 xyz 文件")

    assert response.run is not None
    assert response.run.status == "waiting"
    assert len(intake_feedback) == 2
    assert intake_feedback[0] is None
    assert intake_feedback[1] is not None
    assert client.calls == ["answer"]
    assert [item["role"] for item in agent._session["recent_messages"]] == ["user", "assistant"]
