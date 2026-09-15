from __future__ import annotations

import json
from pathlib import Path

import pytest

from bg6022.agent import Agent, _step_fingerprint
from bg6022.config import load_config
from bg6022.models import Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.planner import intake_message
from bg6022.session import create_run, load_run, save_result, save_run, save_session, utc_now
from bg6022.tools.registry import ToolRegistry


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


def _result_tool() -> Tool:
    return Tool(
        name="measure",
        description="Return a verified scalar measurement.",
        results={"energy": "Eh"},
        result_properties={"energy": "electronic_energy"},
        result_metadata={
            "energy": {
                "label": "测试电子能",
                "description": "测试工具产生的电子能",
                "caveat": "测试事实，不代表真实化学计算",
            }
        },
        requires_compute_permission=False,
    )


def _save_scalar_run(
    data_root: Path,
    *,
    session_id: str,
    run_id: str,
    description: str,
    value_token: str,
) -> Run:
    request = Request(
        id=f"request_{run_id}",
        description=description,
        operation="SP",
        requested_results=[ResultTarget(step_id="measure", field="energy")],
        source="chat",
    )
    step = Step(id="measure", tool="measure")
    plan = Plan(
        id=f"plan_{run_id}",
        request_id=request.id,
        steps=[step],
        requested_results=request.requested_results,
    )
    run = Run(
        id=run_id,
        request=request,
        plan=plan,
        resources={"cores": 4, "memory_mb": 1024, "maxcore_mb": 192, "max_concurrent_jobs": 1},
        status="succeeded",
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(data_root, run)
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        values={"energy": {"value": float(value_token), "unit": "Eh", "token": value_token}},
        attempt_relative_path="measure/attempt-01",
        step_fingerprint=_step_fingerprint(step),
    )
    result_path = save_result(data_root, run, result)
    run.result_index.append(result_path.relative_to(data_root / "runs" / run.id).as_posix())
    run.current_results[step.id] = run.result_index[-1]
    save_run(data_root, run)
    return run


class SelectingClient:
    def __init__(
        self,
        *,
        choose_description: str | None = None,
        ref: str | None = None,
        target_property: str = "electronic_energy",
        evidence: str = "能量",
    ):
        self.choose_description = choose_description
        self.ref = ref
        self.target_property = target_property
        self.evidence = evidence
        self.catalog: list[dict[str, object]] = []

    def complete_json(self, messages, _schema, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        self.catalog = payload["result_catalog"]
        subject_ref = self.ref
        if self.choose_description is not None:
            item = next(
                item
                for item in self.catalog
                if item["task"]["description"] == self.choose_description
            )
            subject_ref = item["subject_ref"]
        if subject_ref is None:
            subject_ref = self.catalog[0]["subject_ref"]
        return {
            "intent": "context_query",
            "query_selection": {
                "status": "selected",
                "targets": [
                    {
                        "subject_ref": subject_ref,
                        "property": self.target_property,
                        "evidence": self.evidence,
                    }
                ],
            },
        }


def test_query_reloads_the_saved_result_after_agent_recreation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tool = _result_tool()
    registry = ToolRegistry([tool])
    run = _save_scalar_run(
        Path(config.data_root_path),
        session_id="session_query",
        run_id="run_water",
        description="water single point",
        value_token="-76.418938720831",
    )
    save_session(
        config.data_root_path,
        "session_query",
        {
            "session_id": "session_query",
            "active_run_id": run.id,
            "recent_results": [{"run_id": run.id, "status": "succeeded", "values": {"energy": 1}}],
            "recent_messages": [],
            "pending_prompt": None,
        },
    )

    client = SelectingClient(ref="t1")
    agent = Agent(config, registry, llm=client, session_id="session_query")
    response = agent.handle_message("这个结构的能量是多少？")

    assert response.result is not None
    assert response.result.values["energy"]["token"] == "-76.418938720831"
    assert "-76.418938720831 Eh" in response.text
    assert "**1 Eh**" not in response.text
    assert "result_path" not in json.dumps(client.catalog, ensure_ascii=False)
    assert "76.418938720831" not in json.dumps(client.catalog, ensure_ascii=False)


def test_query_selection_cannot_use_a_reference_outside_this_round() -> None:
    class InvalidClient:
        def complete_json(self, *_args, **_kwargs):
            return {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": "t999",
                            "property": "electronic_energy",
                            "evidence": "energy",
                        }
                    ],
                },
            }

    with pytest.raises(ValueError, match="outside this catalog"):
        intake_message(
            InvalidClient(),
            "what is the energy?",
            result_catalog=[{"subject_ref": "t1", "result": {"property": "electronic_energy"}}],
        )


def test_query_property_must_match_the_user_cited_property_phrase() -> None:
    class MismatchedPropertyClient:
        def complete_json(self, *_args, **_kwargs):
            return {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": "t1",
                            "property": "electronic_energy",
                            "evidence": "零点能",
                        }
                    ],
                },
            }

    intake = intake_message(
        MismatchedPropertyClient(),
        "零点能是多少？",
        result_catalog=[
            {
                "subject_ref": "t1",
                "result": {"property": "electronic_energy"},
            }
        ],
    )

    assert intake.query_selection is not None
    assert intake.query_selection.status == "clarify"
    assert intake.query_selection.targets == []


def test_structure_as_query_subject_is_not_a_geometry_output_request() -> None:
    class ConfusedSubjectClient:
        def complete_json(self, *_args, **_kwargs):
            return {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": "t1",
                            "property": "molecular_geometry",
                            "evidence": "这个结构",
                        }
                    ],
                },
            }

    intake = intake_message(
        ConfusedSubjectClient(),
        "这个结构的能量是多少？",
        result_catalog=[
            {
                "subject_ref": "t1",
                "result": {"property": "electronic_energy"},
            }
        ],
    )

    assert intake.query_selection is not None
    assert intake.query_selection.status == "clarify"
    assert intake.query_selection.targets == []


def test_generic_energy_phrase_cannot_stand_in_for_free_energy() -> None:
    class GenericEnergyClient:
        def complete_json(self, *_args, **_kwargs):
            return {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": "t1",
                            "property": "electronic_energy",
                            "evidence": "energy",
                        }
                    ],
                },
            }

    intake = intake_message(
        GenericEnergyClient(),
        "What is the free energy?",
        result_catalog=[
            {
                "subject_ref": "t1",
                "result": {"property": "electronic_energy"},
            }
        ],
    )

    assert intake.query_selection is not None
    assert intake.query_selection.status == "clarify"
    assert intake.query_selection.targets == []


def test_selected_electronic_energy_cannot_answer_a_missing_zero_point_property(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = ToolRegistry([_result_tool()])
    run = _save_scalar_run(
        Path(config.data_root_path),
        session_id="session_property_scope",
        run_id="run_property_scope",
        description="water single point",
        value_token="-76.418938720831",
    )
    save_session(
        config.data_root_path,
        "session_property_scope",
        {
            "session_id": "session_property_scope",
            "active_run_id": run.id,
            "recent_results": [{"run_id": run.id}],
            "recent_messages": [],
            "pending_prompt": None,
        },
    )

    response = Agent(
        config,
        registry,
        llm=SelectingClient(ref="t1", target_property="zero_point_energy", evidence="零点能"),
        session_id="session_property_scope",
    ).handle_message("零点能是多少？")

    assert "尚未得到所问性质" in response.text
    assert "-76.418938720831" not in response.text


def test_catalog_selects_the_indexed_water_result_without_falling_back_to_latest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = ToolRegistry([_result_tool()])
    water = _save_scalar_run(
        Path(config.data_root_path),
        session_id="session_multi",
        run_id="run_water",
        description="water single point",
        value_token="-76.418938720831",
    )
    ethanol = _save_scalar_run(
        Path(config.data_root_path),
        session_id="session_multi",
        run_id="run_ethanol",
        description="ethanol single point",
        value_token="-154.000000000000",
    )
    save_session(
        config.data_root_path,
        "session_multi",
        {
            "session_id": "session_multi",
            "active_run_id": ethanol.id,
            "recent_results": [{"run_id": water.id}, {"run_id": ethanol.id}],
            "recent_messages": [],
            "pending_prompt": None,
        },
    )

    client = SelectingClient(choose_description="water single point")
    response = Agent(config, registry, llm=client, session_id="session_multi").handle_message(
        "查询水的上一次能量"
    )

    assert "-76.418938720831 Eh" in response.text
    assert "-154.000000000000" not in response.text


def test_stale_fingerprint_and_foreign_session_are_not_catalog_facts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ToolRegistry([_result_tool()])
    run = _save_scalar_run(
        Path(config.data_root_path),
        session_id="foreign_session",
        run_id="run_foreign",
        description="foreign result",
        value_token="-1.0",
    )
    save_session(
        config.data_root_path,
        "local_session",
        {
            "session_id": "local_session",
            "active_run_id": run.id,
            "recent_results": [{"run_id": run.id}],
            "recent_messages": [],
            "pending_prompt": None,
        },
    )
    agent = Agent(config, registry, llm=None, session_id="local_session")
    assert agent._build_query_catalog() == []


def test_current_result_is_removed_when_the_step_fingerprint_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ToolRegistry([_result_tool()])
    run = _save_scalar_run(
        Path(config.data_root_path),
        session_id="session_stale",
        run_id="run_stale",
        description="stale result",
        value_token="-1.0",
    )
    changed = load_run(config.data_root_path, run.id)
    changed.plan = changed.plan.model_copy(
        update={"steps": [Step(id="measure", tool="measure", goal_checks=["changed"])]}
    )
    save_run(config.data_root_path, changed)
    save_session(
        config.data_root_path,
        "session_stale",
        {
            "session_id": "session_stale",
            "active_run_id": run.id,
            "recent_results": [{"run_id": run.id}],
            "recent_messages": [],
            "pending_prompt": None,
        },
    )

    agent = Agent(config, registry, llm=None, session_id="session_stale")
    assert agent._build_query_catalog() == []
