from __future__ import annotations

import json
from pathlib import Path

import pytest

from bg6022.agent import Agent, _requests_history_geometry, _step_fingerprint
from bg6022.config import load_config
from bg6022.llm import LlmError
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.planner import IntakeOutput, intake_message
from bg6022.session import (
    create_run,
    load_run,
    load_session,
    register_bytes_artifact,
    save_result,
    save_run,
    save_session,
    utc_now,
)
from bg6022.tools.registry import ToolRegistry, build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false
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


def _save_three_step_opt_run(
    config,
    *,
    session_id: str,
    run_id: str,
    description: str,
    energy_token: str | None,
) -> tuple[Run, list[Result]]:
    """Persist a test-only resolve -> geometry -> Opt history entry."""

    data_root = Path(config.data_root_path)
    resolve = Step(
        id="resolve",
        tool="resolve_molecule",
        parameters={"query": "water", "input_kind": "name"},
    )
    geometry = Step(
        id="geometry",
        tool="generate_geometry",
        inputs={"molecule": InputReference(step_id=resolve.id, port="molecule")},
    )
    optimize = Step(
        id="optimize",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(step_id=geometry.id, port="geometry")},
    )
    target = (
        ResultTarget(step_id=optimize.id, field="opt_final_electronic_energy")
        if energy_token is not None
        else ResultTarget(step_id=optimize.id, port="optimized_geometry")
    )
    request = Request(
        id=f"request_{run_id}",
        description=description,
        operation="Opt",
        requested_results=[target],
        source="chat",
    )
    plan = Plan(
        id=f"plan_{run_id}",
        request_id=request.id,
        steps=[resolve, geometry, optimize],
        requested_results=[target],
    )
    run = Run(
        id=run_id,
        request=request,
        plan=plan,
        resources=config.resources,
        status="succeeded",
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(data_root, run)

    molecule = register_bytes_artifact(
        data_root,
        run,
        json.dumps(
            {
                "schema_version": 1,
                "facts": {
                    "formula": "H2O",
                    "title": description,
                    "query": "water",
                    "cid": 962,
                    "atom_count": 3,
                },
                "query": "water",
                "input_kind": "name",
                "source": "test-only synthetic artifact",
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        artifact_type="molecule",
        role="resolved_molecule",
        source="test-only synthetic artifact",
        extension=".json",
        step_id=resolve.id,
        attempt=1,
    )
    xyz = (
        b"3\nwater test geometry\n"
        b"O 0.000000 0.000000 0.000000\n"
        b"H 0.800000 0.600000 0.000000\n"
        b"H -0.800000 0.600000 0.000000\n"
    )
    initial_geometry = register_bytes_artifact(
        data_root,
        run,
        xyz,
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test-only synthetic artifact",
        extension=".xyz",
        step_id=geometry.id,
        attempt=1,
        metadata={"molecule_artifact_id": molecule.id},
    )
    optimized_geometry = register_bytes_artifact(
        data_root,
        run,
        xyz,
        artifact_type="molecular_geometry",
        role="optimized_geometry",
        source="test-only synthetic artifact",
        extension=".xyz",
        step_id=optimize.id,
        attempt=1,
        metadata={"molecule_artifact_id": molecule.id},
    )

    step_results = [
        Result(
            run_id=run.id,
            step_id=resolve.id,
            attempt=1,
            status="succeeded",
            values={"molecule_formula": "H2O", "formal_charge": 0},
            artifact_ids=[molecule.id],
            output_ports={"molecule": molecule.id},
            attempt_relative_path="resolve/attempt-01",
            step_fingerprint=_step_fingerprint(resolve),
        ),
        Result(
            run_id=run.id,
            step_id=geometry.id,
            attempt=1,
            status="succeeded",
            values={"geometry_atom_count": 3},
            artifact_ids=[initial_geometry.id],
            output_ports={"geometry": initial_geometry.id},
            input_artifact_ids=[molecule.id],
            input_bindings={"molecule": molecule.id},
            attempt_relative_path="geometry/attempt-01",
            step_fingerprint=_step_fingerprint(geometry),
        ),
        Result(
            run_id=run.id,
            step_id=optimize.id,
            attempt=1,
            status="succeeded",
            values=(
                {
                    "opt_final_electronic_energy": {
                        "value": float(energy_token),
                        "unit": "Eh",
                        "token": energy_token,
                    }
                }
                if energy_token is not None
                else {}
            ),
            artifact_ids=[optimized_geometry.id],
            output_ports={"optimized_geometry": optimized_geometry.id},
            input_artifact_ids=[initial_geometry.id],
            input_bindings={"geometry": initial_geometry.id},
            attempt_relative_path="optimize/attempt-01",
            step_fingerprint=_step_fingerprint(optimize),
        ),
    ]
    for result in step_results:
        result_path = save_result(data_root, run, result)
        relative = result_path.relative_to(data_root / "runs" / run.id).as_posix()
        run.result_index.append(relative)
        run.current_results[result.step_id] = relative
        run.step_status[result.step_id] = "succeeded"
    save_run(data_root, run)
    return run, step_results


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


class ComparingClient:
    def __init__(self, descriptions: tuple[str, str]) -> None:
        self.descriptions = descriptions
        self.catalog: list[dict[str, object]] = []

    def complete_json(self, messages, _schema, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        self.catalog = payload["result_catalog"]
        refs = {
            description: next(
                item["subject_ref"]
                for item in self.catalog
                if item["task"]["description"] == description
                and item["result"]["property"] == "electronic_energy"
            )
            for description in self.descriptions
        }
        return {
            "intent": "context_query",
            "query_selection": {
                "status": "selected",
                "targets": [
                    {
                        "subject_ref": refs[description],
                        "property": "electronic_energy",
                        "evidence": "能量",
                    }
                    for description in self.descriptions
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

    with pytest.raises(LlmError, match="outside this catalog") as error:
        intake_message(
            InvalidClient(),
            "what is the energy?",
            result_catalog=[{"subject_ref": "t1", "result": {"property": "electronic_energy"}}],
        )
    assert error.value.category == "schema_error"
    assert error.value.diagnostics[0]["path"] == "query_selection.targets"


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


def test_three_step_runs_preserve_current_and_historical_energy_for_queries(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    old, old_results = _save_three_step_opt_run(
        config,
        session_id="session_three_step",
        run_id="run_water_old",
        description="previous water Opt",
        energy_token="-76.400000000000",
    )
    current, current_results = _save_three_step_opt_run(
        config,
        session_id="session_three_step",
        run_id="run_water_current",
        description="current water Opt",
        energy_token="-76.418938720831",
    )
    agent = Agent(config, registry, llm=None, session_id="session_three_step")

    for run, results in ((old, old_results), (current, current_results)):
        for result in results:
            agent._record_result_summary(run, result)

    retained = agent._session["recent_results"]
    assert [item["run_id"] for item in retained] == [old.id, current.id]
    assert len({item["run_id"] for item in retained}) == 2

    historical_client = SelectingClient(choose_description="previous water Opt")
    agent.llm = historical_client
    historical = agent.handle_message("上一个水任务的能量是多少？")
    assert "-76.400000000000 Eh" in historical.text
    assert "-76.418938720831" not in historical.text

    current_catalog_fact = next(
        item
        for item in historical_client.catalog
        if item["task"]["description"] == "current water Opt"
        and item["result"]["property"] == "electronic_energy"
    )
    assert current_catalog_fact["subject_ref"] == "t1"
    current_geometry_fact = next(
        item
        for item in historical_client.catalog
        if item["task"]["description"] == "current water Opt"
        and item["step"]["tool"] == "generate_geometry"
    )
    assert current_geometry_fact["subject_ref"] == "t2"
    assert len({item["subject_ref"] for item in historical_client.catalog}) >= 4

    comparison_client = ComparingClient(("previous water Opt", "current water Opt"))
    agent.llm = comparison_client
    comparison = agent.handle_message("比较这两个水任务的能量")
    assert "-76.400000000000 Eh" in comparison.text
    assert "-76.418938720831 Eh" in comparison.text
    assert (
        len(
            {
                item["subject_ref"]
                for item in comparison_client.catalog
                if item["result"]["property"] == "electronic_energy"
            }
        )
        >= 2
    )


def test_history_geometry_language_does_not_route_result_query_into_compute_selection(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    old_run, old_results = _save_three_step_opt_run(
        config,
        session_id="session_recent_geometry_query",
        run_id="run_recent_geometry_old",
        description="previous water Opt",
        energy_token="-76.400000000000",
    )
    current_run, current_results = _save_three_step_opt_run(
        config,
        session_id="session_recent_geometry_query",
        run_id="run_recent_geometry_current",
        description="current water Opt",
        energy_token="-76.418938720831",
    )
    agent = Agent(config, registry, llm=None, session_id="session_recent_geometry_query")
    for run, results in ((old_run, old_results), (current_run, current_results)):
        for result in results:
            agent._record_result_summary(run, result)
    initial_run_count = len(list((Path(config.data_root_path) / "runs").iterdir()))

    client = SelectingClient(choose_description="current water Opt")
    agent.llm = client
    response = agent.handle_message("刚才优化结构的能量是多少？")

    assert response.result is not None
    assert response.result.run_id == current_run.id
    assert "-76.418938720831 Eh" in response.text
    assert "-76.400000000000" not in response.text
    assert response.run is not None and response.run.id == current_run.id
    assert len(list((Path(config.data_root_path) / "runs").iterdir())) == initial_run_count
    assert len(client.catalog) >= 2


@pytest.mark.parametrize(
    ("category", "expected_text"),
    [
        ("empty_response", "请求解析阶段未获得有效模型响应"),
        ("ambiguous_result", "Opt 和 SP 同时请求时"),
    ],
)
def test_llm_intake_failure_names_stage_and_persists_bounded_diagnostic(
    tmp_path: Path, category: str, expected_text: str
) -> None:
    config = _config(tmp_path)

    class EmptyIntakeClient:
        calls: list[object] = []

        def complete_json(self, *_args, **_kwargs):
            raise LlmError(
                "model returned an empty JSON response",
                category=category,
                purpose="intake",
            )

    session_id = "session_intake_failure"
    response = Agent(
        config,
        build_registry(),
        llm=EmptyIntakeClient(),
        session_id=session_id,
    ).handle_message("计算水分子的能量")

    assert expected_text in response.text
    assert response.run is None
    runs_path = Path(config.data_root_path) / "runs"
    assert not runs_path.exists() or list(runs_path.iterdir()) == []
    saved_session = load_session(config.data_root_path, session_id)
    assert saved_session["llm_diagnostics"][-1] == {
        "stage": "intake",
        "category": category,
    }


def test_llm_planner_failure_names_stage_without_creating_a_run(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class PlannerFailureClient:
        calls: list[object] = []

        def complete_json(self, _messages, schema, **kwargs):
            if kwargs["purpose"] == "intake":
                return schema.model_validate(
                    {
                        "intent": "chemistry_compute",
                        "operations": ["Opt"],
                        "molecule_query": "water",
                        "molecule_input_kind": "name",
                        "molecule_name_evidence": "水分子",
                        "requested_results": ["opt_final_electronic_energy"],
                    },
                    strict=True,
                )
            raise LlmError(
                "model returned an empty JSON response",
                category="empty_response",
                purpose="planner",
            )

    session_id = "session_planner_failure"
    response = Agent(
        config,
        build_registry(),
        llm=PlannerFailureClient(),
        session_id=session_id,
    ).handle_message("优化水分子，电荷为0，多重度为1")

    assert "计划生成阶段未获得有效模型响应" in response.text
    assert response.run is None
    runs_path = Path(config.data_root_path) / "runs"
    assert not runs_path.exists() or list(runs_path.iterdir()) == []
    saved_session = load_session(config.data_root_path, session_id)
    assert saved_session["llm_diagnostics"][-1] == {
        "stage": "planner",
        "category": "empty_response",
    }


@pytest.mark.parametrize(
    "user_message",
    [
        "Optimize fresh water with charge 0 and multiplicity 1; start from scratch.",
        "Do not use or reuse the previous optimized geometry; generate fresh water "
        "with charge 0 and multiplicity 1.",
    ],
)
def test_model_cannot_select_history_geometry_without_user_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_message: str,
) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    source_run, source_results = _save_three_step_opt_run(
        config,
        session_id="session_spurious_history_geometry",
        run_id="run_history_geometry_source",
        description="previous optimized water geometry",
        energy_token=None,
    )
    agent = Agent(config, registry, llm=None, session_id="session_spurious_history_geometry")
    agent._record_result_summary(source_run, source_results[-1])

    def select_history_geometry(*_args, **_kwargs) -> IntakeOutput:
        return IntakeOutput(
            intent="chemistry_compute",
            operations=["Opt"],
            molecule_query="water",
            molecule_input_kind="name",
            history_geometry_alias="geometry_1",
            explicit_parameters={"charge": 0, "multiplicity": 1},
            requested_results=["opt_final_electronic_energy"],
        )

    monkeypatch.setattr("bg6022.agent.intake_message", select_history_geometry)

    response = agent.handle_message(user_message)

    assert "only when the user explicitly requests reuse" in response.text
    assert response.run is None
    assert agent._session["active_run_id"] == source_run.id
    assert len(list((Path(config.data_root_path) / "runs").iterdir())) == 1


def test_model_cannot_choose_between_multiple_history_geometries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    agent = Agent(config, registry, llm=None, session_id="session_ambiguous_history_geometry")
    for run_id, description in (
        ("run_history_geometry_first", "first successful water optimization"),
        ("run_history_geometry_second", "second successful water optimization"),
    ):
        source_run, source_results = _save_three_step_opt_run(
            config,
            session_id="session_ambiguous_history_geometry",
            run_id=run_id,
            description=description,
            energy_token=None,
        )
        agent._record_result_summary(source_run, source_results[-1])
    previous_active_run_id = agent._session.get("active_run_id")

    def choose_first_geometry(*_args, **_kwargs) -> IntakeOutput:
        return IntakeOutput(
            intent="chemistry_compute",
            operations=["SP"],
            molecule_query="water",
            molecule_input_kind="name",
            history_geometry_alias="geometry_1",
            explicit_parameters={"charge": 0, "multiplicity": 1},
            requested_results=["sp_electronic_energy"],
        )

    monkeypatch.setattr("bg6022.agent.intake_message", choose_first_geometry)

    response = agent.handle_message("Run SP using the previous optimized geometry.")

    assert "多个可复用" in response.text
    assert "geometry_1" in response.text
    assert "geometry_2" in response.text
    assert response.run is None
    assert agent._session.get("active_run_id") == previous_active_run_id
    assert len(list((Path(config.data_root_path) / "runs").iterdir())) == 2

    class PlanningStarted(Exception):
        pass

    def start_planning(*_args, **_kwargs):
        raise PlanningStarted

    monkeypatch.setattr("bg6022.agent.plan_message", start_planning)
    with pytest.raises(PlanningStarted):
        agent.handle_message("复用 geometry_1")


def test_history_geometry_request_requires_affirmative_reuse_intent() -> None:
    assert _requests_history_geometry(
        "Run an SP on the successful optimized geometry from my previous water task."
    )
    assert not _requests_history_geometry(
        "Do not use or reuse the previous optimized geometry; start from scratch."
    )
    assert not _requests_history_geometry(
        "Optimize fresh water; start from scratch and do not reuse prior geometry."
    )
    assert not _requests_history_geometry("不要复用之前的结构，重新优化。")
    assert not _requests_history_geometry("之前的结构不要使用，请重新建立。")
    assert _requests_history_geometry("复用 geometry_1")
    assert not _requests_history_geometry("不要复用 geometry_1")


def test_current_task_cannot_borrow_energy_from_a_different_task(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    previous, previous_results = _save_three_step_opt_run(
        config,
        session_id="session_no_cross_fill",
        run_id="run_energy_only",
        description="previous water energy task",
        energy_token="-76.400000000000",
    )
    current, current_results = _save_three_step_opt_run(
        config,
        session_id="session_no_cross_fill",
        run_id="run_geometry_only",
        description="current water geometry task",
        energy_token=None,
    )
    client = SelectingClient(
        choose_description="current water geometry task",
        target_property="electronic_energy",
    )
    agent = Agent(config, registry, llm=client, session_id="session_no_cross_fill")
    for run, results in ((previous, previous_results), (current, current_results)):
        for result in results:
            agent._record_result_summary(run, result)

    response = agent.handle_message("这个任务的能量是多少？")

    assert "尚未得到所问性质" in response.text
    assert "-76.400000000000" not in response.text


def test_recent_result_window_is_bounded_by_distinct_runs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent = Agent(config, ToolRegistry([_result_tool()]), llm=None, session_id="session_bounded")
    run_ids: list[str] = []

    for run_number in range(7):
        steps = [Step(id=f"step_{step_number}", tool="measure") for step_number in range(1, 4)]
        request = Request(
            id=f"request_{run_number}",
            description=f"three-step task {run_number}",
            source="chat",
        )
        run = Run(
            id=f"run_{run_number}",
            request=request,
            plan=Plan(id=f"plan_{run_number}", request_id=request.id, steps=steps),
            resources={"cores": 4, "memory_mb": 1024, "maxcore_mb": 192},
            status="succeeded",
            session_id=agent.session_id,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        run_ids.append(run.id)
        for step in steps:
            agent._record_result_summary(
                run,
                Result(
                    run_id=run.id,
                    step_id=step.id,
                    attempt=1,
                    status="succeeded",
                    attempt_relative_path=f"{step.id}/attempt-01",
                ),
            )

    expected_ids = run_ids[-6:]
    assert [item["run_id"] for item in agent._session["recent_results"]] == expected_ids
    assert [item["step_id"] for item in agent._session["recent_results"]] == ["step_3"] * len(
        expected_ids
    )
    saved = load_session(config.data_root_path, agent.session_id)
    assert [item["run_id"] for item in saved["recent_results"]] == expected_ids


def test_opt_and_sp_energy_keep_distinct_step_subjects_within_one_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry()
    opt = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
    )
    sp = Step(
        id="sp",
        tool="single_point",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
    )
    targets = [
        ResultTarget(step_id=opt.id, field="opt_final_electronic_energy"),
        ResultTarget(step_id=sp.id, field="sp_electronic_energy"),
    ]
    request = Request(
        id="request_opt_and_sp",
        description="water Opt and SP comparison",
        requested_results=targets,
        source="chat",
    )
    run = Run(
        id="run_opt_and_sp",
        request=request,
        plan=Plan(
            id="plan_opt_and_sp",
            request_id=request.id,
            steps=[opt, sp],
            requested_results=targets,
        ),
        resources=config.resources,
        status="succeeded",
        session_id="session_opt_and_sp",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    results = [
        Result(
            run_id=run.id,
            step_id=opt.id,
            attempt=1,
            status="succeeded",
            values={
                "opt_final_electronic_energy": {
                    "value": -76.4,
                    "unit": "Eh",
                    "token": "-76.400000000000",
                }
            },
            attempt_relative_path="opt/attempt-01",
            step_fingerprint=_step_fingerprint(opt),
        ),
        Result(
            run_id=run.id,
            step_id=sp.id,
            attempt=1,
            status="succeeded",
            values={
                "sp_electronic_energy": {
                    "value": -76.3,
                    "unit": "Eh",
                    "token": "-76.300000000000",
                }
            },
            attempt_relative_path="sp/attempt-01",
            step_fingerprint=_step_fingerprint(sp),
        ),
    ]
    for result in results:
        result_path = save_result(config.data_root_path, run, result)
        relative = result_path.relative_to(Path(config.data_root_path) / "runs" / run.id).as_posix()
        run.result_index.append(relative)
        run.current_results[result.step_id] = relative
    save_run(config.data_root_path, run)
    session = {
        "session_id": run.session_id,
        "active_run_id": run.id,
        "recent_results": [{"run_id": run.id}],
        "recent_messages": [],
    }
    save_session(config.data_root_path, run.session_id, session)

    agent = Agent(config, registry, llm=None, session_id=run.session_id)
    catalog = agent._build_query_catalog()
    opt_fact = next(item for item in catalog if item["step"]["tool"] == "optimize_geometry")
    sp_fact = next(item for item in catalog if item["step"]["tool"] == "single_point")

    assert opt_fact["result"]["property"] == sp_fact["result"]["property"] == "electronic_energy"
    assert opt_fact["subject_ref"] != sp_fact["subject_ref"]
    assert agent._load_query_fact(opt_fact["subject_ref"], "electronic_energy")["value"][
        "token"
    ] == ("-76.400000000000")
    assert agent._load_query_fact(sp_fact["subject_ref"], "electronic_energy")["value"][
        "token"
    ] == ("-76.300000000000")


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
        update={"steps": [Step(id="measure", tool="measure", origin_step_id="changed")]}
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
