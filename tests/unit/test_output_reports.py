"""Offline conversation/report tests; synthetic sources never count as real computation."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest
from test_context_query import _config, _result_tool, _save_scalar_run
from test_orca_output_query import lookup, query, saved_output, snapshot

from bg6022.agent import Agent
from bg6022.answer import combine_output_reports, render_confirmation
from bg6022.canonicalize import canonicalize_semantic_request
from bg6022.models import Request, Result, Run, Step, Tool
from bg6022.output_query import (
    build_raw_catalog_entries,
    collect_raw_output_sources,
    merge_report_constraints,
    query_output_sources,
    read_report_queries,
    reports_for_run,
    validate_raw_query_targets,
)
from bg6022.planner import IntakeOutput, QuerySelection, intake_message, request_from_intake
from bg6022.semantic import SemanticProposal, semantic_message
from bg6022.session import execution_fingerprint, load_run, save_result, save_run
from bg6022.tools.registry import ToolRegistry, build_registry


def test_partial_verified_delivery_updates_only_actually_rendered_focus(tmp_path):
    from bg6022.agent import AgentResponse

    agent, _, _ = agent_with_output(tmp_path)
    agent._session["last_delivery"] = [{"output_ref": "old"}]
    shown = {
        "output_ref": "out_1",
        "run_id": "run_raw",
        "step_id": "opt",
        "attempt": 1,
        "step_fingerprint": "verified",
        "property": "electronic_energy",
    }
    agent._record_response(
        AgentResponse(
            text="energy shown",
            delivery={
                "status": "partial",
                "rendered_refs": ["out_1"],
                "outputs": [shown, dict(shown, output_ref="not_shown")],
            },
        )
    )
    assert agent._session["last_delivery"] == [shown]
    agent._record_response(
        AgentResponse(
            text="cancelled",
            delivery={
                "status": "cancelled",
                "rendered_refs": [],
                "outputs": [dict(shown, output_ref="new")],
            },
        )
    )
    assert agent._session["last_delivery"] == [shown]


def test_raw_source_does_not_borrow_parameters_from_modified_step(tmp_path):
    from bg6022.agent import _step_fingerprint

    run, _ = saved_output(tmp_path)
    step = run.plan.steps[0]
    step.parameters["method_profile"] = "pbe0_d3bj_def2svp"
    path = tmp_path / "runs" / run.id / run.result_index[0]
    result = Result.model_validate_json(path.read_bytes())
    result.step_fingerprint = _step_fingerprint(step)
    save_result(tmp_path, run, result)
    context = collect_raw_output_sources(tmp_path, run, "raw_session")[0]["source_context"]
    assert "PBE0" in context["method_label"]
    step.parameters["method_profile"] = "b3lyp_d3bj_def2svp"
    context = collect_raw_output_sources(tmp_path, run, "raw_session")[0]["source_context"]
    assert context["method_label"] == "unknown"


def agent_with_output(tmp_path, *, semantic=True, payload=None):
    config = _config(tmp_path)
    config.runtime.semantic_planner_v1 = semantic
    root = Path(config.data_root_path)
    run, _ = saved_output(root, **({"payload": payload} if payload is not None else {}))
    registry = build_registry(config)
    agent = Agent(config, registry, session_id="raw_session")
    agent._session["active_run_id"] = run.id
    return agent, run, root


def selected(catalog, *, evidence="偶极矩", term="DIPOLE MOMENT", followup=False):
    item = next(e for e in catalog if e.get("access") == "raw_output")
    return {
        "status": "selected",
        "targets": [
            {
                "subject_ref": item["subject_ref"],
                "property": "orca_output",
                "evidence": evidence,
                "reference_mode": "followup" if followup else "explicit",
                "queries": [query(term, evidence)],
            }
        ],
    }


class QueryClient:
    def __init__(self, *, review=False):
        self.messages = []
        self.purposes = []
        self.review = review
        self.routes = 0

    def complete_json(self, messages, schema, **kwargs):
        purpose = kwargs["purpose"]
        self.purposes.append(purpose)
        self.messages.append(messages)
        payload = json.loads(messages[-1]["content"])
        if purpose == "answer":
            assert self.review
            return schema.model_validate(
                {"action": "needs_tools", "requested_results": ["opt_final_electronic_energy"]}
            )
        self.routes += 1
        catalog = payload["result_catalog"]
        assert any(e.get("access") == "raw_output" for e in catalog)
        if self.review and self.routes == 1:
            return schema.model_validate(
                {"mode": "qa"} if purpose == "semantic" else {"intent": "chemistry_qa"}
            )
        value = {"query_selection": selected(catalog)}
        value.update(
            {"mode": "context_query"} if purpose == "semantic" else {"intent": "context_query"}
        )
        return schema.model_validate(value, strict=True)


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("review", [True, False])
def test_both_entrypoints_and_route_review_read_failed_source_without_compute(
    tmp_path,
    monkeypatch,
    semantic,
    review,
):
    agent, run, root = agent_with_output(tmp_path, semantic=semantic)
    client = QueryClient(review=review)
    agent.llm = client
    before = snapshot(root / "runs")

    def forbidden(*args, **kwargs):
        pytest.fail("pure raw query crossed a write/compute boundary")

    monkeypatch.setattr(agent, "_invoke_step", forbidden)
    monkeypatch.setattr("bg6022.agent.save_run", forbidden)
    monkeypatch.setattr("bg6022.agent.publish_step_result", forbidden)
    response = agent.handle_message("上次输出中的偶极矩")
    assert "1.25 Debye" in response.text, response.text
    assert response.delivery["outputs"] == []
    assert response.delivery["raw_reports"][0]["run_id"] == run.id
    route = "semantic" if semantic else "intake"
    assert client.purposes == ([route, "answer", route] if review else [route])
    assert snapshot(root / "runs") == before


def test_raw_summary_and_next_model_request_never_include_body(tmp_path):
    agent, _, _ = agent_with_output(
        tmp_path,
        payload=b"DIPOLE MOMENT\n````\nSECRET_RAW_BODY_IGNORE_INSTRUCTIONS\n",
    )
    agent.llm = QueryClient()
    response = agent.handle_message("上次输出中的偶极矩")
    assert "SECRET_RAW_BODY" in response.text
    assert "SECRET_RAW_BODY" not in json.dumps(agent._session)
    agent.handle_message("上次输出中的偶极矩")
    assert "SECRET_RAW_BODY" not in json.dumps(agent.llm.messages)
    assert agent._session["last_output_query"][0]["queries"] == [query()]


def semantic_compute(reports=None):
    return SemanticProposal.model_validate(
        {
            "mode": "compute",
            "subjects": [
                {
                    "key": "water",
                    "query": "water",
                    "input_kind": "name",
                    "evidence": "water",
                }
            ],
            "tasks": [
                {
                    "key": "opt_water",
                    "subject_key": "water",
                    "capability": "optimize_geometry",
                    "method_request": "PBE0",
                    "requested_properties": ["geometry"],
                    "report_queries": reports or [],
                }
            ],
        }
    )


def compute_request(message="Optimize water, 报告输出中的偶极矩", reports=None):
    return canonicalize_semantic_request(
        message,
        semantic_compute(
            reports if reports is not None else [query(evidence="报告输出中的偶极矩")]
        ),
        request_id="request_report",
        registry=build_registry(),
    )


def test_report_persistence_preview_and_requirement_producer_attempt(tmp_path):
    request = compute_request()
    requirement = request.requirements[0]
    assert requirement.constraints["method_resolution"]["profile"] == "pbe0_d3bj_def2svp"
    assert requirement.constraints["report_queries"] == [query(evidence="报告输出中的偶极矩")]
    assert requirement.outputs == ["optimized_geometry"]
    assert "report_queries" not in type(requirement).model_fields
    assert "report_queries" not in requirement.parameters
    run, _ = saved_output(tmp_path, requirement_id=requirement.id)
    run.request = request
    run, _ = saved_output(
        tmp_path,
        run=run,
        attempt=2,
        payload=b"DIPOLE MOMENT\nSECOND_CURRENT_ATTEMPT\n",
        status="succeeded",
    )
    run.status = "succeeded"
    save_run(tmp_path, run)
    reports = reports_for_run(
        tmp_path, run, session_id="raw_session", cancel=Event(), output_limit_bytes=2**20
    )
    assert reports[0]["attempt"] == 2
    assert "SECOND_CURRENT_ATTEMPT" in reports[0]["evidence"][0]["snippets"][0]["text"]
    assert reports[0]["is_current_attempt"] is True
    catalog, _ = build_raw_catalog_entries(
        collect_raw_output_sources(tmp_path, run, "raw_session"), []
    )
    assert {e["step"]["attempt"] for e in catalog} == {1, 2}
    preview = render_confirmation({"report_queries": requirement.constraints["report_queries"]})
    assert "报告输出中的偶极矩" in preview
    assert "若未打印" in preview


def test_first_report_missing_keeps_run_success_and_delivery_partial(tmp_path, monkeypatch):
    agent, run, root = agent_with_output(tmp_path)
    run.request = compute_request(
        "Optimize water, 报告输出中的X", [query("MISSING_X", "报告输出中的X")]
    )
    run.plan.steps[0].requirement_id = run.request.requirements[0].id
    run.status = "succeeded"
    save_run(root, run)
    before = snapshot(root / "runs")
    monkeypatch.setattr(agent, "_collect_requested_outputs", lambda *a, **k: ([], [], []))
    response = agent._response_for_run(run, None)
    assert response.delivery["status"] == "partial"
    assert "未交付" in response.text
    assert run.status == "succeeded"
    assert snapshot(root / "runs") == before


def test_report_sources_do_not_cross_requirement_boundaries(tmp_path):
    run, _ = saved_output(tmp_path, payload=b"DIPOLE MOMENT\nFIRST\n")
    request = compute_request()
    other = request.requirements[0].model_copy(deep=True)
    other.id = "req_other"
    request.requirements.append(other)
    run.request = request
    run.plan.steps[0].requirement_id = request.requirements[0].id
    save_run(tmp_path, run)
    reports = reports_for_run(
        tmp_path, run, session_id="raw_session", cancel=None, output_limit_bytes=2**20
    )
    assert reports[0]["artifact_id"]
    assert reports[1]["error"] == "invalid_source"
    assert "artifact_id" not in reports[1]


def test_two_method_reports_bind_separate_producers_not_last_result(tmp_path):
    request = compute_request()
    other = request.requirements[0].model_copy(deep=True)
    other.id = "req_other_method"
    other.parameters["method_profile"] = "r2scan3c"
    request.requirements.append(other)
    run, _ = saved_output(
        tmp_path,
        requirement_id=request.requirements[0].id,
        payload=b"DIPOLE MOMENT\nFIRST_METHOD_VALUE\n",
        status="succeeded",
    )
    run.request = request
    run.plan.steps.append(Step(id="opt_other", tool="optimize_geometry", requirement_id=other.id))
    run, _ = saved_output(
        tmp_path,
        run=run,
        step_id="opt_other",
        payload=b"DIPOLE MOMENT\nSECOND_METHOD_VALUE\n",
        status="succeeded",
    )
    reports = reports_for_run(
        tmp_path, run, session_id="raw_session", cancel=None, output_limit_bytes=2**20
    )
    assert len(reports) == 2
    first, second = [r["evidence"][0]["snippets"][0]["text"] for r in reports]
    assert "FIRST_METHOD_VALUE" in first and "SECOND_METHOD_VALUE" not in first
    assert "SECOND_METHOD_VALUE" in second and "FIRST_METHOD_VALUE" not in second


def test_explicit_old_attempt_is_read_as_historical_evidence(tmp_path):
    run, _ = saved_output(tmp_path, payload=b"DIPOLE MOMENT\nFAILED_OLD_VALUE\n")
    run, _ = saved_output(
        tmp_path,
        run=run,
        attempt=2,
        status="succeeded",
        payload=b"DIPOLE MOMENT\nSUCCESS_CURRENT_VALUE\n",
    )
    old = next(
        s for s in collect_raw_output_sources(tmp_path, run, "raw_session") if s["attempt"] == 1
    )
    reports = query_output_sources(
        tmp_path, [(old, [query()])], session_id="raw_session", indexed_ids=[run.id]
    )
    text, _ = combine_output_reports("", {}, reports)
    assert "FAILED_OLD_VALUE" in text and "SUCCESS_CURRENT_VALUE" not in text
    assert "历史 attempt" in text


def test_verified_energy_focus_survives_raw_partial_delivery(tmp_path):
    agent, raw_run, root = agent_with_output(tmp_path)
    agent.registry = ToolRegistry(
        [agent.registry.get(name) for name in agent.registry.names()] + [_result_tool()]
    )
    scalar = _save_scalar_run(
        root,
        session_id="raw_session",
        run_id="energy_run",
        description="water energy",
        value_token="-12.345",
    )
    agent._session["recent_results"] = [{"run_id": scalar.id}, {"run_id": raw_run.id}]
    catalog = agent._build_query_catalog()
    energy = next(e for e in catalog if e["result"]["property"] == "electronic_energy")
    selection = selected(catalog, evidence="Mayer", term="MAYER")
    selection["targets"].append(
        {"subject_ref": energy["subject_ref"], "property": "electronic_energy", "evidence": "能量"}
    )
    agent.llm = None
    response = agent._answer_context(
        "能量和Mayer", selection=QuerySelection(**selection), catalog=catalog
    )
    assert "-12.345" in response.text
    assert response.delivery["status"] == "partial"
    assert response.delivery["verified_status"] == "complete"
    assert agent._session["last_delivery"][0]["run_id"] == scalar.id
    assert len(response.delivery["outputs"]) == 1
    catalog = agent._build_query_catalog()
    assert next(e for e in catalog if e["result"]["property"] == "electronic_energy")[
        "recently_delivered"
    ]


def test_followup_reuses_only_one_unchanged_question(tmp_path):
    agent, _, _ = agent_with_output(tmp_path)
    agent.llm = QueryClient()
    agent.handle_message("偶极矩")
    catalog = agent._build_query_catalog()
    target = selected(catalog, followup=True)["targets"]
    validate_raw_query_targets(target, "再显示一遍", catalog)
    with pytest.raises(ValueError):
        validate_raw_query_targets(target, "那Mayer键级呢", catalog)
    target[0]["queries"] = [query("MAYER", "Mayer")]
    target[0]["reference_mode"] = "explicit"
    target[0]["evidence"] = "Mayer"
    validate_raw_query_targets(target, "那Mayer键级呢", catalog)
    raw = next(e for e in catalog if e.get("access") == "raw_output")
    raw["recent_queries"] = [query(), query("MAYER", "键级")]
    with pytest.raises(ValueError):
        validate_raw_query_targets(selected(catalog, followup=True)["targets"], "刚才那个", catalog)


@pytest.mark.parametrize(
    "message, reports",
    [
        ("Optimize water", [query(evidence="报告输出中的偶极矩")]),
        ("Optimize water, 计算Gibbs并输出", [query("GIBBS", "计算Gibbs并输出")]),
        ("Optimize water, 给出Gibbs", [query("GIBBS", "给出Gibbs")]),
    ],
)
def test_fabricated_or_demoted_report_requirements_rejected(message, reports):
    with pytest.raises(ValueError):
        compute_request(message, reports)


def test_constraint_collision_and_global_query_limit():
    with pytest.raises(ValueError, match="reserved"):
        merge_report_constraints({"report_queries": []}, [], "", "optimize_geometry")
    req = compute_request()
    req.requirements[0].constraints["report_queries"] *= 4
    with pytest.raises(ValueError):
        read_report_queries(req)


def test_history_gibbs_routes_but_new_gibbs_remains_unsupported(tmp_path):
    agent, _, _ = agent_with_output(tmp_path)
    catalog = agent._build_query_catalog()

    class Client:
        calls = 0

        def complete_json(self, messages, schema, **kwargs):
            self.calls += 1
            return schema.model_validate(
                {
                    "mode": "context_query",
                    "query_selection": selected(catalog, evidence="Gibbs", term="GIBBS"),
                }
            )

    client = Client()
    value = semantic_message(
        client, "刚才计算的 Gibbs 在哪里", registry=agent.registry, result_catalog=catalog
    )
    assert value.mode == "context_query"
    assert client.calls == 1
    value = semantic_message(
        client, "计算水的 Gibbs 自由能", registry=agent.registry, result_catalog=catalog
    )
    assert value.mode == "unsupported"
    assert client.calls == 1


def test_legacy_intake_keeps_report_requirement(tmp_path):
    payload = IntakeOutput.model_validate(
        {
            "intent": "chemistry_compute",
            "subjects": {
                "water": {
                    "key": "water",
                    "molecule_query": "water",
                    "molecule_input_kind": "name",
                    "molecule_name_evidence": "water",
                }
            },
            "requirements": [
                {
                    "key": "opt",
                    "subject_key": "water",
                    "capability": "optimize_geometry",
                    "outputs": ["optimized_geometry"],
                    "report_queries": [query(evidence="报告输出中的偶极矩")],
                }
            ],
        }
    )

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            return schema.model_validate(payload.model_dump(), strict=True)

    message = "Optimize water, 报告输出中的偶极矩"
    registry = build_registry()
    intake = intake_message(Client(), message, registry=registry)
    request = request_from_intake(message, intake, request_id="req", registry=registry)
    assert read_report_queries(request)[0][1] == [query(evidence="报告输出中的偶极矩")]


def test_old_request_and_acceptance_snapshot_round_trip_unchanged(tmp_path):
    run, _ = saved_output(tmp_path)
    # No report key existed in this persisted shape; full Request is in the old snapshot.
    old_request = run.request.model_dump(mode="json")
    run.accepted_snapshot = {"request": old_request, "plan": run.plan.model_dump(mode="json")}
    old_snapshot = json.loads(json.dumps(run.accepted_snapshot))
    run.accepted_execution_sha256 = execution_fingerprint(
        run.plan, run.resources, run.artifact_index, snapshot=old_snapshot
    )
    accepted_hash = run.accepted_execution_sha256
    save_run(tmp_path, run)
    restored = load_run(tmp_path, run.id)
    assert read_report_queries(restored.request) == []
    assert Request.model_validate(old_request).model_dump(mode="json") == old_request
    assert restored.accepted_snapshot == old_snapshot
    assert (
        execution_fingerprint(
            restored.plan,
            restored.resources,
            restored.artifact_index,
            snapshot=restored.accepted_snapshot,
        )
        == accepted_hash
    )


def test_baseline_generated_acceptance_snapshot_keeps_original_hash(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures/output_query_old_acceptance.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["baseline_commit"] == "611316191c0680ec77ed2ab1dca9feea94b14b4a"
    old = payload["run"]
    run = Run.model_validate(old, strict=True)
    before = json.dumps(old, sort_keys=True)
    assert read_report_queries(run.request) == []
    assert run.request.model_dump(mode="json") == old["accepted_snapshot"]["request"]
    assert (
        execution_fingerprint(
            run.plan,
            run.resources,
            run.artifact_index,
            snapshot=run.accepted_snapshot,
        )
        == old["accepted_execution_sha256"]
    )
    assert json.dumps(old, sort_keys=True) == before


def test_full_verified_catalog_reserves_raw_slots_and_rejects_unknown_alias(tmp_path):
    agent, raw, root = agent_with_output(tmp_path)
    scalar = _save_scalar_run(
        root,
        session_id="raw_session",
        run_id="many_fields",
        description="Synthetic fields",
        value_token="-1.0",
    )
    tool = Tool(
        name="measure",
        description="Synthetic catalog capacity test",
        results={f"field_{i}": "Eh" for i in range(30)},
        result_properties={f"field_{i}": f"property_{i}" for i in range(30)},
        requires_compute_permission=False,
    )
    agent.registry = ToolRegistry([agent.registry.get(n) for n in agent.registry.names()] + [tool])
    relative = scalar.current_results["measure"]
    result = Result.model_validate_json((root / "runs" / scalar.id / relative).read_bytes())
    result.values = {
        f"field_{i}": {"value": -1.0, "token": "-1.0", "unit": "Eh"} for i in range(30)
    }
    save_result(root, scalar, result)
    agent._session["recent_results"] = [{"run_id": scalar.id}, {"run_id": raw.id}]
    catalog = agent._build_query_catalog()
    assert len(catalog) == 24
    assert any(e.get("access") == "raw_output" for e in catalog)
    assert len(agent._query_bindings) == 24
    selection = selected(catalog)
    selection["targets"][0]["subject_ref"] = "raw_missing_old_task"
    before = snapshot(root / "runs")
    response = agent._answer_context(
        "偶极矩", selection=QuerySelection(**selection), catalog=catalog
    )
    assert not response.delivery.get("raw_reports")
    assert "1.25" not in response.text
    assert snapshot(root / "runs") == before


def test_finished_failed_attempt_without_current_result_can_supply_first_report(tmp_path):
    request = compute_request()
    run, _ = saved_output(tmp_path, requirement_id=request.requirements[0].id)
    run.request = request
    assert not run.current_results
    save_run(tmp_path, run)
    reports = reports_for_run(
        tmp_path, run, session_id="raw_session", cancel=None, output_limit_bytes=2**20
    )
    assert reports[0]["source_status"] == "failed"
    assert reports[0]["is_current_attempt"]
    assert reports[0]["evidence"][0]["lookup_status"] == "found"


def test_cancelled_run_does_not_start_report_reads(tmp_path, monkeypatch):
    request = compute_request()
    run, _ = saved_output(tmp_path, requirement_id=request.requirements[0].id)
    run.request, run.status = request, "cancelled"

    def forbidden(*args, **kwargs):
        pytest.fail("cancelled Run must not start raw source reads")

    monkeypatch.setattr("bg6022.output_query.collect_raw_output_sources", forbidden)
    reports = reports_for_run(
        tmp_path, run, session_id="raw_session", cancel=None, output_limit_bytes=2**20
    )
    assert reports[0]["error"] == "cancelled"


def test_title_and_multiple_candidates_never_claim_final_scientific_value(tmp_path):
    run, _ = saved_output(
        tmp_path,
        payload=b"DIPOLE MOMENT\n"
        + b"x\n" * 30
        + b"DIPOLE MOMENT\n1 Debye\n"
        + b"x\n" * 30
        + b"DIPOLE MOMENT\n2 Debye\n",
    )
    reports = lookup(tmp_path, run)
    text, delivery = combine_output_reports("", {}, reports)
    assert delivery["status"] == "partial"
    assert "不能据此确定唯一最终科学值" in text
    assert len(reports[0]["evidence"][0]["snippets"]) <= 3
    assert delivery["outputs"] == []
