"""Synthetic protocol/continuation regressions; never launch scientific work."""

import json

import pytest

from bg6022.planner import (
    IntentItem,
    QuerySelection,
    RequirementProposal,
    validate_intent_items,
)
from bg6022.tools.registry import build_registry


def shared_goal_task():
    return RequirementProposal(
        key="opt",
        capability="optimize_geometry",
        outputs=["optimized_geometry"],
        report_queries=[{"evidence": "both properties", "search_terms": ["DIPOLE MOMENT"]}],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_quote_preserves_unpromoted_goal_and_is_atomic(reverse):
    """F-R01/F-R02/F-R03: shared evidence is a graph, not a disposable window."""
    task = shared_goal_task()
    items = [IntentItem(kind="compute", evidence="Optimize", task_keys=["opt"])]
    reports = [
        IntentItem(
            kind="report", evidence="both properties", task_keys=["opt"], requested_property=prop
        )
        for prop in ("dipole_moment", "electronic_energy")
    ]
    items += list(reversed(reports)) if reverse else reports
    registry = build_registry()
    validate_intent_items(items, "Optimize both properties", [task], registry=registry)
    assert task.outputs == ["optimized_geometry", "opt_final_electronic_energy"]
    assert len(task.report_queries) == 1
    normalized = task.model_dump()
    validate_intent_items(items, "Optimize both properties", [task], registry=registry)
    assert task.model_dump() == normalized
    bad = shared_goal_task()
    original = bad.model_dump()
    with pytest.raises(ValueError):
        validate_intent_items(
            items + [IntentItem(kind="unresolved", evidence="both properties")],
            "Optimize both properties",
            [bad],
            registry=registry,
        )
    assert bad.model_dump() == original


@pytest.mark.parametrize("status", ["clarify", "unavailable"])
def test_business_query_states_need_no_fake_targets(status):
    """F-R06: missing facts are legal business states."""
    validate_intent_items(
        [IntentItem(kind="query", evidence="electrons", requested_property="total_electrons")],
        "electrons",
        [],
        registry=None,
        query_selection=QuerySelection(status=status),
    )


@pytest.mark.parametrize("view", ["sources", "properties"])
def test_catalog_query_intent_needs_no_result_target(view):
    """F-R04/F-R05: navigation never authorizes computation."""
    selection = QuerySelection(
        status="selected",
        catalog_request={
            "view": view,
            **({"source_ref": "source_1"} if view == "properties" else {}),
        },
    )
    validate_intent_items(
        [IntentItem(kind="query", evidence="show")],
        "show",
        [],
        registry=None,
        query_selection=selection,
    )


def test_absent_query_hints_keep_original_serialization():
    from bg6022.tools.orca_output import dump_query_spec

    old = {"evidence": "dipole", "search_terms": ["DIPOLE MOMENT"]}
    assert dump_query_spec(old) == old


@pytest.mark.parametrize("semantic", [True, False])
def test_repeat_then_new_property_preserves_source_but_changes_goal(
    tmp_path, monkeypatch, semantic
):
    """N16/N20: an explicit new topic consumes old pending and retains the chosen source."""
    from test_readonly_observations import forbid_science, user_water_agent

    agent, _, _ = user_water_agent(tmp_path, semantic, PendingClient())
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    assert agent._session.get("pending_query")

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            context = json.loads(messages[-1]["content"])
            question = context.get("message", context.get("user_message"))
            if question == "再给一遍":
                entry = next(e for e in context["result_catalog"] if e.get("recent_queries"))
                queries = entry["recent_queries"]
            else:
                count = question == "那电子数呢"
                entry = next(
                    e
                    for e in context["result_catalog"]
                    if e.get("access") == ("readonly_observation" if count else "raw_output")
                )
                queries = (
                    [] if count else [{"evidence": question, "search_terms": ["DIPOLE MOMENT"]}]
                )
            target = {
                "subject_ref": entry["subject_ref"],
                "property": "chemical_total_electrons"
                if question == "那电子数呢"
                else "orca_output",
                "reference_mode": "followup",
                "evidence": question,
                "queries": queries,
            }
            return schema.model_validate(
                {
                    ("mode" if semantic else "intent"): "context_query",
                    "intent_items": [
                        {
                            "kind": "query",
                            "evidence": question,
                            "requested_property": "chemical_total_electrons"
                            if question == "那电子数呢"
                            else "dipole_moment",
                        }
                    ],
                    "query_selection": {"status": "selected", "targets": [target]},
                }
            )

    agent.llm = Client()
    first = agent.handle_message("偶极矩")
    assert "1.861296656" in first.text and not agent._session.get("pending_query")
    repeat = agent.handle_message("再给一遍")
    assert "1.861296656" in repeat.text
    count = agent.handle_message("那电子数呢")
    assert "总电子数 10" in count.text and "1.861296656" not in count.text
    assert first.delivery["raw_reports"][0]["run_id"] == count.delivery["raw_reports"][0]["run_id"]


@pytest.mark.parametrize("semantic", [True, False])
def test_knowledge_query_kind_is_valid_without_saved_results(tmp_path, monkeypatch, semantic):
    """N30: a question labelled query in QA cannot authorize or require computation."""
    from test_context_query import _config
    from test_readonly_observations import forbid_science

    from bg6022.agent import Agent

    config = _config(tmp_path)
    config.runtime.semantic_planner_v1 = semantic

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            if kwargs.get("purpose") == "answer":
                return schema.model_validate(
                    {"action": "respond", "sections": [{"text": "中性水分子有 10 个电子。"}]}
                )
            return schema.model_validate(
                {
                    ("mode" if semantic else "intent"): "qa" if semantic else "chemistry_qa",
                    "intent_items": [{"kind": "query", "evidence": "中性水有几个电子"}],
                }
            )

    agent = Agent(config, build_registry(config), llm=Client())
    forbid_science(monkeypatch)
    response = agent.handle_message("中性水有几个电子")
    assert "10" in response.text and not response.delivery.get("raw_reports")
    assert response.run is None and not agent._session.get("last_delivery")


@pytest.mark.parametrize("malformation", ["oversize", "slot_type"])
def test_invalid_restored_pending_preserves_original_session_file(tmp_path, malformation):
    """Restoration limits report a diagnostic and never overwrite the damaged source."""
    from test_readonly_observations import user_water_agent

    from bg6022.agent import Agent
    from bg6022.session import session_path

    agent, root, _ = user_water_agent(tmp_path, llm=PendingClient())
    agent.handle_message("LUMO 能隙呢")
    path = session_path(root, agent.session_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if malformation == "oversize":
        payload["pending_query"]["origin_question"] = "x" * 9000
    else:
        payload["pending_query"]["missing_slots"] = [{"mode": "read_existing"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()
    restored = Agent(agent.config, agent.registry, session_id=agent.session_id, llm=PendingClient())
    assert restored._session_readonly and not restored._session.get("pending_query")
    assert restored._session["history_diagnostic"] == "pending_query_invalid"
    restored.handle_message("无关问题")
    assert path.read_bytes() == original


@pytest.mark.parametrize("stage", ["directory", "content", "presentation"])
@pytest.mark.parametrize("semantic", [True, False])
def test_query_cancellation_does_not_commit_pending_or_delivery_focus(
    tmp_path, monkeypatch, stage, semantic
):
    """N36: cancellation at each checkpoint retains the previous delivery and query."""
    from test_readonly_observations import forbid_science, user_water_agent

    from bg6022 import agent as agent_module
    from bg6022.tools import orca_output

    agent, _, _ = user_water_agent(tmp_path, semantic, PendingClient())
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    previous = agent._session["pending_query"].copy()
    focus = [{"output_ref": "previous_delivery"}]
    agent._session["last_delivery"] = focus.copy()
    if stage == "directory":
        original = agent._build_query_catalog

        def cancelled_catalog(*args, **kwargs):
            result = original(*args, **kwargs)
            agent._query_cancel.set()
            return result

        monkeypatch.setattr(agent, "_build_query_catalog", cancelled_catalog)
    elif stage == "content":
        original = orca_output.read_registered_artifact_bytes

        def cancelled_content(*args, **kwargs):
            result = original(*args, **kwargs)
            kwargs["cancel"].set()
            return result

        monkeypatch.setattr(orca_output, "read_registered_artifact_bytes", cancelled_content)
    else:
        original = agent_module.combine_output_reports

        def cancelled_presentation(*args, **kwargs):
            result = original(*args, **kwargs)
            agent._query_cancel.set()
            return result

        monkeypatch.setattr(agent_module, "combine_output_reports", cancelled_presentation)
    response = agent.handle_message("查询")
    assert response.run is None
    assert not response.delivery or response.delivery["status"] == "cancelled"
    assert agent._session["pending_query"] == previous
    assert agent._session["last_delivery"] == focus


@pytest.mark.parametrize("semantic", [True, False])
def test_electron_schema_failure_keeps_delivered_focus_and_run(tmp_path, monkeypatch, semantic):
    """N31/N32: both actual LlmClient correction receipts reach the session diagnostics."""
    import hashlib

    import httpx
    from test_readonly_observations import forbid_science, user_water_agent

    from bg6022.llm import LlmClient

    agent, root, _ = user_water_agent(tmp_path, semantic)
    forbid_science(monkeypatch)
    focus = [{"output_ref": "previous", "property": "molecular_geometry"}]
    agent._session["last_delivery"] = focus.copy()
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }
    requests = []
    invalid = json.dumps(
        {
            ("mode" if semantic else "intent"): "context_query",
            "query_selection": {
                "status": "selected",
                "targets": [
                    {"subject_ref": "raw_1_a1", "property": 42, "evidence": "水分子的电子数呢"}
                ],
            },
        }
    )

    def handler(request):
        requests.append(True)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": invalid}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    agent.llm = LlmClient(
        agent.config, transport=httpx.MockTransport(handler), api_key="test-only-key"
    )
    response = agent.handle_message("水分子的电子数呢")
    assert response.run is None and "诊断编号" in response.text
    assert requests == [True, True] and agent._session["last_delivery"] == focus
    calls = [c for c in agent._session["llm_diagnostics"] if "structured_correction_count" in c]
    assert [c["structured_correction_count"] for c in calls] == [0, 1]
    assert all(c["schema_errors"][0]["error_type"] == "string_type" for c in calls)
    assert all("property" in c["schema_errors"][0]["path"] for c in calls)
    assert all(c["response_sha256"] for c in calls)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


class PendingClient:
    """Fixed model proposals exercise the production protocol at each entrance."""

    def __init__(self, *, ambiguous=False, forged=False):
        self.ambiguous, self.forged, self.contexts = ambiguous, forged, []

    def complete_json(self, messages, schema, **kwargs):
        context = json.loads(messages[-1]["content"])
        self.contexts.append(context)
        semantic = kwargs["purpose"] == "semantic"
        if kwargs["purpose"] == "answer":
            return schema.model_validate(
                {"action": "respond", "sections": [{"text": "ordinary explanation"}]}
            )
        question = context["message"]
        if question == "无关问题":
            return schema.model_validate({"mode": "qa"} if semantic else {"intent": "chemistry_qa"})
        pending = context.get("pending_query")
        if pending:
            selection = {
                "status": "resume",
                "resume": {
                    "pending_ref": "forged" if self.forged else pending["pending_ref"],
                    "resolution_evidence": question,
                    "slot_updates": {"property_hint": "homo_lumo_gap"}
                    if question == "能隙"
                    else {"mode": "read_existing"},
                },
            }
        else:
            source = next(
                e["subject_ref"]
                for e in context["result_catalog"]
                if e.get("access") == "raw_output"
            )
            selection = {
                "status": "clarify",
                "clarification": "请补充读取方式或性质。",
                "clarification_context": {
                    "original_question": question,
                    "origin_evidence": question,
                    "property_hint": None if self.ambiguous else "homo_lumo_gap",
                    "property_candidates": ["lumo_energy", "homo_lumo_gap"],
                    "candidate_source_refs": [source],
                    "missing_slots": ["mode", "property"] if self.ambiguous else ["mode"],
                },
            }
        return schema.model_validate(
            {
                ("mode" if semantic else "intent"): "context_query",
                "intent_items": [{"kind": "query", "evidence": question}],
                "query_selection": selection,
            }
        )


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("reply", ["查询", "读取刚才的", "查已有结果"])
def test_short_reply_restores_original_goal_source_and_only_fills_mode(
    tmp_path, monkeypatch, semantic, reply
):
    """N13/N15: no regex list in production, no new Run, original question retained."""
    from test_readonly_observations import forbid_science, user_water_agent

    client = PendingClient()
    agent, root, manifest = user_water_agent(tmp_path, semantic, client)
    forbid_science(monkeypatch)
    first = agent.handle_message("LUMO 能隙呢")
    assert "补充" in first.text
    saved = agent._session["pending_query"]
    assert saved["origin_question"] == "LUMO 能隙呢"
    assert saved["source_bindings"][0]["run_id"] == manifest["original_run_id"]
    response = agent.handle_message(reply)
    assert "9.3533 eV" in response.text and response.delivery["status"] == "complete"
    assert response.run is None and not agent._session.get("pending_query")
    report = response.delivery["raw_reports"][0]
    assert report["queries"][0]["evidence"] == "LUMO 能隙呢"
    assert report["queries"][0]["search_terms"] == ["ORBITAL ENERGIES"]
    assert report["run_id"] == manifest["original_run_id"]
    assert len(list((root / "runs").iterdir())) == 1


@pytest.mark.parametrize("semantic", [True, False])
def test_mode_reply_keeps_property_ambiguity_then_property_reply_selects_gap(
    tmp_path, monkeypatch, semantic
):
    """N14: a reading-mode reply is not an answer to the property question."""
    from test_readonly_observations import forbid_science, user_water_agent

    agent, _, _ = user_water_agent(tmp_path, semantic, PendingClient(ambiguous=True))
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    response = agent.handle_message("查询")
    assert not response.delivery and agent._session["pending_query"]["missing_slots"] == [
        "property"
    ]
    assert agent._session["pending_query"]["resolved_slots"] == {"mode": "read_existing"}
    response = agent.handle_message("能隙")
    assert "9.3533 eV" in response.text and "LUMO 轨道能量为" not in response.text


@pytest.mark.parametrize("semantic", [True, False])
def test_pending_survives_restart_and_unrelated_question_but_not_new_session(
    tmp_path, monkeypatch, semantic
):
    """N17/N18/N20: restore read-only state without execution permission."""
    from test_readonly_observations import forbid_science, user_water_agent

    from bg6022.agent import Agent

    agent, _, _ = user_water_agent(tmp_path, semantic, PendingClient())
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    original = agent._session["pending_query"].copy()
    assert "explanation" in agent.handle_message("无关问题").text
    assert agent._session["pending_query"] == original
    restored = Agent(agent.config, agent.registry, session_id=agent.session_id, llm=PendingClient())
    assert restored._session["pending_query"] == original
    assert "9.3533 eV" in restored.handle_message("查询").text
    agent.handle_message("/new")
    assert not agent._session.get("pending_query") and not agent._query_bindings
    assert agent._session["active_run_id"] is None


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("change", ["content", "remove", "metadata"])
def test_pending_changed_source_never_falls_back_to_latest(tmp_path, monkeypatch, semantic, change):
    """N19: full source revalidation after clarification, no nearest-Run fallback."""
    from test_readonly_observations import forbid_science, user_water_agent

    from bg6022.session import load_run, save_run

    agent, root, manifest = user_water_agent(tmp_path, semantic, PendingClient())
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    run = load_run(root, manifest["original_run_id"])
    artifact = next(a for a in run.artifact_index if a.artifact_type == "orca_output")
    path = root / "runs" / run.id / artifact.relative_path
    if change == "remove":
        path.unlink()
    elif change == "metadata":
        artifact.sha256 = "0" * 64
        save_run(root, run)
    else:
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
    response = agent.handle_message("查询")
    assert "9.3533" not in response.text and response.run is None
    assert not response.delivery or response.delivery["status"] == "partial"
    assert len(list((root / "runs").iterdir())) == 1


@pytest.mark.parametrize("semantic", [True, False])
def test_forged_pending_reference_rejected_before_content_reads(tmp_path, monkeypatch, semantic):
    """N21: current evidence and the program-issued token are mandatory."""
    from test_readonly_observations import forbid_science, user_water_agent

    agent, _, _ = user_water_agent(tmp_path, semantic, PendingClient())
    forbid_science(monkeypatch)
    agent.handle_message("LUMO 能隙呢")
    original = agent._session["pending_query"].copy()
    agent.llm = PendingClient(forged=True)
    monkeypatch.setattr(
        "bg6022.tools.orca_output.read_registered_artifact_bytes",
        lambda *a, **k: pytest.fail("forged pending caused a content read"),
    )
    response = agent.handle_message("查询")
    assert not response.delivery and response.run is None
    assert agent._session["pending_query"] == original
