"""History metadata is separate from recent conversational context."""

from threading import Event

import pytest


def test_directory_lists_more_than_six_raw_attempts_and_marks_default_limit(tmp_path):
    """F-R10: directory expansion cannot silently discard the seventh source."""
    import json

    from test_orca_output_query import saved_output
    from test_output_reports import agent_with_output

    from bg6022.planner import CatalogRequest

    agent, run, root = agent_with_output(tmp_path)
    for attempt in range(2, 9):
        saved_output(root, run=run, attempt=attempt)
    catalog = agent._build_query_catalog()
    assert len([e for e in catalog if e.get("access") == "raw_output"]) == 6
    assert agent._query_catalog_limited
    agent._answer_catalog(CatalogRequest(view="sources"))
    source_ref = next(iter(agent._catalog_sources))
    response = agent._answer_catalog(CatalogRequest(view="properties", source_ref=source_ref))
    payload = json.loads(response.text.split("\n", 1)[1])
    assert payload["complete"] and payload["status"] == "ok"
    raw = [e for e in payload["items"] if e.get("access") == "raw_output"]
    assert {e["step"]["attempt"] for e in raw} == set(range(1, 9))


def test_history_index_keeps_runs_outside_recent_window(tmp_path):
    from test_orca_output_query import saved_output

    from bg6022.session import (
        index_session_run,
        list_session_run_summaries,
        load_session,
        save_session,
    )

    session = {"session_id": "raw_session", "recent_results": []}
    for index in range(9):
        run, _ = saved_output(tmp_path, run_id=f"history_{index}")
        index_session_run(session, run)
        session["recent_results"].append({"run_id": run.id})
    save_session(tmp_path, "raw_session", session)
    restored = load_session(tmp_path, "raw_session")
    assert len(restored["recent_results"]) == 6
    page = list_session_run_summaries(tmp_path, restored, limit=24)
    assert {entry["run_id"] for entry in page["items"]} == {f"history_{i}" for i in range(9)}


def test_history_discovery_is_cancelled_and_rejects_invented_cursor(tmp_path):
    from bg6022.session import list_session_run_summaries

    session = {"session_id": "s"}
    cancel = Event()
    cancel.set()
    page = list_session_run_summaries(tmp_path, session, cancel=cancel)
    assert page["status"] == "cancelled" and not page["complete"]
    with pytest.raises(ValueError):
        list_session_run_summaries(tmp_path, session, cursor="../../secret")


def test_selected_raw_source_does_not_enumerate_history_again(tmp_path, monkeypatch):
    from test_orca_output_query import saved_output

    from bg6022.output_query import collect_raw_output_sources, resolve_raw_output_source

    run, _ = saved_output(tmp_path)
    binding = collect_raw_output_sources(tmp_path, run, "raw_session")[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("selected binding must not enumerate all results")

    monkeypatch.setattr("bg6022.output_query.collect_raw_output_sources", forbidden)
    source, result, artifact = resolve_raw_output_source(tmp_path, binding, "raw_session", [run.id])
    assert source.id == run.id and artifact.id == binding["artifact_id"]


def test_discovery_paginates_and_excludes_unindexed_legacy_or_foreign(tmp_path):
    from test_orca_output_query import saved_output

    from bg6022.session import list_session_run_summaries, save_run

    for index in range(27):
        saved_output(tmp_path, run_id=f"saved_{index}")
    foreign, _ = saved_output(tmp_path, run_id="foreign")
    foreign.session_id = "other"
    save_run(tmp_path, foreign)
    legacy, _ = saved_output(tmp_path, run_id="legacy")
    legacy.session_id = None
    save_run(tmp_path, legacy)
    session = {"session_id": "raw_session"}
    first = list_session_run_summaries(tmp_path, session)
    assert len(first["items"]) == 24 and not first["complete"]
    second = list_session_run_summaries(tmp_path, session, cursor=first["cursor"])
    assert len(second["items"]) == 3 and second["complete"]
    assert not {"legacy", "foreign"} & {entry["run_id"] for entry in session["run_history"]}


def test_metadata_budget_rejects_oversize_before_read_and_cancels_cached_read(tmp_path):
    from bg6022.session import ArtifactReadError, new_metadata_budget, read_metadata_json

    path = tmp_path / "large.json"
    with path.open("wb") as handle:
        handle.truncate(4 * 1024 * 1024 + 1)
    budget = new_metadata_budget()
    with pytest.raises(ArtifactReadError) as error:
        read_metadata_json(path, budget)
    assert error.value.category == "metadata_file_bytes" and budget["bytes"] == 0
    path.write_text('{"ok": true}', encoding="utf-8")
    assert read_metadata_json(path, budget)["ok"]
    budget["cancel"].set()
    with pytest.raises(ArtifactReadError, match="cancelled"):
        read_metadata_json(path, budget)


def test_directory_properties_reaches_field_30_without_execution(tmp_path):
    from test_context_query import _save_scalar_run
    from test_output_reports import agent_with_output

    from bg6022.models import Result, Tool
    from bg6022.planner import CatalogRequest, QuerySelection
    from bg6022.semantic import compact_result_catalog
    from bg6022.session import save_result
    from bg6022.tools.registry import ToolRegistry

    agent, raw, root = agent_with_output(tmp_path)
    scalar = _save_scalar_run(
        root,
        session_id="raw_session",
        run_id="many_fields",
        description="Thirty fields",
        value_token="-1.0",
    )
    tool = Tool(
        name="measure",
        description="Synthetic test only",
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
    default = agent._build_query_catalog()
    assert any(e.get("access") == "raw_output" for e in default)
    response = agent._answer_catalog(CatalogRequest(view="sources"))
    assert response.delivery["status"] == "catalog"
    source = next(
        ref for ref, item in agent._catalog_sources.items() if item["run_id"] == scalar.id
    )
    agent._answer_catalog(CatalogRequest(view="properties", source_ref=source))
    assert len(agent._catalog_page) == 24 and agent._catalog_next
    cursor = agent._catalog_next
    compact = compact_result_catalog(compact_result_catalog(agent._build_query_catalog()))
    assert any(e.get("catalog_next_cursor") == cursor for e in compact)
    agent._answer_catalog(CatalogRequest(view="properties", source_ref=source, cursor=cursor))
    assert len(agent._catalog_page) == 6 and agent._catalog_next is None
    last = agent._catalog_page[-1]
    assert last["result"]["property"] == "property_29"
    fact = agent._load_query_fact(last["subject_ref"], "property_29")
    assert fact["value"]["token"] == "-1.0"
    assert (
        agent._answer_catalog(CatalogRequest(view="properties", source_ref="invented")).delivery
        == {}
    )
    with pytest.raises(ValueError):
        QuerySelection(
            status="selected",
            targets=[{"subject_ref": "x", "property": "p", "evidence": "x"}],
            catalog_request=CatalogRequest(view="sources"),
        )


def test_directory_can_be_selected_from_empty_semantic_catalog():
    from bg6022.semantic import semantic_schema
    from bg6022.tools.registry import build_registry

    schema = semantic_schema(build_registry(), result_catalog=[], message="列出历史任务")
    value = schema.model_validate(
        {
            "mode": "context_query",
            "query_selection": {"status": "selected", "catalog_request": {"view": "sources"}},
        }
    )
    assert value.query_selection.catalog_request.view == "sources"


def test_index_limit_preserves_old_entries_and_discovery_can_still_find_new(tmp_path, monkeypatch):
    from test_orca_output_query import saved_output

    from bg6022.session import index_session_run, list_session_run_summaries

    old, _ = saved_output(tmp_path, run_id="old")
    session = {"session_id": "raw_session"}
    index_session_run(session, old)
    monkeypatch.setattr("bg6022.session.MAX_HISTORY_INDEX_BYTES", 1)
    saved_output(tmp_path, run_id="new")
    page = list_session_run_summaries(tmp_path, session)
    assert [entry["run_id"] for entry in session["run_history"]] == ["old"]
    assert {entry["run_id"] for entry in page["items"]} == {"old", "new"}
    assert session["history_diagnostic"] == "history_index_bytes"


def test_discovery_budget_continuation_makes_progress(tmp_path):
    from test_orca_output_query import saved_output

    from bg6022.session import list_session_run_summaries

    for i in range(66):
        saved_output(tmp_path, run_id=f"source_{i}")
    session = {"session_id": "raw_session"}
    page = list_session_run_summaries(tmp_path, session)
    assert page["metadata_candidates"] <= 64 and not page["complete"]
    seen = {item["run_id"] for item in page["items"]}
    for _ in range(4):
        page = list_session_run_summaries(tmp_path, session, cursor=page["cursor"])
        seen.update(item["run_id"] for item in page["items"])
        if page["complete"]:
            break
    assert page["complete"] and len(seen) == 66


def test_directory_reports_save_failure_without_changing_run(tmp_path, monkeypatch):
    from test_output_reports import agent_with_output

    from bg6022.planner import CatalogRequest

    agent, run, root = agent_with_output(tmp_path)
    original = (root / "runs" / run.id / "run.json").read_bytes()

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("bg6022.agent.save_session", fail)
    response = agent._answer_catalog(CatalogRequest(view="sources"))
    assert "session_save_failed" in response.text
    assert (root / "runs" / run.id / "run.json").read_bytes() == original
