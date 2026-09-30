"""Bounded read-only views: user Run copies and clearly synthetic boundary cases."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from threading import Event

import pytest

from bg6022.agent import Agent
from bg6022.answer import combine_output_reports, render_output_evidence
from bg6022.session import load_run
from bg6022.tools.orca_output import MAX_CANDIDATES, OutputQuerySpec, search_output_bytes
from bg6022.tools.registry import build_registry

FIXTURE = Path(__file__).parents[1] / "fixtures/query_delivery/user_water"


def user_water_agent(tmp_path, semantic=True, llm=None):
    from test_context_query import _config

    config = _config(tmp_path)
    config.runtime.semantic_planner_v1 = semantic
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    root = Path(config.data_root_path)
    shutil.copytree(FIXTURE / "runs", root / "runs")
    agent = Agent(
        config, build_registry(config), session_id=manifest["original_session_id"], llm=llm
    )
    agent._session["active_run_id"] = manifest["original_run_id"]
    return agent, root, manifest


def forbid_science(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("read-only query crossed a scientific execution/publish boundary")

    monkeypatch.setattr(Agent, "_invoke_step", forbidden)
    monkeypatch.setattr("bg6022.tools.orca.run_orca", forbidden)
    monkeypatch.setattr("bg6022.session.publish_step_result", forbidden)
    monkeypatch.setattr("bg6022.agent.publish_step_result", forbidden)


def scan(payload, hint=None, terms=None, **kwargs):
    if isinstance(payload, str):
        payload = payload.encode()
    return search_output_bytes(
        payload,
        [
            OutputQuerySpec(
                evidence="synthetic question",
                search_terms=terms or ["DIPOLE MOMENT"],
                property_hint=hint,
            )
        ],
        deadline=time.monotonic() + 5,
        cancel=Event(),
        limits={"snippets": 3, "lines": 120, "bytes": 8192},
        **kwargs,
    )[0]


def test_user_dipole_source_hash_and_exact_token():
    """N01/N08/N33: this is the user's actual Run, not the older water fixture."""
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    run_root = FIXTURE / "runs" / manifest["original_run_id"]
    run = load_run(FIXTURE, manifest["original_run_id"])
    stdout = next(a for a in run.artifact_index if a.artifact_type == "orca_output")
    payload = (run_root / stdout.relative_path).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["stdout_sha256"]
    item = scan(payload, "dipole_moment", ["Dipole moment", "Magnitude (Debye)"])
    assert item["candidate_status"] == "unique"
    assert item["observations"][0]["token"] == "1.861296656"
    report = {
        "queries": [{"evidence": "偶极矩", "property_hint": "dipole_moment"}],
        "evidence": [item],
    }
    text = render_output_evidence([report])
    assert text.startswith("该次 ORCA 输出报告") and "1.861296656" in text
    assert "SUGGESTED CITATIONS" not in text and "... YES" not in text
    assert "绑定到最终优化结构" in text
    assert combine_output_reports("", {}, [report])[1]["status"] == "partial"
    full = render_output_evidence([report], detail="full")
    assert "Magnitude (Debye)" in full and "原文第" in full


@pytest.mark.parametrize("count", [5, MAX_CANDIDATES, MAX_CANDIDATES + 1])
def test_candidate_capacity_is_not_the_old_five_hit_cutoff(count):
    """N03/N04: saturation requires a real extra candidate."""
    payload = ("needle\n" + "other\n" * 25) * count
    item = scan(payload, terms=["needle"])
    assert (item["search_status"] == "limited") == (count > MAX_CANDIDATES)
    if count == 5:
        assert "candidate_limit" not in item["limitations"]
    compact = scan("needle\n" * 5, terms=["needle"])
    assert compact["excerpt_complete"] and not compact["truncated"]


def test_long_unrelated_line_does_not_truncate_a_complete_numeric_block():
    """N05/N06: search completeness and result-block completeness are separate."""
    item = scan(
        "x" * 5000 + "\n---\nDIPOLE MOMENT\n---\nMagnitude (Debye): 1.25\n",
        "dipole_moment",
        ["DIPOLE MOMENT", "Magnitude"],
    )
    assert item["search_status"] == "limited" and item["excerpt_complete"]
    assert not item["truncated"] and not item["ambiguous"]
    assert item["observations"][0]["token"] == "1.25"


def test_title_total_and_magnitude_are_one_numeric_candidate():
    """N06: three literal terms in one synthetic result block never imply ambiguity."""
    item = scan(
        "DIPOLE MOMENT\nTotal Dipole Moment : 0 0 0.5\nMagnitude (Debye): 1.25\n",
        "dipole_moment",
        ["DIPOLE MOMENT", "Total", "Magnitude (Debye)"],
    )
    assert item["candidate_status"] == "unique" and not item["ambiguous"]
    assert len(item["snippets"]) == len(item["observations"]) == 1
    assert item["observations"][0]["token"] == "1.25"


@pytest.mark.parametrize(
    "payload",
    [
        "DIPOLE MOMENT\n---\n",
        "DIPOLE MOMENT\nMagnitude (Debye): 1\n---\nDIPOLE MOMENT\nMagnitude (Debye): 2\n",
        "DIPOLE MOMENT\nType of density: A\nMagnitude (Debye): 1\n"
        "Type of density: B\nMagnitude (Debye): 2\n",
    ],
)
def test_titles_stages_and_densities_are_not_unique_final_values(payload):
    """N07/N09/N10: do not select the last numeric value or mix densities."""
    item = scan(payload, "dipole_moment")
    assert not item["observations"] and not item["required_scope_complete"]


def orbit_table(rows):
    return "---\nORBITAL ENERGIES\n---\nNO OCC E(Eh) E(eV)\n" + rows + "---\nNEXT SECTION\n"


def test_synthetic_frontier_difference_preserves_source_precision():
    payload = (
        "ORBITAL ENERGIES\nNO OCC E(Eh) E(eV)\n"
        "0 2 -1 -6.123456789012345678901234567890\n"
        "1 0 0 1.098765432109876543210987654321\n"
    )
    item = scan(payload, "homo_lumo_gap", ["ORBITAL ENERGIES"])
    assert item["observations"][0]["token"] == "7.222222221122222222112222222211"


def test_synthetic_frontier_gap_and_long_table():
    """N23/N24: synthetic -6 and +1 eV, and frontier past the old window."""
    for count in (1, 80):
        rows = "".join(
            f"{i} 2.0000 {-100 - i / 100:.4f} {-100 + i:.4f}\n" for i in range(count - 1)
        )
        # A valid monotonic table needs ordered energies in its selected unit.
        rows += f"{count - 1} 2.0000 -0.2 -6\n{count} 0.0000 0.04 1\n"
        item = scan(orbit_table(rows), "frontier_orbitals", ["ORBITAL ENERGIES"])
        values = {v["property_hint"]: v for v in item["observations"]}
        assert values["homo_lumo_gap"]["token"] == "7"
        assert values["homo_lumo_gap"]["operands"] == {"HOMO": "-6", "LUMO": "1"}
        assert values["lumo_energy"]["unit"] == "eV"
        assert "-6" in item["snippets"][0]["text"]


@pytest.mark.parametrize(
    "rows",
    [
        "0 2 -0.2 -6\n",
        "0 1.5 -0.2 -6\n1 0 0.04 1\n",
        "0 0 0.04 1\n1 2 -0.2 -6\n",
        "0 2 -0.2 -6\n2 0 0.04 1\n",
        "SPIN UP ORBITALS\n0 2 -0.2 -6\n1 0 0.04 1\n",
    ],
)
def test_unsupported_orbital_tables_do_not_produce_a_gap(rows):
    """N25: no missing, fractional, broken, spin or non-Aufbau fallback."""
    item = scan(orbit_table(rows), "homo_lumo_gap", ["ORBITAL ENERGIES"])
    assert not item["observations"]


def test_two_methods_or_tables_never_supply_mixed_operands():
    """N26: two complete tables require a stage binding, never cross-subtraction."""
    item = scan(
        orbit_table("0 2 -0.2 -6\n1 0 0.04 1\n") + orbit_table("0 2 -0.3 -7\n1 0 0.08 2\n"),
        "homo_lumo_gap",
        ["ORBITAL ENERGIES"],
    )
    assert item["candidate_status"] == "multiple" and not item["observations"]


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("goal", ["dipole_moment", "frontier_orbitals", "chemical_total_electrons"])
def test_agent_delivers_each_readonly_adapter_with_zero_new_steps(
    tmp_path, monkeypatch, semantic, goal
):
    """N01/N23/N27: real Agent, fixed model proposals, exact user Run bytes."""
    agent, root, manifest = user_water_agent(tmp_path, semantic)
    forbid_science(monkeypatch)
    before = {
        p.relative_to(root / "runs"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            context = json.loads(messages[-1]["content"])
            entry = next(
                e
                for e in context["result_catalog"]
                if e.get("access")
                == ("readonly_observation" if goal == "chemical_total_electrons" else "raw_output")
            )
            target = {
                "subject_ref": entry["subject_ref"],
                "property": goal if goal == "chemical_total_electrons" else "orca_output",
                "evidence": "question",
            }
            if goal != "chemical_total_electrons":
                target["queries"] = [
                    {
                        "evidence": "question",
                        "property_hint": goal,
                        "search_terms": [
                            "ORBITAL ENERGIES" if goal == "frontier_orbitals" else "DIPOLE MOMENT"
                        ],
                    }
                ]
            return schema.model_validate(
                {
                    ("mode" if semantic else "intent"): "context_query",
                    "intent_items": [
                        {"kind": "query", "evidence": "question", "requested_property": goal}
                    ],
                    "query_selection": {"status": "selected", "targets": [target]},
                }
            )

    agent.llm = Client()
    response = agent.handle_message("question")
    assert response.run is None and response.delivery["raw_reports"]
    report = response.delivery["raw_reports"][0]
    assert report["run_id"] == manifest["original_run_id"] and report["attempt"] == 1
    assert report["source_step_id"] == "s03_optimize_geometry"
    if goal == "chemical_total_electrons":
        assert "总电子数 10" in response.text and "ORCA 输出报告" not in response.text
        assert response.delivery["status"] == "complete"
    elif goal == "dipole_moment":
        assert "1.861296656" in response.text and response.delivery["status"] == "partial"
    else:
        assert "1.9216 eV" in response.text and "9.3533 eV" in response.text
        assert response.delivery["status"] == "complete"
    after = {
        p.relative_to(root / "runs"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize("charge", [1, None, "changed_step"])
def test_electron_count_uses_executed_charge_without_default_neutral(tmp_path, charge):
    """N28: synthetic mutations of a test COPY, never edits of the user's source."""
    from bg6022.agent import _step_fingerprint
    from bg6022.models import Result
    from bg6022.output_query import (
        build_readonly_observation_entries,
        collect_raw_output_sources,
        query_readonly_observations,
    )
    from bg6022.session import save_result, save_run

    agent, root, manifest = user_water_agent(tmp_path)
    run = load_run(root, manifest["original_run_id"])
    step = next(s for s in run.plan.steps if s.tool == "optimize_geometry")
    relative = run.current_results[step.id]
    result = Result.model_validate_json((root / "runs" / run.id / relative).read_bytes())
    if charge is None:
        step.parameters.pop("charge")
    else:
        step.parameters["charge"] = 1
    if charge != "changed_step":
        result.step_fingerprint = _step_fingerprint(step)
    save_result(root, run, result)
    save_run(root, run)
    sources = collect_raw_output_sources(root, run, agent.session_id)
    entries, bindings = build_readonly_observation_entries(sources, root)
    if charge != 1:
        assert not entries and not bindings
    else:
        binding = next(iter(bindings.values()))
        reports = query_readonly_observations(
            root, [(binding, "count")], session_id=agent.session_id, indexed_ids=[run.id]
        )
        assert "总电子数 9" in render_output_evidence(reports)
        observation = reports[0]["evidence"][0]["observations"][0]
        assert observation["operands"]["charge"] == 1
        assert observation["view_kind"] == "derived_value"


@pytest.mark.parametrize(
    "hint,field,token",
    [
        ("orca_printed_electron_count", "Number of Electrons", "10"),
        ("orca_printed_correlated_electrons", "Number of Correlated Electrons", "8"),
        ("orca_printed_alpha_electrons", "Number of Alpha Electrons", "5"),
        ("orca_printed_beta_electrons", "Number of Beta Electrons", "4"),
    ],
)
def test_printed_electron_definitions_are_not_interchanged(hint, field, token):
    """N29: distinctly labelled synthetic printed fields, no chemical substitution."""
    payload = (
        "Number of Electrons : 10\nNumber of Correlated Electrons : 8\n"
        "Number of Alpha Electrons : 5\nNumber of Beta Electrons : 4\n"
    )
    evidence = scan(payload, hint, [field])
    assert evidence["observations"][0]["token"] == token
    assert evidence["observations"][0]["definition"] == field
    assert evidence["observations"][0]["view_kind"] == "observed_value"


def test_three_questions_receive_fair_excerpts_from_one_verified_read(tmp_path, monkeypatch):
    """N12: one prolific question cannot consume the other questions' entire display."""
    from test_orca_output_query import lookup, saved_output

    from bg6022.tools import orca_output

    payload = ("FIRST\n" + "filler\n" * 24) * 18 + "SECOND\nanswer two\nTHIRD\nanswer three\n"
    run, _ = saved_output(tmp_path, payload=payload.encode())
    reads = []
    original = orca_output.read_registered_artifact_bytes
    monkeypatch.setattr(
        orca_output,
        "read_registered_artifact_bytes",
        lambda *args, **kwargs: reads.append(True) or original(*args, **kwargs),
    )
    reports = lookup(
        tmp_path,
        run,
        [
            {"evidence": t, "search_terms": [t], "property_hint": "raw_excerpt"}
            for t in ("FIRST", "SECOND", "THIRD")
        ],
    )
    evidence = reports[0]["evidence"]
    assert reads == [True] and all(e["snippets"] for e in evidence)
    assert len(evidence[0]["snippets"]) == 1
    assert evidence[1]["required_scope_complete"] and evidence[2]["required_scope_complete"]


@pytest.mark.parametrize("semantic", [True, False])
def test_explicit_method_query_then_ambiguous_source_preserves_candidates(
    tmp_path, monkeypatch, semantic
):
    """N22: recorded PBE0 source selected explicitly; ambiguous next source stays pending."""
    agent, root, _ = user_water_agent(tmp_path, semantic)
    method_fixture = FIXTURE.parent / "existing_methods"
    methods = json.loads((method_fixture / "manifest.json").read_text())
    shutil.copytree(method_fixture / "runs" / methods["run_id"], root / "runs" / methods["run_id"])
    agent._session["recent_results"] = [{"run_id": methods["run_id"]}]
    forbid_science(monkeypatch)

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            context = json.loads(messages[-1]["content"])
            question = context.get("message", context.get("user_message"))
            raw = [e for e in context["result_catalog"] if e.get("access") == "raw_output"]
            if question == "指定 PBE0 来源的偶极矩":
                entry = next(
                    e for e in raw if e["source_context"]["method_label"].startswith("PBE0")
                )
                selection = {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": entry["subject_ref"],
                            "property": "orca_output",
                            "evidence": question,
                            "queries": [
                                {
                                    "evidence": question,
                                    "property_hint": "dipole_moment",
                                    "search_terms": ["DIPOLE MOMENT"],
                                }
                            ],
                        }
                    ],
                }
            else:
                selection = {
                    "status": "clarify",
                    "reason": "ambiguous_subject",
                    "clarification_context": {
                        "original_question": question,
                        "origin_evidence": question,
                        "property_hint": "dipole_moment",
                        "search_terms": ["DIPOLE MOMENT"],
                        "candidate_source_refs": [e["subject_ref"] for e in raw],
                        "missing_slots": ["source"],
                    },
                }
            return schema.model_validate(
                {
                    ("mode" if semantic else "intent"): "context_query",
                    "intent_items": [
                        {
                            "kind": "query",
                            "evidence": question,
                            "requested_property": "dipole_moment",
                        }
                    ],
                    "query_selection": selection,
                }
            )

    agent.llm = Client()
    response = agent.handle_message("指定 PBE0 来源的偶极矩")
    assert "1.953232539" in response.text
    assert response.delivery["raw_reports"][0]["source_step_id"] == "sp_b"
    response = agent.handle_message("那个来源的偶极矩")
    assert not response.delivery and response.run is None
    assert agent._session["pending_query"]["missing_slots"] == ["source"]
    assert len(agent._session["pending_query"]["source_bindings"]) == 3
