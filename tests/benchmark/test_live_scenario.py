from __future__ import annotations

import json
from pathlib import Path

from live_helpers import install_json_handler, make_live_config

from bg6022.benchmark.loader import load_cases
from bg6022.benchmark.runner import _case_data_root, run_case


def _case(case_id: str):
    return next(item for item in load_cases("benchmarks/v1") if item.id == case_id)


def test_b010_continuation_updates_only_the_selected_requirement(
    monkeypatch, tmp_path: Path
) -> None:
    def scripted(purpose, messages):
        context = json.loads(messages[-1]["content"])
        if purpose == "intake":
            requirements = context["pending_context"]["requirements"]
            target_id = requirements[1]["requirement_id"]
            return {
                "intent": "chemistry_compute",
                "parameter_target_requirement_key": target_id,
                "parameter_patch": {"method_profile": "pbe0_d3bj_def2svp"},
            }
        raise AssertionError(f"parameter continuation unexpectedly called {purpose}")

    install_json_handler(monkeypatch, scripted)
    observation = run_case(
        _case("B010_modify_second_requirement").model_copy(update={"repeat": 1}),
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "completed"
    assert observation.run is not None
    assert observation.run["waiting_for"] == "confirmation"
    requirements = observation.request["requirements"]
    profiles = [item["parameters"].get("method_profile") for item in requirements]
    assert profiles == ["r2scan3c", "pbe0_d3bj_def2svp"]
    assert observation.orca_attempts == 0


def test_b022_ambiguous_iteration_keeps_the_seeded_opt_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    def scripted(purpose, _messages):
        assert purpose == "intake"
        return {
            "intent": "chemistry_compute",
            "unresolved_requirements": [
                "迭代上限可能指 SCF 收敛迭代或几何优化迭代，需要用户澄清。"
            ],
        }

    install_json_handler(monkeypatch, scripted)
    observation = run_case(
        _case("B022_ambiguous_iteration"),
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "blocked"
    assert observation.run is not None
    assert observation.run["status"] == "waiting"
    assert observation.run["waiting_for"] == "confirmation"
    step = observation.plan["steps"][0]
    assert "scf_maxiter" not in step["parameters"]
    assert "geom_maxiter" not in step["parameters"]
    assert observation.orca_attempts == 0


def test_b028_queries_a_published_verified_result_without_recomputation(
    monkeypatch, tmp_path: Path
) -> None:
    contexts = []

    def scripted(purpose, messages):
        context = json.loads(messages[-1]["content"])
        if purpose == "intake":
            contexts.append(context)
            result_entry = next(
                item
                for item in context["result_catalog"]
                if item["result"]["name"] == "opt_final_electronic_energy"
            )
            return {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": result_entry["subject_ref"],
                            "property": result_entry["result"]["property"],
                            "evidence": context["message"],
                            "reference_mode": "followup",
                        }
                    ],
                },
            }
        if purpose == "answer":
            available = context["available_outputs"]
            return {
                "action": "respond",
                "requested_results": [],
                "clarification": None,
                "sections": [
                    {
                        "format": "plain",
                        "heading": "results",
                        "output_refs": [available[0]["output_ref"]],
                        "detail": "normal",
                        "text": None,
                    }
                ],
            }
        raise AssertionError(f"saved-result query unexpectedly called {purpose}")

    install_json_handler(monkeypatch, scripted)
    observation = run_case(
        _case("B028_history_energy_query").model_copy(update={"repeat": 1}),
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "completed"
    assert observation.stage == "query"
    assert observation.run is not None
    assert observation.run["status"] == "succeeded"
    assert observation.run["result_index"]
    assert observation.orca_attempts == 0
    assert contexts[0]["result_catalog"]
    energy_entry = next(
        item
        for item in contexts[0]["result_catalog"]
        if item["result"]["name"] == "opt_final_electronic_energy"
    )
    assert energy_entry["recently_delivered"] is True
    assert [item["purpose"] for item in observation.llm_structured_outputs] == ["intake"]
    assert "-76.418938720985" in observation.response_text

    session_path = _case_data_root(
        tmp_path / "benchmark-data", "B028_history_energy_query", 1
    ) / "sessions" / "bench_live_B028_history_energy_query_1.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert len(session["recent_results"]) == 1
    assert session["recent_results"][0]["status"] == "succeeded"
