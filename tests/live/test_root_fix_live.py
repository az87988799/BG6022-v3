"""Explicit live acceptance: distinguish live parsing from deterministic proposals."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from threading import Event

import pytest

from bg6022.agent import Agent
from bg6022.canonicalize import canonicalize_semantic_request
from bg6022.config import load_config
from bg6022.llm import LlmClient
from bg6022.models import Result
from bg6022.output_query import reports_for_run
from bg6022.plan_builder import build_plan
from bg6022.planner import intake_message, request_from_intake
from bg6022.semantic import SemanticProposal, semantic_message
from bg6022.session import new_id, utc_now
from bg6022.tools.registry import build_registry


@pytest.mark.live_llm
@pytest.mark.parametrize(
    "message",
    [
        "优化水，然后给出它的偶极矩",
        "优化水，然后告诉我偶极矩",
        "优化水并报告偶极矩",
        "优化水，然后计算它的偶极矩",
    ],
)
@pytest.mark.parametrize("semantic", [True, False])
def test_live_natural_dipole_intake(pytestconfig, message, semantic):
    config = load_config(pytestconfig.getoption("--orca-config") or "config.toml")
    if not os.environ.get(config.llm.api_key_env):
        pytest.skip("configured LLM API key is unavailable")
    registry = build_registry(config)
    client = LlmClient(config)
    if semantic:
        proposal = semantic_message(client, message, registry=registry)
        tasks = proposal.tasks
        assert proposal.mode == "compute"
        request = canonicalize_semantic_request(
            message, proposal, request_id="live_dipole", registry=registry
        )
    else:
        proposal = intake_message(client, message, registry=registry)
        tasks = proposal.requirements
        assert proposal.intent == "chemistry_compute"
        request = request_from_intake(
            message, proposal, request_id="live_dipole", registry=registry
        )
    assert [task.capability for task in tasks] == ["optimize_geometry"]
    assert len(tasks[0].report_queries) == 1
    assert any(item.kind == "report" for item in proposal.intent_items)
    plan = build_plan(request, registry=registry, plan_id="live_dipole")
    assert [
        step.tool for step in plan.steps if registry.get(step.tool).requires_compute_permission
    ] == ["optimize_geometry"]
    evidence = {
        "message": message,
        "entrypoint": "semantic" if semantic else "intake",
        "proposal": proposal.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "llm_calls": [asdict(c) for c in client.calls],
        "execution": "planning only; no ORCA run",
    }
    out = Path(config.data_root_path) / "root-fix-evidence" / "llm"
    out.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((str(semantic) + message).encode()).hexdigest()[:16]
    (out / f"{name}.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.mark.live_orca
def test_real_shared_optimized_geometry_and_dipole_report(pytestconfig):
    config_path = pytestconfig.getoption("--orca-config")
    if not config_path:
        pytest.fail("--orca-config is required")
    config = load_config(config_path)
    config.repair.enabled = False
    registry = build_registry(config)
    agent = Agent(config, registry)
    xyz = Path("examples/water.xyz").read_text(encoding="utf-8")
    message = (
        "Optimize water, report dipole moment; compare PBE0 and B3LYP energies "
        "at that optimized geometry. charge=0 multiplicity=1\n" + xyz
    )
    tasks = [
        {
            "key": key,
            "subject_key": "water",
            "capability": capability,
            "method_request": method,
            "parameters": {"charge": 0, "multiplicity": 1},
        }
        for key, capability, method in [
            ("opt", "optimize_geometry", "r2SCAN-3c"),
            ("a", "single_point", "PBE0"),
            ("b", "single_point", "B3LYP"),
        ]
    ]
    tasks[0]["report_queries"] = [
        {"evidence": "dipole moment", "search_terms": ["Total Dipole Moment", "Magnitude (Debye)"]}
    ]
    proposal = SemanticProposal(
        mode="compute",
        subjects=[{"key": "water", "query": "water", "evidence": "water"}],
        tasks=tasks,
        relations=[
            {"type": "difference", "tasks": ["a", "b"]},
            *(
                {
                    "type": "use_output",
                    "source_task": "opt",
                    "target_task": key,
                    "property": "geometry",
                }
                for key in ("a", "b")
            ),
        ],
    )
    request = canonicalize_semantic_request(
        message, proposal, request_id="root_fix_real", registry=registry
    )
    plan = build_plan(request, registry=registry, plan_id="root_fix_real")
    run = agent._create_chat_run(request, plan)
    result = agent.advance(run)
    assert run.waiting_for == "confirmation"
    response = agent.confirm(run)
    run, result = response.run, response.result
    root = Path(config.data_root_path) / "runs" / run.id
    results = [Result.model_validate_json((root / path).read_bytes()) for path in run.result_index]
    reports = reports_for_run(
        config.data_root_path,
        run,
        session_id=agent.session_id,
        cancel=Event(),
        output_limit_bytes=config.output_limit_bytes,
    )
    evidence = {
        "run_id": run.id,
        "run_status": run.status,
        "resources": run.resources,
        "intake": "deterministic proposal; not live LLM evidence",
        "result_status": result.status if result else None,
        "results": [
            {
                "step_id": r.step_id,
                "status": r.status,
                "input_bindings": r.input_bindings,
                "input_hashes": {
                    name: next(a.sha256 for a in run.artifact_index if a.id == artifact_id)
                    for name, artifact_id in r.input_bindings.items()
                },
                "values": r.values,
            }
            for r in results
        ],
        "reports": reports,
        "attempts": run.attempts,
        "artifacts": [{"id": a.id, "role": a.role, "sha256": a.sha256} for a in run.artifact_index],
    }
    out = Path(config.data_root_path) / "root-fix-evidence"
    out.mkdir(exist_ok=True)
    (out / "real-shared-geometry.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert run.status == "succeeded", evidence
    assert len([s for s in run.plan.steps if registry.get(s.tool).requires_compute_permission]) == 3
    sp_steps = {s.id for s in run.plan.steps if s.tool == "single_point"}
    sp = [r for r in results if r.step_id in sp_steps]
    assert len(sp) == 2 and sp[0].input_bindings["geometry"] == sp[1].input_bindings["geometry"]
    sp_attempts = [a for a in run.attempts if a["step_id"] in sp_steps]
    assert sp_attempts[0]["geometry_sha256"] == sp_attempts[1]["geometry_sha256"]
    initial = next(a for a in run.artifact_index if a.role == "input_geometry")
    assert hashlib.sha256((root / initial.relative_path).read_bytes()).hexdigest() == initial.sha256
    assert all(r["evidence"][0]["lookup_status"] in {"found", "ambiguous"} for r in reports)
    assert run.resources["cores"] == 4 and run.resources["memory_mb"] == 1024
    assert run.resources["maxcore_mb"] == 192 and run.resources["max_concurrent_jobs"] == 1


def _saved_dialogue_agent(tmp_path, config_path, semantic, recorder, two_methods):
    """Copy existing registered evidence, never write to the production data root."""
    config = load_config(config_path)
    if not os.environ.get(config.llm.api_key_env):
        env_path = Path(config_path).resolve().parent / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == config.llm.api_key_env:
                    os.environ[key.strip()] = value.strip().strip("\"'")
    if not os.environ.get(config.llm.api_key_env):
        pytest.skip("configured LLM API key is unavailable")
    config.runtime.data_root = str(tmp_path / "readonly-data")
    config.data_root_path = config.runtime.data_root
    assert Path(config.data_root_path).resolve().is_relative_to(tmp_path.resolve())
    config.runtime.semantic_planner_v1 = semantic
    fixture = Path(__file__).parents[1] / "fixtures/query_delivery"
    manifest = json.loads((fixture / "user_water/manifest.json").read_text(encoding="utf-8"))
    root = Path(config.data_root_path)
    shutil.copytree(fixture / "user_water/runs", root / "runs")
    client = LlmClient(config, response_recorder=recorder)
    agent = Agent(
        config, build_registry(config), session_id=manifest["original_session_id"], llm=client
    )
    agent._session["active_run_id"] = manifest["original_run_id"]
    if two_methods:
        methods = json.loads((fixture / "existing_methods/manifest.json").read_text())
        shutil.copytree(
            fixture / "existing_methods/runs" / methods["run_id"], root / "runs" / methods["run_id"]
        )
        agent._session["recent_results"] = [{"run_id": methods["run_id"]}]
    return agent, client, root, manifest


@pytest.mark.live_llm
@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("chain", ["L-A", "L-B", "L-C"])
@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_saved_result_dialogue(pytestconfig, tmp_path, monkeypatch, semantic, chain, repeat):
    """18 bounded real-model sessions; scientific execution is blocked throughout."""
    captures, turns, science_calls = [], [], []
    config_path = pytestconfig.getoption("--orca-config") or "config.toml"
    agent, client, root, manifest = _saved_dialogue_agent(
        tmp_path, config_path, semantic, captures.append, chain == "L-B"
    )

    def forbidden(*args, **kwargs):
        science_calls.append(True)
        pytest.fail("saved-result dialogue attempted new science")

    monkeypatch.setattr(Agent, "_invoke_step", forbidden)
    monkeypatch.setattr("bg6022.tools.orca.run_orca", forbidden)
    monkeypatch.setattr("bg6022.session.publish_step_result", forbidden)
    monkeypatch.setattr("bg6022.agent.publish_step_result", forbidden)
    before = {
        p.relative_to(root / "runs").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }
    evidence = {
        "schema": "bg6022.saved_dialogue_test.v1",
        "chain": chain,
        "entrypoint": "semantic" if semantic else "intake",
        "repeat": repeat,
        "started_at": utc_now(),
        "source_kind": "user_run_copy",
        "user_stdout_sha256": manifest["stdout_sha256"],
        "model": client.settings.model,
        "outcome": "failed",
        "execution": "existing Run copies; no new ORCA",
        "code_sha256": {
            p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *(
                    Path("src/bg6022") / name
                    for name in (
                        "agent.py",
                        "answer.py",
                        "canonicalize.py",
                        "llm.py",
                        "output_contracts.py",
                        "output_query.py",
                        "planner.py",
                        "semantic.py",
                        "session.py",
                        "tools/orca_output.py",
                        "prompts/answer.md",
                        "prompts/intake.md",
                        "prompts/semantic.md",
                    )
                )
            ]
        },
        "resources": agent.config.resources,
    }

    def ask(message):
        response = agent.handle_message(message)
        turns.append(
            {
                "question": message,
                "text": response.text,
                "delivery": response.delivery,
                "pending_query": agent._session.get("pending_query"),
                "new_run": response.run.id if response.run else None,
            }
        )
        assert response.run is None or response.run.id in {manifest["original_run_id"]}
        assert "字段匹配失败" not in response.text
        assert not science_calls
        return response

    def reports(response):
        result = response.delivery.get("raw_reports", [])
        assert result, response.text
        return result

    try:
        if chain == "L-A":
            response = ask("读取刚才水优化计算的偶极矩。")
            assert "1.861296656" in response.text and response.delivery["status"] == "partial"
            assert reports(response)[0]["run_id"] == manifest["original_run_id"]
            response = ask("LUMO 能隙呢")
            if agent._session.get("pending_query"):
                response = ask("查询")
                if agent._session.get("pending_query"):
                    response = ask("能隙")
            assert "9.3533" in response.text, response.text
            response = ask("水分子的电子数呢")
            assert "总电子数 10" in response.text and response.delivery["status"] == "complete"
        elif chain == "L-B":
            response = ask("读取已保存的 PBE0 单点水计算输出中的偶极矩。")
            if agent._session.get("pending_query"):
                response = ask("选 PBE0 单点结果，读取已有输出。")
            assert "1.953232539" in response.text
            first_source = reports(response)[0]
            assert first_source["source_context"]["method_label"].startswith("PBE0")
            response = ask("只查这次 PBE0 单点水体系的总电子数。")
            assert "总电子数 10" in response.text
            assert reports(response)[0]["run_id"] == first_source["run_id"]
            assert reports(response)[0]["source_step_id"] == first_source["source_step_id"]
            response = ask("只重复刚才的总电子数。")
            assert "总电子数 10" in response.text and response.delivery["status"] == "complete"
            assert reports(response)[0]["source_step_id"] == first_source["source_step_id"]
        else:
            response = ask(
                "只读查询刚才水优化输出的 UNREGISTERED_TEST_PROPERTY_MISSING_X 字段原文。"
            )
            assert response.delivery.get("status") != "complete"
            assert not any(
                e.get("observations")
                for r in response.delivery.get("raw_reports", [])
                for e in r.get("evidence", [])
            )
            response = ask("那这次水优化输出中的偶极矩呢")
            assert "1.861296656" in response.text and response.delivery["status"] == "partial"
            ask("/new")
            assert not agent._session.get("pending_query") and not agent._session.get(
                "active_run_id"
            )
            response = ask("一般中性水分子有几个电子？")
            assert "10" in response.text and not response.delivery.get("raw_reports")
        evidence["outcome"] = "passed"
    except Exception as error:
        evidence["error"] = client._sanitize(str(error))[:2000]
        raise
    finally:
        after = {
            p.relative_to(root / "runs").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "runs").rglob("*")
            if p.is_file()
        }
        evidence.update(
            turns=turns,
            model_responses=captures,
            llm_calls=[asdict(c) for c in client.calls],
            science_calls=len(science_calls),
            run_files_unchanged=(before == after),
            finished_at=utc_now(),
        )
        out = Path(os.environ.get("BG6022_QUERY_EVIDENCE_DIR", "docs/evidence/query-delivery/live"))
        out.mkdir(parents=True, exist_ok=True)
        (
            out / f"{chain}-{'semantic' if semantic else 'intake'}-{repeat}-{new_id('sample')}.json"
        ).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert before == after
