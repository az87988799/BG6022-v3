"""Explicit live acceptance: distinguish live parsing from deterministic proposals."""

from __future__ import annotations

import hashlib
import json
import os
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
