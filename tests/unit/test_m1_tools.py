from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bg6022.agent import Agent
from bg6022.config import LlmSettings, load_config
from bg6022.llm import LlmClient
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Run, Step
from bg6022.orca.profiles import resolve_parameters
from bg6022.orca.repair_rules import applicable_repairs
from bg6022.orca.runner import ProcessFacts
from bg6022.repair import RepairProposal, apply_repair_proposal
from bg6022.session import (
    artifact_path,
    clear_execution_guard,
    find_artifact,
    register_bytes_artifact,
    run_directory,
    utc_now,
)
from bg6022.tools.pubchem import fetch_pubchem
from bg6022.tools.registry import build_registry


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


def test_old_config_gets_bounded_m1_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.llm.model == "deepseek-flash"
    assert config.repair.max_attempts_per_science_step == 3
    assert config.molecule.embedding_seeds == [61453, 61454]


def test_deferred_orca_fields_keep_full_validation(tmp_path: Path) -> None:
    tool = build_registry(_config(tmp_path)).get("optimize_geometry")
    with pytest.raises(ValueError):
        tool.validate_parameters({"geom_maxiter": 0}, allow_deferred=True)
    with pytest.raises(ValueError):
        tool.validate_parameters({"multiplicity": 0}, allow_deferred=True)
    assert tool.validate_parameters({"geom_maxiter": 2}, allow_deferred=True) == {"geom_maxiter": 2}


def test_explicit_smiles_resolve_and_geometry_are_real_tools(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(
        id="request_chat",
        description="prepare water",
        source="chat",
    )
    plan = Plan(
        id="plan_prepare",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
        ],
        requested_results=[ResultTarget(step_id="geometry", port="geometry")],
    )
    plan = registry.validate_plan(plan)
    agent = Agent(config, registry, llm=None)
    run = agent._create_chat_run(request, plan)
    result = agent.advance(run)
    assert run.status == "succeeded"
    assert result is not None and result.status == "succeeded"
    geometry_id = result.output_ports["geometry"]
    geometry = next(item for item in run.artifact_index if item.id == geometry_id)
    assert geometry.metadata["initial_guess_only"] is True
    assert geometry.metadata["embedding"] == "ETKDGv3"
    assert geometry.size_bytes > 0


def test_parameter_resolution_preserves_explicit_zero_and_reports_sources() -> None:
    resolved = resolve_parameters(
        {"charge": 0},
        {"formal_charge": -1, "radical_electrons": 0, "atom_symbols": ["O"]},
        {"method_profile": "r2scan-3c"},
        {"method_profile": "r2scan3c", "environment": "gas"},
    )
    assert resolved.effective_parameters["charge"] == 0
    assert resolved.parameter_sources["charge"] == "request_explicit"
    assert resolved.effective_parameters["multiplicity"] == 1
    assert not resolved.missing_fields


def test_pubchem_url_encoding_and_bounded_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 962,
                            "Title": "water",
                            "ConnectivitySMILES": "O",
                            "MolecularFormula": "H2O",
                            "Charge": 0,
                        }
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lookup = fetch_pubchem("water & tea", "name", config=config, client=client)
    assert lookup.attempts == 2
    assert "%26" in calls[0]
    assert lookup.candidates[0]["CID"] == 962
    client.close()


def test_llm_json_schema_correction_is_bounded_and_does_not_record_secret() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"intent":"wrong"}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"intent":"chemistry_qa","answer":null}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return next(responses)

    config = LlmSettings(api_key_env="TEST_BG6022_KEY")
    client = LlmClient(
        config,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        api_key="do-not-record-this",
    )
    from bg6022.planner import IntakeOutput

    value = client.complete_json(
        [{"role": "user", "content": "question"}],
        IntakeOutput,
        purpose="test",
    )
    assert value.intent == "chemistry_qa"
    assert len(client.calls) == 2
    assert all("do-not-record-this" not in repr(call) for call in client.calls)


def test_restart_rule_requires_evidence_and_validated_candidate(tmp_path: Path) -> None:
    request = Request(id="request_repair", description="test")
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
            "geom_maxiter": 1,
        },
    )
    plan = Plan(id="plan_repair", request_id=request.id, steps=[step])
    run = Run(
        id="run_repair",
        request=request,
        plan=plan,
        resources={"cores": 4},
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    data_root = tmp_path / "data"
    (run_directory(data_root, run.id) / "artifacts").mkdir(parents=True, exist_ok=True)
    # The path is only needed for the durable model in this unit rule test.
    candidate = register_bytes_artifact(
        data_root,
        run,
        b"1\nH\nH 0 0 0\n",
        artifact_type="molecular_geometry",
        role="restart_candidate",
        source="test",
        extension=".xyz",
        step_id="opt",
        attempt=1,
        metadata={"eligible_for": "optimization_restart_only"},
    )
    result = Result(
        run_id=run.id,
        step_id="opt",
        attempt=1,
        status="failed",
        artifact_ids=[candidate.id],
        attempt_relative_path="opt/attempt-01",
        diagnostics={
            "facts": {
                "opt_iteration_limit_reached": True,
                "effective_geom_maxiter": 1,
            },
            "process": {
                "status": "failed",
                "process_tree_empty": True,
                "stop_confirmed": True,
            },
            "input_hashes": {"match": True},
        },
    )
    options = applicable_repairs(run, step, result)
    assert options and options[0].action == "restart_optimization"
    proposal = RepairProposal(
        action="restart_optimization",
        failed_step_key="opt",
        candidate_alias="last_complete_geometry",
        parameter_patch=options[0].parameter_patch,
        evidence_refs=list(options[0].evidence_refs),
    )
    candidate_plan, record = apply_repair_proposal(
        proposal,
        option=options[0],
        run=run,
        step=step,
        result=result,
        tool=build_registry().get("optimize_geometry"),
    )
    replacement = next(item for item in candidate_plan.steps if item.id == step.id)
    assert replacement.parameters["geom_maxiter"] == 100
    assert replacement.inputs["geometry"].artifact_id == candidate.id
    assert record["validated"] is True


def test_agent_repair_continues_same_run_after_failed_opt(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(
        id="request_opt",
        description="optimize water",
        source="chat",
        operation="Opt",
        requested_results=[ResultTarget(field="energy")],
        explicit_parameters={"charge": 0, "multiplicity": 1},
    )
    plan = Plan(
        id="plan_opt",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                    "geom_maxiter": 1,
                },
                inputs={"geometry": InputReference(step_id="geometry", port="geometry")},
            ),
        ],
        requested_results=[
            ResultTarget(step_id="opt", field="opt_final_electronic_energy"),
            ResultTarget(step_id="opt", port="optimized_geometry"),
        ],
    )
    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda _config: None)
    first_stdout = (
        b"GEOMETRY OPTIMIZATION CYCLE 1\n"
        b"CARTESIAN COORDINATES (ANGSTROEM)\n"
        b"------------------------------\n"
        b"O 0.000000 0.100000 0.000000\n"
        b"H 0.750000 -0.200000 0.000000\n"
        b"H -0.750000 -0.200000 0.000000\n"
        b"CARTESIAN COORDINATES (A.U.)\n"
        b"SCF CONVERGED AFTER 1 CYCLES\n"
        b"FINAL SINGLE POINT ENERGY -1.234000000000\n"
        b"MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED\n"
    )
    final_geometry = (
        b"3\nfinal\nO 0.000000 0.100000 0.000000\n"
        b"H 0.750000 -0.200000 0.000000\n"
        b"H -0.750000 -0.200000 0.000000\n"
    )
    second_stdout = (
        b"GEOMETRY OPTIMIZATION CYCLE 1\n"
        b"OPTIMIZATION HAS CONVERGED\n"
        b"CARTESIAN COORDINATES (ANGSTROEM)\n"
        b"------------------------------\n"
        b"O 0.000000 0.100000 0.000000\n"
        b"H 0.750000 -0.200000 0.000000\n"
        b"H -0.750000 -0.200000 0.000000\n"
        b"CARTESIAN COORDINATES (A.U.)\n"
        b"SCF CONVERGED AFTER 1 CYCLES\n"
        b"FINAL SINGLE POINT ENERGY -1.235000000000\n"
        b"****ORCA TERMINATED NORMALLY****\n"
        b"TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 0 msec\n"
    )
    calls = 0

    def fake_runner(**kwargs):
        nonlocal calls
        calls += 1
        attempt_dir = Path(kwargs["attempt_dir"])
        if calls == 1:
            process = ProcessFacts(
                status="failed",
                exit_code=1,
                stop_reason="nonzero_exit",
                process_tree_empty=True,
                stop_confirmed=True,
            )
            stdout = first_stdout
        else:
            process = ProcessFacts(
                status="succeeded",
                exit_code=0,
                process_tree_empty=True,
                stop_confirmed=True,
            )
            stdout = second_stdout
            (attempt_dir / "input.xyz").write_bytes(final_geometry)
        kwargs["on_started"](process)
        (attempt_dir / "stdout.out").write_bytes(stdout)
        (attempt_dir / "stderr.txt").write_bytes(b"")
        clear_execution_guard(kwargs["data_root"], kwargs["execution_id"])
        return process

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)

    class FakeRepairClient:
        def complete_json(self, *_args, **_kwargs):
            return RepairProposal(
                action="restart_optimization",
                failed_step_key="opt",
                candidate_alias="last_complete_geometry",
                parameter_patch={"geom_maxiter": 100},
                evidence_refs=[
                    "opt_iteration_limit_reached",
                    "candidate_geometry_valid",
                    "process_cleanup_confirmed",
                    "input_hashes_match",
                ],
            )

    agent = Agent(config, registry, llm=FakeRepairClient())
    run = agent._create_chat_run(request, registry.validate_plan(plan))
    run_id = run.id
    preview = agent.advance(run)
    assert preview is not None and preview.step_id == "geometry"
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    response = agent.confirm(run)
    assert response.run is not None and response.run.id == run_id
    run, result = response.run, response.result
    assert calls == 2
    assert run.status == "succeeded"
    assert run.id == run_id
    assert result is not None
    assert result.status == "succeeded"
    assert run.extra_executions_by_category["electronic_structure"] == 1
    assert len(run.repair_records) == 1
    assert run.current_results["opt"].endswith("attempt-02/result.json")
    energy_artifact = find_artifact(run, result.output_ports["energy_data"])
    geometry_artifact = find_artifact(run, result.output_ports["optimized_geometry"])
    energy_data = json.loads(artifact_path(config.data_root_path, run, energy_artifact).read_text())
    assert energy_artifact.id in result.artifact_ids
    assert energy_data["value"] == result.values["opt_final_electronic_energy"]["value"]
    assert energy_data["geometry"]["sha256"] == geometry_artifact.sha256
    assert energy_data["method_profile"] == "r2scan3c"
    assert energy_data["source"] == {"step_id": "opt", "attempt": result.attempt}
