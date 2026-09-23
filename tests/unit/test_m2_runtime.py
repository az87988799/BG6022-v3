from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bg6022.agent import (
    Agent,
    _invalidate_current_results,
    _next_ready_step,
    _step_fingerprint,
    _unmet_goal_checks,
)
from bg6022.config import load_config
from bg6022.models import (
    GoalCheckRequirement,
    InputReference,
    Plan,
    Request,
    Result,
    ResultTarget,
    Run,
    ScientificCheckResult,
    Step,
)
from bg6022.orca.frequency_parser import parse_vibrational_frequencies
from bg6022.planner import IntakeOutput, PlanProposal, PlanStepProposal, PlanTargetProposal
from bg6022.repair import RepairProposal
from bg6022.repair import _context as repair_context
from bg6022.session import (
    artifact_path,
    create_run,
    register_bytes_artifact,
    save_result,
    save_run,
    utc_now,
)
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.orca import FrequencyParameters, _local_minimum_check
from bg6022.tools.registry import ToolRegistry, build_registry

WATER_XYZ = (
    b"3\nwater\n"
    b"O 0.000000 0.000000 0.000000\n"
    b"H 0.758602 0.000000 0.504284\n"
    b"H -0.758602 0.000000 0.504284\n"
)


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


def _save_result(data_root: Path, run: Run, result: Result) -> str:
    path = save_result(data_root, run, result)
    relative = path.relative_to(data_root / "runs" / run.id).as_posix()
    run.result_index.append(relative)
    run.current_results[result.step_id] = relative
    return relative


def _source_opt_run(config: Any, session_id: str) -> tuple[Run, Any]:
    data_root = Path(config.data_root_path)
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(artifact_id="seed")},
    )
    target = ResultTarget(step_id=step.id, port="optimized_geometry")
    request = Request(
        id="request_history_source",
        description="verified water optimization",
        operations=["Opt"],
        requested_results=[target],
        source="chat",
    )
    run = Run(
        id="run_history_source",
        request=request,
        plan=Plan(
            id="plan_history_source",
            request_id=request.id,
            steps=[step],
            requested_results=[target],
        ),
        resources=config.resources,
        status="succeeded",
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    run.plan = run.plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"inputs": {"geometry": InputReference(artifact_id="seed")}})
            ]
        }
    )
    create_run(data_root, run)
    seed = register_bytes_artifact(
        data_root,
        run,
        WATER_XYZ,
        artifact_type="molecular_geometry",
        role="input_geometry",
        source="test:input_geometry",
        extension=".xyz",
    )
    run.plan = run.plan.model_copy(
        update={
            "steps": [
                step.model_copy(
                    update={"inputs": {"geometry": InputReference(artifact_id=seed.id)}}
                )
            ]
        }
    )
    optimized = register_bytes_artifact(
        data_root,
        run,
        WATER_XYZ,
        artifact_type="molecular_geometry",
        role="optimized_geometry",
        source="test:successful_opt",
        extension=".xyz",
        step_id=step.id,
        attempt=1,
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        artifact_ids=[optimized.id],
        output_ports={"optimized_geometry": optimized.id},
        input_artifact_ids=[seed.id],
        input_bindings={"geometry": seed.id},
        attempt_relative_path="opt/attempt-01",
        step_fingerprint=_step_fingerprint(run.plan.steps[0]),
    )
    _save_result(data_root, run, result)
    run.attempts.append(
        {
            "step_id": step.id,
            "attempt": 1,
            "phase": "finished",
            "status": "succeeded",
            "artifact_ids": [optimized.id],
        }
    )
    save_run(data_root, run)
    return run, optimized


def test_history_geometry_is_verified_copied_and_bound_to_the_new_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    session_id = "session_history"
    source_run, source_artifact = _source_opt_run(config, session_id)
    agent = Agent(config, build_registry(config), llm=None, session_id=session_id)
    agent._session["active_run_id"] = source_run.id

    catalog, bindings = agent._build_geometry_catalog()
    assert len(catalog) == 1
    assert catalog[0]["alias"] == "geometry_1"

    step = Step(
        id="sp",
        tool="single_point",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(artifact_id="geometry_1")},
    )
    target = ResultTarget(step_id=step.id, field="sp_electronic_energy")
    request = Request(
        id="request_history_reuse",
        description="single point on the selected historical geometry",
        operations=["SP"],
        requested_results=[target],
        structure_input={"history_geometry_alias": "geometry_1"},
        source="chat",
    )
    plan = Plan(
        id="plan_history_reuse",
        request_id=request.id,
        steps=[step],
        requested_results=[target],
    )

    new_run = agent._create_chat_run(request, plan, history_geometry_binding=bindings["geometry_1"])
    new_step = new_run.plan.steps[0]
    copied = next(
        artifact
        for artifact in new_run.artifact_index
        if artifact.id == new_step.inputs["geometry"].artifact_id
    )

    assert copied.id != source_artifact.id
    assert copied.run_id == new_run.id
    assert copied.role == "input_geometry"
    assert copied.sha256 == source_artifact.sha256
    assert copied.metadata["history_source_sha256"] == source_artifact.sha256
    assert artifact_path(config.data_root_path, new_run, copied).read_bytes() == WATER_XYZ
    assert (
        artifact_path(config.data_root_path, source_run, source_artifact).read_bytes() == WATER_XYZ
    )


@pytest.mark.parametrize("tamper", ["missing", "fabricated", "wrong_role", "altered_bytes"])
def test_invalid_historical_geometry_bindings_stop_before_a_new_run(
    tmp_path: Path, tamper: str
) -> None:
    config = _config(tmp_path)
    session_id = f"session_history_{tamper}"
    source_run, source_artifact = _source_opt_run(config, session_id)
    agent = Agent(config, build_registry(config), llm=None, session_id=session_id)
    agent._session["active_run_id"] = source_run.id
    catalog, bindings = agent._build_geometry_catalog()
    assert len(catalog) == 1
    binding = dict(bindings["geometry_1"])

    if tamper == "wrong_role":
        source_run.artifact_index = [
            item.model_copy(update={"role": "initial_geometry"})
            if item.id == source_artifact.id
            else item
            for item in source_run.artifact_index
        ]
        save_run(config.data_root_path, source_run)
    elif tamper == "altered_bytes":
        path = artifact_path(config.data_root_path, source_run, source_artifact)
        original = path.read_bytes()
        path.write_bytes(original.replace(b"0.758602", b"0.758603", 1))
    elif tamper == "fabricated":
        binding["artifact_id"] = "artifact_fabricated"
    elif tamper == "missing":
        binding = None  # type: ignore[assignment]

    step = Step(
        id="sp",
        tool="single_point",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(artifact_id="geometry_1")},
    )
    target = ResultTarget(step_id=step.id, field="sp_electronic_energy")
    request = Request(
        id=f"request_history_{tamper}",
        description="run only SP on the selected previous optimized geometry",
        operations=["SP"],
        requested_results=[target],
        structure_input={"history_geometry_alias": "geometry_1"},
        source="chat",
    )
    plan = Plan(
        id=f"plan_history_{tamper}",
        request_id=request.id,
        steps=[step],
        requested_results=[target],
    )

    with pytest.raises(ValueError, match="history geometry|artifact hash or size mismatch"):
        agent._create_chat_run(request, plan, history_geometry_binding=binding)

    run_dirs = list((Path(config.data_root_path) / "runs").iterdir())
    assert run_dirs == [Path(config.data_root_path) / "runs" / source_run.id]


def test_local_minimum_check_excludes_supported_external_modes_and_preserves_imaginary_modes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run, optimized = _source_opt_run(config, "session_minimum")
    frequency_step = Step(
        id="freq",
        tool="frequency",
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
    )
    parameters = FrequencyParameters(charge=0, multiplicity=1)

    def parse(values: list[str]):
        rows = [f"{index}: {value} cm**-1" for index, value in enumerate(values)]
        stdout = (
            "VIBRATIONAL FREQUENCIES\n-----------------------\n"
            + "\n".join(rows)
            + "\nScaling factor for frequencies = 1.0 (already applied!)\n"
            "NORMAL MODES\n-------------\n"
        )
        return parse_vibrational_frequencies(stdout, expected_atom_count=3)

    saddle = _local_minimum_check(
        config,
        run,
        frequency_step,
        optimized,
        parse_xyz_bytes(WATER_XYZ),
        parameters,
        parse(["0.00"] * 6 + ["-83.22", "435.79", "544.50"]),
        True,
        optimized.sha256,
    )
    supported = _local_minimum_check(
        config,
        run,
        frequency_step,
        optimized,
        parse_xyz_bytes(WATER_XYZ),
        parameters,
        parse(["0.00"] * 6 + ["100.0", "435.79", "544.50"]),
        True,
        optimized.sha256,
    )
    unsupported = _local_minimum_check(
        config,
        run,
        frequency_step,
        optimized,
        parse_xyz_bytes(WATER_XYZ),
        parameters,
        parse(
            [
                "5.0",
                "0.00",
                "0.00",
                "0.00",
                "0.00",
                "0.00",
                "100.0",
                "435.79",
                "544.50",
            ]
        ),
        True,
        optimized.sha256,
    )

    assert saddle.status == "not_met"
    assert saddle.conditions["negative_mode_indices"] == [6]
    assert supported.status == "passed"
    assert supported.conditions["external_mode_count"] == 6
    assert unsupported.status == "unverified"


@pytest.mark.parametrize("check_status", ["not_met", "unverified"])
def test_failed_local_minimum_gate_never_calls_the_sp_executor(
    tmp_path: Path, check_status: str
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    base_registry = build_registry(config)

    def forbidden_sp(*args: Any, **kwargs: Any) -> Result:
        calls.append("single_point")
        raise AssertionError("SP must stay blocked by an unmet scientific check")

    replacement = base_registry.get("single_point").model_copy(
        update={"execute_function": forbidden_sp}
    )
    registry = ToolRegistry(
        [
            replacement if tool.name == "single_point" else tool
            for tool in (base_registry.get(name) for name in base_registry.names())
        ]
    )
    data_root = Path(config.data_root_path)
    opt = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(artifact_id="seed")},
    )
    freq = Step(
        id="freq",
        tool="frequency",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
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
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
        goal_checks=[GoalCheckRequirement(source_step_id=freq.id, check="local_minimum_supported")],
    )
    request = Request(
        id=f"request_gate_{check_status}",
        description="verify a local minimum before an independent single point",
        operations=["Opt", "Freq", "SP"],
        requested_results=[
            ResultTarget(step_id=freq.id, check="local_minimum_supported"),
            ResultTarget(step_id=sp.id, field="sp_electronic_energy"),
        ],
        source="chat",
    )
    run = Run(
        id=f"run_gate_{check_status}",
        request=request,
        plan=Plan(
            id=f"plan_gate_{check_status}",
            request_id=request.id,
            steps=[opt, freq, sp],
            requested_results=request.requested_results,
        ),
        resources=config.resources,
        status="planned",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(data_root, run)
    seed = register_bytes_artifact(
        data_root,
        run,
        WATER_XYZ,
        artifact_type="molecular_geometry",
        role="input_geometry",
        source="test:input_geometry",
        extension=".xyz",
    )
    optimized = register_bytes_artifact(
        data_root,
        run,
        WATER_XYZ,
        artifact_type="molecular_geometry",
        role="optimized_geometry",
        source="test:successful_opt",
        extension=".xyz",
        step_id=opt.id,
        attempt=1,
    )
    run.plan = run.plan.model_copy(
        update={
            "steps": [
                opt.model_copy(
                    update={"inputs": {"geometry": InputReference(artifact_id=seed.id)}}
                ),
                freq,
                sp,
            ]
        }
    )
    opt_result = Result(
        run_id=run.id,
        step_id=opt.id,
        attempt=1,
        status="succeeded",
        artifact_ids=[optimized.id],
        output_ports={"optimized_geometry": optimized.id},
        input_artifact_ids=[seed.id],
        input_bindings={"geometry": seed.id},
        attempt_relative_path="opt/attempt-01",
        step_fingerprint=_step_fingerprint(run.plan.steps[0]),
    )
    _save_result(data_root, run, opt_result)
    check = ScientificCheckResult(
        status=check_status,  # type: ignore[arg-type]
        input_geometry_sha256=optimized.sha256,
        reason=f"test scientific check status is {check_status}",
    )
    freq_result = Result(
        run_id=run.id,
        step_id=freq.id,
        attempt=1,
        status="succeeded",
        scientific_checks={"local_minimum_supported": check},
        artifact_ids=[],
        input_artifact_ids=[optimized.id],
        input_bindings={"geometry": optimized.id},
        attempt_relative_path="freq/attempt-01",
        step_fingerprint=_step_fingerprint(freq),
    )
    _save_result(data_root, run, freq_result)
    save_run(data_root, run)
    agent = Agent(config, registry, llm=None, session_id=f"session_gate_{check_status}")

    agent.advance(run)

    assert run.status == "failed"
    assert run.pending_data["category"] == "goal_not_met"
    assert run.pending_data["blocked_goal_checks"][0]["actual_status"] == check_status
    assert calls == []


def test_passed_check_unblocks_sp_and_upstream_invalidation_cascades(tmp_path: Path) -> None:
    config = _config(tmp_path)
    # The graph itself proves readiness once its source check has passed.
    source_run, optimized = _source_opt_run(config, "session_ready")
    freq = Step(
        id="freq",
        tool="frequency",
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
    )
    sp = Step(
        id="sp",
        tool="single_point",
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(step_id="opt", port="optimized_geometry")},
        goal_checks=[GoalCheckRequirement(source_step_id=freq.id, check="local_minimum_supported")],
    )
    # Add valid frequency evidence bound to the exact output geometry.
    source_run.plan = Plan(
        id="plan_ready",
        request_id=source_run.request.id,
        steps=[source_run.plan.steps[0], freq, sp],
        requested_results=[],
    )
    source_run.request = source_run.request.model_copy(update={"operations": ["Opt", "Freq", "SP"]})
    source_run.plan = source_run.plan.model_copy(update={"request_id": source_run.request.id})
    freq_result = Result(
        run_id=source_run.id,
        step_id=freq.id,
        attempt=1,
        status="succeeded",
        scientific_checks={
            "local_minimum_supported": ScientificCheckResult(
                status="passed", input_geometry_sha256=optimized.sha256
            )
        },
        input_artifact_ids=[optimized.id],
        input_bindings={"geometry": optimized.id},
        attempt_relative_path="freq/attempt-01",
        step_fingerprint=_step_fingerprint(freq),
    )
    _save_result(Path(config.data_root_path), source_run, freq_result)
    assert _next_ready_step(source_run, config.data_root_path) is sp

    mismatched_sp = sp.model_copy(
        update={"inputs": {"geometry": InputReference(artifact_id="another_geometry")}}
    )
    source_run.plan = source_run.plan.model_copy(
        update={"steps": [source_run.plan.steps[0], freq, mismatched_sp]}
    )
    assert _next_ready_step(source_run, config.data_root_path) is None
    blocked = _unmet_goal_checks(config.data_root_path, source_run, build_registry(config))
    assert blocked[0]["actual_status"] == "unverified"
    assert "not bound to the dependent Step geometry" in blocked[0]["reason"]

    source_run.plan = source_run.plan.model_copy(
        update={"steps": [source_run.plan.steps[0], freq, sp]}
    )

    _invalidate_current_results(source_run, "opt")

    assert source_run.current_results == {}
    assert source_run.step_status["freq"] == "planned"
    assert source_run.step_status["sp"] == "planned"


def test_repair_context_includes_prior_attempts_remaining_limits_and_original_constraints() -> None:
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={"charge": 0, "multiplicity": 1, "geom_maxiter": 1},
        origin_step_id="opt",
    )
    request = Request(
        id="request_repair_context",
        description="optimize water without changing its electronic state",
        operations=["Opt"],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
        explicit_parameters={"charge": 0, "multiplicity": 1},
    )
    plan = Plan(id="plan_repair_context", request_id=request.id, steps=[step])
    run = Run(
        id="run_repair_context",
        request=request,
        plan=plan,
        resources={"run_active_timeout_seconds": 600},
        accepted_snapshot={
            "request": request.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
        },
        budget={"max_attempts_per_science_step": 3, "max_extra_orca_executions": 2},
        attempt_counts={"opt": 1},
        attempts=[{"step_id": "opt", "attempt": 1, "phase": "finished", "status": "failed"}],
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="failed",
        diagnostics={"category": "opt_not_converged"},
    )

    context = json.loads(repair_context(run, step, result, [], remaining_timeout_seconds=321.5))

    assert context["prior_attempts"][0]["status"] == "failed"
    assert context["attempt_budget"] == {
        "origin_step_id": "opt",
        "used": 1,
        "maximum": 3,
        "remaining": 2,
    }
    assert context["run_budget"]["remaining_active_seconds"] == 321.5
    assert context["original_scientific_constraints"]["request"]["operations"] == ["Opt"]
    assert context["original_scientific_constraints"]["step"]["parameters"]["multiplicity"] == 1


def test_science_attempt_and_extra_execution_budgets_are_cumulative(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    agent = Agent(config, registry, llm=None, session_id="session_budgets")
    request = Request(id="request_budgets", description="bounded calculations")
    retry_step = Step(id="opt_retry", tool="optimize_geometry")
    new_step = Step(id="freq", tool="frequency")
    run = Run(
        id="run_budgets",
        request=request,
        plan=Plan(id="plan_budgets", request_id=request.id, steps=[retry_step, new_step]),
        resources=config.resources,
        status="planned",
        accepted_snapshot={"plan": {"steps": []}},
        budget={"max_attempts_per_science_step": 2, "max_extra_orca_executions": 1},
        attempt_counts={"opt": 1},
        origin_step_map={"opt_retry": "opt"},
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    assert agent._reserve_attempt(run, retry_step, registry.get(retry_step.tool))
    assert run.attempt_counts["opt"] == 2
    assert run.extra_orca_executions == 1
    assert not agent._reserve_attempt(run, retry_step, registry.get(retry_step.tool))
    assert run.pending_data["budget_exhausted"] == "max_attempts_per_science_step"
    assert not agent._reserve_attempt(run, new_step, registry.get(new_step.tool))
    assert run.pending_data["budget_exhausted"] == "max_extra_orca_executions"


def test_expired_active_time_budget_stops_before_tool_execution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base_registry = build_registry(config)
    calls: list[str] = []

    def forbidden_resolve(*_args: Any, **_kwargs: Any) -> Result:
        calls.append("resolve_molecule")
        raise AssertionError("an expired Run must not start another Tool")

    replacement = base_registry.get("resolve_molecule").model_copy(
        update={"execute_function": forbidden_resolve}
    )
    registry = ToolRegistry(
        [
            replacement if tool.name == "resolve_molecule" else tool
            for tool in (base_registry.get(name) for name in base_registry.names())
        ]
    )
    step = Step(
        id="molecule", tool="resolve_molecule", parameters={"query": "water", "input_kind": "name"}
    )
    target = ResultTarget(step_id=step.id, port="molecule")
    request = Request(
        id="request_timeout",
        description="resolve water",
        requested_results=[target],
    )
    run = Run(
        id="run_timeout",
        request=request,
        plan=Plan(
            id="plan_timeout", request_id=request.id, steps=[step], requested_results=[target]
        ),
        resources={**config.resources, "run_active_timeout_seconds": 1},
        status="planned",
        active_seconds=2,
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    Agent(config, registry, llm=None).advance(run)

    assert run.status == "failed"
    assert run.pending_data["category"] == "timeout"
    assert calls == []


def test_recovered_opt_geometry_flows_to_frequency_and_sp_without_repeating_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    base_registry = build_registry(config)
    calls: list[str] = []
    artifacts: dict[str, Any] = {}
    initial_geometry = (
        b"3\ninitial\nO 0.000000 0.000000 0.000000\n"
        b"H 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284\n"
    )
    restart_geometry = (
        b"3\nrestart candidate\nO 0.010000 0.000000 0.000000\n"
        b"H 0.768602 0.000000 0.504284\nH -0.748602 0.000000 0.504284\n"
    )
    retry_geometry = (
        b"3\nrecovered optimum\nO 0.020000 0.000000 0.000000\n"
        b"H 0.778602 0.000000 0.504284\nH -0.738602 0.000000 0.504284\n"
    )

    def result(
        run: Run,
        step: Step,
        attempt: int,
        *,
        status: str = "succeeded",
        values: dict[str, Any] | None = None,
        artifacts_out: list[Any] | None = None,
        output_ports: dict[str, str] | None = None,
        diagnostics: dict[str, Any] | None = None,
        scientific_checks: dict[str, ScientificCheckResult] | None = None,
    ) -> Result:
        output_artifacts = artifacts_out or []
        run.attempts.append(
            {
                "step_id": step.id,
                "attempt": attempt,
                "phase": "finished",
                "status": status,
                "artifact_ids": [item.id for item in output_artifacts],
            }
        )
        return Result(
            run_id=run.id,
            step_id=step.id,
            attempt=attempt,
            status=status,  # type: ignore[arg-type]
            values=values or {},
            artifact_ids=[item.id for item in output_artifacts],
            output_ports=output_ports or {},
            diagnostics=diagnostics or {},
            scientific_checks=scientific_checks or {},
            attempt_relative_path=f"{step.id}/attempt-{attempt:02d}",
        )

    def resolve_molecule(step: Step, run: Run, _cancel: Any) -> Result:
        calls.append(step.tool)
        molecule = register_bytes_artifact(
            config.data_root_path,
            run,
            b'{"formula":"H2O","charge":0}',
            artifact_type="molecule",
            role="resolved_molecule",
            source="test:offline_molecule",
            extension=".json",
            step_id=step.id,
            attempt=1,
        )
        artifacts["molecule"] = molecule
        return result(
            run,
            step,
            1,
            values={"molecule_formula": "H2O", "formal_charge": 0},
            artifacts_out=[molecule],
            output_ports={"molecule": molecule.id},
        )

    def generate_geometry(step: Step, run: Run, _cancel: Any) -> Result:
        calls.append(step.tool)
        geometry = register_bytes_artifact(
            config.data_root_path,
            run,
            initial_geometry,
            artifact_type="molecular_geometry",
            role="initial_geometry",
            source="test:offline_initial_geometry",
            extension=".xyz",
            step_id=step.id,
            attempt=1,
            metadata={"initial_guess_only": True},
        )
        artifacts["initial"] = geometry
        return result(
            run,
            step,
            1,
            values={"geometry_atom_count": 3},
            artifacts_out=[geometry],
            output_ports={"geometry": geometry.id},
        )

    def optimize_geometry(step: Step, run: Run, _cancel: Any) -> Result:
        attempt = run.attempt_counts[step.id]
        calls.append(f"{step.tool}:{attempt}")
        if attempt == 1:
            candidate = register_bytes_artifact(
                config.data_root_path,
                run,
                restart_geometry,
                artifact_type="molecular_geometry",
                role="restart_candidate",
                source="test:offline_failed_opt",
                extension=".xyz",
                step_id=step.id,
                attempt=attempt,
                metadata={"eligible_for": "optimization_restart_only"},
            )
            artifacts["candidate"] = candidate
            return result(
                run,
                step,
                attempt,
                status="failed",
                artifacts_out=[candidate],
                diagnostics={
                    "category": "opt_not_converged",
                    "facts": {
                        "opt_iteration_limit_reached": True,
                        "effective_geom_maxiter": 1,
                        "effective_geom_maxiter_error": None,
                    },
                    "process": {
                        "status": "failed",
                        "process_tree_empty": True,
                        "stop_confirmed": True,
                    },
                    "input_hashes": {"match": True},
                },
            )
        assert attempt == 2
        assert step.inputs["geometry"].artifact_id == artifacts["candidate"].id
        optimized = register_bytes_artifact(
            config.data_root_path,
            run,
            retry_geometry,
            artifact_type="molecular_geometry",
            role="optimized_geometry",
            source="test:offline_recovered_opt",
            extension=".xyz",
            step_id=step.id,
            attempt=attempt,
        )
        artifacts["retry"] = optimized
        return result(
            run,
            step,
            attempt,
            values={"opt_final_electronic_energy": {"value": -76.0, "unit": "Eh"}},
            artifacts_out=[optimized],
            output_ports={"optimized_geometry": optimized.id},
        )

    def frequency(step: Step, run: Run, _cancel: Any) -> Result:
        calls.append(f"{step.tool}:{run.attempt_counts[step.id]}")
        assert step.inputs["geometry"].step_id == "opt"
        optimized = artifacts["retry"]
        return result(
            run,
            step,
            run.attempt_counts[step.id],
            values={"vibrational_frequencies": {"complete": True, "unit": "cm^-1"}},
            scientific_checks={
                "local_minimum_supported": ScientificCheckResult(
                    status="passed",
                    input_geometry_sha256=optimized.sha256,
                    reason="simulated offline check for the recovered geometry",
                )
            },
        )

    def single_point(step: Step, run: Run, _cancel: Any) -> Result:
        calls.append(f"{step.tool}:{run.attempt_counts[step.id]}")
        assert step.inputs["geometry"].step_id == "opt"
        return result(
            run,
            step,
            run.attempt_counts[step.id],
            values={"sp_electronic_energy": {"value": -76.0, "unit": "Eh"}},
        )

    executor_by_name = {
        "resolve_molecule": resolve_molecule,
        "generate_geometry": generate_geometry,
        "optimize_geometry": optimize_geometry,
        "frequency": frequency,
        "single_point": single_point,
    }
    registry = ToolRegistry(
        [
            base_registry.get(name).model_copy(
                update={
                    "execute_function": executor_by_name.get(
                        name, base_registry.get(name).execute_function
                    )
                }
            )
            for name in base_registry.names()
        ]
    )

    def deterministic_repair(
        _client: Any,
        *,
        run: Run,
        step: Step,
        result: Result,
        options: list[Any],
        **_kwargs: Any,
    ) -> RepairProposal:
        option = options[0]
        return RepairProposal(
            action=option.action,
            failed_step_key=step.id,
            candidate_alias="last_complete_geometry",
            parameter_patch=option.parameter_patch,
            evidence_refs=list(option.evidence_refs),
        )

    monkeypatch.setattr("bg6022.agent.propose_repair", deterministic_repair)
    opt = Step(
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
    )
    freq = Step(
        id="freq",
        tool="frequency",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
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
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
        goal_checks=[GoalCheckRequirement(source_step_id=freq.id, check="local_minimum_supported")],
    )
    request = Request(
        id="request_opt_retry_composition",
        description=(
            "Optimize water, verify its local minimum, then calculate an independent SP energy."
        ),
        operations=["Opt", "Freq", "SP"],
        requested_results=[
            ResultTarget(step_id=opt.id, port="optimized_geometry"),
            ResultTarget(step_id=freq.id, check="local_minimum_supported"),
            ResultTarget(step_id=sp.id, field="sp_electronic_energy"),
        ],
        explicit_parameters={"charge": 0, "multiplicity": 1},
        source="chat",
    )
    plan = Plan(
        id="plan_opt_retry_composition",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "water", "input_kind": "name"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            opt,
            freq,
            sp,
        ],
        requested_results=request.requested_results,
    )
    agent = Agent(config, registry, llm=object(), session_id="session_opt_retry_composition")
    run = agent._create_chat_run(request, registry.validate_plan(plan))
    run.execution_permission = True

    agent.advance(run)

    assert run.status == "succeeded"
    assert calls == [
        "resolve_molecule",
        "generate_geometry",
        "optimize_geometry:1",
        "optimize_geometry:2",
        "frequency:1",
        "single_point:1",
    ]
    assert len(run.repair_records) == 1
    assert run.repair_records[0]["candidate_artifact_id"] == artifacts["candidate"].id
    assert run.attempt_counts == {"opt": 2, "freq": 1, "sp": 1}
    assert run.current_results["molecule"].endswith("attempt-01/result.json")
    assert run.current_results["geometry"].endswith("attempt-01/result.json")
    assert (
        artifact_path(config.data_root_path, run, artifacts["initial"]).read_bytes()
        == initial_geometry
    )
    assert artifacts["retry"].role == "optimized_geometry"
    assert artifacts["retry"].step_id == "opt"
    assert artifacts["retry"].attempt == 2

    frequency_result = Result.model_validate(
        json.loads(
            (Path(config.data_root_path) / "runs" / run.id / run.current_results["freq"]).read_text(
                encoding="utf-8"
            )
        ),
        strict=True,
    )
    sp_result = Result.model_validate(
        json.loads(
            (Path(config.data_root_path) / "runs" / run.id / run.current_results["sp"]).read_text(
                encoding="utf-8"
            )
        ),
        strict=True,
    )
    assert frequency_result.input_bindings["geometry"] == artifacts["retry"].id
    assert sp_result.input_bindings["geometry"] == artifacts["retry"].id


def test_chat_freq_only_request_runs_on_supplied_xyz_without_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    base_registry = build_registry(config)
    calls: list[str] = []
    agent: Agent | None = None

    def unexpected_tool(step: Step, _run: Run, _cancel: Any) -> Result:
        calls.append(step.tool)
        raise AssertionError(f"Freq-only request unexpectedly ran {step.tool}")

    def execute_frequency(step: Step, run: Run, _cancel: Any) -> Result:
        calls.append(step.tool)
        assert agent is not None
        assert [item.tool for item in run.plan.steps] == ["frequency"]
        assert step.parameters["charge"] == 0
        assert step.parameters["multiplicity"] == 1
        geometry = agent._artifact_from_reference(run, step.inputs["geometry"])
        assert geometry is not None
        assert geometry.role == "input_geometry"
        assert geometry.source == "chat:inline_xyz"
        assert artifact_path(config.data_root_path, run, geometry).read_bytes() == WATER_XYZ

        attempt = run.attempt_counts[step.id]
        hessian = register_bytes_artifact(
            config.data_root_path,
            run,
            b"offline verified Hessian",
            artifact_type="orca_hessian",
            role="verified_hessian",
            source="test:offline_frequency",
            step_id=step.id,
            attempt=attempt,
            metadata={
                "validated_for_input_geometry_sha256": geometry.sha256,
                "dimension": 9,
            },
        )
        run.attempts.append(
            {
                "step_id": step.id,
                "attempt": attempt,
                "phase": "finished",
                "status": "succeeded",
                "artifact_ids": [hessian.id],
                "output_ports": {"hessian": hessian.id},
            }
        )
        modes = [
            {"index": index, "value": 0.0 if index < 6 else 100.0 + index, "unit": "cm^-1"}
            for index in range(9)
        ]
        return Result(
            run_id=run.id,
            step_id=step.id,
            attempt=attempt,
            status="succeeded",
            values={
                "vibrational_frequencies": {
                    "modes": modes,
                    "unit": "cm^-1",
                    "scaling_factor": 1.0,
                    "scaling_applied": True,
                    "complete": True,
                }
            },
            checks={
                name: True
                for name in (
                    "runner_succeeded",
                    "exit_code_zero",
                    "process_tree_empty",
                    "normal_termination",
                    "stdout_valid_utf8",
                    "stdout_within_size_limit",
                    "stderr_within_size_limit",
                    "scf_converged",
                    "input_hashes_match",
                    "frequency_section_complete",
                    "frequency_values_finite",
                    "frequency_mode_indices_match_hessian",
                    "hessian_present",
                    "hessian_valid",
                )
            },
            scientific_checks={
                "frequency_complete": ScientificCheckResult(
                    status="passed",
                    input_geometry_sha256=geometry.sha256,
                    conditions={"mode_count": 9, "hessian_valid": True},
                ),
                "local_minimum_supported": ScientificCheckResult(
                    status="unverified",
                    input_geometry_sha256=geometry.sha256,
                    reason="A standalone frequency run does not establish optimization provenance.",
                ),
            },
            artifact_ids=[hessian.id],
            output_ports={"hessian": hessian.id},
            attempt_relative_path=f"{step.id}/attempt-{attempt:02d}",
        )

    registry = ToolRegistry(
        [
            base_registry.get(name).model_copy(
                update={
                    "execute_function": (
                        execute_frequency if name == "frequency" else unexpected_tool
                    )
                }
            )
            for name in base_registry.names()
        ]
    )
    user_message = (
        "Calculate vibrational frequencies for the supplied XYZ; charge 0, multiplicity 1"
    )
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["Freq"],
        explicit_parameters={"charge": 0, "multiplicity": 1},
        structure_input={"xyz_text": WATER_XYZ.decode("ascii")},
        requested_results=["vibrational_frequencies", "frequency_complete"],
    )
    proposal = PlanProposal(
        steps=[
            PlanStepProposal(
                key="freq",
                tool="frequency",
                parameters={"method_profile": "r2scan3c", "environment": "gas"},
                inputs={"geometry": {"artifact_alias": "request_geometry"}},
            )
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", field="vibrational_frequencies"),
            PlanTargetProposal(step_key="freq", check="frequency_complete"),
        ],
    )

    def fake_intake(*_args: Any, **_kwargs: Any) -> IntakeOutput:
        return intake

    def fake_plan(_client: Any, request: Request, **_kwargs: Any) -> PlanProposal:
        assert request.operations == ["Freq"]
        assert request.structure_input["xyz_text"] == WATER_XYZ.decode("ascii")
        assert request.requested_results == [
            ResultTarget(field="vibrational_frequencies"),
            ResultTarget(check="frequency_complete"),
        ]
        return proposal

    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr("bg6022.agent.plan_message", fake_plan)
    agent = Agent(config, registry, llm=object(), session_id="session_freq_only_inline_xyz")

    waiting = agent.handle_message(user_message)

    assert waiting.run is not None
    assert waiting.run.status == "waiting"
    assert waiting.run.waiting_for == "confirmation"
    assert [item.tool for item in waiting.run.plan.steps] == ["frequency"]
    assert waiting.run.plan.steps[0].inputs["geometry"].artifact_id != "request_geometry"
    assert calls == []

    completed = agent.confirm(waiting.run)

    assert completed.run is not None
    assert completed.run.status == "succeeded"
    assert [item.tool for item in completed.run.plan.steps] == ["frequency"]
    assert calls == ["frequency"]
    frequency_step = completed.run.plan.steps[0]
    result_path = (
        Path(config.data_root_path)
        / "runs"
        / completed.run.id
        / completed.run.current_results[frequency_step.id]
    )
    frequency_result = Result.model_validate(
        json.loads(result_path.read_text(encoding="utf-8")), strict=True
    )
    geometry_id = frequency_result.input_bindings["geometry"]
    geometry = next(item for item in completed.run.artifact_index if item.id == geometry_id)
    assert geometry.role == "input_geometry"
    assert geometry.source == "chat:inline_xyz"
    assert artifact_path(config.data_root_path, completed.run, geometry).read_bytes() == WATER_XYZ
    assert frequency_result.scientific_checks["frequency_complete"].status == "passed"
    assert (
        frequency_result.scientific_checks["frequency_complete"].input_geometry_sha256
        == geometry.sha256
    )
    assert frequency_result.scientific_checks["local_minimum_supported"].status == "unverified"
