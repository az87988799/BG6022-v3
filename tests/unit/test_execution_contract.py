from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from bg6022 import execution
from bg6022.agent import Agent, _step_fingerprint
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Run, Step
from bg6022.orca.runner import ProcessFacts
from bg6022.session import (
    clear_execution_guard,
    execution_fingerprint,
    load_run,
    register_file_artifact,
    run_directory,
    save_run,
    utc_now,
)
from bg6022.tools.orca import OptimizeParameters, SinglePointParameters, execute_orca_step
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path, *, run_timeout: int = 3600):
    executable = tmp_path / "missing-orca.exe"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[orca]
executable = '{executable.as_posix()}'

[runtime]
data_root = 'data'
run_active_timeout_seconds = {run_timeout}

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def _run_with_geometry(config, *, tool_name: str, permission: bool = True) -> tuple[Run, Step]:
    source = Path(config.config_path).parent / "water.xyz"
    source.write_bytes(b"3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n")
    request = Request(id="request_1", description="test")
    step = Step(
        id="compute",
        tool=tool_name,
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(artifact_id="initial")},
    )
    plan = Plan(id="plan_1", request_id=request.id, steps=[step])
    run = Run(
        id="run_1",
        request=request,
        plan=plan,
        resources=config.resources,
        execution_permission=permission,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    (run_directory(config.data_root_path, run.id) / "artifacts").mkdir(parents=True, exist_ok=True)
    artifact = register_file_artifact(
        config.data_root_path,
        run,
        source,
        artifact_type="molecular_geometry",
        role="input_geometry",
        source=str(source),
    )
    step = step.model_copy(update={"inputs": {"geometry": InputReference(artifact_id=artifact.id)}})
    plan = plan.model_copy(update={"steps": [step]})
    run.plan = plan
    run.accepted_execution_sha256 = execution_fingerprint(plan, run.resources, [artifact])
    save_run(config.data_root_path, run)
    return run, step


def _call_orca_adapter_under_test_admission(config, run, step, callback):
    run.status = "running"
    tool = build_registry(config).get(step.tool)
    owner = execution.RunOwner(config.data_root_path, run.id)
    with owner:
        if not tool.reserve_attempt(run, step):
            raise AssertionError("test ORCA attempt was not admitted by its budget")
        context = execution.begin_attempt(
            config.data_root_path,
            run,
            step,
            owner=owner,
            budget_reserved=True,
            step_fingerprint=_step_fingerprint(step),
        )
        try:
            result = callback()
            result.step_fingerprint = _step_fingerprint(step)
            result.input_bindings = {
                name: reference.artifact_id
                for name, reference in step.inputs.items()
                if reference.artifact_id is not None
            }
            result.input_artifact_ids = list(result.input_bindings.values())
            execution.finish_attempt(run, context, result)
            return result
        except Exception as error:
            execution.fail_attempt(
                run,
                context,
                category="test_adapter_failure",
                reason=str(error),
                persist=True,
            )
            raise


def test_tool_requires_explicit_permission_and_accepted_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run, step = _run_with_geometry(config, tool_name="single_point", permission=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("runner must not be called")

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fail_if_called)
    with pytest.raises(PermissionError):
        execute_orca_step(
            config,
            step=step,
            run=run,
            cancel=Event(),
            operation="SP",
            parameter_model=SinglePointParameters,
        )
    assert not (run_directory(config.data_root_path, run.id) / "compute").exists()

    run.execution_permission = True
    changed = step.model_copy(update={"parameters": {"charge": 1, "multiplicity": 1}})
    with pytest.raises(ValueError, match="fingerprint|accepted Plan"):
        execute_orca_step(
            config,
            step=changed,
            run=run,
            cancel=Event(),
            operation="SP",
            parameter_model=SinglePointParameters,
        )


def test_orca_adapter_without_gateway_cannot_bypass_attempt_budget_or_start_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run, step = _run_with_geometry(config, tool_name="single_point")
    run.attempt_counts[step.id] = 1
    run.budget["max_attempts_per_science_step"] = 1
    runner_calls: list[str] = []

    def fake_runner(**_kwargs):
        runner_calls.append("started")
        raise AssertionError("direct adapter call must not start ORCA")

    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda _config: None)
    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)

    with pytest.raises(execution.AttemptLifecycleError, match="Agent.advance"):
        execute_orca_step(
            config,
            step=step,
            run=run,
            cancel=Event(),
            operation="SP",
            parameter_model=SinglePointParameters,
        )

    assert runner_calls == []
    assert run.attempts == []
    assert not (run_directory(config.data_root_path, run.id) / step.id).exists()


def test_cancelled_stale_run_cannot_reenter_orca_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    stale, step = _run_with_geometry(config, tool_name="single_point")
    current = load_run(config.data_root_path, stale.id)
    current.status = "cancelled"
    save_run(config.data_root_path, current)
    runner_calls: list[str] = []

    def fake_runner(**_kwargs):
        runner_calls.append("started")
        raise AssertionError("cancelled stale Run must not start ORCA")

    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda _config: None)
    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)

    with pytest.raises(execution.AttemptLifecycleError, match="Agent.advance"):
        execute_orca_step(
            config,
            step=step,
            run=stale,
            cancel=Event(),
            operation="SP",
            parameter_model=SinglePointParameters,
        )

    assert load_run(config.data_root_path, stale.id).status == "cancelled"
    assert runner_calls == []
    assert stale.attempts == []


def test_agent_gateway_respects_exhausted_orca_budget_before_attempt_or_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run, step = _run_with_geometry(config, tool_name="single_point")
    run.attempt_counts[step.id] = 1
    run.budget["max_attempts_per_science_step"] = 1
    run.execution_permission = True
    registry = build_registry(config)
    agent = Agent(config, registry)
    run.accepted_snapshot = agent._acceptance_snapshot(run)
    run.accepted_execution_sha256 = execution_fingerprint(
        run.plan,
        run.resources,
        run.artifact_index,
        snapshot=run.accepted_snapshot,
    )
    save_run(config.data_root_path, run)
    runner_calls: list[str] = []

    def fake_runner(**_kwargs):
        runner_calls.append("started")
        raise AssertionError("exhausted budget must stop before ORCA")

    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda _config: None)
    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)

    agent.advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "failed", durable.pending_data
    assert durable.pending_data["budget_exhausted"] == "max_attempts_per_science_step"
    assert durable.attempts == []
    assert runner_calls == []
    assert not (run_directory(config.data_root_path, run.id) / step.id).exists()


def test_active_clock_accumulates_one_interval(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = Request(id="request_1", description="test")
    plan = Plan(id="plan_1", request_id=request.id, steps=[])
    run = Run(
        id="run_clock",
        request=request,
        plan=plan,
        resources=config.resources,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    run.start_active_interval(now=10.0)
    assert run.current_active_seconds(now=11.0) == pytest.approx(1.0)
    run.checkpoint_active(now=12.0)
    assert run.active_seconds == pytest.approx(2.0)
    run.finish_active_interval(now=13.0)
    assert run.active_seconds == pytest.approx(3.0)
    run.finish_active_interval(now=99.0)
    assert run.active_seconds == pytest.approx(3.0)


def test_config_loading_is_pure_for_missing_orca_and_data_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert not Path(config.data_root_path).exists()


def test_failed_opt_with_missing_xyz_can_publish_only_restart_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run, step = _run_with_geometry(config, tool_name="optimize_geometry")
    stdout = (
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

    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda config: None)

    def fake_runner(**kwargs):
        attempt_dir = Path(kwargs["attempt_dir"])
        facts = ProcessFacts(
            status="failed",
            exit_code=1,
            stop_reason="nonzero_exit",
            process_tree_empty=True,
            stop_confirmed=True,
        )
        kwargs["on_started"](facts)
        (attempt_dir / "stdout.out").write_bytes(stdout)
        (attempt_dir / "stderr.txt").write_bytes(b"")
        clear_execution_guard(kwargs["data_root"], kwargs["execution_id"])
        return facts

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)
    result = _call_orca_adapter_under_test_admission(
        config,
        run,
        step,
        lambda: execute_orca_step(
            config,
            step=step,
            run=run,
            cancel=Event(),
            operation="Opt",
            parameter_model=OptimizeParameters,
        ),
    )
    assert result.status == "failed"
    assert result.output_ports == {}
    candidates = [item for item in run.artifact_index if item.role == "restart_candidate"]
    assert len(candidates) == 1
    assert candidates[0].metadata["eligible_for"] == "optimization_restart_only"
    assert candidates[0].size_bytes > 0


def test_residual_process_failure_has_no_public_values_and_stops_following_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source = Path(config.config_path).parent / "water.xyz"
    source.write_bytes(b"1\nhydrogen\nH 0 0 0\n")
    request = Request(id="request_agent", description="test")
    parameters = {"charge": -1, "multiplicity": 1}
    plan = Plan(
        id="plan_agent",
        request_id=request.id,
        steps=[
            Step(
                id="first",
                tool="single_point",
                parameters=parameters,
                inputs={
                    "geometry": InputReference(artifact_id="__input_geometry__"),
                },
            ),
            Step(
                id="second",
                tool="single_point",
                parameters=parameters,
                inputs={
                    "geometry": InputReference(artifact_id="__input_geometry__"),
                },
            ),
        ],
    )
    monkeypatch.setattr("bg6022.agent.validate_execution_environment", lambda config: None)
    monkeypatch.setattr("bg6022.tools.orca.validate_execution_environment", lambda config: None)
    calls = 0
    stdout = (
        b"SCF CONVERGED AFTER 1 CYCLES\n"
        b"FINAL SINGLE POINT ENERGY -76.000000000000\n"
        b"****ORCA TERMINATED NORMALLY****\n"
        b"TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 0 msec\n"
    )

    def fake_runner(**kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("a failed step must not start its successor")
        attempt_dir = Path(kwargs["attempt_dir"])
        started = ProcessFacts(
            status="failed",
            exit_code=0,
            stop_reason="residual_processes_terminated",
            process_tree_empty=True,
            stop_confirmed=True,
            pid=1234,
        )
        kwargs["on_started"](started)
        (attempt_dir / "stdout.out").write_bytes(stdout)
        (attempt_dir / "stderr.txt").write_bytes(b"")
        clear_execution_guard(kwargs["data_root"], kwargs["execution_id"])
        return started

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fake_runner)
    run, result = Agent(config, build_registry(config)).execute_plan(
        request,
        plan,
        xyz_path=source,
        execute=True,
    )

    assert calls == 1
    assert result.status == "failed"
    assert result.values == {}
    assert result.output_ports == {}
    assert run.step_status == {"first": "failed"}
