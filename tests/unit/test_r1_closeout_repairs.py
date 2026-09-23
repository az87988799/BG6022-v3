from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from bg6022 import execution
from bg6022.agent import Agent, AgentResponse, _step_fingerprint
from bg6022.cli import _DisplayedConfirmation
from bg6022.config import load_config
from bg6022.models import Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.session import (
    RunRecordInvalidError,
    create_run,
    load_run,
    save_run,
    utc_now,
)
from bg6022.tools.registry import ToolRegistry


def _config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
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
    return load_config(config_path)


def _tool(
    name: str,
    execute_function: Any,
    *,
    requires_compute_permission: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description="R1 closeout probe",
        results={"value": "text"},
        requires_compute_permission=requires_compute_permission,
        execute_function=execute_function,
    )


def _run(config: Any, tool: Tool, *, status: str = "planned") -> Run:
    request = Request(
        id=f"request_{tool.name}",
        description="R1 closeout probe",
        requested_results=[ResultTarget(step_id="step", field="value")],
        source="cli",
    )
    step = Step(id="step", tool=tool.name)
    plan = Plan(
        id=f"plan_{tool.name}",
        request_id=request.id,
        steps=[step],
        requested_results=request.requested_results,
    )
    run = Run(
        id=f"run_{tool.name}",
        request=request,
        plan=plan,
        resources=config.resources,
        execution_permission=True,
        status=status,  # type: ignore[arg-type]
        waiting_for="confirmation" if status == "waiting" else None,
        budget={
            "max_attempts_per_science_step": 3,
            "max_extra_orca_executions": 3,
            "max_plan_revisions": 2,
        },
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    return run


def _success(step: Step, run: Run, attempt: int, *, value: str = "ok") -> Result:
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        status="succeeded",
        values={"value": value},
        step_fingerprint=_step_fingerprint(step),
        attempt_relative_path=f"{step.id}/attempt-{attempt:02d}",
    )


def _store_unindexed_result(
    config: Any, run: Run, step: Step, *, status: str, value: str = "ok"
) -> Result:
    context = execution.begin_attempt(config.data_root_path, run, step)
    result = _success(step, run, context.attempt, value=value).model_copy(
        update={
            "status": status,
            "values": {"value": value} if status == "succeeded" else {},
            "diagnostics": {"category": status, "reason": f"stored {status}"},
        }
    )
    result.step_fingerprint = _step_fingerprint(step)
    execution.finish_attempt(run, context, result, persist=True, release=False)
    execution.persist_result(config.data_root_path, run, result)
    execution.release_attempt(run, context)
    return result


def test_advance_reloads_authoritative_state_and_never_falls_back_on_corruption(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(run.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("authoritative", execute)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    stale = load_run(config.data_root_path, run.id)

    current = load_run(config.data_root_path, run.id)
    current.status = "cancelled"
    current.waiting_for = None
    save_run(config.data_root_path, current)

    agent = Agent(config, registry, llm=object())
    assert agent.advance(stale) is None
    assert stale.status == "cancelled"
    assert calls == []

    run_path = Path(config.data_root_path) / "runs" / run.id / "run.json"
    run_path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(RunRecordInvalidError):
        agent.advance(stale)
    assert calls == []


def test_direct_advance_cancel_signals_owner_without_mutating_its_run(tmp_path: Path) -> None:
    started = Event()
    calls: list[str] = []

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        started.set()
        cancel.wait(timeout=2)
        return Result(
            run_id=run.id,
            step_id=step.id,
            attempt=1,
            status="cancelled",
            diagnostics={"category": "cancelled"},
            attempt_relative_path=f"{step.id}/attempt-01",
        )

    config = _config(tmp_path)
    tool = _tool("cancel_owner", execute)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    agent = Agent(config, registry, llm=object())
    worker = Thread(target=lambda: agent.advance(run), daemon=True)
    worker.start()
    assert started.wait(timeout=2)

    response = agent.cancel(run.id)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert response.text.startswith("已请求取消")
    assert calls == ["step"]
    assert load_run(config.data_root_path, run.id).status == "cancelled"


def test_run_owner_is_scoped_by_normalized_root_and_run_id(tmp_path: Path) -> None:
    first = execution.RunOwner(tmp_path / "data", "run-a")
    second = execution.RunOwner((tmp_path / "data").resolve(), "run-b")
    with first:
        with second:
            pass


def test_public_tool_entry_rejects_execution_without_owner_admission(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("unadmitted", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]

    with pytest.raises(execution.AttemptLifecycleError, match="Agent.advance"):
        tool.execute(step, run, cancel=Event())

    assert calls == []
    assert run.attempts == []


def test_agent_budget_admission_stops_before_allocating_attempt(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("budget_denied", execute).model_copy(
        update={"attempt_reservation_function": lambda _run, _step: False}
    )
    registry = ToolRegistry([tool])
    run = _run(config, tool)

    Agent(config, registry, llm=object()).advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "failed"
    assert durable.attempts == []
    assert calls == []


def test_agent_busy_owner_does_not_allocate_or_execute(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("busy_owner", execute)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    agent = Agent(config, registry, llm=object())

    with execution.RunOwner(config.data_root_path, run.id):
        agent.advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "planned"
    assert durable.attempts == []
    assert calls == []


def test_cancel_busy_owner_does_not_claim_remote_signal_was_delivered(tmp_path: Path) -> None:
    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("cancel_busy_owner", execute)
    run = _run(config, tool)
    agent = Agent(config, ToolRegistry([tool]), llm=object())

    with execution.RunOwner(config.data_root_path, run.id):
        response = agent.cancel(run.id)

    assert "无法确认对方已收到取消请求" in response.text
    assert load_run(config.data_root_path, run.id).status == "planned"


def test_candidate_result_is_rejected_before_result_publish(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def execute(_step: Step, _run: Run, _cancel: Event) -> Result:
        raise AssertionError("not called")

    tool = _tool("candidate_binding", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    context = execution.begin_attempt(config.data_root_path, run, step)
    candidate = _success(step, run, context.attempt).model_copy(update={"run_id": "other-run"})

    with pytest.raises(execution.AttemptLifecycleError):
        execution.commit_attempt_result(
            config.data_root_path,
            run,
            step,
            candidate,
            expected_input_bindings={},
            tool=tool,
            context=context,
        )
    assert not (context.directory / "result.json").exists()
    execution.release_attempt(run, context)


def test_recovery_indexes_verified_result_without_restarting_tool(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 2)

    config = _config(tmp_path)
    tool = _tool("recovery", execute)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    step = run.plan.steps[0]
    context = execution.begin_attempt(config.data_root_path, run, step)
    execution.persist_result(
        config.data_root_path,
        run,
        _success(step, run, context.attempt, value="already-finished"),
    )
    execution.release_attempt(run, context)

    agent = Agent(config, registry, llm=object())
    result = agent.advance(run)
    assert result is not None and result.values["value"] == "already-finished"
    assert calls == []
    assert load_run(config.data_root_path, run.id).status == "succeeded"


def test_recovery_of_latest_failed_result_stops_without_restarting_tool(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("recovery_failed", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    _store_unindexed_result(config, run, step, status="failed")

    Agent(config, ToolRegistry([tool]), llm=object()).advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "failed"
    assert durable.step_status[step.id] == "failed"
    assert durable.pending_data["recovered_attempt"] == 1
    assert calls == []


@pytest.mark.parametrize("status", ["cancelled", "interrupted"])
def test_recovery_of_latest_stopped_result_does_not_restart_tool(
    tmp_path: Path, status: str
) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool(f"recovery_{status}", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    _store_unindexed_result(config, run, step, status=status)

    Agent(config, ToolRegistry([tool]), llm=object()).advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == status
    assert durable.step_status[step.id] == status
    assert calls == []


def test_recovered_needs_input_waits_until_user_changes_step_revision(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        context = execution.active_attempt(run, step.id)
        assert context is not None
        calls.append(context.attempt)
        return _success(step, run, context.attempt)

    config = _config(tmp_path)
    tool = _tool("recovery_needs_input", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    _store_unindexed_result(config, run, step, status="needs_input")
    agent = Agent(config, ToolRegistry([tool]), llm=object())

    agent.advance(run)
    waiting = load_run(config.data_root_path, run.id)
    assert waiting.status == "waiting"
    assert waiting.waiting_for == "clarification"
    assert calls == []

    agent.advance(waiting)
    assert calls == []
    assert load_run(config.data_root_path, run.id).attempts[-1]["attempt"] == 1

    revised = load_run(config.data_root_path, run.id)
    changed_step = revised.plan.steps[0].model_copy(
        update={"parameters": {"user_supplement": "verified"}}
    )
    revised.plan = revised.plan.model_copy(update={"revision": 2, "steps": [changed_step]})
    revised.status = "running"
    revised.waiting_for = None
    revised.pending_data = {}
    save_run(config.data_root_path, revised)

    result = agent.advance(revised)

    durable = load_run(config.data_root_path, run.id)
    assert result is not None and result.status == "succeeded"
    assert durable.status == "succeeded"
    assert calls == [2]


def test_newer_unknown_attempt_blocks_older_verified_success(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 2)

    config = _config(tmp_path)
    tool = _tool("recovery_new_unknown", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    _store_unindexed_result(config, run, step, status="succeeded", value="older")
    context = execution.begin_attempt(config.data_root_path, run, step)
    context.record["phase"] = "started"
    execution.persist_run(config.data_root_path, run)
    execution.release_attempt(run, context)

    Agent(config, ToolRegistry([tool]), llm=object()).advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "interrupted"
    assert durable.pending_data["category"] == "recovery_unknown"
    assert step.id not in durable.current_results
    assert calls == []


def test_newer_verified_success_supersedes_older_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 3)

    config = _config(tmp_path)
    tool = _tool("recovery_new_success", execute)
    run = _run(config, tool)
    step = run.plan.steps[0]
    _store_unindexed_result(config, run, step, status="failed")
    latest = _store_unindexed_result(config, run, step, status="succeeded", value="latest")

    recovered = Agent(config, ToolRegistry([tool]), llm=object()).advance(run)

    durable = load_run(config.data_root_path, run.id)
    assert durable.status == "succeeded"
    assert durable.step_status[step.id] == "succeeded"
    assert durable.current_results[step.id].endswith("attempt-02/result.json")
    assert recovered is not None and recovered.values["value"] == latest.values["value"]
    assert calls == []


def test_queued_confirmation_requires_the_captured_preview_version(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        calls.append(step.id)
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("queued_confirmation", execute, requires_compute_permission=True)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    run.execution_permission = False
    agent = Agent(config, registry, llm=object())
    agent._prepare_confirmation(run, run.plan.steps[0])
    save_run(config.data_root_path, run)
    displayed = _DisplayedConfirmation()
    displayed.observe(agent, AgentResponse("preview A", run=run))
    token = displayed.snapshot()
    assert token is not None

    changed = load_run(config.data_root_path, run.id)
    changed.plan = changed.plan.model_copy(
        update={"steps": [changed.plan.steps[0].model_copy(update={"parameters": {"revision": 2}})]}
    )
    save_run(config.data_root_path, changed)
    assert displayed.snapshot() == token

    response = agent.confirm(
        presented_run_id=token[0],
        presented_fingerprint=token[1],
        queued_confirmation=True,
    )
    assert response.run is not None
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert calls == []


def test_final_run_checkpoint_failure_preserves_committed_result_but_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(step: Step, run: Run, _cancel: Event) -> Result:
        return _success(step, run, 1)

    config = _config(tmp_path)
    tool = _tool("final_checkpoint", execute)
    registry = ToolRegistry([tool])
    run = _run(config, tool)
    original_persist = execution.persist_run
    persist_calls = 0

    def fail_final(root: str | Path, current: Run) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls >= 4:
            raise execution.PersistenceFailure("final_run_checkpoint", OSError("read-only"))
        original_persist(root, current)

    monkeypatch.setattr(execution, "persist_run", fail_final)
    result = Agent(config, registry, llm=object()).advance(run)

    assert result is not None and result.status == "succeeded"
    assert run.status == "failed"
    assert run.step_status["step"] == "succeeded"
    assert run.current_results["step"].endswith("result.json")
    assert load_run(config.data_root_path, run.id).current_results["step"].endswith("result.json")
