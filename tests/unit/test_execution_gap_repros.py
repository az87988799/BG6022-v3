from __future__ import annotations

from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from typing import Any

import pytest

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.session import create_run, load_run, save_run, save_session, utc_now
from bg6022.tools.registry import ToolRegistry


class ExpectedToolBoundaryGap(AssertionError):
    """Only the known Tool exception escape is allowed to xfail."""


class ExpectedPersistenceGap(AssertionError):
    """Only the known persistence escape is allowed to xfail."""


class ExpectedConfirmationRaceGap(AssertionError):
    """Only duplicate execution of one confirmation may xfail."""


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


def _tool(
    name: str,
    result_name: str,
    execute_function: Any,
    *,
    requires_compute_permission: bool,
) -> Tool:
    return Tool(
        name=name,
        description=f"R0 isolated test tool {name}",
        operations=["SP"],
        results={result_name: "text"},
        requires_compute_permission=requires_compute_permission,
        execute_function=execute_function,
    )


def _successful_result(step: Step, run: Run, result_name: str) -> Result:
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        values={result_name: "ok"},
        attempt_relative_path=f"{step.id}/attempt-01",
    )


def _run(
    config: Any,
    steps: list[Step],
    *,
    session_id: str,
    waiting_for: str | None = None,
    execution_permission: bool = False,
) -> Run:
    request = Request(
        id=f"request_{session_id}",
        description="R0 execution-boundary probe",
        operations=["SP"],
        requested_results=[
            ResultTarget(step_id=step.id, field=step.parameters["result_name"]) for step in steps
        ],
        source="cli",
    )
    plan_steps = [step.model_copy(update={"parameters": {}}) for step in steps]
    plan = Plan(
        id=f"plan_{session_id}",
        request_id=request.id,
        steps=plan_steps,
        requested_results=request.requested_results,
    )
    run = Run(
        id=f"run_{session_id}",
        request=request,
        plan=plan,
        resources=config.resources,
        execution_permission=execution_permission,
        status="waiting" if waiting_for else "planned",
        waiting_for=waiting_for,
        budget={
            "max_attempts_per_science_step": 3,
            "max_extra_orca_executions": 3,
            "max_plan_revisions": 2,
        },
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
            "pending_prompt": None,
        },
    )
    return run


@pytest.mark.parametrize(
    ("error_type", "error_message"),
    [
        pytest.param(RuntimeError, "synthetic tool runtime failure", id="runtime-error"),
        pytest.param(TypeError, "synthetic tool type failure", id="type-error"),
    ],
)
@pytest.mark.xfail(
    strict=True,
    raises=ExpectedToolBoundaryGap,
    reason="R1-GAP-TOOL-UNEXPECTED-EXCEPTION: ordinary Tool exceptions escape Agent.advance",
)
def test_unexpected_tool_exception_is_closed_at_tool_boundary(
    tmp_path: Path,
    error_type: type[Exception],
    error_message: str,
) -> None:
    config = _config(tmp_path)
    injection = {
        "hits": 0,
        "error_type": None,
        "error_message": None,
    }
    tool_calls: list[str] = []

    def raise_tool_error(step: Step, _run: Run, cancel: Event) -> Result:
        tool_calls.append(step.id)
        injection["hits"] += 1
        try:
            raise error_type(error_message)
        except Exception as error:
            injection["error_type"] = type(error).__name__
            injection["error_message"] = str(error)
            raise

    tool = _tool(
        "runtime_error_tool",
        "runtime_value",
        raise_tool_error,
        requires_compute_permission=False,
    )
    registry = ToolRegistry([tool])
    run = _run(
        config,
        [Step(id="boom", tool=tool.name, parameters={"result_name": "runtime_value"})],
        session_id="runtime_error",
        execution_permission=True,
    )
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    escaped: Exception | None = None
    try:
        agent.advance(run)
    except error_type as error:
        escaped = error

    assert injection["hits"] == 1
    assert tool_calls == ["boom"]
    assert injection["error_type"] == error_type.__name__
    assert injection["error_message"] == error_message
    if escaped is not None:
        assert type(escaped) is error_type
        assert str(escaped) == error_message
        raise ExpectedToolBoundaryGap(
            f"{error_type.__name__} escaped Agent.advance after the Tool injection"
        )

    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.step_status == {"boom": "failed"}
    assert persisted.pending_data["category"] == "execution_boundary"


@pytest.mark.xfail(
    strict=True,
    raises=ExpectedPersistenceGap,
    reason="R1-GAP-RESULT-PERSISTENCE: save_result failure is not closed as a Run outcome",
)
def test_result_persistence_failure_is_acknowledged_and_stops_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    result_names = {"first": "first_value", "second": "second_value"}

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        return _successful_result(step, run, result_names[step.id])

    first = _tool(
        "first_persist_tool",
        "first_value",
        execute,
        requires_compute_permission=False,
    )
    second = _tool(
        "second_persist_tool",
        "second_value",
        execute,
        requires_compute_permission=False,
    )
    registry = ToolRegistry([first, second])
    run = _run(
        config,
        [
            Step(id="first", tool=first.name, parameters={"result_name": "first_value"}),
            Step(id="second", tool=second.name, parameters={"result_name": "second_value"}),
        ],
        session_id="result_persistence",
        execution_permission=True,
    )

    injection = {"hits": 0, "error_type": None, "error_message": None}

    def fail_result_save(*_args: Any, **_kwargs: Any) -> None:
        injection["hits"] += 1
        try:
            raise OSError("synthetic result persistence failure")
        except OSError as error:
            injection["error_type"] = type(error).__name__
            injection["error_message"] = str(error)
            raise

    monkeypatch.setattr("bg6022.agent.save_result", fail_result_save)
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    escaped: OSError | None = None
    try:
        agent.advance(run)
    except OSError as error:
        escaped = error

    assert injection["hits"] == 1
    assert injection["error_type"] == "OSError"
    assert injection["error_message"] == "synthetic result persistence failure"
    assert calls == ["first"]
    assert run.status != "succeeded"
    if escaped is not None:
        assert type(escaped) is OSError
        assert str(escaped) == "synthetic result persistence failure"
        raise ExpectedPersistenceGap("save_result OSError escaped Agent.advance")

    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.pending_data["category"] == "result_persistence"
    assert persisted.current_results == {}


@pytest.mark.xfail(
    strict=True,
    raises=ExpectedPersistenceGap,
    reason="R1-GAP-RUN-PERSISTENCE: save_run failure is not closed as a Run outcome",
)
def test_run_persistence_failure_stops_successor_without_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    result_names = {"first": "first_value", "second": "second_value"}

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        return _successful_result(step, run, result_names[step.id])

    first = _tool(
        "first_run_persist_tool",
        "first_value",
        execute,
        requires_compute_permission=False,
    )
    second = _tool(
        "second_run_persist_tool",
        "second_value",
        execute,
        requires_compute_permission=False,
    )
    registry = ToolRegistry([first, second])
    run = _run(
        config,
        [
            Step(id="first", tool=first.name, parameters={"result_name": "first_value"}),
            Step(id="second", tool=second.name, parameters={"result_name": "second_value"}),
        ],
        session_id="run_persistence",
        execution_permission=True,
    )

    original_save_run = save_run
    injection = {
        "hits": 0,
        "save_calls": 0,
        "error_type": None,
        "error_message": None,
    }

    def fail_after_first_result(data_root: str, current: Run) -> None:
        injection["save_calls"] += 1
        first_result_persisted = (
            current.step_status.get("first") == "succeeded"
            and "first" in current.current_results
            and current.status == "running"
            and not current.pending_data
        )
        if first_result_persisted and injection["hits"] == 0:
            injection["hits"] += 1
            try:
                raise OSError("synthetic Run persistence failure")
            except OSError as error:
                injection["error_type"] = type(error).__name__
                injection["error_message"] = str(error)
                raise
        original_save_run(data_root, current)

    monkeypatch.setattr("bg6022.agent.save_run", fail_after_first_result)
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    escaped: OSError | None = None
    try:
        agent.advance(run)
    except OSError as error:
        escaped = error

    assert injection["hits"] == 1
    assert injection["save_calls"] >= 1
    assert injection["error_type"] == "OSError"
    assert injection["error_message"] == "synthetic Run persistence failure"
    assert calls == ["first"]
    assert run.status != "succeeded"
    if escaped is not None:
        assert type(escaped) is OSError
        assert str(escaped) == "synthetic Run persistence failure"
        raise ExpectedPersistenceGap("save_run OSError escaped Agent.advance")

    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.pending_data["category"] == "run_persistence"
    assert persisted.current_results == {}


def test_cancel_between_confirmation_and_tool_start_prevents_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        return _successful_result(step, run, "value")

    tool = _tool(
        "cancel_gate_tool",
        "value",
        execute,
        requires_compute_permission=True,
    )
    registry = ToolRegistry([tool])
    run = _run(
        config,
        [Step(id="gated", tool=tool.name, parameters={"result_name": "value"})],
        session_id="cancel_confirmation",
        waiting_for="confirmation",
        execution_permission=False,
    )
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    original_save_run = save_run
    save_entered = Event()
    release_save = Event()
    save_calls = 0

    def gate_first_save(data_root: str, current: Run) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            save_entered.set()
            if not release_save.wait(timeout=2):
                raise RuntimeError("timed out waiting for cancellation probe")
        original_save_run(data_root, current)

    monkeypatch.setattr("bg6022.agent.save_run", gate_first_save)
    errors: list[Exception] = []

    def confirm() -> None:
        try:
            agent.confirm(run.id)
        except Exception as error:
            errors.append(error)

    worker = Thread(target=confirm, daemon=True)
    started = False
    try:
        worker.start()
        started = True
        assert save_entered.wait(timeout=2), "confirmation did not reach the gated save"
        agent.cancel(run.id)
    finally:
        release_save.set()
        if started:
            worker.join(timeout=2)

    assert not worker.is_alive(), "confirmation worker exceeded the bounded probe timeout"
    assert errors == []
    assert calls == []
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "cancelled"


@pytest.mark.xfail(
    strict=True,
    raises=ExpectedConfirmationRaceGap,
    reason="R1-GAP-CONFIRM-RACE: concurrent confirmations can execute one Run twice",
)
def test_duplicate_confirmation_cannot_consume_one_waiting_authorization_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    calls_lock = Lock()

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        with calls_lock:
            calls.append(step.id)
        return _successful_result(step, run, "value")

    tool = _tool(
        "duplicate_confirmation_tool",
        "value",
        execute,
        requires_compute_permission=True,
    )
    registry = ToolRegistry([tool])
    run = _run(
        config,
        [Step(id="confirmed", tool=tool.name, parameters={"result_name": "value"})],
        session_id="duplicate_confirmation",
        waiting_for="confirmation",
        execution_permission=False,
    )
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)
    confirmation_barrier = Barrier(2)
    original_snapshot = agent._acceptance_snapshot

    def gated_snapshot(current: Run) -> dict[str, Any]:
        confirmation_barrier.wait(timeout=2)
        return original_snapshot(current)

    monkeypatch.setattr(agent, "_acceptance_snapshot", gated_snapshot)
    errors: list[Exception] = []

    def confirm() -> None:
        try:
            agent.confirm(run.id)
        except Exception as error:
            errors.append(error)

    workers = [Thread(target=confirm, daemon=True) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)
    finally:
        confirmation_barrier.abort()
        for worker in workers:
            worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers), (
        "duplicate confirmation probe exceeded its bounded timeout"
    )
    assert errors == []
    if calls == ["confirmed", "confirmed"]:
        raise ExpectedConfirmationRaceGap("one waiting authorization started the same Tool twice")
    assert calls == ["confirmed"]
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "succeeded"
