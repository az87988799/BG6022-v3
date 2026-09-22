from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import pytest

from bg6022 import execution
from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.session import create_run, load_run, save_result, save_run, save_session, utc_now
from bg6022.tools.registry import ToolRegistry


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

    agent.advance(run)

    assert injection["hits"] == 1
    assert tool_calls == ["boom"]
    assert injection["error_type"] == error_type.__name__
    assert injection["error_message"] == error_message
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.step_status == {"boom": "failed"}
    assert persisted.pending_data["category"] == "execution_boundary"


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

    monkeypatch.setattr(execution, "save_result", fail_result_save)
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    agent.advance(run)

    assert injection["hits"] == 1
    assert injection["error_type"] == "OSError"
    assert injection["error_message"] == "synthetic result persistence failure"
    assert calls == ["first"]
    assert run.status != "succeeded"
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.pending_data["category"] == "result_persistence"
    assert persisted.current_results == {}


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

    original_save_run = execution.save_run
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

    monkeypatch.setattr(execution, "save_run", fail_after_first_result)
    agent = Agent(config, registry, llm=object(), session_id=run.session_id)

    agent.advance(run)

    assert injection["hits"] == 1
    assert injection["save_calls"] >= 1
    assert injection["error_type"] == "OSError"
    assert injection["error_message"] == "synthetic Run persistence failure"
    assert calls == ["first"]
    assert run.status != "succeeded"
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

    original_save_run = execution.save_run
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

    monkeypatch.setattr(execution, "save_run", gate_first_save)
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


def test_duplicate_confirmation_cannot_consume_one_waiting_authorization_twice(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    calls_lock = Lock()
    tool_started = Event()
    release_tool = Event()

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        tool_started.set()
        assert release_tool.wait(timeout=2), "held Tool exceeded its bounded probe timeout"
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
        assert tool_started.wait(timeout=2), "first confirmation did not reach the Tool"
        release_tool.set()
    finally:
        release_tool.set()
        for worker in workers:
            worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers), (
        "duplicate confirmation probe exceeded its bounded timeout"
    )
    assert errors == []
    assert calls == ["confirmed"]
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "succeeded"


def test_run_checkpoint_failure_preserves_an_existing_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    names = {
        "existing": "existing_value",
        "candidate": "candidate_value",
        "successor": "successor_value",
    }

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        return _successful_result(step, run, names[step.id])

    tools = [
        _tool(
            f"{step_id}_tool",
            result_name,
            execute,
            requires_compute_permission=False,
        )
        for step_id, result_name in names.items()
    ]
    registry = ToolRegistry(tools)
    run = _run(
        config,
        [
            Step(id=step_id, tool=f"{step_id}_tool", parameters={"result_name": result_name})
            for step_id, result_name in names.items()
        ],
        session_id="existing_success_checkpoint",
        execution_permission=True,
    )
    existing_step = run.plan.steps[0]
    existing_result = _successful_result(existing_step, run, names["existing"])
    existing_path = save_result(config.data_root_path, run, existing_result)
    existing_relative = (
        existing_path.relative_to(Path(config.data_root_path) / "runs" / run.id).parent.as_posix()
        + "/result.json"
    )
    run.result_index = [existing_relative]
    run.current_results = {"existing": existing_relative}
    run.step_status = {"existing": "succeeded"}
    save_run(config.data_root_path, run)

    original_save_run = execution.save_run
    injection = {"hits": 0}

    def fail_candidate_checkpoint(data_root: str, current: Run) -> None:
        candidate_published = (
            current.step_status.get("candidate") == "succeeded"
            and "candidate" in current.current_results
            and current.status == "running"
            and not current.pending_data
        )
        if candidate_published and injection["hits"] == 0:
            injection["hits"] += 1
            raise OSError("synthetic candidate Run persistence failure")
        original_save_run(data_root, current)

    monkeypatch.setattr(execution, "save_run", fail_candidate_checkpoint)
    Agent(config, registry, llm=object(), session_id=run.session_id).advance(run)

    assert injection["hits"] == 1
    assert calls == ["candidate"]
    persisted = load_run(config.data_root_path, run.id)
    assert persisted.status == "failed"
    assert persisted.pending_data["category"] == "run_persistence"
    assert persisted.current_results == {"existing": existing_relative}
    assert persisted.step_status["existing"] == "succeeded"
    assert "candidate" not in persisted.current_results


def test_cross_agent_confirmation_observes_the_same_run_owner(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    tool_started = Event()
    release_tool = Event()

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        tool_started.set()
        assert release_tool.wait(timeout=2), "cross-agent Tool exceeded its bounded probe timeout"
        return _successful_result(step, run, "value")

    tool = _tool(
        "cross_agent_confirmation_tool",
        "value",
        execute,
        requires_compute_permission=True,
    )
    registry = ToolRegistry([tool])
    run = _run(
        config,
        [Step(id="cross", tool=tool.name, parameters={"result_name": "value"})],
        session_id="cross_agent_confirmation",
        waiting_for="confirmation",
        execution_permission=False,
    )
    first_agent = Agent(config, registry, llm=object(), session_id=run.session_id)
    second_agent = Agent(config, registry, llm=object(), session_id=run.session_id)
    first_errors: list[Exception] = []

    def first_confirm() -> None:
        try:
            first_agent.confirm(run.id)
        except Exception as error:
            first_errors.append(error)

    worker = Thread(target=first_confirm, daemon=True)
    worker.start()
    assert tool_started.wait(timeout=2), "first Agent did not reach the Tool"
    competing = second_agent.confirm(run.id)
    release_tool.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert first_errors == []
    assert "占用" in competing.text
    assert calls == ["cross"]
    assert load_run(config.data_root_path, run.id).status == "succeeded"


def test_continuous_run_checkpoint_failure_is_bounded_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def execute(step: Step, run: Run, cancel: Event) -> Result:
        calls.append(step.id)
        return _successful_result(step, run, "value")

    tool = _tool(
        "continuous_checkpoint_tool",
        "value",
        execute,
        requires_compute_permission=False,
    )
    registry = ToolRegistry([tool])
    run = _run(
        config,
        [Step(id="bounded", tool=tool.name, parameters={"result_name": "value"})],
        session_id="continuous_checkpoint",
        execution_permission=True,
    )
    hits = {"count": 0}

    def always_fail(_data_root: str, _current: Run) -> None:
        hits["count"] += 1
        raise OSError("synthetic continuous disk failure")

    monkeypatch.setattr(execution, "save_run", always_fail)
    Agent(config, registry, llm=object(), session_id=run.session_id).advance(run)

    assert hits["count"] == 2
    assert calls == []
    assert run.status == "failed"
    assert run.pending_data["category"] == "run_persistence"
    assert run.pending_data["persistence"]["status"] == "not_persisted"
