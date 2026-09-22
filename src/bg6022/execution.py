"""Small lifecycle primitives shared by the Agent and production Tools.

This module deliberately stays below the durable domain model.  It owns only
short-lived admission and persistence boundaries; ``Run`` remains the single
durable source of lifecycle state and Tools remain the execution boundary.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .models import Result, Run, Step, Tool
from .session import attempt_directory, run_directory


class ExecutionBusy(RuntimeError):
    """A Run is currently owned by another execution or confirmation."""

    def __init__(self, run_id: str, path: Path) -> None:
        self.run_id = run_id
        self.path = path
        super().__init__(f"Run {run_id} is currently owned by another execution")


class PersistenceFailure(RuntimeError):
    """A lifecycle checkpoint could not be persisted."""

    def __init__(self, stage: str, error: Exception) -> None:
        self.stage = stage
        self.error = error
        super().__init__(f"{stage}: {error}")


class AttemptLifecycleError(RuntimeError):
    """A Tool result cannot be associated with its admitted attempt."""


@dataclass
class AttemptContext:
    """Short-lived context for exactly one owner-admitted Tool attempt."""

    data_root: Path
    run_id: str
    step_id: str
    attempt: int
    relative_path: str
    directory: Path
    record: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitResultOutcome:
    """The durable paths and publication fact for one candidate Result."""

    result_relative_path: str
    published: bool


_active_attempts: dict[tuple[int, str], AttemptContext] = {}
_active_attempts_lock = Lock()


class RunOwner:
    """Non-blocking cross-process owner for one Run.

    The handle is intentionally short-lived and is never serialized.  The
    operating-system lock, rather than an in-memory flag, arbitrates two
    Agents in one process and two worker processes sharing a data root.
    """

    _process_locks: dict[tuple[str, str], Lock] = {}
    _process_locks_guard = Lock()

    def __init__(self, data_root: str | Path, run_id: str) -> None:
        self.data_root = Path(data_root).resolve()
        self.run_id = str(run_id)
        self.path = run_directory(self.data_root, self.run_id) / ".execution-owner.lock"
        self._handle: Any = None
        self.token = uuid.uuid4().hex
        key = (str(self.data_root), self.run_id)
        with self._process_locks_guard:
            self._local_lock = self._process_locks.setdefault(key, Lock())

    def __enter__(self) -> RunOwner:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The process-local guard avoids platform-specific reentrant locking
        # surprises when two threads open the same lock file simultaneously.
        if not self._local_lock.acquire(blocking=False):
            raise ExecutionBusy(self.run_id, self.path)
        try:
            self._handle = self.path.open("a+b")
            self._handle.seek(0)
            self._handle.write(b"0")
            self._handle.flush()
            self._handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                self._handle.close()
                self._handle = None
                raise ExecutionBusy(self.run_id, self.path) from error
            return self
        except Exception:
            self._local_lock.release()
            raise

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            if self._handle is None:
                return
            try:
                if os.name == "nt":
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
        finally:
            self._local_lock.release()


def save_run(data_root: str | Path, run: Run) -> None:
    """The patchable authoritative Run checkpoint entry point."""

    from .session import save_run as _save_run

    _save_run(data_root, run)


def save_result(data_root: str | Path, run: Run, result: Result) -> Path:
    """The patchable authoritative Result checkpoint entry point."""

    from .session import save_result as _save_result

    return _save_result(data_root, run, result)


def persist_run(data_root: str | Path, run: Run) -> None:
    try:
        save_run(data_root, run)
    except Exception as error:
        raise PersistenceFailure("run_persistence", error) from error


def persist_result(data_root: str | Path, run: Run, result: Result) -> Path:
    try:
        return save_result(data_root, run, result)
    except Exception as error:
        raise PersistenceFailure("result_persistence", error) from error


def begin_attempt(data_root: str | Path, run: Run, step: Step) -> AttemptContext:
    """Allocate, create, register, and persist one prepared attempt.

    This is the only production entry that allocates an attempt directory.  A
    Tool adapter receives the resulting context through ``active_attempt`` and
    may add scientific facts to its record, but it cannot allocate a second
    attempt or publish a Run pointer.
    """

    key = (id(run), step.id)
    with _active_attempts_lock:
        if key in _active_attempts:
            raise AttemptLifecycleError(f"attempt already active for Step {step.id!r}")
    root = Path(data_root).resolve()
    attempt = allocate_attempt(root, run, step.id)
    directory = attempt_directory(root, run.id, step.id, attempt)
    directory.mkdir(parents=True, exist_ok=False)
    relative = directory.relative_to(run_directory(root, run.id)).as_posix()
    record = {
        "step_id": step.id,
        "attempt": attempt,
        "relative_path": relative,
        "phase": "prepared",
        "status": None,
        "artifact_ids": [],
        "output_ports": {},
        "input_artifact_ids": [],
    }
    run.attempts.append(record)
    run.step_status[step.id] = "running"
    context = AttemptContext(
        data_root=root,
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        relative_path=relative,
        directory=directory,
        record=record,
    )
    with _active_attempts_lock:
        _active_attempts[key] = context
    try:
        persist_run(root, run)
    except Exception:
        with _active_attempts_lock:
            _active_attempts.pop(key, None)
        raise
    return context


def ensure_attempt(data_root: str | Path, run: Run, step: Step) -> tuple[AttemptContext, bool]:
    """Return the Agent-admitted context or enter the same gateway directly."""

    context = active_attempt(run, step.id)
    if context is not None:
        return context, False
    return begin_attempt(data_root, run, step), True


def active_attempt(run: Run, step_id: str) -> AttemptContext | None:
    with _active_attempts_lock:
        return _active_attempts.get((id(run), step_id))


def checkpoint_attempt(run: Run, context: AttemptContext) -> None:
    """Persist an adapter's observed prepared/started facts through one entry."""

    _assert_context(run, context)
    if active_attempt(run, context.step_id) is not context:
        raise AttemptLifecycleError("attempt context is not the active lifecycle owner")
    persist_run(context.data_root, run)


def finish_attempt(
    run: Run,
    context: AttemptContext,
    result: Result,
    *,
    persist: bool = True,
    release: bool = True,
) -> None:
    """Record one Tool's finished facts; Result publication remains separate."""

    _assert_context(run, context)
    context.record.update(
        {
            "phase": "finished",
            "status": result.status,
            "artifact_ids": list(result.artifact_ids),
            "output_ports": dict(result.output_ports),
            "input_artifact_ids": list(result.input_artifact_ids),
        }
    )
    try:
        if persist:
            checkpoint_attempt(run, context)
    finally:
        if release:
            release_attempt(run, context)


def fail_attempt(
    run: Run,
    context: AttemptContext,
    *,
    category: str,
    reason: str,
    persist: bool = False,
    release: bool = True,
) -> None:
    """Close an attempt as failed without inventing a successful Result."""

    _assert_context(run, context)
    context.record.update(
        {
            "phase": "finished",
            "status": "failed",
            "failure_category": category,
            "failure_reason": reason,
        }
    )
    try:
        if persist:
            checkpoint_attempt(run, context)
    finally:
        if release:
            release_attempt(run, context)


def release_attempt(run: Run, context: AttemptContext) -> None:
    with _active_attempts_lock:
        _active_attempts.pop((id(run), context.step_id), None)


def _assert_context(run: Run, context: AttemptContext) -> None:
    if context.run_id != run.id:
        raise AttemptLifecycleError("attempt context belongs to another Run")
    if context.record not in run.attempts:
        raise AttemptLifecycleError("attempt context record is not in the current Run")


def validate_result_candidate(
    run: Run,
    step: Step,
    result: Result,
    *,
    context: AttemptContext | None = None,
    expected_input_bindings: dict[str, str] | None = None,
    tool: Tool | None = None,
) -> None:
    """Validate Result identity, attempt path, inputs, outputs, and artifacts."""

    if result.run_id != run.id:
        raise AttemptLifecycleError("candidate Result belongs to another Run")
    if result.step_id != step.id:
        raise AttemptLifecycleError("candidate Result belongs to another Step")
    if result.attempt < 1:
        raise AttemptLifecycleError("candidate Result has an invalid attempt number")
    expected_relative = (
        context.relative_path if context is not None else f"{step.id}/attempt-{result.attempt:02d}"
    )
    if context is not None and result.attempt != context.attempt:
        raise AttemptLifecycleError("candidate Result attempt does not match the admitted attempt")
    if result.attempt_relative_path != expected_relative:
        raise AttemptLifecycleError("candidate Result path does not match the admitted attempt")
    relative_path = Path(result.attempt_relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AttemptLifecycleError("candidate Result path escapes the Run directory")
    if expected_input_bindings is not None:
        if result.input_bindings != expected_input_bindings:
            raise AttemptLifecycleError("candidate Result input bindings do not match the Step")
        if set(result.input_artifact_ids) != set(expected_input_bindings.values()):
            raise AttemptLifecycleError("candidate Result input artifacts do not match the Step")
    if tool is not None:
        if not set(result.values) <= set(tool.results):
            raise AttemptLifecycleError("candidate Result contains an undeclared field")
        if not set(result.output_ports) <= set(tool.output_ports):
            raise AttemptLifecycleError("candidate Result contains an undeclared output port")
        if not set(result.scientific_checks) <= set(tool.scientific_checks):
            raise AttemptLifecycleError("candidate Result contains an undeclared scientific check")
    artifact_by_id = {artifact.id: artifact for artifact in run.artifact_index}
    for artifact_id in result.input_artifact_ids:
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None or artifact.run_id != run.id:
            raise AttemptLifecycleError("candidate Result references an unknown input Artifact")
    for artifact_id in result.artifact_ids:
        artifact = artifact_by_id.get(artifact_id)
        if (
            artifact is None
            or artifact.run_id != run.id
            or artifact.step_id != step.id
            or artifact.attempt != result.attempt
        ):
            raise AttemptLifecycleError("candidate Result output Artifact has the wrong owner")
    for port, artifact_id in result.output_ports.items():
        if artifact_id not in result.artifact_ids:
            raise AttemptLifecycleError(f"output port {port!r} is not present in artifact_ids")


def commit_attempt_result(
    data_root: str | Path,
    run: Run,
    step: Step,
    result: Result,
    *,
    expected_input_bindings: dict[str, str],
    tool: Tool | None = None,
    context: AttemptContext | None = None,
) -> CommitResultOutcome:
    """Save a validated Result, then atomically checkpoint its Run pointers."""

    context = context or active_attempt(run, step.id)
    validate_result_candidate(
        run,
        step,
        result,
        context=context,
        expected_input_bindings=expected_input_bindings,
        tool=tool,
    )
    result_path = f"{result.attempt_relative_path}/result.json"
    persist_result(data_root, run, result)

    missing = object()
    previous = {
        "status": run.status,
        "waiting_for": run.waiting_for,
        "step_status": run.step_status.get(step.id, missing),
        "current_result": run.current_results.get(step.id, missing),
        "result_index": list(run.result_index),
        "pending_data": dict(run.pending_data),
    }
    if result_path not in run.result_index:
        run.result_index.append(result_path)
    run.step_status[step.id] = result.status
    if result.status == "succeeded":
        run.current_results[step.id] = result_path
        run.pending_data = {}
    elif result.status == "needs_input":
        run.status = "waiting"
        run.waiting_for = "clarification"
        run.pending_data = {
            **result.diagnostics,
            **result.clarification,
            "step_id": step.id,
            "result_path": result_path,
        }
    else:
        run.pending_data = dict(result.diagnostics)
    try:
        persist_run(data_root, run)
    except PersistenceFailure:
        run.status = previous["status"]  # type: ignore[assignment]
        run.waiting_for = previous["waiting_for"]  # type: ignore[assignment]
        run.result_index[:] = previous["result_index"]  # type: ignore[index]
        run.pending_data = previous["pending_data"]  # type: ignore[assignment]
        if previous["step_status"] is missing:
            run.step_status.pop(step.id, None)
        else:
            run.step_status[step.id] = previous["step_status"]  # type: ignore[assignment]
        if previous["current_result"] is missing:
            run.current_results.pop(step.id, None)
        else:
            run.current_results[step.id] = previous["current_result"]  # type: ignore[assignment]
        raise
    return CommitResultOutcome(result_relative_path=result_path, published=True)


def allocate_attempt(data_root: str | Path, run: Run, step_id: str) -> int:
    """Allocate the next attempt number from durable records and directories.

    Adapters call this one boundary and never implement their own scan.  The
    directory check prevents a stale Run record from causing a previous raw
    attempt directory to be overwritten after a restart or partial checkpoint.
    """

    attempts = [
        int(item.get("attempt", 0)) for item in run.attempts if item.get("step_id") == step_id
    ]
    step_directory = run_directory(data_root, run.id) / step_id
    if step_directory.is_dir():
        for child in step_directory.glob("attempt-*"):
            if not child.is_dir():
                continue
            try:
                attempts.append(int(child.name.removeprefix("attempt-")))
            except ValueError:
                continue
    return max(attempts, default=0) + 1


__all__ = [
    "AttemptContext",
    "AttemptLifecycleError",
    "CommitResultOutcome",
    "ExecutionBusy",
    "PersistenceFailure",
    "RunOwner",
    "allocate_attempt",
    "active_attempt",
    "begin_attempt",
    "checkpoint_attempt",
    "commit_attempt_result",
    "ensure_attempt",
    "fail_attempt",
    "finish_attempt",
    "persist_result",
    "persist_run",
    "release_attempt",
    "save_result",
    "save_run",
    "validate_result_candidate",
]
