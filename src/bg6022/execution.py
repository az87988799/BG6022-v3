"""Small lifecycle primitives shared by the Agent and production Tools.

This module deliberately stays below the durable domain model.  It owns only
short-lived admission and persistence boundaries; ``Run`` remains the single
durable source of lifecycle state and Tools remain the execution boundary.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from .models import Result, Run
from .session import run_directory


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


class RunOwner:
    """Non-blocking cross-process owner for one Run.

    The handle is intentionally short-lived and is never serialized.  The
    operating-system lock, rather than an in-memory flag, arbitrates two
    Agents in one process and two worker processes sharing a data root.
    """

    _process_lock = Lock()

    def __init__(self, data_root: str | Path, run_id: str) -> None:
        self.data_root = Path(data_root).resolve()
        self.run_id = run_id
        self.path = run_directory(self.data_root, run_id) / ".execution-owner.lock"
        self._handle: Any = None
        self.token = uuid.uuid4().hex

    def __enter__(self) -> RunOwner:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The process-local guard avoids platform-specific reentrant locking
        # surprises when two threads open the same lock file simultaneously.
        if not self._process_lock.acquire(blocking=False):
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
            self._process_lock.release()
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
            self._process_lock.release()


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
    "ExecutionBusy",
    "PersistenceFailure",
    "RunOwner",
    "allocate_attempt",
    "persist_result",
    "persist_run",
    "save_result",
    "save_run",
]
