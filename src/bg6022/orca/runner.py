"""Controlled ORCA process execution; no scientific interpretation lives here."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bg6022.config import environment_for_child
from bg6022.session import (
    RuntimeLock,
    clear_execution_guard,
    new_id,
    write_execution_guard,
)

from .windows_job import (
    CREATE_NEW_PROCESS_GROUP,
    CREATE_SUSPENDED,
    WindowsJobObject,
    process_start_marker,
)


@dataclass(frozen=True)
class RunnerResources:
    cores: int
    memory_mb: int
    output_limit_bytes: int
    workdir_limit_bytes: int


@dataclass
class ProcessFacts:
    status: str = "failed"
    exit_code: int | None = None
    stop_reason: str | None = None
    pid: int | None = None
    process_created_at: float | None = None
    stop_request_sent: bool = False
    process_tree_empty: bool | None = None
    stop_confirmed: bool = False
    cleanup_unconfirmed: bool = False
    platform_supported: bool = True
    stdout_path: str | None = None
    stderr_path: str | None = None
    exception: str | None = None
    cleanup_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_orca(
    *,
    executable: str | Path,
    attempt_dir: str | Path,
    resources: RunnerResources,
    cancel: Any,
    deadline: float,
    data_root: str | Path,
    on_started: Callable[[ProcessFacts], None] | None = None,
    execution_id: str | None = None,
) -> ProcessFacts:
    """Run exactly one ORCA process under a Windows Job Object."""

    directory = Path(attempt_dir).resolve()
    root = Path(data_root).resolve()
    executable_path = Path(executable).resolve()
    facts = ProcessFacts(
        stdout_path=str(directory / "stdout.out"),
        stderr_path=str(directory / "stderr.txt"),
    )
    stdout_handle = None
    stderr_handle = None
    process: subprocess.Popen[bytes] | None = None
    job: WindowsJobObject | None = None
    job_assigned = False
    process_resumed = False
    try:
        if os.name != "nt":
            facts.platform_supported = False
            facts.stop_reason = "platform_not_supported"
            _mark_no_process(facts)
            return facts
        if not executable_path.is_file() or executable_path.suffix.casefold() != ".exe":
            facts.stop_reason = "launch_failed"
            _mark_no_process(facts)
            return facts
        input_path = directory / "input.inp"
        geometry_path = directory / "geometry.xyz"
        if not input_path.is_file() or not geometry_path.is_file():
            facts.stop_reason = "launch_failed"
            _mark_no_process(facts)
            return facts
        if cancel.is_set():
            facts.status = "cancelled"
            facts.stop_reason = "cancel_requested_before_spawn"
            _mark_no_process(facts)
            return facts
        if time.monotonic() >= deadline:
            facts.status = "timed_out"
            facts.stop_reason = "wall_time_deadline_before_spawn"
            _mark_no_process(facts)
            return facts

        stdout_handle = (directory / "stdout.out").open("wb")
        stderr_handle = (directory / "stderr.txt").open("wb")
        job = WindowsJobObject(memory_limit_bytes=resources.memory_mb * 1024 * 1024)
        process = subprocess.Popen(
            [str(executable_path), str(input_path)],
            cwd=str(directory),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            close_fds=True,
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED,
            env=environment_for_child(),
        )
        facts.pid = process.pid
        facts.process_created_at = process_start_marker(process.pid)
        job.assign_pid(process.pid)
        job_assigned = True
        if on_started is not None:
            on_started(facts)
        job.resume_pid(process.pid)
        process_resumed = True
        return _monitor_process(
            process=process,
            job=job,
            job_assigned=job_assigned,
            process_resumed=process_resumed,
            facts=facts,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            resources=resources,
            cancel=cancel,
            deadline=deadline,
            data_root=root,
            execution_id=execution_id,
        )
    except KeyboardInterrupt:
        facts.status = "cancelled"
        facts.stop_reason = facts.stop_reason or "cancel_requested_keyboard_interrupt"
        if process is not None:
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
        else:
            _mark_no_process(facts)
        return facts
    except Exception as error:
        facts.status = "interrupted" if process is not None else "failed"
        facts.stop_reason = facts.stop_reason or (
            "runner_exception" if process is not None else "launch_failed"
        )
        facts.exception = f"{type(error).__name__}: {error}"
        if process is not None:
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
        else:
            _mark_no_process(facts)
        return facts
    finally:
        # The monitor normally performs teardown. This second guard makes the
        # callback/launch exception paths safe and idempotent.
        if process is not None and not facts.stop_confirmed:
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
        _close_handle(stdout_handle, facts, "stdout handle close")
        _close_handle(stderr_handle, facts, "stderr handle close")
        if job is not None:
            try:
                job.close()
            except Exception as error:  # pragma: no cover - Windows API failure
                _record_cleanup_error(facts, f"Job Object close failed: {error}")
        _finalize_execution_guard(root, execution_id, facts)


def _monitor_process(
    *,
    process: subprocess.Popen[bytes],
    job: WindowsJobObject,
    job_assigned: bool,
    process_resumed: bool,
    facts: ProcessFacts,
    stdout_handle: Any,
    stderr_handle: Any,
    resources: RunnerResources,
    cancel: Any,
    deadline: float,
    data_root: Path,
    execution_id: str | None,
) -> ProcessFacts:
    while True:
        stdout_handle.flush()
        stderr_handle.flush()
        return_code = process.poll()
        if return_code is not None:
            facts.exit_code = int(return_code)
            facts.stop_reason = "normal_exit" if return_code == 0 else "nonzero_exit"
            facts.status = "succeeded" if return_code == 0 else "failed"
            limit_reason = _limit_reason(Path(facts.stdout_path).parent, facts, resources)
            if limit_reason is not None:
                facts.status = "failed"
                facts.stop_reason = limit_reason
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=False,
            )
            return facts
        if _limit_reason(Path(facts.stdout_path).parent, facts, resources) is not None:
            facts.stop_reason = _limit_reason(Path(facts.stdout_path).parent, facts, resources)
            facts.status = "failed"
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
            return facts
        if cancel.is_set():
            facts.stop_reason = "cancel_requested"
            facts.status = "cancelled"
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
            return facts
        if time.monotonic() >= deadline:
            facts.stop_reason = "wall_time_deadline"
            facts.status = "timed_out"
            _teardown_process(
                process,
                job,
                job_assigned=job_assigned,
                process_resumed=process_resumed,
                facts=facts,
                needs_stop=True,
            )
            return facts
        time.sleep(0.25)


def _teardown_process(
    process: subprocess.Popen[bytes],
    job: WindowsJobObject | None,
    *,
    job_assigned: bool,
    process_resumed: bool,
    facts: ProcessFacts,
    needs_stop: bool,
) -> None:
    """Stop and verify the main process and, when assigned, its whole Job tree."""

    del process_resumed  # retained as an explicit state boundary for callers/tests
    try:
        if needs_stop and process.poll() is None:
            if job is not None and job_assigned:
                try:
                    job.terminate(1)
                    facts.stop_request_sent = True
                except OSError as error:
                    _record_cleanup_error(facts, f"Job termination failed: {error}")
            if process.poll() is None:
                try:
                    process.terminate()
                    facts.stop_request_sent = True
                except OSError as error:
                    _record_cleanup_error(facts, f"process termination failed: {error}")

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            _record_cleanup_error(facts, f"process wait timed out: {error}")
        facts.exit_code = process.returncode
        main_stopped = process.poll() is not None

        if job is None or not job_assigned:
            facts.process_tree_empty = main_stopped
        else:
            active = job.active_process_count()
            if active and not facts.stop_request_sent:
                # A parent can exit before a child. Give the Job a bounded
                # chance to drain, then terminate the remaining tree.
                drain_deadline = time.monotonic() + 0.5
                while active and time.monotonic() < drain_deadline:
                    time.sleep(0.02)
                    active = job.active_process_count()
            if active:
                try:
                    job.terminate(1)
                    facts.stop_request_sent = True
                except OSError as error:
                    _record_cleanup_error(facts, f"remaining Job termination failed: {error}")
            confirm_deadline = time.monotonic() + 5.0
            while active and time.monotonic() < confirm_deadline:
                time.sleep(0.02)
                active = job.active_process_count()
            facts.process_tree_empty = active == 0

        confirmed = main_stopped and facts.process_tree_empty is True
        facts.stop_confirmed = confirmed
        if not confirmed:
            _mark_cleanup_unconfirmed(facts)
        else:
            facts.cleanup_unconfirmed = False
    except (OSError, subprocess.TimeoutExpired) as error:
        _record_cleanup_error(facts, f"cleanup query failed: {error}")
        _mark_cleanup_unconfirmed(facts)


def _mark_no_process(facts: ProcessFacts) -> None:
    facts.process_tree_empty = True
    facts.stop_confirmed = True
    facts.cleanup_unconfirmed = False


def _mark_cleanup_unconfirmed(facts: ProcessFacts) -> None:
    facts.stop_confirmed = False
    facts.process_tree_empty = False
    facts.cleanup_unconfirmed = True
    facts.status = "interrupted"
    if facts.stop_reason is None:
        facts.stop_reason = "cleanup_unconfirmed"


def _record_cleanup_error(facts: ProcessFacts, message: str) -> None:
    if facts.cleanup_error:
        facts.cleanup_error = f"{facts.cleanup_error}; {message}"
    else:
        facts.cleanup_error = message


def _close_handle(handle: Any, facts: ProcessFacts, name: str) -> None:
    if handle is None:
        return
    try:
        handle.flush()
    except Exception as error:  # pragma: no cover - filesystem failure
        _record_cleanup_error(facts, f"{name} flush failed: {error}")
    try:
        handle.close()
    except Exception as error:  # pragma: no cover - filesystem failure
        _record_cleanup_error(facts, f"{name} close failed: {error}")


def _finalize_execution_guard(
    data_root: Path, execution_id: str | None, facts: ProcessFacts
) -> None:
    if execution_id is None:
        return
    if not facts.stop_confirmed or facts.process_tree_empty is not True:
        return
    try:
        clear_execution_guard(data_root, execution_id)
    except (OSError, RuntimeError) as error:
        _record_cleanup_error(facts, f"execution guard clear failed: {error}")
        _mark_cleanup_unconfirmed(facts)


def _write_execution_guard(
    data_root: Path, facts: ProcessFacts, reason: str, execution_id: str | None = None
) -> None:
    """Compatibility helper for diagnostics; guard writes are never swallowed."""

    payload = {"reason": reason, "facts": facts.to_dict(), "created_at": time.time()}
    if execution_id is not None:
        payload["execution_id"] = execution_id
    write_execution_guard(data_root, payload)


def _limit_reason(directory: Path, facts: ProcessFacts, resources: RunnerResources) -> str | None:
    if _directory_size(directory) > resources.workdir_limit_bytes:
        return "workdir_limit_exceeded"
    output_size = _file_size(Path(facts.stdout_path)) + _file_size(Path(facts.stderr_path))
    if output_size > resources.output_limit_bytes:
        return "output_limit_exceeded"
    return None


def _directory_size(directory: Path) -> int:
    total = 0
    if not directory.exists():
        return 0
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def probe_orca_version(
    executable: str | Path,
    *,
    timeout_seconds: float = 5.0,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Ask ORCA for its banner with a missing input, never a calculation."""

    if os.name != "nt":
        return {"ok": False, "reason": "platform_not_supported"}
    exe = Path(executable).resolve()
    if not exe.is_file() or exe.suffix.casefold() != ".exe":
        return {"ok": False, "reason": "executable_missing"}
    if data_root is None:
        return _probe_once(exe, timeout_seconds, None)
    try:
        with RuntimeLock(data_root):
            return _probe_once(exe, timeout_seconds, Path(data_root).resolve())
    except RuntimeError as error:
        return {"ok": False, "reason": f"probe_blocked: {error}"}


def _probe_once(exe: Path, timeout_seconds: float, data_root: Path | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bg6022-probe-") as temporary:
        directory = Path(temporary)
        missing = directory / "input-does-not-exist.inp"
        stdout_path = directory / "stdout.out"
        stderr_path = directory / "stderr.txt"
        facts = ProcessFacts(stdout_path=str(stdout_path), stderr_path=str(stderr_path))
        execution_id = new_id("probe") if data_root is not None else None
        if data_root is not None:
            write_execution_guard(
                data_root,
                {
                    "run_id": None,
                    "step_id": None,
                    "attempt": None,
                    "execution_id": execution_id,
                    "phase": "probe_prepared",
                    "created_at": time.time(),
                },
            )
        process: subprocess.Popen[bytes] | None = None
        job: WindowsJobObject | None = None
        stdout_handle = None
        stderr_handle = None
        assigned = False
        resumed = False
        try:
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            job = WindowsJobObject(memory_limit_bytes=512 * 1024 * 1024)
            process = subprocess.Popen(
                [str(exe), str(missing)],
                cwd=str(directory),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                close_fds=True,
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED,
                env=environment_for_child(),
            )
            facts.pid = process.pid
            facts.process_created_at = process_start_marker(process.pid)
            job.assign_pid(process.pid)
            assigned = True
            job.resume_pid(process.pid)
            resumed = True
            end = time.monotonic() + timeout_seconds
            while process.poll() is None and time.monotonic() < end:
                if _file_size(stdout_path) + _file_size(stderr_path) > 1024 * 1024:
                    facts.status = "failed"
                    facts.stop_reason = "probe_output_limit_exceeded"
                    break
                time.sleep(0.02)
            if process.poll() is None:
                facts.status = "timed_out"
                facts.stop_reason = "probe_timeout"
                _teardown_process(
                    process,
                    job,
                    job_assigned=assigned,
                    process_resumed=resumed,
                    facts=facts,
                    needs_stop=True,
                )
            else:
                facts.exit_code = process.returncode
                facts.status = "succeeded" if process.returncode == 0 else "failed"
                facts.stop_reason = facts.stop_reason or "normal_exit"
                _teardown_process(
                    process,
                    job,
                    job_assigned=assigned,
                    process_resumed=resumed,
                    facts=facts,
                    needs_stop=False,
                )
        except Exception as error:
            facts.status = "interrupted" if process is not None else "failed"
            facts.stop_reason = facts.stop_reason or "probe_launch_failed"
            facts.exception = f"{type(error).__name__}: {error}"
            if process is not None:
                _teardown_process(
                    process,
                    job,
                    job_assigned=assigned,
                    process_resumed=resumed,
                    facts=facts,
                    needs_stop=True,
                )
            else:
                _mark_no_process(facts)
        finally:
            if process is not None and not facts.stop_confirmed:
                _teardown_process(
                    process,
                    job,
                    job_assigned=assigned,
                    process_resumed=resumed,
                    facts=facts,
                    needs_stop=True,
                )
            _close_handle(stdout_handle, facts, "probe stdout handle")
            _close_handle(stderr_handle, facts, "probe stderr handle")
            if job is not None:
                try:
                    job.close()
                except Exception as error:  # pragma: no cover
                    _record_cleanup_error(facts, f"probe Job Object close failed: {error}")
            if data_root is not None:
                _finalize_execution_guard(data_root, execution_id, facts)

        stdout = _read_limited(stdout_path, 64 * 1024).decode("utf-8", errors="replace")
        stderr = _read_limited(stderr_path, 64 * 1024).decode("utf-8", errors="replace")
        text = f"{stdout}\n{stderr}"
        match = re.search(
            r"(?:Program Version|ORCA Version|Version)\s*[:=]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
            text,
            re.I,
        )
        if match is None:
            match = re.search(r"ORCA\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text, re.I)
        return {
            "ok": match is not None and facts.stop_confirmed,
            "version": None if match is None else match.group(1),
            "exit_code": facts.exit_code,
            "stdout_preview": stdout[-2000:],
            "stderr_preview": stderr[-2000:],
            "reason": None if match is not None else "version_not_found",
            "process": facts.to_dict(),
        }


def _read_limited(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except OSError:
        return b""


__all__ = ["ProcessFacts", "RunnerResources", "probe_orca_version", "run_orca"]
