"""Controlled ORCA process execution; no scientific interpretation lives here."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any

from bg6022.config import environment_for_child

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_orca(
    *,
    executable: str | Path,
    attempt_dir: str | Path,
    resources: RunnerResources,
    cancel: Event,
    deadline: float,
    data_root: str | Path,
    on_started: Callable[[ProcessFacts], None] | None = None,
) -> ProcessFacts:
    """Run exactly one ORCA process under a Windows Job Object."""

    directory = Path(attempt_dir).resolve()
    root = Path(data_root).resolve()
    executable_path = Path(executable).resolve()
    facts = ProcessFacts(
        stdout_path=str(directory / "stdout.out"),
        stderr_path=str(directory / "stderr.txt"),
    )
    if os.name != "nt":
        facts.status = "failed"
        facts.stop_reason = "platform_not_supported"
        facts.platform_supported = False
        return facts
    if not executable_path.is_file() or executable_path.suffix.casefold() != ".exe":
        facts.stop_reason = "launch_failed"
        return facts
    input_path = directory / "input.inp"
    geometry_path = directory / "geometry.xyz"
    if not input_path.is_file() or not geometry_path.is_file():
        facts.stop_reason = "launch_failed"
        return facts
    if cancel.is_set():
        facts.status = "cancelled"
        facts.stop_reason = "cancel_requested_before_spawn"
        return facts

    stdout_handle = None
    stderr_handle = None
    process: subprocess.Popen[bytes] | None = None
    job: WindowsJobObject | None = None
    try:
        stdout_handle = (directory / "stdout.out").open("wb")
        stderr_handle = (directory / "stderr.txt").open("wb")
        job = WindowsJobObject(memory_limit_bytes=resources.memory_mb * 1024 * 1024)
        creation_flags = CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        process = subprocess.Popen(
            [str(executable_path), str(input_path)],
            cwd=str(directory),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            close_fds=True,
            creationflags=creation_flags,
            env=environment_for_child(),
        )
        facts.pid = process.pid
        facts.process_created_at = process_start_marker(process.pid)
        job.assign_pid(process.pid)
        if on_started is not None:
            on_started(facts)
        job.resume_pid(process.pid)
        return _monitor_process(
            process,
            job,
            facts,
            stdout_handle,
            stderr_handle,
            resources,
            cancel,
            deadline,
            root,
        )
    except KeyboardInterrupt:
        facts.stop_reason = "cancel_requested_keyboard_interrupt"
        facts.status = "cancelled"
        if process is not None:
            _stop_and_confirm(process, job, facts, root)
        return facts
    except Exception as error:
        facts.stop_reason = "launch_failed" if process is None else "runner_exception"
        facts.status = "interrupted" if process is not None else "failed"
        facts.cleanup_unconfirmed = process is not None
        if process is not None:
            _stop_and_confirm(process, job, facts, root)
        facts.exception = f"{type(error).__name__}: {error}"
        if facts.cleanup_unconfirmed:
            _write_execution_guard(root, facts, "runner_exception")
        return facts
    finally:
        if process is not None and process.poll() is None:
            _stop_and_confirm(process, job, facts, root)
        if stdout_handle is not None:
            stdout_handle.flush()
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.flush()
            stderr_handle.close()
        if job is not None:
            job.close()


def _monitor_process(
    process: subprocess.Popen[bytes],
    job: WindowsJobObject,
    facts: ProcessFacts,
    stdout_handle: Any,
    stderr_handle: Any,
    resources: RunnerResources,
    cancel: Event,
    deadline: float,
    data_root: Path,
) -> ProcessFacts:
    while True:
        stdout_handle.flush()
        stderr_handle.flush()
        return_code = process.poll()
        if return_code is not None:
            facts.exit_code = int(return_code)
            facts.stop_reason = "normal_exit" if return_code == 0 else "nonzero_exit"
            facts.status = "succeeded" if return_code == 0 else "failed"
            _confirm_natural_exit(process, job, facts, data_root)
            return facts
        if _directory_size(Path(facts.stdout_path).parent) > resources.workdir_limit_bytes:
            facts.stop_reason = "workdir_limit_exceeded"
            facts.status = "failed"
            _stop_and_confirm(process, job, facts, data_root)
            return facts
        output_size = _file_size(Path(facts.stdout_path)) + _file_size(Path(facts.stderr_path))
        if output_size > resources.output_limit_bytes:
            facts.stop_reason = "output_limit_exceeded"
            facts.status = "failed"
            _stop_and_confirm(process, job, facts, data_root)
            return facts
        if cancel.is_set():
            facts.stop_reason = "cancel_requested"
            facts.status = "cancelled"
            _stop_and_confirm(process, job, facts, data_root)
            return facts
        if time.monotonic() >= deadline:
            facts.stop_reason = "wall_time_deadline"
            facts.status = "timed_out"
            _stop_and_confirm(process, job, facts, data_root)
            return facts
        time.sleep(0.25)


def _confirm_natural_exit(
    process: subprocess.Popen[bytes], job: WindowsJobObject, facts: ProcessFacts, data_root: Path
) -> None:
    try:
        deadline = time.monotonic() + 5.0
        while job.active_process_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        facts.process_tree_empty = job.active_process_count() == 0
        facts.stop_confirmed = facts.process_tree_empty
        if not facts.process_tree_empty:
            facts.status = "interrupted"
            facts.stop_reason = "cleanup_unconfirmed"
            facts.cleanup_unconfirmed = True
            _write_execution_guard(data_root, facts, "cleanup_unconfirmed")
    except OSError:
        facts.process_tree_empty = False
        facts.stop_confirmed = False
        facts.status = "interrupted"
        facts.stop_reason = "cleanup_unconfirmed"
        facts.cleanup_unconfirmed = True
        _write_execution_guard(data_root, facts, "cleanup_query_failed")


def _stop_and_confirm(
    process: subprocess.Popen[bytes],
    job: WindowsJobObject | None,
    facts: ProcessFacts,
    data_root: Path,
) -> None:
    if process.poll() is not None and job is not None:
        _confirm_natural_exit(process, job, facts, data_root)
        return
    try:
        if job is not None:
            job.terminate(1)
            facts.stop_request_sent = True
        process.wait(timeout=10)
        facts.exit_code = process.returncode
        if job is None:
            facts.process_tree_empty = True
        else:
            deadline = time.monotonic() + 5.0
            while job.active_process_count() and time.monotonic() < deadline:
                time.sleep(0.02)
            facts.process_tree_empty = job.active_process_count() == 0
        facts.stop_confirmed = bool(facts.process_tree_empty)
        if not facts.stop_confirmed:
            facts.status = "interrupted"
            facts.stop_reason = "cleanup_unconfirmed"
            facts.cleanup_unconfirmed = True
            _write_execution_guard(data_root, facts, "cleanup_unconfirmed")
    except (OSError, subprocess.TimeoutExpired):
        facts.stop_confirmed = False
        facts.process_tree_empty = False
        facts.status = "interrupted"
        facts.stop_reason = "cleanup_unconfirmed"
        facts.cleanup_unconfirmed = True
        _write_execution_guard(data_root, facts, "cleanup_unconfirmed")


def _write_execution_guard(data_root: Path, facts: ProcessFacts, reason: str) -> None:
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        target = data_root / "execution_guard.json"
        temporary = data_root / ".execution_guard.json.tmp"
        payload = {"reason": reason, "facts": facts.to_dict(), "created_at": time.time()}
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError:
        pass


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


def probe_orca_version(executable: str | Path, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Ask ORCA for its banner with a missing input, never a calculation."""

    import re
    import tempfile

    if os.name != "nt":
        return {"ok": False, "reason": "platform_not_supported"}
    exe = Path(executable).resolve()
    if not exe.is_file():
        return {"ok": False, "reason": "executable_missing"}
    with tempfile.TemporaryDirectory(prefix="bg6022-probe-") as temporary:
        missing = Path(temporary) / "input-does-not-exist.inp"
        process: subprocess.Popen[bytes] | None = None
        job: WindowsJobObject | None = None
        try:
            job = WindowsJobObject(memory_limit_bytes=512 * 1024 * 1024)
            process = subprocess.Popen(
                [str(exe), str(missing)],
                cwd=temporary,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED,
                env=environment_for_child(),
            )
            job.assign_pid(process.pid)
            job.resume_pid(process.pid)
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if process is not None and job is not None:
                job.terminate(1)
                process.wait(timeout=5)
            return {"ok": False, "reason": "probe_timeout"}
        except OSError as error:
            return {"ok": False, "reason": f"probe_launch_failed: {error}"}
        finally:
            if job is not None:
                try:
                    end = time.monotonic() + 5
                    while job.active_process_count() and time.monotonic() < end:
                        time.sleep(0.02)
                except OSError:
                    pass
                job.close()
        stdout = stdout_bytes[: 1024 * 1024].decode("utf-8", errors="replace")
        stderr = stderr_bytes[: 1024 * 1024].decode("utf-8", errors="replace")
        text = f"{stdout}\n{stderr}"
        match = re.search(
            r"(?:Program Version|ORCA Version|Version)\s*[:=]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
            text,
            re.I,
        )
        if match is None:
            match = re.search(r"ORCA\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text, re.I)
        return {
            "ok": match is not None,
            "version": None if match is None else match.group(1),
            "exit_code": process.returncode if process is not None else None,
            "stdout_preview": stdout[-2000:],
            "stderr_preview": stderr[-2000:],
        }


__all__ = ["ProcessFacts", "RunnerResources", "probe_orca_version", "run_orca"]
