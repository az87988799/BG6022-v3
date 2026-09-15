from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bg6022.orca import runner
from bg6022.orca.runner import ProcessFacts
from bg6022.session import execution_guard_path, write_execution_guard


class _FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.pid = 1234
        self._returncode = returncode

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        if self._returncode is None:
            self._returncode = 1
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 1


class _FakeJob:
    def __init__(self, mode: str, *, effective_termination: bool = True) -> None:
        self.mode = mode
        self.effective_termination = effective_termination
        self.process: _FakeProcess | None = None
        self.terminate_calls = 0
        self._active_calls = 0
        self._terminated = False

    def assign_pid(self, pid: int) -> None:
        del pid

    def resume_pid(self, pid: int) -> None:
        del pid

    def terminate(self, exit_code: int = 1) -> None:
        del exit_code
        self.terminate_calls += 1
        if self.effective_termination:
            self._terminated = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def active_process_count(self) -> int:
        if self.mode == "none":
            return 0
        if self.mode == "natural_child":
            self._active_calls += 1
            return 1 if self._active_calls == 1 else 0
        if self.mode == "residual":
            return 0 if self._terminated else 1
        if self.mode == "unconfirmed":
            return 1
        raise AssertionError(f"unknown fake Job mode: {self.mode}")

    def close(self) -> None:
        return None


class _SteppingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        self.value += 0.1
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0.1)


def _teardown(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
    job: _FakeJob,
    facts: ProcessFacts,
    *,
    clock: _SteppingClock | None = None,
    needs_stop: bool = False,
) -> None:
    if clock is not None:
        monkeypatch.setattr(runner.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(runner.time, "sleep", clock.sleep)
    else:
        monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    job.process = process
    runner._teardown_process(
        process,
        job,
        job_assigned=True,
        process_resumed=True,
        facts=facts,
        needs_stop=needs_stop,
    )


def test_natural_parent_and_child_exit_keeps_success(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(returncode=0)
    job = _FakeJob("natural_child")
    facts = ProcessFacts(status="succeeded", stop_reason="normal_exit")

    _teardown(monkeypatch, process, job, facts)

    assert facts.status == "succeeded"
    assert facts.stop_reason == "normal_exit"
    assert facts.stop_request_sent is False
    assert facts.process_tree_empty is True
    assert facts.stop_confirmed is True
    assert job.terminate_calls == 0


def test_residual_child_forced_termination_revokes_natural_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=0)
    job = _FakeJob("residual")
    facts = ProcessFacts(status="succeeded", stop_reason="normal_exit")

    _teardown(monkeypatch, process, job, facts, clock=_SteppingClock())

    assert facts.status == "failed"
    assert facts.stop_reason == "residual_processes_terminated"
    assert facts.exit_code == 0
    assert facts.stop_request_sent is True
    assert facts.process_tree_empty is True
    assert facts.stop_confirmed is True
    assert facts.cleanup_unconfirmed is False
    assert job.terminate_calls == 1


def test_unconfirmed_residual_cleanup_stays_interrupted_and_guard_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess(returncode=0)
    job = _FakeJob("unconfirmed", effective_termination=False)
    facts = ProcessFacts(status="succeeded", stop_reason="normal_exit")
    _teardown(monkeypatch, process, job, facts, clock=_SteppingClock())

    data_root = tmp_path / "data"
    write_execution_guard(data_root, {"execution_id": "execution_1", "phase": "started"})
    runner._finalize_execution_guard(data_root, "execution_1", facts)

    assert facts.status == "interrupted"
    assert facts.process_tree_empty is False
    assert facts.stop_confirmed is False
    assert facts.cleanup_unconfirmed is True
    assert execution_guard_path(data_root).exists()


def test_cancel_request_stops_process_tree_and_clears_execution_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "orca.exe"
    executable.write_bytes(b"test executable placeholder")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    (attempt_dir / "input.inp").write_text("! offline cancellation test\n", encoding="utf-8")
    (attempt_dir / "geometry.xyz").write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
    data_root = tmp_path / "data"
    execution_id = "execution_cancelled"
    write_execution_guard(data_root, {"execution_id": execution_id, "phase": "started"})
    harness = _ProbeHarness(
        monkeypatch,
        returncode=None,
        output=b"",
        job_mode="residual",
    )

    class CancelAfterSpawn:
        calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls >= 2

    facts = runner.run_orca(
        executable=executable,
        attempt_dir=attempt_dir,
        resources=runner.RunnerResources(
            cores=4,
            memory_mb=1024,
            output_limit_bytes=1024 * 1024,
            workdir_limit_bytes=1024 * 1024,
        ),
        cancel=CancelAfterSpawn(),
        deadline=runner.time.monotonic() + 10,
        data_root=data_root,
        execution_id=execution_id,
    )

    assert facts.status == "cancelled"
    assert facts.stop_reason == "cancel_requested"
    assert facts.stop_request_sent is True
    assert facts.process_tree_empty is True
    assert facts.stop_confirmed is True
    assert harness.job.terminate_calls == 1
    assert not execution_guard_path(data_root).exists()


class _ProbeHarness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        returncode: int | None,
        output: bytes,
        job_mode: str = "none",
        effective_termination: bool = True,
    ) -> None:
        self.job = _FakeJob(job_mode, effective_termination=effective_termination)
        self.process: _FakeProcess | None = None

        def fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
            del args
            self.process = _FakeProcess(returncode)
            self.job.process = self.process
            kwargs["stdout"].write(output)
            kwargs["stdout"].flush()
            return self.process

        monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            runner,
            "WindowsJobObject",
            lambda *, memory_limit_bytes: self.job,
        )
        monkeypatch.setattr(runner, "process_start_marker", lambda _pid: 1.0)
        monkeypatch.setattr(runner, "environment_for_child", lambda: {})


def _probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    returncode: int | None,
    output: bytes,
    job_mode: str = "none",
    effective_termination: bool = True,
    timeout_seconds: float = 1.0,
) -> tuple[dict[str, Any], _ProbeHarness]:
    harness = _ProbeHarness(
        monkeypatch,
        returncode=returncode,
        output=output,
        job_mode=job_mode,
        effective_termination=effective_termination,
    )
    result = runner._probe_once(tmp_path / "orca.exe", timeout_seconds, None)
    return result, harness


def test_probe_allows_natural_nonzero_exit_when_version_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=2,
        output=b"Program Version 6.1.1\ninput file missing\n",
    )

    assert result["ok"] is True
    assert result["version"] == "6.1.1"
    assert result["reason"] is None
    assert result["process"]["stop_reason"] == "nonzero_exit"


def test_probe_timeout_after_version_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=None,
        output=b"Program Version 6.1.1\n",
        timeout_seconds=0.0,
    )

    assert result["ok"] is False
    assert result["version"] == "6.1.1"
    assert result["reason"] == "probe_timeout"
    assert result["process"]["stop_request_sent"] is True
    assert harness.job.terminate_calls == 1


def test_probe_output_limit_reason_is_not_overwritten_by_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = b"Program Version 6.1.1\n" + b"x" * (1024 * 1024)
    result, _harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=None,
        output=output,
        timeout_seconds=60.0,
    )

    assert result["ok"] is False
    assert result["version"] == "6.1.1"
    assert result["reason"] == "probe_output_limit_exceeded"
    assert result["process"]["stop_reason"] == "probe_output_limit_exceeded"


def test_probe_fast_output_limit_is_checked_after_natural_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = b"Program Version 6.1.1\n" + b"x" * (1024 * 1024)
    result, _harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=0,
        output=output,
    )

    assert result["ok"] is False
    assert result["version"] == "6.1.1"
    assert result["reason"] == "probe_output_limit_exceeded"
    assert result["process"]["stop_reason"] == "probe_output_limit_exceeded"


def test_probe_without_version_reports_version_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=1,
        output=b"input file missing\n",
    )

    assert result["ok"] is False
    assert result["version"] is None
    assert result["reason"] == "version_not_found"


def test_probe_with_residual_child_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _SteppingClock()
    monkeypatch.setattr(runner.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runner.time, "sleep", clock.sleep)
    result, harness = _probe(
        monkeypatch,
        tmp_path,
        returncode=0,
        output=b"Program Version 6.1.1\n",
        job_mode="residual",
    )

    assert result["ok"] is False
    assert result["version"] == "6.1.1"
    assert result["reason"] == "residual_processes_terminated"
    assert result["process"]["status"] == "failed"
    assert result["process"]["process_tree_empty"] is True
    assert result["process"]["stop_confirmed"] is True
    assert harness.job.terminate_calls == 1
