from __future__ import annotations

import pytest

from bg6022.benchmark.loader import load_cases
from bg6022.benchmark.runner import BenchmarkModeDisabled, run_case


def test_live_llm_case_requires_an_explicit_flag() -> None:
    case = next(item for item in load_cases("benchmarks/v1") if item.id == "B001_single_opt")
    with pytest.raises(BenchmarkModeDisabled, match="--live-llm"):
        run_case(case, config=None, benchmark_dir="benchmarks/v1")


def test_offline_case_uses_production_plan_pipeline_without_live_calls(monkeypatch) -> None:
    from bg6022.benchmark import runner

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline benchmark must not call a live execution boundary")

    monkeypatch.setattr(runner, "_run_live_llm", forbidden)
    monkeypatch.setattr(runner.Agent, "execute_plan", forbidden)
    case = next(item for item in load_cases("benchmarks/v1") if item.id == "B002_single_sp")
    observations = run_case(case, config=None, benchmark_dir="benchmarks/v1")
    assert len(observations) == 1
    assert observations[0].status == "completed"
    assert observations[0].llm_calls == []
    assert observations[0].orca_attempts == 0
