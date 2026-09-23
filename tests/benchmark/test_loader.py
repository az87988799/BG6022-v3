from __future__ import annotations

import json

import pytest

from bg6022.benchmark.loader import (
    BenchmarkConfigurationError,
    fixture_path,
    load_cases,
    load_fixture,
)
from bg6022.benchmark.models import BenchmarkCase, LiveSetup


def test_case_sets_keep_holdout_separate() -> None:
    cases = load_cases("benchmarks/v1")
    all_cases = load_cases("benchmarks/v1", include_holdout=True)
    assert len(cases) == 39
    assert len(all_cases) == len(cases) + 4
    assert sum(case.mode in {"offline", "replay"} for case in cases) >= 20
    assert sum(case.mode == "live_llm" for case in cases) >= 10


def test_live_compute_cases_declare_loadable_fixtures() -> None:
    cases = load_cases("benchmarks/v1")
    live_compute_cases = [case for case in cases if case.requires_orca]
    assert live_compute_cases
    for case in live_compute_cases:
        assert case.fixture is not None
        load_fixture(case, "benchmarks/v1")

    missing_executable = next(case for case in live_compute_cases if case.id.startswith("B032"))
    assert load_fixture(missing_executable, "benchmarks/v1")["force_missing_executable"] is True


def test_live_scenario_fixtures_validate_and_resolve_seed_references() -> None:
    cases = {item.id: item for item in load_cases("benchmarks/v1")}
    expected = {
        "B010_modify_second_requirement": "waiting",
        "B022_ambiguous_iteration": "waiting",
        "B028_history_energy_query": "succeeded",
    }
    for case_id, status in expected.items():
        fixture = load_fixture(cases[case_id], "benchmarks/v1")
        setup = LiveSetup.model_validate(fixture["live_setup"], strict=True)
        assert setup.active_run_status == status
        assert setup.active_run_fixture


def test_live_setup_references_cannot_escape_benchmark_root(tmp_path) -> None:
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "case.json").write_text(
        json.dumps(
            {
                "live_setup": {
                    "active_run_fixture": "../outside.json",
                    "active_run_status": "waiting",
                    "waiting_for": "confirmation",
                }
            }
        ),
        encoding="utf-8",
    )
    case = BenchmarkCase.model_validate(
        {
            "id": "live_path_case",
            "category": "planning",
            "support": "supported",
            "mode": "live_llm",
            "prompt": "fixture path containment",
            "fixture": "case.json",
            "assertions": [{"type": "step_count", "value": 1}],
        },
        strict=True,
    )
    with pytest.raises(BenchmarkConfigurationError, match="remain inside"):
        load_fixture(case, root)


def test_loader_rejects_unknown_assertion(tmp_path) -> None:
    (tmp_path / "cases.jsonl").write_text(
        json.dumps(
            {
                "id": "bad_case",
                "category": "planning",
                "support": "supported",
                "mode": "offline",
                "prompt": "bad",
                "assertions": [{"type": "evaluate", "expression": "True"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkConfigurationError, match="invalid benchmark case"):
        load_cases(tmp_path)


def test_fixture_paths_cannot_escape_benchmark_root(tmp_path) -> None:
    case = BenchmarkCase.model_validate(
        {
            "id": "path_case",
            "category": "planning",
            "support": "supported",
            "mode": "offline",
            "prompt": "fixture containment",
            "fixture": "../outside.json",
            "assertions": [{"type": "step_count", "value": 1}],
        },
        strict=True,
    )
    with pytest.raises(BenchmarkConfigurationError, match="remain inside"):
        fixture_path(case, tmp_path / "v1")
