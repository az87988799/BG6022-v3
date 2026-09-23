from __future__ import annotations

import json

import pytest

from bg6022.benchmark.loader import (
    BenchmarkConfigurationError,
    fixture_path,
    load_cases,
)
from bg6022.benchmark.models import BenchmarkCase


def test_case_sets_keep_holdout_separate() -> None:
    cases = load_cases("benchmarks/v1")
    all_cases = load_cases("benchmarks/v1", include_holdout=True)
    assert len(cases) == 39
    assert len(all_cases) == len(cases) + 4
    assert sum(case.mode in {"offline", "replay"} for case in cases) >= 20
    assert sum(case.mode == "live_llm" for case in cases) >= 10


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
