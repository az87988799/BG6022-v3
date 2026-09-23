"""Load strict JSONL cases and safe, repository-local fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import BenchmarkCase


class BenchmarkConfigurationError(ValueError):
    """A benchmark definition or fixture is malformed."""


def load_cases(benchmark_dir: str | Path, *, include_holdout: bool = False) -> list[BenchmarkCase]:
    root = Path(benchmark_dir).resolve()
    paths = [root / "cases.jsonl"]
    if include_holdout:
        holdout = root / "holdout.jsonl"
        if holdout.is_file():
            paths.append(holdout)
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            if path.name == "holdout.jsonl" and not include_holdout:
                continue
            raise BenchmarkConfigurationError(f"benchmark case file does not exist: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                case = BenchmarkCase.model_validate(raw, strict=True)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
                raise BenchmarkConfigurationError(
                    f"invalid benchmark case at {path}:{line_number}: {error}"
                ) from error
            if case.id in seen:
                raise BenchmarkConfigurationError(f"duplicate benchmark case id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    return cases


def fixture_path(case: BenchmarkCase, benchmark_dir: str | Path) -> Path:
    if case.fixture is None:
        raise BenchmarkConfigurationError(f"case {case.id} does not declare a fixture")
    root = Path(benchmark_dir).resolve()
    raw_candidate = root / case.fixture
    if _has_symlink_component(raw_candidate):
        raise BenchmarkConfigurationError(f"fixture for {case.id} cannot use a symlink")
    candidate = raw_candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise BenchmarkConfigurationError(
            f"fixture for {case.id} must remain inside {root}: {case.fixture}"
        )
    if not candidate.is_file() or candidate.is_symlink():
        raise BenchmarkConfigurationError(f"fixture for {case.id} is not a regular file")
    return candidate


def _has_symlink_component(path: Path) -> bool:
    current = path
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return current.is_symlink()


def load_fixture(case: BenchmarkCase, benchmark_dir: str | Path) -> dict[str, Any]:
    path = fixture_path(case, benchmark_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkConfigurationError(f"fixture is invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkConfigurationError(f"fixture root must be a JSON object: {path}")
    return payload


__all__ = [
    "BenchmarkConfigurationError",
    "fixture_path",
    "load_cases",
    "load_fixture",
]
