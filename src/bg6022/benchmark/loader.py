"""Load strict JSONL cases and safe, repository-local fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import BenchmarkCase, LiveSetup


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
    return fixture_reference_path(case.fixture, benchmark_dir, label=f"fixture for {case.id}")


def fixture_reference_path(
    reference: str, benchmark_dir: str | Path, *, label: str = "fixture reference"
) -> Path:
    if not reference.strip():
        raise BenchmarkConfigurationError(f"{label} must not be blank")
    root = Path(benchmark_dir).resolve()
    raw_candidate = root / reference
    if _has_symlink_component(raw_candidate):
        raise BenchmarkConfigurationError(f"{label} cannot use a symlink")
    candidate = raw_candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise BenchmarkConfigurationError(f"{label} must remain inside {root}: {reference}")
    if not candidate.is_file() or candidate.is_symlink():
        raise BenchmarkConfigurationError(f"{label} is not a regular file")
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
    return _load_fixture_file(path, benchmark_dir)


def load_fixture_reference(reference: str, benchmark_dir: str | Path) -> dict[str, Any]:
    """Load a fixture referenced by a finite live_setup field."""

    path = fixture_reference_path(reference, benchmark_dir)
    return _load_fixture_file(path, benchmark_dir)


def _load_fixture_file(path: Path, benchmark_dir: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkConfigurationError(f"fixture is invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkConfigurationError(f"fixture root must be a JSON object: {path}")
    if "live_setup" in payload:
        try:
            setup = LiveSetup.model_validate(payload["live_setup"], strict=True)
        except (ValidationError, TypeError, ValueError) as error:
            raise BenchmarkConfigurationError(f"invalid live_setup in {path}: {error}") from error
        for label, reference in (
            ("active_run_fixture", setup.active_run_fixture),
            ("published_result_fixture", setup.published_result_fixture),
        ):
            if reference is not None:
                fixture_reference_path(
                    reference,
                    benchmark_dir,
                    label=f"{label} in {path}",
                )
    return payload


__all__ = [
    "BenchmarkConfigurationError",
    "fixture_path",
    "fixture_reference_path",
    "load_cases",
    "load_fixture",
    "load_fixture_reference",
]
