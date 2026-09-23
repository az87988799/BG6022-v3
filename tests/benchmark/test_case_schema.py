from __future__ import annotations

import pytest
from pydantic import ValidationError

from bg6022.benchmark.models import BenchmarkCase


def _case(**overrides):
    payload = {
        "id": "schema_case",
        "category": "planning",
        "support": "supported",
        "mode": "offline",
        "prompt": "Plan a single point calculation",
        "assertions": [{"type": "step_count", "value": 1}],
    }
    payload.update(overrides)
    return payload


def test_case_schema_accepts_bounded_assertion() -> None:
    case = BenchmarkCase.model_validate(_case(), strict=True)
    assert case.assertions[0].type == "step_count"


def test_case_schema_rejects_unknown_assertion() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(
            _case(assertions=[{"type": "arbitrary_python", "value": "True"}]), strict=True
        )


def test_case_schema_rejects_expression_fields_and_excessive_repeat() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(
            _case(assertions=[{"type": "step_count", "expression": "len(plan.steps) == 1"}]),
            strict=True,
        )
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(_case(repeat=6), strict=True)
