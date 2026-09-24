from __future__ import annotations

import pytest
from pydantic import ValidationError

from bg6022.benchmark.dataset12 import (
    Benchmark12Object,
    Benchmark12TaskTemplate,
    PromptVariant,
    render_prompt,
)
from bg6022.benchmark.loader import BenchmarkConfigurationError


def test_object_schema_rejects_charged_or_open_shell_structures() -> None:
    with pytest.raises(ValidationError):
        Benchmark12Object.model_validate(
            {
                "id": "cation",
                "label": "cation",
                "geometry": "geometries/h2.xyz",
                "charge": 1,
                "multiplicity": 1,
            },
            strict=True,
        )


def test_task_schema_rejects_duplicate_variant_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Benchmark12TaskTemplate.model_validate(
            {
                "id": "T001",
                "label": "task",
                "input_mode": "inline_xyz",
                "prompt_variants": [
                    {"id": "canonical", "style": "canonical", "template": "A"},
                    {"id": "canonical", "style": "natural", "template": "B"},
                ],
                "ground_truth": {"roles": []},
                "cost_class": "low",
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "template", ["", "use {object.__class__}", "use {unknown}", "use {label!r}"]
)
def test_prompt_renderer_rejects_empty_unknown_or_unsafe_markers(template: str) -> None:
    with pytest.raises(BenchmarkConfigurationError):
        render_prompt(template, {"label": "water"})


def test_prompt_renderer_substitutes_only_declared_markers() -> None:
    prompt = render_prompt(
        "{label} {object_id} q={charge} m={multiplicity}\n{xyz_text}",
        {
            "label": "water",
            "object_id": "water",
            "charge": 0,
            "multiplicity": 1,
            "xyz_text": "3\nwater\nO 0 0 0\n",
        },
    )
    assert prompt.startswith("water water q=0 m=1\n3")


def test_prompt_renderer_rejects_missing_marker_value() -> None:
    with pytest.raises(BenchmarkConfigurationError, match="no value"):
        render_prompt("Calculate {label}", {})


def test_prompt_variant_rejects_unknown_style() -> None:
    with pytest.raises(ValidationError):
        PromptVariant.model_validate(
            {"id": "test", "style": "random", "template": "prompt"},
            strict=True,
        )
