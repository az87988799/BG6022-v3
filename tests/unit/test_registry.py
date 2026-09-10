from __future__ import annotations

import pytest

from bg6022.models import InputReference, Plan, Request, Step
from bg6022.tools.registry import describe_tools


def test_only_two_m0_tools_are_public() -> None:
    names = {item["name"] for item in describe_tools()}
    assert names == {"single_point", "optimize_geometry"}


def test_input_reference_rejects_arbitrary_path_shape() -> None:
    with pytest.raises(ValueError):
        InputReference(step_id="compute")


def test_plan_keeps_step_reference_logical() -> None:
    request = Request(id="request_1", description="test")
    plan = Plan(
        id="plan_1",
        request_id=request.id,
        steps=[
            Step(
                id="compute",
                tool="single_point",
                inputs={"geometry": InputReference(artifact_id="artifact_water")},
            )
        ],
    )
    assert plan.steps[0].inputs["geometry"].artifact_id == "artifact_water"
