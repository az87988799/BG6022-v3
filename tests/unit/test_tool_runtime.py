from __future__ import annotations

from pathlib import Path

import pytest
from tool_context import make_tool_context

from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Run, Step
from bg6022.session import artifact_path, create_run, register_bytes_artifact, utc_now


def test_tool_call_context_freezes_inputs_and_binds_results(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    request = Request(id="request_context", description="verify Tool context")
    step = Step(
        id="measure",
        tool="geometry_distance",
        inputs={"geometry": InputReference(artifact_id="__geometry__")},
    )
    run = Run(
        id="run_context",
        request=request,
        plan=Plan(id="plan_context", request_id=request.id, steps=[step]),
        resources=config.resources,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    geometry = register_bytes_artifact(
        config.data_root_path,
        run,
        b"original geometry",
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test:context",
        extension=".xyz",
        step_id="source",
        attempt=1,
    )
    step.inputs["geometry"] = InputReference(artifact_id=geometry.id)
    run.plan = Plan(id="plan_context", request_id=request.id, steps=[step])

    context = make_tool_context(config, run, step)

    assert context.read_input("geometry") == b"original geometry"
    output = context.register_bytes(
        b"tool output",
        artifact_type="geometry_report",
        role="measurement_report",
        source="test:context_output",
    )
    assert output.step_id == step.id
    assert output.attempt == context.attempt
    assert output.relative_path.endswith(".bin")

    result = context.make_result("failed")
    assert result.run_id == run.id
    assert result.step_id == step.id
    assert result.attempt == context.attempt
    assert result.attempt_relative_path == context.relative_attempt_path
    assert result.input_bindings == {"geometry": geometry.id}
    assert result.input_artifact_ids == [geometry.id]

    artifact_path(config.data_root_path, run, geometry).write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="hash or size mismatch|changed after it was frozen"):
        context.read_input("geometry")
