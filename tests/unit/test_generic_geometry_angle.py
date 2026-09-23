from __future__ import annotations

from pathlib import Path

from tool_context import make_tool_context

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, ResultTarget, Run, Step
from bg6022.planner import QuerySelection, QueryTarget
from bg6022.session import create_run, register_bytes_artifact, save_run, utc_now
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_geometry_angle_proves_scalar_records_and_csv_delivery(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(
        id="request_angle",
        description="measure the angle and show the selected atom CSV",
        requested_results=[
            ResultTarget(step_id="angle", field="angle_value"),
            ResultTarget(step_id="angle", field="selected_atoms"),
            ResultTarget(step_id="angle", port="atom_report"),
        ],
        source="chat",
    )
    step = Step(
        id="angle",
        tool="geometry_angle",
        parameters={"atom_i": 1, "atom_j": 2, "atom_k": 3},
        inputs={"geometry": InputReference(artifact_id="__geometry__")},
    )
    plan = Plan(
        id="plan_angle",
        request_id=request.id,
        steps=[step],
        requested_results=request.requested_results,
    )
    run = Run(
        id="run_angle",
        request=request,
        plan=plan,
        resources=config.resources,
        status="planned",
        session_id="session_angle",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    xyz = b"3\nright angle\nH 1.0 0.0 0.0\nO 0.0 0.0 0.0\nH 0.0 1.0 0.0\n"
    geometry = register_bytes_artifact(
        config.data_root_path,
        run,
        xyz,
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test:geometry_angle",
        extension=".xyz",
        step_id="source",
        attempt=1,
    )
    run.plan = plan.model_copy(
        update={
            "steps": [
                step.model_copy(
                    update={"inputs": {"geometry": InputReference(artifact_id=geometry.id)}}
                )
            ]
        }
    )
    run.accepted_snapshot = {}
    save_run(config.data_root_path, run)

    agent = Agent(config, registry, llm=object(), session_id=run.session_id)
    result = agent.advance(run)

    assert result is not None and result.status == "succeeded"
    assert result.values["angle_value"]["value"] == 90.0
    assert len(result.values["selected_atoms"]) == 3
    report_id = result.output_ports["atom_report"]
    report = next(item for item in run.artifact_index if item.id == report_id)
    assert (
        Path(config.data_root_path, "runs", run.id, report.relative_path)
        .read_text(encoding="utf-8")
        .startswith("atom_index,element,x,y,z,raw_line\n")
    )

    response = agent._response_for_run(run, result)
    assert response.delivery["status"] == "complete"
    assert "原子夹角" in response.text
    assert "所选原子记录" in response.text
    assert "原子坐标 CSV" in response.text
    assert "atom_index,element,x,y,z,raw_line" in response.text
    assert "file_1" not in response.text
    assert "geometry_angle" in registry.names()

    catalog = agent._build_query_catalog()
    angle_entry = next(item for item in catalog if item["result"]["property"] == "angle")
    report_entry = next(item for item in catalog if item["result"]["property"] == "atom_report")
    selection = QuerySelection(
        status="selected",
        targets=[
            QueryTarget(
                subject_ref=angle_entry["subject_ref"],
                property="angle",
                evidence="原子夹角",
            ),
            QueryTarget(
                subject_ref=report_entry["subject_ref"],
                property="atom_report",
                evidence="原子坐标 CSV",
            ),
        ],
    )

    # A missing current artifact is a delivery failure, not a successful
    # rerun or a fabricated file link.
    report_path = Path(config.data_root_path, "runs", run.id, report.relative_path)
    report_path.unlink()
    partial_query = agent._answer_context(
        "给我原子夹角和原子坐标 CSV",
        selection=selection,
        catalog=catalog,
    )
    assert partial_query.delivery["status"] == "partial"
    assert "原子夹角" in partial_query.text
    assert "尚未交付" in partial_query.text

    missing = agent._response_for_run(run, result)
    assert missing.delivery["status"] == "partial"
    assert "未重新计算" in missing.text
    assert "尚未交付" in missing.text

    cancelled_run = run.model_copy(update={"status": "cancelled", "current_results": {}})
    cancelled = agent._response_for_run(cancelled_run, None)
    assert cancelled.delivery["status"] == "cancelled"
    assert cancelled.delivery["rendered_refs"] == []
    assert "complete" not in cancelled.delivery["status"]


def test_registered_geometry_angle_rejects_missing_and_out_of_range_inputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    tool = registry.get("geometry_angle")
    request = Request(id="request_angle_failures", description="measure angle", source="chat")
    step = Step(
        id="angle",
        tool="geometry_angle",
        parameters={"atom_i": 1, "atom_j": 2, "atom_k": 3},
        inputs={},
    )
    plan = Plan(id="plan_angle_failures", request_id=request.id, steps=[step])
    run = Run(
        id="run_angle_failures",
        request=request,
        plan=plan,
        resources=config.resources,
        status="planned",
        session_id="session_angle_failures",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)

    missing = tool.execute(step, make_tool_context(config, run, step))
    assert missing.status == "failed"
    assert missing.diagnostics["category"] == "angle_measurement_failed"
    assert "requires a geometry input" in missing.diagnostics["reason"]

    xyz = b"2\nnot enough atoms\nH 0.0 0.0 0.0\nO 0.0 0.0 1.0\n"
    geometry = register_bytes_artifact(
        config.data_root_path,
        run,
        xyz,
        artifact_type="molecular_geometry",
        role="initial_geometry",
        source="test:geometry_angle_failure",
        extension=".xyz",
        step_id="source",
        attempt=1,
    )
    invalid_step = step.model_copy(
        update={"inputs": {"geometry": InputReference(artifact_id=geometry.id)}}
    )
    run.plan = plan.model_copy(update={"steps": [invalid_step]})
    save_run(config.data_root_path, run)
    invalid = tool.execute(invalid_step, make_tool_context(config, run, invalid_step))
    assert invalid.status == "failed"
    assert invalid.diagnostics["category"] == "angle_measurement_failed"
    assert "2 atoms" in invalid.diagnostics["reason"]
