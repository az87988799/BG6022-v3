from __future__ import annotations

from pathlib import Path

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, ResultTarget, Run, Step
from bg6022.session import create_run, register_bytes_artifact, save_run, utc_now
from bg6022.tools.registry import ToolRegistry, build_registry
from tests.support.geometry_angle_tool import make_geometry_angle_tool


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
    base = build_registry(config)
    registry = ToolRegistry(
        [base.get(name) for name in base.names()] + [make_geometry_angle_tool(config)]
    )
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

    # A missing current artifact is a delivery failure, not a successful
    # rerun or a fabricated file link.
    report_path = Path(config.data_root_path, "runs", run.id, report.relative_path)
    report_path.unlink()
    missing = agent._response_for_run(run, result)
    assert missing.delivery["status"] == "partial"
    assert "未重新计算" in missing.text
    assert "尚未交付" in missing.text
