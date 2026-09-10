from __future__ import annotations

from pathlib import Path

from bg6022.models import Plan, Request, Run, Step
from bg6022.session import (
    artifact_path,
    find_artifact,
    load_run,
    register_file_artifact,
    run_directory,
    save_run,
    utc_now,
)


def test_artifact_copy_and_run_json_are_atomic(tmp_path: Path) -> None:
    source = tmp_path / "water.xyz"
    source.write_bytes(b"1\nhydrogen\nH 0 0 0\n")
    data_root = tmp_path / "data"
    request = Request(id="request_1", description="test")
    plan = Plan(id="plan_1", request_id=request.id, steps=[Step(id="compute", tool="single_point")])
    run = Run(
        id="run_1",
        request=request,
        plan=plan,
        resources={"cores": 4},
        execution_permission=True,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    (run_directory(data_root, run.id) / "artifacts").mkdir(parents=True)
    artifact = register_file_artifact(
        data_root,
        run,
        source,
        artifact_type="molecular_geometry",
        role="input_geometry",
        source="test",
    )
    save_run(data_root, run)
    loaded = load_run(data_root, run.id)
    loaded_artifact = find_artifact(loaded, artifact.id)
    copied = artifact_path(data_root, loaded, loaded_artifact)
    assert copied.read_bytes() == source.read_bytes()
    assert loaded_artifact.sha256 == artifact.sha256
