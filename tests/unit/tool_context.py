from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Artifact, Run, Step
from bg6022.session import find_artifact, run_directory
from bg6022.tools.runtime import ToolCallContext


def make_tool_context(
    config: AppConfig, run: Run, step: Step, cancel: Event | None = None
) -> ToolCallContext:
    """Allocate the same invocation facts as Agent._invoke_step for direct Tool tests."""

    attempt = (
        max(
            (
                int(item.get("attempt", 0))
                for item in run.attempts
                if item.get("step_id") == step.id
            ),
            default=0,
        )
        + 1
    )
    relative = f"{step.id}/attempt-{attempt:02d}"
    workdir = run_directory(config.data_root_path, run.id) / relative
    workdir.mkdir(parents=True, exist_ok=False)
    frozen_inputs: dict[str, Artifact] = {}
    for name, reference in step.inputs.items():
        artifact_id = reference.artifact_id
        if artifact_id is None:
            result_relative = run.current_results.get(reference.step_id)
            if result_relative is None:
                raise ValueError(f"no current result for {reference.step_id}")
            result_path = (run_directory(config.data_root_path, run.id) / result_relative).resolve()
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_id = payload.get("output_ports", {}).get(reference.port)
            if not isinstance(artifact_id, str):
                raise ValueError(f"no current artifact at {reference.step_id}.{reference.port}")
        frozen_inputs[name] = find_artifact(run, artifact_id)
    attempt_record: dict[str, Any] = {
        "step_id": step.id,
        "attempt": attempt,
        "phase": "prepared",
        "status": "running",
        "relative_path": workdir.relative_to(Path(config.data_root_path).resolve()).as_posix(),
        "result_relative_path": relative,
        "artifact_ids": [],
        "output_ports": {},
        "input_artifact_ids": list(dict.fromkeys(a.id for a in frozen_inputs.values())),
    }
    run.attempts.append(attempt_record)
    run.step_status[step.id] = "running"
    return ToolCallContext(
        data_root=Path(config.data_root_path),
        run=run,
        step=step,
        attempt=attempt,
        cancel=cancel or Event(),
        workdir=workdir,
        relative_attempt_path=relative,
        frozen_inputs=frozen_inputs,
        attempt_record=attempt_record,
    )
