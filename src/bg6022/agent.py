"""The short, reusable Plan -> Tool execution path used by M0 and M1."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from .config import AppConfig, validate_execution_environment
from .models import InputReference, Plan, Request, Result, Run, Step
from .session import (
    execution_fingerprint,
    register_file_artifact,
    run_directory,
    save_result,
    save_run,
    utc_now,
)
from .tools.molecule import parse_xyz_bytes
from .tools.registry import ToolRegistry

INPUT_GEOMETRY_PLACEHOLDER = "__input_geometry__"


class Agent:
    """Execute a supplied Plan without adding a second executor abstraction."""

    def __init__(self, config: AppConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def execute_plan(
        self,
        request: Request,
        plan: Plan,
        *,
        xyz_path: str | Path,
        cancel: Event | None = None,
        execute: bool = False,
    ) -> tuple[Run, Result]:
        if not request.id or plan.request_id != request.id:
            raise ValueError("Plan.request_id must match Request.id")
        if not plan.steps:
            raise ValueError("Plan must contain at least one Step")
        if not any(step.inputs.get("geometry") is not None for step in plan.steps):
            raise ValueError("M0 ORCA execution requires a geometry input")
        if not execute:
            raise PermissionError("execution permission was not explicitly granted")
        self.registry.validate_plan(plan)
        validate_execution_environment(self.config)
        geometry_path = Path(xyz_path).resolve()
        geometry = parse_xyz_bytes(geometry_path.read_bytes())
        del geometry
        cancel_event = cancel or Event()
        run = Run(
            id=_new_run_id(),
            request=request,
            plan=plan,
            resources=self.config.resources,
            execution_permission=execute,
            status="planned",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        run_dir = run_directory(self.config.data_root_path, run.id)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=False)
        initial_artifact = register_file_artifact(
            self.config.data_root_path,
            run,
            geometry_path,
            artifact_type="molecular_geometry",
            role="input_geometry",
            source=str(geometry_path),
            metadata={"imported_as_raw_bytes": True},
        )
        run.plan = _bind_input_geometry(plan, initial_artifact.id)
        run.accepted_execution_sha256 = execution_fingerprint(
            run.plan, run.resources, [initial_artifact]
        )
        save_run(self.config.data_root_path, run)
        run.status = "running"
        run.start_active_interval()
        save_run(self.config.data_root_path, run)
        last_result: Result | None = None
        try:
            for step in run.plan.steps:
                tool = self.registry.get(step.tool)
                run.step_status[step.id] = "running"
                save_run(self.config.data_root_path, run)
                result = tool.execute(step, run, cancel=cancel_event)
                last_result = result
                save_result(self.config.data_root_path, run, result)
                run.result_index.append(result.attempt_relative_path + "/result.json")
                run.step_status[step.id] = result.status
                if result.status != "succeeded":
                    run.status = result.status
                    save_run(self.config.data_root_path, run)
                    return run, result
                save_run(self.config.data_root_path, run)
            if last_result is None:
                raise ValueError("Plan executed no steps")
            run.status = "succeeded"
            save_run(self.config.data_root_path, run)
            return run, last_result
        except KeyboardInterrupt:
            cancel_event.set()
            run.status = "cancelled"
            save_run(self.config.data_root_path, run)
            raise
        except Exception:
            run.status = "failed"
            save_run(self.config.data_root_path, run)
            raise
        finally:
            run.finish_active_interval()
            save_run(self.config.data_root_path, run)


def _bind_input_geometry(plan: Plan, artifact_id: str) -> Plan:
    steps: list[Step] = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        reference = inputs.get("geometry")
        if reference is not None and reference.artifact_id == INPUT_GEOMETRY_PLACEHOLDER:
            inputs["geometry"] = InputReference(artifact_id=artifact_id)
        steps.append(step.model_copy(update={"inputs": inputs}))
    return plan.model_copy(update={"steps": steps})


def _new_run_id() -> str:
    import uuid

    return f"run_{uuid.uuid4().hex}"


__all__ = ["Agent", "INPUT_GEOMETRY_PLACEHOLDER"]
