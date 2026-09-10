"""Thin single-point and geometry-optimization Tools sharing one ORCA path."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from bg6022.config import AppConfig, validate_execution_environment
from bg6022.models import Result, Run, Step, Tool
from bg6022.orca.checks import evaluate_success
from bg6022.orca.input import OrcaInputSpec, render_input
from bg6022.orca.parser import inspect_attempt
from bg6022.orca.profiles import get_profile
from bg6022.orca.runner import ProcessFacts, RunnerResources, run_orca
from bg6022.session import (
    RuntimeLock,
    artifact_path,
    attempt_directory,
    execution_fingerprint,
    find_artifact,
    new_id,
    register_bytes_artifact,
    register_file_artifact,
    save_run,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_execution_guard,
)
from bg6022.tools.molecule import parse_xyz_bytes, validate_electronic_state


class OrcaParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    method_profile: StrictStr = "r2scan3c"
    environment: StrictStr = "gas"
    charge: StrictInt
    multiplicity: StrictInt
    scf_maxiter: StrictInt | None = Field(default=None, ge=1, le=1000)

    @field_validator("multiplicity")
    @classmethod
    def _positive_multiplicity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("multiplicity must be positive")
        return value


class SinglePointParameters(OrcaParameters):
    pass


class OptimizeParameters(OrcaParameters):
    geom_maxiter: StrictInt | None = Field(default=None, ge=1, le=1000)


ParametersModel = TypeVar("ParametersModel", bound=OrcaParameters)


def make_single_point_tool(config: AppConfig | None = None) -> Tool:
    return _make_tool(
        config,
        operation="SP",
        name="single_point",
        description="Run an independent ORCA single-point calculation on a registered geometry.",
        parameter_model=SinglePointParameters,
        output_ports={},
        results={"sp_electronic_energy": "Eh"},
    )


def make_optimize_tool(config: AppConfig | None = None) -> Tool:
    return _make_tool(
        config,
        operation="Opt",
        name="optimize_geometry",
        description=(
            "Optimize a registered geometry and return its converged final geometry and energy."
        ),
        parameter_model=OptimizeParameters,
        output_ports={"optimized_geometry": "molecular_geometry"},
        results={"opt_final_electronic_energy": "Eh", "optimized_geometry": "molecular_geometry"},
    )


def _make_tool(
    config: AppConfig | None,
    *,
    operation: str,
    name: str,
    description: str,
    parameter_model: type[OrcaParameters],
    output_ports: dict[str, str],
    results: dict[str, str],
) -> Tool:
    def execute(step: Step, run: Run, cancel: Event) -> Result:
        if config is None:
            raise RuntimeError(f"tool {name!r} is a description-only Tool")
        return execute_orca_step(
            config,
            step=step,
            run=run,
            cancel=cancel,
            operation=operation,
            parameter_model=parameter_model,
        )

    return Tool(
        name=name,
        description=description,
        parameter_model=parameter_model.__name__,
        parameter_schema=parameter_model.model_json_schema(),
        parameter_type=parameter_model,
        input_ports={"geometry": "molecular_geometry"},
        output_ports=output_ports,
        results=results,
        success_conditions=[
            "normal ORCA termination",
            "exit code 0",
            "SCF convergence",
            "selected finite final electronic energy",
            "unchanged execution inputs",
        ]
        + (["optimization convergence", "final geometry binding"] if operation == "Opt" else []),
        repair_capabilities=["failed optimization may provide a restart-only candidate"],
        execute_function=execute if config is not None else None,
    )


def execute_orca_step(
    config: AppConfig,
    *,
    step: Step,
    run: Run,
    cancel: Event,
    operation: str,
    parameter_model: type[OrcaParameters],
) -> Result:
    _check_execution_contract(run, step)
    parameters = parameter_model.model_validate(step.parameters, strict=True)
    profile = get_profile(parameters.method_profile)
    if parameters.environment not in profile.supported_environments:
        raise ValueError(
            f"environment {parameters.environment!r} is not implemented for {profile.name!r}"
        )
    reference = step.inputs.get("geometry")
    if reference is None:
        raise ValueError("ORCA Tool requires a geometry input reference")
    geometry_artifact = _resolve_geometry_reference(config, run, reference)
    geometry_source = artifact_path(config.data_root_path, run, geometry_artifact)
    geometry_bytes = geometry_source.read_bytes()
    geometry = parse_xyz_bytes(geometry_bytes, supported_elements=profile.supported_elements)
    validate_electronic_state(
        geometry, charge=parameters.charge, multiplicity=parameters.multiplicity
    )
    if any(symbol not in profile.supported_elements for symbol in geometry.symbols):
        raise ValueError("geometry contains an element outside the selected method profile")

    owns_active_interval = not run.active_interval_open
    if owns_active_interval:
        run.start_active_interval()
    try:
        return _execute_prepared_attempt(
            config,
            step=step,
            run=run,
            cancel=cancel,
            operation=operation,
            parameters=parameters,
            geometry_artifact=geometry_artifact,
            geometry=geometry,
            geometry_bytes=geometry_bytes,
            owns_active_interval=owns_active_interval,
        )
    finally:
        if owns_active_interval:
            run.finish_active_interval()


def _execute_prepared_attempt(
    config: AppConfig,
    *,
    step: Step,
    run: Run,
    cancel: Event,
    operation: str,
    parameters: OrcaParameters,
    geometry_artifact: Any,
    geometry: Any,
    geometry_bytes: bytes,
    owns_active_interval: bool,
) -> Result:
    attempt = _next_attempt(run, step.id)
    attempt_dir = attempt_directory(config.data_root_path, run.id, step.id, attempt)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "geometry.xyz").write_bytes(geometry_bytes)
    expected_geometry_sha = geometry_artifact.sha256
    if sha256_file(attempt_dir / "geometry.xyz") != expected_geometry_sha:
        raise ValueError("attempt geometry hash differs from the registered input artifact")

    input_spec = OrcaInputSpec(
        operation=operation,
        method_profile=parameters.method_profile,
        environment=parameters.environment,
        charge=parameters.charge,
        multiplicity=parameters.multiplicity,
        cores=int(run.resources["cores"]),
        maxcore_mb=int(run.resources["maxcore_mb"]),
        scf_maxiter=parameters.scf_maxiter,
        geom_maxiter=getattr(parameters, "geom_maxiter", None),
    )
    input_bytes = render_input(input_spec)
    (attempt_dir / "input.inp").write_bytes(input_bytes)
    attempt_record = {
        "step_id": step.id,
        "attempt": attempt,
        "relative_path": attempt_dir.relative_to(Path(config.data_root_path).resolve()).as_posix(),
        "operation": operation,
        "phase": "prepared",
        "input_geometry_artifact_id": geometry_artifact.id,
        "input_sha256": sha256_bytes(input_bytes),
        "geometry_sha256": sha256_bytes(geometry_bytes),
    }
    run.step_status[step.id] = "running"
    run.attempts.append(attempt_record)
    # A prepared attempt is durable before any process can be created.
    save_run(config.data_root_path, run)

    runner_resources = RunnerResources(
        cores=int(run.resources["cores"]),
        memory_mb=int(run.resources["memory_mb"]),
        output_limit_bytes=int(run.resources["output_limit_bytes"]),
        workdir_limit_bytes=int(run.resources["workdir_limit_bytes"]),
    )
    execution_id = new_id("execution")
    process_facts: ProcessFacts
    try:
        with RuntimeLock(config.data_root_path):
            validate_execution_environment(config)
            allowed_seconds = _allowed_seconds(run)
            if allowed_seconds <= 0:
                process_facts = ProcessFacts(
                    status="timed_out",
                    stop_reason="run_active_timeout_exhausted",
                    process_tree_empty=True,
                    stop_confirmed=True,
                )
            else:
                guard = {
                    "run_id": run.id,
                    "step_id": step.id,
                    "attempt": attempt,
                    "execution_id": execution_id,
                    "phase": "prepared",
                    "created_at": utc_now(),
                }
                write_execution_guard(config.data_root_path, guard)

                def on_started(started: ProcessFacts) -> None:
                    attempt_record.update(
                        {
                            "phase": "started",
                            "pid": started.pid,
                            "process_created_at": started.process_created_at,
                        }
                    )
                    write_execution_guard(
                        config.data_root_path,
                        {
                            **guard,
                            "phase": "started",
                            "pid": started.pid,
                            "process_created_at": started.process_created_at,
                        },
                    )
                    save_run(config.data_root_path, run)

                deadline = time.monotonic() + allowed_seconds
                process_facts = run_orca(
                    executable=config.executable_path,
                    attempt_dir=attempt_dir,
                    resources=runner_resources,
                    cancel=cancel,
                    deadline=deadline,
                    data_root=config.data_root_path,
                    execution_id=execution_id,
                    on_started=on_started,
                )
    except RuntimeError as error:
        process_facts = _lock_failure_facts(str(error))

    stdout_path = attempt_dir / "stdout.out"
    stderr_path = attempt_dir / "stderr.txt"
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    output_xyz_path = attempt_dir / "input.xyz"
    output_path_present = output_xyz_path.exists() or output_xyz_path.is_symlink()
    input_hashes_match, input_hash_error = _check_input_hashes(
        attempt_dir,
        expected_geometry_sha=expected_geometry_sha,
        expected_input_sha=attempt_record["input_sha256"],
    )
    facts = inspect_attempt(
        operation=operation,
        stdout=stdout_path,
        stderr=stderr_path,
        exit_code=process_facts.exit_code,
        runner_status=process_facts.status,
        input_geometry=geometry,
        stop_reason=process_facts.stop_reason,
        process_tree_empty=process_facts.process_tree_empty,
        output_xyz=output_xyz_path if output_path_present else None,
        input_hashes_match=input_hashes_match,
        input_hash_error=input_hash_error,
        max_output_bytes=int(run.resources["output_limit_bytes"]),
    )
    outcome = evaluate_success(facts, operation=operation)
    artifact_ids: list[str] = []
    for path, artifact_type, role in (
        (attempt_dir / "input.inp", "orca_input", "input"),
        (attempt_dir / "geometry.xyz", "molecular_geometry", "input_geometry_snapshot"),
        (stdout_path, "orca_output", "stdout"),
        (stderr_path, "orca_output", "stderr"),
    ):
        artifact = register_file_artifact(
            config.data_root_path,
            run,
            path,
            artifact_type=artifact_type,
            role=role,
            source=f"{step.id}/attempt-{attempt:02d}/{path.name}",
            step_id=step.id,
            attempt=attempt,
        )
        artifact_ids.append(artifact.id)

    output_ports: dict[str, str] = {}
    output_is_regular = output_xyz_path.is_file() and not output_xyz_path.is_symlink()
    if operation == "Opt" and input_hashes_match:
        if (
            outcome.success
            and output_is_regular
            and facts.output_geometry is not None
            and facts.stdout_geometry is not None
            and facts.geometry_consistent is True
        ):
            artifact = register_file_artifact(
                config.data_root_path,
                run,
                output_xyz_path,
                artifact_type="molecular_geometry",
                role="optimized_geometry",
                source=f"{step.id}/attempt-{attempt:02d}/input.xyz",
                step_id=step.id,
                attempt=attempt,
            )
            artifact_ids.append(artifact.id)
            output_ports["optimized_geometry"] = artifact.id
        elif (
            not outcome.success
            and facts.output_xyz_exists
            and output_is_regular
            and facts.output_geometry is not None
            and facts.stdout_geometry is not None
            and facts.geometry_consistent is True
        ):
            candidate = register_file_artifact(
                config.data_root_path,
                run,
                output_xyz_path,
                artifact_type="molecular_geometry",
                role="restart_candidate",
                source=f"{step.id}/attempt-{attempt:02d}/input.xyz",
                step_id=step.id,
                attempt=attempt,
                metadata={
                    "eligible_for": "optimization_restart_only",
                    "source_locations": facts.source_locations,
                },
            )
            artifact_ids.append(candidate.id)
        elif (
            not outcome.success
            and not facts.output_xyz_exists
            and facts.stdout_geometry_bytes is not None
            and facts.stdout_geometry is not None
        ):
            candidate = register_bytes_artifact(
                config.data_root_path,
                run,
                facts.stdout_geometry_bytes,
                artifact_type="molecular_geometry",
                role="restart_candidate",
                source=f"{step.id}/attempt-{attempt:02d}/stdout_final_geometry",
                extension=".xyz",
                step_id=step.id,
                attempt=attempt,
                metadata={
                    "eligible_for": "optimization_restart_only",
                    "source_locations": facts.source_locations,
                },
            )
            artifact_ids.append(candidate.id)

    status = "succeeded" if outcome.success else _result_status(process_facts.status)
    observed = facts.final_energy
    diagnostics = {
        "category": outcome.failure_category,
        "reason": _failure_reason(outcome.failure_category, facts, process_facts),
        "raw_paths": {
            "attempt": str(attempt_dir),
            "input": str(attempt_dir / "input.inp"),
            "geometry": str(attempt_dir / "geometry.xyz"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "facts": facts.to_dict(),
        "process": process_facts.to_dict(),
        "observed_energy": None if observed is None else observed.to_dict(),
        "input_hashes": {
            "expected_geometry": expected_geometry_sha,
            "actual_geometry": _safe_sha256(attempt_dir / "geometry.xyz"),
            "expected_input": attempt_record["input_sha256"],
            "actual_input": _safe_sha256(attempt_dir / "input.inp"),
            "match": input_hashes_match,
        },
    }
    values: dict[str, Any] = {}
    if outcome.success and observed is not None and observed.value is not None:
        key = "sp_electronic_energy" if operation == "SP" else "opt_final_electronic_energy"
        values[key] = {
            "value": observed.value,
            "unit": "Eh",
            "token": observed.token,
            "source_line": observed.line,
        }
    attempt_record.update(
        {
            "phase": "finished",
            "status": status,
            "result_category": outcome.failure_category,
            "artifact_ids": artifact_ids,
            "output_ports": output_ports,
        }
    )
    if owns_active_interval:
        run.checkpoint_active()
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        status=status,
        values=values,
        checks=outcome.checks,
        diagnostics=diagnostics,
        artifact_ids=artifact_ids,
        output_ports=output_ports,
        input_artifact_ids=[geometry_artifact.id],
        attempt_relative_path=f"{step.id}/attempt-{attempt:02d}",
    )


def _check_execution_contract(run: Run, step: Step) -> None:
    if not run.execution_permission:
        raise PermissionError("Run does not have explicit execution permission")
    if run.accepted_execution_sha256 is None:
        raise PermissionError("Run has no accepted execution fingerprint")
    expected = execution_fingerprint(run.plan, run.resources, run.artifact_index)
    if expected != run.accepted_execution_sha256:
        raise ValueError("Run execution fingerprint does not match the current Plan or inputs")
    planned = next((item for item in run.plan.steps if item.id == step.id), None)
    if planned is None:
        raise ValueError(f"step is not part of the current Plan: {step.id}")
    if planned.model_dump(mode="json") != step.model_dump(mode="json"):
        raise ValueError(f"step {step.id} differs from the accepted Plan")


def _allowed_seconds(run: Run) -> float:
    run_remaining = (
        float(run.resources["run_active_timeout_seconds"]) - run.current_active_seconds()
    )
    return min(float(run.resources["attempt_timeout_seconds"]), run_remaining)


def _next_attempt(run: Run, step_id: str) -> int:
    attempts = [item.get("attempt", 0) for item in run.attempts if item.get("step_id") == step_id]
    return max(attempts, default=0) + 1


def _check_input_hashes(
    attempt_dir: Path, *, expected_geometry_sha: str, expected_input_sha: str
) -> tuple[bool, str | None]:
    actual_geometry = _safe_sha256(attempt_dir / "geometry.xyz")
    actual_input = _safe_sha256(attempt_dir / "input.inp")
    errors: list[str] = []
    if actual_geometry != expected_geometry_sha:
        errors.append("geometry.xyz changed after launch")
    if actual_input != expected_input_sha:
        errors.append("input.inp changed after launch")
    return not errors, "; ".join(errors) if errors else None


def _safe_sha256(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except (FileNotFoundError, OSError):
        return None


def _resolve_geometry_reference(config: AppConfig, run: Run, reference: Any):
    if reference.artifact_id is not None:
        artifact = find_artifact(run, reference.artifact_id)
    else:
        artifact = None
        for item in reversed(run.attempts):
            if item.get("step_id") == reference.step_id:
                artifact_id = item.get("output_ports", {}).get(reference.port)
                if artifact_id:
                    artifact = find_artifact(run, artifact_id)
                    break
        if artifact is None:
            raise ValueError(
                "upstream port "
                f"{reference.step_id}.{reference.port} has not produced a successful artifact"
            )
    if artifact.artifact_type != "molecular_geometry":
        raise ValueError(f"artifact {artifact.id} is not a molecular geometry")
    if artifact.role == "restart_candidate":
        raise ValueError("restart_candidate is not a successful geometry input")
    return artifact


def _result_status(process_status: str) -> str:
    if process_status == "cancelled":
        return "cancelled"
    if process_status == "interrupted":
        return "interrupted"
    return "failed"


def _failure_reason(category: str | None, facts: Any, process: ProcessFacts) -> str | None:
    if category is None:
        return None
    if process.exception:
        return process.exception
    if facts.final_energy_error and category == "invalid_output":
        return facts.final_energy_error
    if facts.output_geometry_error:
        return facts.output_geometry_error
    if facts.stdout_geometry_error:
        return facts.stdout_geometry_error
    if facts.input_hash_error:
        return facts.input_hash_error
    if facts.geometry_mismatch_reason:
        return facts.geometry_mismatch_reason
    return process.stop_reason or category


def _lock_failure_facts(message: str) -> ProcessFacts:
    return ProcessFacts(
        status="failed",
        stop_reason="resource_lock",
        process_tree_empty=True,
        stop_confirmed=True,
        exception=message,
    )


__all__ = [
    "OptimizeParameters",
    "OrcaParameters",
    "SinglePointParameters",
    "execute_orca_step",
    "make_optimize_tool",
    "make_single_point_tool",
]
