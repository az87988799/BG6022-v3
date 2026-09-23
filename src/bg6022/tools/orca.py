"""ORCA Tools sharing one validated input, execution, and parsing path."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from threading import Event
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from bg6022.config import AppConfig, validate_execution_environment
from bg6022.models import (
    InputReference,
    Plan,
    RepairOption,
    Request,
    Result,
    ResultProperty,
    Run,
    ScientificCheckResult,
    Step,
    Tool,
    ToolPreparation,
)
from bg6022.orca.checks import CheckOutcome, evaluate_success
from bg6022.orca.input import OrcaInputSpec, render_input
from bg6022.orca.parser import EnergyObservation, inspect_attempt
from bg6022.orca.profiles import get_profile, resolve_parameters
from bg6022.orca.repair_rules import applicable_repairs, applicable_scf_repair
from bg6022.orca.runner import ProcessFacts, RunnerResources, run_orca
from bg6022.output_contracts import is_compatible_value
from bg6022.session import (
    RuntimeLock,
    artifact_path,
    attempt_directory,
    execution_fingerprint,
    find_artifact,
    new_id,
    register_bytes_artifact,
    register_file_artifact,
    run_directory,
    save_run,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_execution_guard,
)
from bg6022.tools.molecule import ParsedGeometry, parse_xyz_bytes, validate_electronic_state


class OrcaParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    method_profile: StrictStr = "r2scan3c"
    environment: StrictStr = "gas"
    charge: StrictInt
    multiplicity: StrictInt
    scf_maxiter: StrictInt | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="SCF electronic iterations; applies to SP, Opt, and Freq.",
    )

    @field_validator("multiplicity")
    @classmethod
    def _positive_multiplicity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("multiplicity must be positive")
        return value


class SinglePointParameters(OrcaParameters):
    pass


class OptimizeParameters(OrcaParameters):
    geom_maxiter: StrictInt | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Geometry optimization iterations; applies only to Opt.",
    )


class FrequencyParameters(OrcaParameters):
    pass


ParametersModel = TypeVar("ParametersModel", bound=OrcaParameters)


def make_single_point_tool(config: AppConfig | None = None) -> Tool:
    return _make_tool(
        config,
        operation="SP",
        name="single_point",
        display_name="独立单点计算",
        description="Run an independent ORCA single-point calculation on a registered geometry.",
        parameter_model=SinglePointParameters,
        output_ports={"energy_data": "energy_data"},
        results={"sp_electronic_energy": "Eh"},
        result_properties={
            "sp_electronic_energy": "electronic_energy",
            "energy_data": "energy_data",
        },
        result_check_prerequisites={
            "sp_electronic_energy": ["local_minimum_supported"],
        },
        result_metadata={
            "sp_electronic_energy": {
                "label": "单点电子能",
                "description": "给定输入几何上的电子能，不含零点能和热校正",
                "caveat": "这是固定几何的单点结果，不代表几何优化或频率验证",
            },
            "energy_data": {
                "label": "单点电子能数据",
                "description": "含数值、方法、电子态、结构哈希和生成来源的结构化能量数据",
            },
        },
        request_parameters=[
            "method_profile",
            "environment",
            "charge",
            "multiplicity",
            "scf_maxiter",
        ],
    )


def make_optimize_tool(config: AppConfig | None = None) -> Tool:
    return _make_tool(
        config,
        operation="Opt",
        name="optimize_geometry",
        display_name="几何优化",
        description=(
            "Optimize a registered geometry and return its converged final geometry and energy."
        ),
        parameter_model=OptimizeParameters,
        output_ports={
            "optimized_geometry": "molecular_geometry",
            "energy_data": "energy_data",
        },
        results={"opt_final_electronic_energy": "Eh", "optimized_geometry": "molecular_geometry"},
        result_properties={
            "opt_final_electronic_energy": "electronic_energy",
            "optimized_geometry": "molecular_geometry",
            "energy_data": "energy_data",
        },
        result_metadata={
            "opt_final_electronic_energy": {
                "label": "优化后的电子能",
                "description": "几何优化末态的电子能，不含零点能和热校正",
                "caveat": "本次仅完成几何优化，尚未进行频率验证",
            },
            "optimized_geometry": {
                "label": "优化后的几何",
                "description": "通过几何优化收敛检查的输出结构",
                "caveat": "频率稳定性、全局最低点和热力学性质未验证",
            },
            "energy_data": {
                "label": "优化末态电子能数据",
                "description": (
                    "含优化末态数值、方法、电子态、末态结构哈希和生成来源的结构化能量数据"
                ),
            },
        },
        request_parameters=[
            "method_profile",
            "environment",
            "charge",
            "multiplicity",
            "scf_maxiter",
            "geom_maxiter",
        ],
    )


def make_frequency_tool(config: AppConfig | None = None) -> Tool:
    return _make_tool(
        config,
        operation="Freq",
        name="frequency",
        display_name="频率计算",
        description=(
            "Run an ORCA vibrational frequency calculation on the registered input geometry; "
            "this Tool does not optimize the geometry."
        ),
        parameter_model=FrequencyParameters,
        output_ports={"hessian": "orca_hessian"},
        results={"vibrational_frequencies": "frequency"},
        result_properties={
            "hessian": "vibrational_hessian",
            "vibrational_frequencies": "frequency",
            "frequency_complete": "frequency_complete",
            "local_minimum_supported": "local_minimum_supported",
        },
        result_metadata={
            "hessian": {
                "label": "振动 Hessian 文件",
                "description": "与本次输入几何和频率结果绑定的已验证 Hessian 文件",
            },
            "vibrational_frequencies": {
                "label": "振动频率",
                "description": "ORCA 本次 Hessian 计算的有符号频率，单位 cm⁻¹",
                "caveat": "负频率会保留；频率本身不证明全局最低点或热力学自由能",
            },
            "frequency_complete": {
                "label": "完整频率检查",
                "description": "频率模式和 Hessian 均完整且与输入几何匹配",
            },
            "local_minimum_supported": {
                "label": "局部极小值检查",
                "description": "在当前适用检查范围内没有负频率并支持局部极小值判断",
                "caveat": "该检查不证明全局最低点",
            },
        },
        scientific_checks={
            "frequency_complete": (
                "Complete finite frequency modes and matching Hessian for this input geometry."
            ),
            "local_minimum_supported": (
                "A current successful optimized geometry with complete frequencies "
                "and no negative vibrational modes after ORCA's supported external-mode check."
            ),
        },
        scientific_check_input_ports={
            "frequency_complete": "geometry",
            "local_minimum_supported": "geometry",
        },
        request_parameters=[
            "method_profile",
            "environment",
            "charge",
            "multiplicity",
            "scf_maxiter",
        ],
    )


def _make_tool(
    config: AppConfig | None,
    *,
    operation: str,
    name: str,
    display_name: str,
    description: str,
    parameter_model: type[OrcaParameters],
    output_ports: dict[str, str],
    results: dict[str, str],
    result_properties: dict[str, ResultProperty],
    result_metadata: dict[str, dict[str, str]],
    scientific_checks: dict[str, str] | None = None,
    scientific_check_input_ports: dict[str, str] | None = None,
    result_check_prerequisites: dict[str, list[str]] | None = None,
    request_parameters: list[str],
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

    tool_holder: dict[str, Tool] = {}

    def prepare(step: Step, context: Any) -> ToolPreparation:
        tool = tool_holder["tool"]
        original_parameters = dict(step.parameters)
        checked = tool.validate_parameters(original_parameters, allow_deferred=True)
        supplied = {name: checked[name] for name in original_parameters if name in checked}
        request = context.get("request")
        if not isinstance(request, Request):
            raise TypeError("ORCA parameter preparation needs a Request")
        pending = context.get("pending_parameters")
        locked = bool(context.get("parameters_locked"))
        if locked or (
            isinstance(pending, dict)
            and pending.get("step_id") == step.id
            and pending.get("parameters") == step.parameters
            and pending.get("parameter_sources")
        ):
            validated = tool.validate_parameters(step.parameters)
            _validate_orca_parameters(validated, operation)
            return ToolPreparation(
                step=step.model_copy(update={"parameters": validated}),
                parameter_sources=(
                    dict(pending.get("parameter_sources", {})) if isinstance(pending, dict) else {}
                ),
            )
        resolution = resolve_parameters(
            {
                **request.explicit_parameters,
                **(
                    next(
                        (
                            item.parameters
                            for item in request.requirements
                            if item.id == step.requirement_id
                        ),
                        {},
                    )
                ),
            },
            dict(context.get("structure_facts", {})),
            supplied,
            dict(context.get("defaults", {})),
            user_modifications={
                **request.user_modifications,
                **request.user_modifications_by_requirement.get(step.requirement_id or "", {}),
            },
            parameter_fields=tuple(parameter_model.model_fields),
        )
        if resolution.missing_fields:
            _validate_orca_parameters(resolution.effective_parameters, operation)
            return ToolPreparation(
                step=step.model_copy(update={"parameters": supplied}),
                missing_fields=tuple(resolution.missing_fields),
                parameter_sources=dict(resolution.parameter_sources),
                question="Please provide the missing electronic state parameters.",
            )
        validated = tool.validate_parameters(resolution.effective_parameters)
        _validate_orca_parameters(validated, operation)
        prepared = Step.model_validate(
            {**step.model_dump(mode="python"), "parameters": validated}, strict=True
        )
        return ToolPreparation(
            step=prepared,
            parameter_sources=dict(resolution.parameter_sources),
        )

    def repair_options(run: Run, step: Step, result: Result) -> list[RepairOption]:
        return applicable_repairs(run, step, result) + applicable_scf_repair(run, step, result)

    def apply_repair(
        option: RepairOption,
        run: Run,
        step: Step,
        result: Result,
        proposal: Any,
    ) -> tuple[Step, dict[str, Any]]:
        aliases = dict(option.input_aliases)
        selected_aliases = list(proposal.get("input_aliases", []))
        if any(alias not in aliases for alias in selected_aliases):
            raise ValueError("repair proposal selected an input alias not offered by the Tool")
        candidate_id = aliases[selected_aliases[0]] if selected_aliases else None
        from bg6022.orca.repair_rules import validate_repair_option

        return validate_repair_option(
            option,
            run=run,
            step=step,
            result=result,
            requested_action=str(proposal.get("option_id", "")),
            requested_patch=dict(proposal.get("parameters", {})),
            requested_candidate_id=candidate_id,
            evidence_refs=list(proposal.get("evidence_refs", [])),
        )

    tool = Tool(
        name=name,
        display_name=display_name,
        description=description,
        operations=[operation],
        parameter_model=parameter_model.__name__,
        parameter_schema=parameter_model.model_json_schema(),
        parameter_type=parameter_model,
        input_ports={"geometry": "molecular_geometry"},
        output_ports=output_ports,
        results=results,
        result_properties=result_properties,
        result_metadata=result_metadata,
        scientific_checks=scientific_checks or {},
        scientific_check_input_ports=scientific_check_input_ports or {},
        result_check_prerequisites=result_check_prerequisites or {},
        success_conditions=(
            [
                "normal ORCA termination",
                "exit code 0",
                "SCF convergence",
                "unchanged execution inputs",
                "complete frequency modes",
                "matching Hessian bound to the input geometry",
            ]
            if operation == "Freq"
            else [
                "normal ORCA termination",
                "exit code 0",
                "SCF convergence",
                "selected finite final electronic energy",
                "unchanged execution inputs",
            ]
            + (["optimization convergence", "final geometry binding"] if operation == "Opt" else [])
        ),
        repair_capabilities={
            "Opt": ["restart_optimization", "increase_scf_maxiter"],
            "SP": ["increase_scf_maxiter"],
            "Freq": [],
        }[operation],
        repair_parameter_fields={
            "Opt": {
                "restart_optimization": ["geom_maxiter"],
                "increase_scf_maxiter": ["scf_maxiter"],
            },
            "SP": {"increase_scf_maxiter": ["scf_maxiter"]},
            "Freq": {},
        }[operation],
        repair_parameter_limits={
            "Opt": {
                "restart_optimization": {"geom_maxiter": 1000},
                "increase_scf_maxiter": {"scf_maxiter": 1000},
            },
            "SP": {"increase_scf_maxiter": {"scf_maxiter": 1000}},
            "Freq": {},
        }[operation],
        repair_input_aliases={
            "Opt": {"restart_optimization": ["last_complete_geometry"]},
            "SP": {},
            "Freq": {},
        }[operation],
        requires_compute_permission=True,
        execution_budget="electronic_structure",
        deferred_parameters=["charge", "multiplicity"],
        request_parameters=request_parameters,
        geometry_output_input_ports=(
            {"optimized_geometry": "geometry"} if operation == "Opt" else {}
        ),
        repair_capabilities_function=lambda parameters: _profile_repair_capabilities(
            parameters,
            operation=operation,
            declared={
                "Opt": ["restart_optimization", "increase_scf_maxiter"],
                "SP": ["increase_scf_maxiter"],
                "Freq": [],
            }[operation],
        ),
        preparation_function=prepare,
        repair_options_function=repair_options,
        apply_repair_function=apply_repair,
        result_validation_function=lambda run, step, result: _validate_orca_result(
            config, operation, run, step, result
        ),
        preflight_function=(
            (lambda: validate_execution_environment(config)) if config is not None else None
        ),
        execute_function=execute if config is not None else None,
    )
    tool_holder["tool"] = tool
    return tool


def _validate_orca_result(
    config: AppConfig | None, operation: str, run: Run, step: Step, result: Result
) -> bool:
    if config is None:
        return True
    if operation == "Freq":
        return _validate_frequency_result(config, run, step, result)
    return _validate_energy_result(config, operation, run, step, result)


def _validate_energy_result(
    config: AppConfig, operation: str, run: Run, step: Step, result: Result
) -> bool:
    required_checks = [
        "runner_succeeded",
        "exit_code_zero",
        "process_tree_empty",
        "normal_termination",
        "stdout_valid_utf8",
        "stdout_within_size_limit",
        "stderr_within_size_limit",
        "scf_converged",
        "input_hashes_match",
        "final_energy_selected",
        "finite_final_energy",
    ]
    if operation == "Opt":
        required_checks.extend(
            [
                "optimization_converged",
                "output_geometry_present",
                "stdout_geometry_present",
                "geometry_consistent",
            ]
        )
    if any(result.checks.get(name) is not True for name in required_checks):
        return False

    energy_name = {
        "SP": "sp_electronic_energy",
        "Opt": "opt_final_electronic_energy",
    }.get(operation)
    if energy_name is None:
        return False
    energy_value = result.values.get(energy_name)
    if not is_compatible_value(energy_value, "Eh"):
        return False

    energy_artifact_id = result.output_ports.get("energy_data")
    if not isinstance(energy_artifact_id, str) or energy_artifact_id not in result.artifact_ids:
        return False
    try:
        energy_artifact = find_artifact(run, energy_artifact_id)
        if (
            energy_artifact.run_id != run.id
            or energy_artifact.step_id != step.id
            or energy_artifact.attempt != result.attempt
            or energy_artifact.artifact_type != "energy_data"
            or energy_artifact.role != "verified_energy_data"
        ):
            return False
        payload = json.loads(
            artifact_path(config.data_root_path, run, energy_artifact).read_text(encoding="utf-8")
        )
        parameters = step.parameters
        profile = get_profile(str(parameters["method_profile"]))
        geometry_source = payload.get("geometry") if isinstance(payload, dict) else None
        if not isinstance(geometry_source, dict):
            return False
        geometry_id = geometry_source.get("artifact_id")
        expected_geometry_id = (
            result.input_bindings.get("geometry")
            if operation == "SP"
            else result.output_ports.get("optimized_geometry")
        )
        if not isinstance(geometry_id, str) or geometry_id != expected_geometry_id:
            return False
        geometry_artifact = find_artifact(run, geometry_id)
        expected_geometry_role = "optimized_geometry" if operation == "Opt" else None
        if (
            geometry_artifact.run_id != run.id
            or geometry_artifact.artifact_type != "molecular_geometry"
            or geometry_artifact.sha256 != geometry_source.get("sha256")
            or (
                expected_geometry_role is not None
                and geometry_artifact.role != expected_geometry_role
            )
            or (operation == "SP" and geometry_id not in result.input_artifact_ids)
            or (operation == "Opt" and geometry_id not in result.artifact_ids)
        ):
            return False
        artifact_path(config.data_root_path, run, geometry_artifact)

        expected_value = energy_value["value"]
        observation = payload.get("observation") if isinstance(payload, dict) else None
        expected_fields = {
            "schema": "bg6022.energy_data.v1",
            "property": "electronic_energy",
            "value": expected_value,
            "unit": "Eh",
            "method_profile": profile.name,
            "method_keyword": profile.orca_keyword,
            "operation": operation,
            "charge": parameters.get("charge"),
            "multiplicity": parameters.get("multiplicity"),
            "source": {"step_id": step.id, "attempt": result.attempt},
        }
        if any(payload.get(name) != value for name, value in expected_fields.items()):
            return False
        if (
            not isinstance(observation, dict)
            or observation != energy_value
            or not math.isfinite(float(expected_value))
            or energy_artifact.metadata.get("geometry_sha256") != geometry_artifact.sha256
            or energy_artifact.metadata.get("method_profile") != profile.name
            or energy_artifact.metadata.get("operation") != operation
            or energy_artifact.metadata.get("charge") != parameters.get("charge")
            or energy_artifact.metadata.get("multiplicity") != parameters.get("multiplicity")
            or energy_artifact.metadata.get("unit") != "Eh"
        ):
            return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _validate_frequency_result(config: AppConfig, run: Run, step: Step, result: Result) -> bool:
    required_checks = (
        "runner_succeeded",
        "exit_code_zero",
        "process_tree_empty",
        "normal_termination",
        "stdout_valid_utf8",
        "stdout_within_size_limit",
        "stderr_within_size_limit",
        "scf_converged",
        "input_hashes_match",
        "frequency_section_complete",
        "frequency_values_finite",
        "frequency_mode_indices_match_hessian",
        "hessian_present",
        "hessian_valid",
    )
    if any(result.checks.get(name) is not True for name in required_checks):
        return False
    if not is_compatible_value(result.values.get("vibrational_frequencies"), "frequency"):
        return False
    input_id = result.input_bindings.get("geometry")
    check = result.scientific_checks.get("frequency_complete")
    if not isinstance(input_id, str) or check is None or check.status != "passed":
        return False
    try:
        input_geometry = find_artifact(run, input_id)
        if (
            input_geometry.artifact_type != "molecular_geometry"
            or check.input_geometry_sha256 != input_geometry.sha256
        ):
            return False
        artifact_path(config.data_root_path, run, input_geometry)
        hessian_id = result.output_ports.get("hessian")
        if not isinstance(hessian_id, str) or hessian_id not in result.artifact_ids:
            return False
        hessian = find_artifact(run, hessian_id)
        if (
            hessian.run_id != run.id
            or hessian.step_id != step.id
            or hessian.attempt != result.attempt
            or hessian.artifact_type != "orca_hessian"
            or hessian.role != "verified_hessian"
        ):
            return False
        artifact_path(config.data_root_path, run, hessian)
    except (OSError, ValueError):
        return False
    return True


def _validate_orca_parameters(parameters: dict[str, Any], operation: str) -> None:
    profile = get_profile(str(parameters["method_profile"]))
    if parameters["environment"] not in profile.supported_environments:
        raise ValueError(
            f"environment {parameters['environment']!r} is not implemented for {profile.name!r}"
        )
    if operation not in profile.supported_operations:
        raise ValueError(f"operation {operation!r} is not implemented for {profile.name!r}")


def _profile_repair_capabilities(
    parameters: dict[str, Any], *, operation: str, declared: list[str]
) -> list[str]:
    try:
        profile = get_profile(str(parameters.get("method_profile", "r2scan3c")))
    except ValueError:
        return []
    allowed = set(profile.repair_options_by_operation.get(operation, ()))
    return [action for action in declared if action in allowed]


def execute_orca_step(
    config: AppConfig,
    *,
    step: Step,
    run: Run,
    cancel: Event,
    operation: str,
    parameter_model: type[OrcaParameters],
) -> Result:
    _check_execution_contract(config, run, step)
    parameters = parameter_model.model_validate(step.parameters, strict=True)
    profile = get_profile(parameters.method_profile)
    if parameters.environment not in profile.supported_environments:
        raise ValueError(
            f"environment {parameters.environment!r} is not implemented for {profile.name!r}"
        )
    if operation not in profile.supported_operations:
        raise ValueError(f"operation {operation!r} is not implemented for {profile.name!r}")
    reference = step.inputs.get("geometry")
    if reference is None:
        raise ValueError("ORCA Tool requires a geometry input reference")
    geometry_artifact = _resolve_geometry_reference(
        config,
        run,
        reference,
        allow_restart_candidate=_authorized_restart_candidate(run, step, reference),
    )
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
    hessian_path = attempt_dir / "input.hess"
    hessian_path_present = hessian_path.is_file() and not hessian_path.is_symlink()
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
        hessian=hessian_path if operation == "Freq" and hessian_path_present else None,
        expected_atom_count=geometry.atom_count if operation == "Freq" else None,
        input_hashes_match=input_hashes_match,
        input_hash_error=input_hash_error,
        max_output_bytes=int(run.resources["output_limit_bytes"]),
        effective_geom_maxiter=getattr(parameters, "geom_maxiter", None),
        effective_scf_maxiter=parameters.scf_maxiter,
    )
    outcome = evaluate_success(facts, operation=operation)
    if profile_versions := get_profile(parameters.method_profile).validated_orca_versions:
        version_supported = facts.orca_version in profile_versions
        outcome = CheckOutcome(
            success=outcome.success and version_supported,
            checks={**outcome.checks, "orca_version_supported": version_supported},
            failure_category=(
                None
                if outcome.success and version_supported
                else "unsupported_orca_version"
                if not version_supported
                else outcome.failure_category
            ),
        )
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

    hessian_artifact = None
    if operation == "Freq" and hessian_path_present:
        hessian_artifact = register_file_artifact(
            config.data_root_path,
            run,
            hessian_path,
            artifact_type="orca_hessian",
            role="verified_hessian" if outcome.success else "raw_hessian",
            source=f"{step.id}/attempt-{attempt:02d}/input.hess",
            step_id=step.id,
            attempt=attempt,
            metadata={
                "validated_for_input_geometry_sha256": (
                    expected_geometry_sha if facts.hessian_valid is True else None
                ),
                "dimension": facts.hessian_dimension,
                "validation_error": facts.hessian_error,
            },
        )
        artifact_ids.append(hessian_artifact.id)

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

    if operation == "Freq" and outcome.success and hessian_artifact is not None:
        output_ports["hessian"] = hessian_artifact.id

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
            "hessian": str(hessian_path),
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
    if outcome.success:
        energy_values = _electronic_energy_value(operation, observed)
        values.update(energy_values)
        energy_result_name = {
            "SP": "sp_electronic_energy",
            "Opt": "opt_final_electronic_energy",
        }.get(operation)
        if energy_result_name is not None and energy_result_name in energy_values:
            if operation == "Opt":
                optimized_geometry_id = output_ports.get("optimized_geometry")
                energy_geometry = (
                    find_artifact(run, optimized_geometry_id)
                    if optimized_geometry_id is not None
                    else None
                )
                if energy_geometry is None or energy_geometry.role != "optimized_geometry":
                    raise ValueError(
                        "successful Opt result has no verified optimized geometry for energy_data"
                    )
            else:
                energy_geometry = geometry_artifact
            energy_data = _energy_data_payload(
                energy_values[energy_result_name],
                operation=operation,
                parameters=parameters,
                geometry_artifact=energy_geometry,
                step=step,
                attempt=attempt,
            )
            energy_artifact = register_bytes_artifact(
                config.data_root_path,
                run,
                json.dumps(
                    energy_data,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n",
                artifact_type="energy_data",
                role="verified_energy_data",
                source=f"{step.id}/attempt-{attempt:02d}/energy_data.json",
                extension=".json",
                step_id=step.id,
                attempt=attempt,
                metadata={
                    "property": "electronic_energy",
                    "unit": "Eh",
                    "method_profile": parameters.method_profile,
                    "operation": operation,
                    "charge": parameters.charge,
                    "multiplicity": parameters.multiplicity,
                    "geometry_sha256": energy_geometry.sha256,
                },
            )
            artifact_ids.append(energy_artifact.id)
            output_ports["energy_data"] = energy_artifact.id
    scientific_checks: dict[str, ScientificCheckResult] = {}
    if operation == "Freq":
        section = facts.frequency_section
        modes = (
            []
            if section is None
            else [
                {"index": mode.index, "value": mode.value, "unit": mode.unit}
                for mode in section.modes
            ]
        )
        if outcome.success and section is not None:
            values["vibrational_frequencies"] = {
                "modes": modes,
                "unit": "cm^-1",
                "scaling_factor": section.scaling_factor,
                "scaling_applied": section.scaling_applied,
                "complete": section.complete,
            }
            frequency_status = "passed"
            frequency_reason = None
        else:
            frequency_status = "unverified"
            frequency_reason = (
                section.error
                if section is not None and section.error
                else "frequency success checks did not pass"
            )
        scientific_checks["frequency_complete"] = ScientificCheckResult(
            status=frequency_status,
            input_geometry_sha256=expected_geometry_sha,
            conditions={
                "mode_count": len(modes),
                "expected_mode_count": 3 * geometry.atom_count,
                "hessian_valid": facts.hessian_valid is True,
                "scaling_factor": None if section is None else section.scaling_factor,
                "scaling_applied": False if section is None else section.scaling_applied,
            },
            reason=frequency_reason,
        )
        scientific_checks["local_minimum_supported"] = _local_minimum_check(
            config,
            run,
            step,
            geometry_artifact,
            geometry,
            parameters,
            section,
            outcome.success,
            expected_geometry_sha,
        )
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
        scientific_checks=scientific_checks,
        diagnostics=diagnostics,
        artifact_ids=artifact_ids,
        output_ports=output_ports,
        input_bindings={"geometry": geometry_artifact.id},
        input_artifact_ids=[geometry_artifact.id],
        attempt_relative_path=f"{step.id}/attempt-{attempt:02d}",
    )


def _check_execution_contract(config: AppConfig, run: Run, step: Step) -> None:
    if not run.execution_permission:
        raise PermissionError("Run does not have explicit execution permission")
    if run.accepted_execution_sha256 is None:
        raise PermissionError("Run has no accepted execution fingerprint")
    if run.accepted_snapshot:
        try:
            accepted_plan = Plan.model_validate(run.accepted_snapshot["plan"], strict=True)
            accepted_resources = dict(run.accepted_snapshot["resources"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Run accepted snapshot is invalid") from error
        if run.resources != accepted_resources:
            raise ValueError("Run resources differ from the accepted execution budget")
        accepted_request = run.accepted_snapshot.get("request")
        if accepted_request is not None and accepted_request != run.request.model_dump(mode="json"):
            raise ValueError("Run Request differs from the accepted scientific request")
        accepted_budgets = run.accepted_snapshot.get("budgets")
        if isinstance(accepted_budgets, dict) and dict(run.budget) != accepted_budgets:
            raise ValueError("Run repair budget differs from the accepted budget")
        expected = execution_fingerprint(
            accepted_plan,
            accepted_resources,
            run.artifact_index,
            snapshot=run.accepted_snapshot,
        )
        current_allowed = _plan_fingerprint(run.plan)
        base_allowed = _plan_fingerprint(accepted_plan)
        if expected != run.accepted_execution_sha256:
            raise ValueError("Run execution fingerprint does not match the accepted Plan or inputs")
        if current_allowed != base_allowed:
            derived_plan = _reconstruct_repair_plan(config, run, accepted_plan)
            if current_allowed != _plan_fingerprint(derived_plan):
                raise ValueError("Run Plan is not derived from the accepted repair chain")
    else:
        expected = execution_fingerprint(run.plan, run.resources, run.artifact_index)
        if expected != run.accepted_execution_sha256:
            raise ValueError("Run execution fingerprint does not match the current Plan or inputs")
    planned = next((item for item in run.plan.steps if item.id == step.id), None)
    if planned is None:
        raise ValueError(f"step is not part of the current Plan: {step.id}")
    if planned.model_dump(mode="json") != step.model_dump(mode="json"):
        raise ValueError(f"step {step.id} differs from the accepted Plan")


def _reconstruct_repair_plan(config: AppConfig, run: Run, accepted_plan: Plan) -> Plan:
    """Derive the current Plan from recorded repair facts, not trusted hashes."""

    from bg6022.tools.registry import build_registry

    snapshot = run.accepted_snapshot
    try:
        start = int(snapshot.get("repair_record_start", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("accepted repair record start is invalid") from error
    if start < 0 or start > len(run.repair_records):
        raise ValueError("accepted repair record start is outside the Run history")
    plan = accepted_plan
    registry = build_registry(config)
    for record in run.repair_records[start:]:
        if record.get("validated") is not True:
            raise ValueError("repair chain contains an unvalidated record")
        step_id = record.get("failed_step_id")
        step = next((item for item in plan.steps if item.id == step_id), None)
        if step is None:
            raise ValueError("repair chain targets a step outside the accepted Plan")
        if record.get("old_parameters") != step.parameters:
            raise ValueError("repair chain old parameters do not match the preceding Step")
        patch = record.get("parameter_patch")
        new_parameters = record.get("new_parameters")
        action = record.get("action")
        if not isinstance(patch, dict) or not isinstance(new_parameters, dict):
            raise ValueError("repair chain parameters are malformed")
        if any(type(value) is not int for value in patch.values()):
            raise ValueError("repair chain iteration patches must be integers")
        expected_parameters = {**step.parameters, **patch}
        if new_parameters != expected_parameters:
            raise ValueError("repair chain new parameters do not match its patch")
        failed_result = _load_repair_result(config, run, record)
        if failed_result.status != "failed":
            raise ValueError("repair chain refers to a non-failed Result")
        if failed_result.step_fingerprint != _step_fingerprint(step):
            raise ValueError("repair chain failed Result is stale for the preceding Step")
        if (
            record.get("failed_attempt") is not None
            and record.get("failed_attempt") != failed_result.attempt
        ):
            raise ValueError("repair chain failed attempt does not match its Result")
        _validate_record_scope(run, step, action, patch)
        applicable = applicable_repairs(run, step, failed_result) + applicable_scf_repair(
            run, step, failed_result
        )
        matching = next((item for item in applicable if item.action == action), None)
        if (
            matching is None
            or matching.parameter_patch != patch
            or matching.candidate_artifact_id != record.get("candidate_artifact_id")
            or set(matching.evidence_refs) != set(record.get("evidence_refs", []))
        ):
            raise ValueError("repair chain is not reproducible from the failed Result facts")

        new_inputs = dict(step.inputs)
        candidate_id = record.get("candidate_artifact_id")
        if action == "restart_optimization":
            if not isinstance(candidate_id, str):
                raise ValueError("optimization restart record has no candidate artifact")
            candidate = find_artifact(run, candidate_id)
            if (
                candidate.role != "restart_candidate"
                or candidate.step_id != step.id
                or candidate.id != candidate_id
                or candidate.sha256 != record.get("candidate_sha256")
            ):
                raise ValueError("repair candidate is not bound to the recorded failed attempt")
            if (
                failed_result.status != "failed"
                or candidate.id not in failed_result.artifact_ids
                or candidate.attempt != failed_result.attempt
            ):
                raise ValueError("repair candidate is not present in the recorded failed Result")
            new_inputs["geometry"] = InputReference(artifact_id=candidate.id)
        elif action == "increase_scf_maxiter":
            if candidate_id is not None:
                raise ValueError("SCF repair record cannot contain a geometry candidate")
        else:
            raise ValueError(f"unsupported repair action in chain: {action}")

        replacement = Step.model_validate(
            {
                **step.model_dump(mode="python"),
                "parameters": new_parameters,
                "inputs": new_inputs,
            },
            strict=True,
        )
        plan = Plan.model_validate(
            {
                **plan.model_dump(mode="python"),
                "steps": [replacement if item.id == step.id else item for item in plan.steps],
            },
            strict=True,
        )
        # The repair removes the failed Step's dependency on its previous
        # geometry producer. Reproduce the Agent's canonical topological order
        # before checking the recorded derived-Plan fingerprint.
        plan = registry.validate_plan(plan)
        plan = Plan.model_validate(
            {**plan.model_dump(mode="python"), "revision": plan.revision + 1},
            strict=True,
        )
        if record.get("derived_plan_sha256") != _plan_fingerprint(plan):
            raise ValueError("repair chain derived Plan hash is not reproducible")
    return plan


def _validate_record_scope(run: Run, step: Step, action: Any, patch: dict[str, Any]) -> None:
    snapshot = run.accepted_snapshot
    scope = snapshot.get("repair_scope") if isinstance(snapshot, dict) else None
    if not isinstance(scope, dict):
        raise ValueError("accepted repair scope is missing")
    steps = scope.get("steps")
    step_scope = steps.get(step.id) if isinstance(steps, dict) else None
    actions = step_scope.get("actions") if isinstance(step_scope, dict) else None
    action_scope = actions.get(action) if isinstance(actions, dict) else None
    if not isinstance(action_scope, dict):
        raise ValueError("repair action is outside the accepted scope")
    fields = action_scope.get("fields")
    maximum = action_scope.get("maximum")
    if not isinstance(fields, list) or set(patch) != set(fields) or type(maximum) is not int:
        raise ValueError("repair fields are outside the accepted scope")
    if any(type(value) is not int or value < 1 or value > maximum for value in patch.values()):
        raise ValueError("repair value is outside the accepted numeric scope")


def _load_repair_result(config: AppConfig, run: Run, record: dict[str, Any]) -> Result:
    relative = record.get("failed_result_path")
    if not isinstance(relative, str):
        raise ValueError("repair record has no failed Result path")
    root = run_directory(config.data_root_path, run.id).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or path.name != "result.json":
        raise ValueError("repair Result path escapes the Run directory")
    try:
        result = Result.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("recorded repair Result cannot be read") from error
    if result.run_id != run.id or result.step_id != record.get("failed_step_id"):
        raise ValueError("recorded repair Result belongs to a different Run or Step")
    if result.attempt_relative_path + "/result.json" != relative:
        raise ValueError("repair Result path does not match its recorded attempt")
    return result


def _plan_fingerprint(plan: Plan) -> str:
    import hashlib
    import json

    encoded = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _resolve_geometry_reference(
    config: AppConfig,
    run: Run,
    reference: Any,
    *,
    allow_restart_candidate: bool = False,
):
    if reference.artifact_id is not None:
        artifact = find_artifact(run, reference.artifact_id)
    else:
        result_relative = run.current_results.get(reference.step_id)
        if result_relative is None:
            raise ValueError(
                "upstream port "
                f"{reference.step_id}.{reference.port} has not produced a successful artifact"
            )
        root = run_directory(config.data_root_path, run.id).resolve()
        result_path = (root / result_relative).resolve()
        if root not in result_path.parents or result_path.name != "result.json":
            raise ValueError("current result path is outside the Run directory")
        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"current result for {reference.step_id} cannot be read") from error
        if result_payload.get("status") != "succeeded":
            raise ValueError(f"current result for {reference.step_id} is not successful")
        upstream = next((item for item in run.plan.steps if item.id == reference.step_id), None)
        if upstream is None or result_payload.get("step_fingerprint") != _step_fingerprint(
            upstream
        ):
            raise ValueError(f"current result for {reference.step_id} is stale")
        artifact_id = result_payload.get("output_ports", {}).get(reference.port)
        if not artifact_id or artifact_id not in result_payload.get("artifact_ids", []):
            raise ValueError(
                "upstream port "
                f"{reference.step_id}.{reference.port} has not produced a successful artifact"
            )
        artifact = find_artifact(run, artifact_id)
        if artifact.step_id != reference.step_id or artifact.attempt != result_payload.get(
            "attempt"
        ):
            raise ValueError("current result port is not bound to its successful producing attempt")
    if artifact.artifact_type != "molecular_geometry":
        raise ValueError(f"artifact {artifact.id} is not a molecular geometry")
    if artifact.role == "restart_candidate" and not allow_restart_candidate:
        raise ValueError("restart_candidate is not a successful geometry input")
    return artifact


def _authorized_restart_candidate(run: Run, step: Step, reference: Any) -> bool:
    if reference.artifact_id is None:
        return False
    return any(
        record.get("validated") is True
        and record.get("action") == "restart_optimization"
        and record.get("failed_step_id") == step.id
        and record.get("candidate_artifact_id") == reference.artifact_id
        for record in run.repair_records
    )


def _step_fingerprint(step: Step) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        step.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _local_minimum_check(
    config: AppConfig,
    run: Run,
    frequency_step: Step,
    geometry_artifact: Any,
    geometry: ParsedGeometry,
    parameters: OrcaParameters,
    section: Any,
    frequency_success: bool,
    geometry_sha256: str,
) -> ScientificCheckResult:
    conditions: dict[str, Any] = {
        "input_geometry_sha256": geometry_sha256,
        "mode_count": 0 if section is None else len(section.modes),
        "mode_sign_rule": (
            "vibrational modes after classified external zero modes must be nonnegative"
        ),
        "verified_optimized_geometry": False,
    }
    if section is not None:
        conditions["scaling_factor"] = section.scaling_factor
        conditions["scaling_applied"] = section.scaling_applied
    if not frequency_success or section is None or not section.complete:
        return ScientificCheckResult(
            status="unverified",
            input_geometry_sha256=geometry_sha256,
            conditions=conditions,
            reason="complete frequency modes and a matching Hessian were not verified",
        )
    if not _is_current_optimized_geometry(config, run, geometry_artifact, parameters):
        return ScientificCheckResult(
            status="unverified",
            input_geometry_sha256=geometry_sha256,
            conditions=conditions,
            reason=(
                "the frequency input is not a current successful optimized_geometry from this Run"
            ),
        )
    conditions["verified_optimized_geometry"] = True
    external_mode_count = _external_mode_count(geometry)
    if external_mode_count is None or external_mode_count >= len(section.modes):
        return ScientificCheckResult(
            status="unverified",
            input_geometry_sha256=geometry_sha256,
            conditions=conditions,
            reason="this geometry has no supported vibrational-mode classification",
        )
    external_modes = section.modes[:external_mode_count]
    external_mode_limit = 1.0
    conditions.update(
        {
            "external_mode_count": external_mode_count,
            "external_mode_indices": [mode.index for mode in external_modes],
            "external_mode_values_cm1": [mode.value for mode in external_modes],
            "external_mode_tolerance_cm1": external_mode_limit,
        }
    )
    if any(abs(mode.value) > external_mode_limit for mode in external_modes):
        return ScientificCheckResult(
            status="unverified",
            input_geometry_sha256=geometry_sha256,
            conditions=conditions,
            reason=(
                "the leading ORCA translation/rotation modes do not match the supported "
                "near-zero layout"
            ),
        )
    vibrational_modes = section.modes[external_mode_count:]
    negative = [mode for mode in vibrational_modes if mode.value < 0.0]
    conditions["vibrational_mode_indices"] = [mode.index for mode in vibrational_modes]
    conditions["negative_mode_indices"] = [mode.index for mode in negative]
    if negative:
        values = ", ".join(f"{mode.value:g} cm^-1" for mode in negative[:6])
        suffix = " …" if len(negative) > 6 else ""
        return ScientificCheckResult(
            status="not_met",
            input_geometry_sha256=geometry_sha256,
            conditions=conditions,
            reason=f"negative frequencies were observed: {values}{suffix}",
        )
    return ScientificCheckResult(
        status="passed",
        input_geometry_sha256=geometry_sha256,
        conditions=conditions,
        reason=(
            "all classified vibrational modes are nonnegative for the verified optimized "
            "input geometry; "
            "this does not establish a global minimum"
        ),
    )


def _electronic_energy_value(
    operation: str, observed: EnergyObservation | None
) -> dict[str, dict[str, Any]]:
    """Expose energy only under the operation's declared, scientifically precise field."""

    key = {
        "Opt": "opt_final_electronic_energy",
        "SP": "sp_electronic_energy",
    }.get(operation)
    if key is None or observed is None or observed.value is None:
        return {}
    return {
        key: {
            "value": observed.value,
            "unit": "Eh",
            "token": observed.token,
            "source_line": observed.line,
        }
    }


def _energy_data_payload(
    energy: dict[str, Any],
    *,
    operation: str,
    parameters: OrcaParameters,
    geometry_artifact: Any,
    step: Step,
    attempt: int,
) -> dict[str, Any]:
    """Bind one verified energy observation to its method, state, and geometry."""

    profile = get_profile(parameters.method_profile)
    return {
        "schema": "bg6022.energy_data.v1",
        "property": "electronic_energy",
        "value": energy["value"],
        "unit": "Eh",
        "observation": dict(energy),
        "method_profile": profile.name,
        "method_keyword": profile.orca_keyword,
        "operation": operation,
        "charge": parameters.charge,
        "multiplicity": parameters.multiplicity,
        "geometry": {
            "artifact_id": geometry_artifact.id,
            "sha256": geometry_artifact.sha256,
        },
        "source": {"step_id": step.id, "attempt": attempt},
    }


def _external_mode_count(geometry: ParsedGeometry) -> int | None:
    """Classify ORCA's leading rigid-body modes from molecular geometry."""

    if geometry.atom_count <= 0:
        return None
    if geometry.atom_count == 1:
        return 3
    if geometry.atom_count == 2:
        return 5

    origin = geometry.coordinates[0]
    vectors = [
        tuple(coordinate[axis] - origin[axis] for axis in range(3))
        for coordinate in geometry.coordinates[1:]
    ]
    reference = max(vectors, key=lambda vector: sum(value * value for value in vector))
    reference_norm = math.sqrt(sum(value * value for value in reference))
    if reference_norm <= 1e-8:
        return None
    max_cross_norm = 0.0
    for vector in vectors:
        cross = (
            reference[1] * vector[2] - reference[2] * vector[1],
            reference[2] * vector[0] - reference[0] * vector[2],
            reference[0] * vector[1] - reference[1] * vector[0],
        )
        max_cross_norm = max(max_cross_norm, math.sqrt(sum(value * value for value in cross)))
    return 5 if max_cross_norm <= 1e-6 * reference_norm**2 else 6


def _is_current_optimized_geometry(
    config: AppConfig,
    run: Run,
    artifact: Any,
    parameters: OrcaParameters,
) -> bool:
    if (
        artifact.role != "optimized_geometry"
        or artifact.run_id != run.id
        or not isinstance(artifact.step_id, str)
        or artifact.attempt is None
    ):
        return False
    source_step = next((item for item in run.plan.steps if item.id == artifact.step_id), None)
    if source_step is None or source_step.tool != "optimize_geometry":
        return False
    for field in ("method_profile", "environment", "charge", "multiplicity"):
        if source_step.parameters.get(field) != getattr(parameters, field, None):
            return False
    relative = run.current_results.get(source_step.id)
    if not isinstance(relative, str):
        return False
    root = run_directory(config.data_root_path, run.id).resolve()
    result_path = (root / relative).resolve()
    if root not in result_path.parents or result_path.name != "result.json":
        return False
    try:
        result = Result.model_validate(
            json.loads(result_path.read_text(encoding="utf-8")), strict=True
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if (
        result.run_id != run.id
        or result.step_id != source_step.id
        or result.status != "succeeded"
        or result.attempt != artifact.attempt
        or result.step_fingerprint != _step_fingerprint(source_step)
        or result.output_ports.get("optimized_geometry") != artifact.id
        or artifact.id not in result.artifact_ids
    ):
        return False
    return any(
        item.get("step_id") == source_step.id
        and item.get("attempt") == result.attempt
        and item.get("phase") == "finished"
        and item.get("status") == "succeeded"
        and artifact.id in item.get("artifact_ids", [])
        for item in run.attempts
    )


def _lock_failure_facts(message: str) -> ProcessFacts:
    return ProcessFacts(
        status="failed",
        stop_reason="resource_lock",
        process_tree_empty=True,
        stop_confirmed=True,
        exception=message,
    )


__all__ = [
    "FrequencyParameters",
    "OptimizeParameters",
    "OrcaParameters",
    "SinglePointParameters",
    "execute_orca_step",
    "make_frequency_tool",
    "make_optimize_tool",
    "make_single_point_tool",
]
