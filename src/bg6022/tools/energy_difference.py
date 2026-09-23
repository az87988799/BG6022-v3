"""Compare two verified electronic-energy Artifacts on the same structure."""

from __future__ import annotations

import hashlib
import json
import math
import re
from threading import Event
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Result, Run, Step, Tool
from bg6022.orca.profiles import get_profile
from bg6022.output_contracts import is_compatible_value
from bg6022.session import artifact_path, find_artifact, run_directory, save_run
from bg6022.tools.molecule import resolve_artifact_reference

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def make_energy_difference_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, run: Run, cancel: Event) -> Result:
        if config is None:
            raise RuntimeError("tool 'energy_difference' is a description-only Tool")
        return execute_energy_difference(config, step=step, run=run, cancel=cancel)

    return Tool(
        name="energy_difference",
        display_name="方法间电子能差",
        description=(
            "Subtract two verified electronic-energy Artifacts on the same molecular "
            "geometry and electronic state. The result is method-dependent and does not "
            "identify which method is more accurate."
        ),
        input_ports={"energy_a": "energy_data", "energy_b": "energy_data"},
        results={"method_energy_difference": "Eh"},
        result_properties={"method_energy_difference": "method_energy_difference"},
        result_metadata={
            "method_energy_difference": {
                "label": "方法间电子能差",
                "description": "定义为方法 B 电子能减去方法 A 电子能",
                "caveat": "方法之间的差异不表示任一方法更准确",
            }
        },
        success_conditions=[
            "both inputs are verified energy_data Artifacts",
            "both observations use Eh and a registered method profile",
            "methods are distinct",
            "geometry hashes, charge, and multiplicity match",
        ],
        requires_compute_permission=False,
        execution_budget="none",
        execute_function=execute if config is not None else None,
    )


def execute_energy_difference(config: AppConfig, *, step: Step, run: Run, cancel: Event) -> Result:
    attempt = _next_attempt(run, step.id)
    relative = f"{step.id}/attempt-{attempt:02d}"
    (run_directory(config.data_root_path, run.id) / relative).mkdir(parents=True, exist_ok=True)
    input_bindings: dict[str, str] = {}
    status = "failed"
    values: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        if cancel.is_set():
            status = "cancelled"
            diagnostics = {
                "category": "cancelled",
                "reason": "cancelled before energy comparison",
            }
        else:
            energy_artifacts = {}
            for input_name in ("energy_a", "energy_b"):
                reference = step.inputs.get(input_name)
                if reference is None:
                    raise ValueError(f"energy_difference requires {input_name!r} input")
                if reference.step_id is None or reference.port != "energy_data":
                    raise ValueError(
                        f"{input_name} must reference a current successful energy_data output port"
                    )
                artifact = resolve_artifact_reference(
                    config, run, reference, expected_type="energy_data"
                )
                input_bindings[input_name] = artifact.id
                energy_artifacts[input_name] = artifact
            energy_records = []
            for input_name in ("energy_a", "energy_b"):
                artifact = energy_artifacts[input_name]
                path = artifact_path(config.data_root_path, run, artifact)
                payload = json.loads(path.read_text(encoding="utf-8"))
                energy = _validated_energy_data(payload, artifact)
                _validate_current_energy_source(config, run, artifact, energy)
                energy_records.append(energy)

            energy_a, energy_b = energy_records
            _validate_pair(energy_a, energy_b)
            if cancel.is_set():
                status = "cancelled"
                diagnostics = {
                    "category": "cancelled",
                    "reason": "cancelled before energy difference calculation",
                }
            else:
                difference = float(energy_b["value"]) - float(energy_a["value"])
                if not math.isfinite(difference):
                    raise ValueError("method energy difference is not finite")
                values["method_energy_difference"] = {
                    "value": difference,
                    "unit": "Eh",
                    "direction": "B - A",
                    "method_a": energy_a["method_profile"],
                    "method_b": energy_b["method_profile"],
                    "geometry_sha256": energy_a["geometry_sha256"],
                    "charge": energy_a["charge"],
                    "multiplicity": energy_a["multiplicity"],
                    "input_energy_artifact_ids": [
                        input_bindings["energy_a"],
                        input_bindings["energy_b"],
                    ],
                }
                diagnostics = {
                    "category": None,
                    "direction": "B - A",
                    "method_a": energy_a["method_profile"],
                    "method_b": energy_b["method_profile"],
                    "geometry_sha256": energy_a["geometry_sha256"],
                    "charge": energy_a["charge"],
                    "multiplicity": energy_a["multiplicity"],
                }
                status = "succeeded"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        status = "failed"
        diagnostics = {"category": "energy_comparison_failed", "reason": str(error)}

    run.attempts.append(
        {
            "step_id": step.id,
            "attempt": attempt,
            "phase": "finished",
            "status": status,
            "artifact_ids": [],
            "output_ports": {},
            "input_artifact_ids": list(input_bindings.values()),
        }
    )
    save_run(config.data_root_path, run)
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        status=status,  # type: ignore[arg-type]
        values=values,
        diagnostics=diagnostics,
        input_bindings=input_bindings,
        input_artifact_ids=list(input_bindings.values()),
        attempt_relative_path=relative,
    )


def _validated_energy_data(payload: Any, artifact: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != "bg6022.energy_data.v1":
        raise ValueError(f"Artifact {artifact.id} has an unsupported energy_data schema")
    if payload.get("property") != "electronic_energy" or payload.get("unit") != "Eh":
        raise ValueError(f"Artifact {artifact.id} is not an electronic energy in Eh")
    value = payload.get("value")
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"Artifact {artifact.id} has no finite verified energy value")
    observation = payload.get("observation")
    if (
        not isinstance(observation, dict)
        or observation.get("unit") != "Eh"
        or observation.get("value") != value
        or type(observation.get("value")) not in {int, float}
    ):
        raise ValueError(f"Artifact {artifact.id} energy disagrees with its source observation")
    method = payload.get("method_profile")
    if not isinstance(method, str):
        raise ValueError(f"Artifact {artifact.id} has no registered method profile")
    profile = get_profile(method)
    if payload.get("method_keyword") != profile.orca_keyword:
        raise ValueError(f"Artifact {artifact.id} method keyword disagrees with its profile")
    if payload.get("operation") not in {"SP", "Opt"}:
        raise ValueError(f"Artifact {artifact.id} does not come from a supported energy operation")
    if payload["operation"] not in profile.supported_operations:
        raise ValueError(f"Artifact {artifact.id} method does not support its energy operation")
    charge, multiplicity = payload.get("charge"), payload.get("multiplicity")
    if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError(f"Artifact {artifact.id} has an invalid electronic state")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Artifact {artifact.id} has no geometry source")
    geometry_sha256 = geometry.get("sha256")
    if (
        not isinstance(geometry_sha256, str)
        or _SHA256_RE.fullmatch(geometry_sha256) is None
        or geometry_sha256 != artifact.metadata.get("geometry_sha256")
    ):
        raise ValueError(f"Artifact {artifact.id} geometry hash is invalid or inconsistent")
    source = payload.get("source")
    if (
        not isinstance(source, dict)
        or source.get("step_id") != artifact.step_id
        or source.get("attempt") != artifact.attempt
        or type(source.get("attempt")) is not int
    ):
        raise ValueError(f"Artifact {artifact.id} producer attempt is inconsistent")
    expected_metadata = {
        "unit": "Eh",
        "method_profile": profile.name,
        "operation": payload["operation"],
        "charge": charge,
        "multiplicity": multiplicity,
        "geometry_sha256": geometry_sha256,
    }
    if any(artifact.metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"Artifact {artifact.id} provenance metadata is inconsistent")
    return {
        "value": float(value),
        "method_profile": profile.name,
        "geometry_sha256": geometry_sha256,
        "charge": charge,
        "multiplicity": multiplicity,
        "operation": payload["operation"],
        "geometry_artifact_id": geometry.get("artifact_id"),
    }


def _validate_current_energy_source(
    config: AppConfig, run: Run, artifact: Any, energy: dict[str, Any]
) -> None:
    """Bind energy JSON back to its current successful producer Result and geometry."""

    step = next((item for item in run.plan.steps if item.id == artifact.step_id), None)
    relative = run.current_results.get(artifact.step_id or "")
    if step is None or relative is None:
        raise ValueError(f"Artifact {artifact.id} has no current successful producer Step")
    root = run_directory(config.data_root_path, run.id).resolve()
    result_path = (root / relative).resolve()
    if root not in result_path.parents or result_path.name != "result.json":
        raise ValueError("energy producer Result path escapes the Run directory")
    try:
        result = Result.model_validate(
            json.loads(result_path.read_text(encoding="utf-8")), strict=True
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("energy producer Result is invalid") from error
    if (
        relative != f"{result.attempt_relative_path.rstrip('/')}/result.json"
        or result.run_id != run.id
        or result.step_id != step.id
        or result.status != "succeeded"
        or result.attempt != artifact.attempt
        or result.step_fingerprint != _step_fingerprint(step)
        or result.output_ports.get("energy_data") != artifact.id
        or artifact.id not in result.artifact_ids
    ):
        raise ValueError(f"Artifact {artifact.id} is not bound to its current successful Result")

    operation = energy["operation"]
    expected_tool = {"SP": "single_point", "Opt": "optimize_geometry"}[operation]
    if (
        step.tool != expected_tool
        or step.parameters.get("method_profile") != energy["method_profile"]
        or step.parameters.get("charge") != energy["charge"]
        or step.parameters.get("multiplicity") != energy["multiplicity"]
    ):
        raise ValueError(f"Artifact {artifact.id} disagrees with its producer Step parameters")
    value_name = {
        "SP": "sp_electronic_energy",
        "Opt": "opt_final_electronic_energy",
    }[operation]
    source_value = result.values.get(value_name)
    if (
        not is_compatible_value(source_value, "Eh")
        or source_value["value"] != energy["value"]
        or source_value["unit"] != "Eh"
    ):
        raise ValueError(f"Artifact {artifact.id} disagrees with its producer's energy value")

    geometry_id = energy["geometry_artifact_id"]
    if operation == "SP":
        if result.input_bindings.get("geometry") != geometry_id:
            raise ValueError(f"Artifact {artifact.id} does not match the SP input geometry")
    elif result.output_ports.get("optimized_geometry") != geometry_id:
        raise ValueError(f"Artifact {artifact.id} does not match the optimized output geometry")
    if geometry_id not in result.artifact_ids and operation == "Opt":
        raise ValueError(f"Artifact {artifact.id} optimized geometry is not a Result output")
    if geometry_id not in result.input_artifact_ids and operation == "SP":
        raise ValueError(f"Artifact {artifact.id} input geometry is not a Result input")
    geometry = find_artifact(run, geometry_id)
    if (
        geometry.artifact_type != "molecular_geometry"
        or geometry.sha256 != energy["geometry_sha256"]
        or (operation == "Opt" and geometry.role != "optimized_geometry")
    ):
        raise ValueError(f"Artifact {artifact.id} does not identify a verified geometry")
    artifact_path(config.data_root_path, run, geometry)


def _step_fingerprint(step: Step) -> str:
    payload = json.dumps(
        step.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_pair(energy_a: dict[str, Any], energy_b: dict[str, Any]) -> None:
    if energy_a["method_profile"] == energy_b["method_profile"]:
        raise ValueError("method energy difference requires two different method profiles")
    for name in ("geometry_sha256", "charge", "multiplicity"):
        if energy_a[name] != energy_b[name]:
            raise ValueError(
                f"energy inputs differ in {name}; a controlled method comparison is invalid"
            )


def _next_attempt(run: Run, step_id: str) -> int:
    attempts = [
        int(item.get("attempt", 0)) for item in run.attempts if item.get("step_id") == step_id
    ]
    return max(attempts, default=0) + 1


__all__ = ["execute_energy_difference", "make_energy_difference_tool"]
