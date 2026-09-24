"""Deterministic interatomic-distance measurement on a verified XYZ artifact."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from bg6022.config import AppConfig
from bg6022.models import Result, Step, Tool
from bg6022.tools.molecule import parse_xyz_bytes, resolve_artifact_reference
from bg6022.tools.runtime import ToolCallContext


class GeometryDistanceParameters(BaseModel):
    """Strict 1-based XYZ atom selection for one distance measurement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    atom_i: StrictInt = Field(ge=1, description="1-based XYZ atom index")
    atom_j: StrictInt = Field(ge=1, description="1-based XYZ atom index")

    @model_validator(mode="after")
    def _distinct_atoms(self) -> GeometryDistanceParameters:
        if self.atom_i == self.atom_j:
            raise ValueError("atom_i and atom_j must identify two different atoms")
        return self


def validate_distance_parameters(parameters: dict[str, Any], context: Mapping[str, Any]) -> None:
    """Validate index upper bounds when a trusted geometry count is known."""

    atom_count = context.get("geometry_atom_count")
    if atom_count is None:
        return
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count < 1:
        raise ValueError("geometry_atom_count must be a positive integer")
    if max(parameters["atom_i"], parameters["atom_j"]) > atom_count:
        raise ValueError(
            f"atom index is outside the known XYZ geometry; it contains {atom_count} atoms"
        )


def measure_distance(geometry: Any, parameters: GeometryDistanceParameters) -> float:
    """Measure the straight-line distance in the XYZ coordinate system."""

    count = len(geometry.symbols)
    if max(parameters.atom_i, parameters.atom_j) > count:
        raise ValueError(f"atom index is outside the XYZ geometry; it contains {count} atoms")
    left = parameters.atom_i - 1
    right = parameters.atom_j - 1
    value = math.dist(geometry.coordinates[left], geometry.coordinates[right])
    if not math.isfinite(value):
        raise ValueError("interatomic distance is not finite")
    return value


def make_geometry_distance_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, context: ToolCallContext) -> Result:
        if config is None:
            raise RuntimeError("tool 'geometry_distance' is a description-only Tool")
        return execute_geometry_distance(config, step=step, context=context)

    return Tool(
        name="geometry_distance",
        display_name="原子间距离测量",
        description=(
            "Measure the straight-line distance between two explicitly selected atoms "
            "in a verified molecular geometry."
        ),
        parameter_model=GeometryDistanceParameters.__name__,
        parameter_schema=GeometryDistanceParameters.model_json_schema(),
        parameter_type=GeometryDistanceParameters,
        request_parameters=["atom_i", "atom_j"],
        input_ports={"geometry": "molecular_geometry"},
        results={"interatomic_distance": "angstrom"},
        default_outputs=["interatomic_distance"],
        result_properties={"interatomic_distance": "distance"},
        result_metadata={
            "interatomic_distance": {
                "label": "原子间距离",
                "description": "指定 XYZ 原子顺序中两个原子的直线距离",
                "caveat": "这是笛卡尔坐标直线距离，不自动判定化学键或周期性最小镜像",
            }
        },
        success_conditions=[
            "verified molecular-geometry artifact",
            "strict 1-based distinct atom indices",
            "finite distance in angstrom",
        ],
        repair_capabilities=[],
        requires_compute_permission=False,
        execution_budget="none",
        execute_function=execute if config is not None else None,
        parameter_validation_function=validate_distance_parameters,
    )


def execute_geometry_distance(config: AppConfig, *, step: Step, context: ToolCallContext) -> Result:
    """Read, verify, and measure one geometry without changing its bytes."""

    parameters = GeometryDistanceParameters.model_validate(step.parameters, strict=True)
    status = "failed"
    values: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    input_artifact_id: str | None = None
    try:
        if context.cancel.is_set():
            status = "cancelled"
            diagnostics = {
                "category": "cancelled",
                "reason": "cancelled before distance measurement",
            }
        else:
            reference = step.inputs.get("geometry")
            if reference is None:
                raise ValueError("geometry_distance requires a geometry input reference")
            geometry_artifact = resolve_artifact_reference(
                config, context.run, reference, expected_type="molecular_geometry"
            )
            input_artifact_id = geometry_artifact.id
            geometry_bytes = context.read_input("geometry")
            geometry = parse_xyz_bytes(geometry_bytes)
            value = measure_distance(geometry, parameters)
            if context.cancel.is_set():
                status = "cancelled"
                diagnostics = {
                    "category": "cancelled",
                    "reason": "cancelled after geometry verification",
                }
            else:
                status = "succeeded"
                values["interatomic_distance"] = {
                    "value": value,
                    "unit": "angstrom",
                    "atom_indices": [parameters.atom_i, parameters.atom_j],
                    "atom_symbols": [
                        geometry.symbols[parameters.atom_i - 1],
                        geometry.symbols[parameters.atom_j - 1],
                    ],
                    "geometry_artifact_id": geometry_artifact.id,
                    "geometry_sha256": geometry_artifact.sha256,
                }
                diagnostics = {
                    "category": None,
                    "geometry_artifact_id": geometry_artifact.id,
                    "geometry_sha256": geometry_artifact.sha256,
                    "atom_indices": [parameters.atom_i, parameters.atom_j],
                    "atom_symbols": [
                        geometry.symbols[parameters.atom_i - 1],
                        geometry.symbols[parameters.atom_j - 1],
                    ],
                }
    except (OSError, TypeError, ValueError) as error:
        status = "failed"
        diagnostics = {
            "category": "distance_measurement_failed",
            "reason": str(error),
            "geometry_artifact_id": input_artifact_id,
        }

    return context.make_result(
        status=status,  # type: ignore[arg-type]
        values=values,
        diagnostics=diagnostics,
        input_bindings={"geometry": input_artifact_id} if input_artifact_id else {},
        input_artifact_ids=[input_artifact_id] if input_artifact_id else [],
    )


__all__ = [
    "GeometryDistanceParameters",
    "execute_geometry_distance",
    "make_geometry_distance_tool",
    "measure_distance",
    "validate_distance_parameters",
]
