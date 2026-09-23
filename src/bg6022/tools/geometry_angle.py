"""Deterministic angle measurement on a verified XYZ geometry."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from bg6022.config import AppConfig
from bg6022.models import Result, Step, Tool
from bg6022.tools.molecule import parse_xyz_bytes, resolve_artifact_reference
from bg6022.tools.runtime import ToolCallContext


class GeometryAngleParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    atom_i: StrictInt = Field(ge=1, description="1-based XYZ atom index")
    atom_j: StrictInt = Field(ge=1, description="1-based XYZ atom index; angle vertex")
    atom_k: StrictInt = Field(ge=1, description="1-based XYZ atom index")

    @model_validator(mode="after")
    def _distinct_atoms(self) -> GeometryAngleParameters:
        indices = (self.atom_i, self.atom_j, self.atom_k)
        if len(set(indices)) != len(indices):
            raise ValueError("atom_i, atom_j, and atom_k must be distinct")
        return self


def validate_angle_parameters(parameters: dict[str, Any], context: Mapping[str, Any]) -> None:
    atom_count = context.get("geometry_atom_count")
    if atom_count is None:
        return
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count < 1:
        raise ValueError("geometry_atom_count must be a positive integer")
    if max(parameters["atom_i"], parameters["atom_j"], parameters["atom_k"]) > atom_count:
        raise ValueError(
            f"atom index is outside the known XYZ geometry; it contains {atom_count} atoms"
        )


def make_geometry_angle_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, context: ToolCallContext) -> Result:
        if config is None:
            raise RuntimeError("tool 'geometry_angle' is a description-only Tool")
        return execute_geometry_angle(config, step=step, context=context)

    return Tool(
        name="geometry_angle",
        display_name="几何夹角测量",
        description=(
            "Measure the angle at an explicitly selected vertex atom in a verified XYZ "
            "geometry and export the selected atom records."
        ),
        parameter_model=GeometryAngleParameters.__name__,
        parameter_schema=GeometryAngleParameters.model_json_schema(),
        parameter_type=GeometryAngleParameters,
        request_parameters=["atom_i", "atom_j", "atom_k"],
        input_ports={"geometry": "molecular_geometry"},
        results={"angle_value": "degree", "selected_atoms": "record_list"},
        output_ports={"atom_report": "text_file"},
        result_properties={
            "angle_value": "angle",
            "selected_atoms": "selected_atoms",
            "atom_report": "atom_report",
        },
        result_metadata={
            "angle_value": {
                "label": "原子夹角",
                "description": "以第二个所选原子为顶点的夹角",
            },
            "selected_atoms": {
                "label": "所选原子记录",
                "description": "原子序号、元素及 XYZ 原始坐标",
            },
            "atom_report": {
                "label": "原子坐标 CSV",
                "description": "所选原子的实际坐标表文件",
            },
        },
        success_conditions=[
            "verified molecular-geometry Artifact",
            "strict 1-based distinct atom indices",
            "finite angle in degrees",
            "CSV Artifact is bound to the same geometry hash",
        ],
        requires_compute_permission=False,
        execution_budget="none",
        execute_function=execute if config is not None else None,
        parameter_validation_function=validate_angle_parameters,
    )


def execute_geometry_angle(config: AppConfig, *, step: Step, context: ToolCallContext) -> Result:
    parameters = GeometryAngleParameters.model_validate(step.parameters, strict=True)
    status = "failed"
    values: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    artifact_ids: list[str] = []
    output_ports: dict[str, str] = {}
    input_artifact_id: str | None = None
    try:
        if context.cancel.is_set():
            status = "cancelled"
            diagnostics = {"category": "cancelled", "reason": "cancelled before angle measurement"}
        else:
            reference = step.inputs.get("geometry")
            if reference is None:
                raise ValueError("geometry_angle requires a geometry input reference")
            geometry_artifact = resolve_artifact_reference(
                config, context.run, reference, expected_type="molecular_geometry"
            )
            input_artifact_id = geometry_artifact.id
            geometry_bytes = context.read_input("geometry")
            geometry = parse_xyz_bytes(geometry_bytes)
            indices = (parameters.atom_i, parameters.atom_j, parameters.atom_k)
            if max(indices) > geometry.atom_count:
                raise ValueError(
                    "atom index is outside the XYZ geometry; "
                    f"it contains {geometry.atom_count} atoms"
                )
            selected = _selected_records(geometry, indices)
            vertex = geometry.coordinates[parameters.atom_j - 1]
            left = _vector(geometry.coordinates[parameters.atom_i - 1], vertex)
            right = _vector(geometry.coordinates[parameters.atom_k - 1], vertex)
            left_norm = math.sqrt(sum(value * value for value in left))
            right_norm = math.sqrt(sum(value * value for value in right))
            if left_norm == 0.0 or right_norm == 0.0:
                raise ValueError("angle vector has zero length")
            cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
            if not math.isfinite(cosine):
                raise ValueError("angle cosine is not finite")
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
                raise ValueError("angle is outside the physical 0-180 degree range")
            if context.cancel.is_set():
                status = "cancelled"
                diagnostics = {"category": "cancelled", "reason": "cancelled before CSV export"}
            else:
                report = context.register_bytes(
                    _records_csv(selected),
                    artifact_type="text_file",
                    role="angle_atom_report",
                    source=f"geometry_angle:{geometry_artifact.id}:{geometry_artifact.sha256}",
                    extension=".csv",
                    metadata={
                        "geometry_artifact_id": geometry_artifact.id,
                        "geometry_sha256": geometry_artifact.sha256,
                        "format": "csv",
                    },
                )
                artifact_ids.append(report.id)
                output_ports["atom_report"] = report.id
                values = {
                    "angle_value": {
                        "value": angle,
                        "unit": "degree",
                        "atom_indices": list(indices),
                        "atom_symbols": [geometry.symbols[index - 1] for index in indices],
                        "vertex_atom_index": parameters.atom_j,
                        "geometry_artifact_id": geometry_artifact.id,
                        "geometry_sha256": geometry_artifact.sha256,
                    },
                    "selected_atoms": selected,
                }
                diagnostics = {
                    "category": None,
                    "geometry_artifact_id": geometry_artifact.id,
                    "geometry_sha256": geometry_artifact.sha256,
                    "atom_indices": list(indices),
                    "report_artifact_id": report.id,
                }
                status = "succeeded"
    except (OSError, TypeError, ValueError) as error:
        status = "failed"
        diagnostics = {
            "category": "angle_measurement_failed",
            "reason": str(error),
            "geometry_artifact_id": input_artifact_id,
        }

    return context.make_result(
        status=status,  # type: ignore[arg-type]
        values=values,
        artifact_ids=artifact_ids,
        output_ports=output_ports,
        diagnostics=diagnostics,
        input_bindings={"geometry": input_artifact_id} if input_artifact_id else {},
        input_artifact_ids=[input_artifact_id] if input_artifact_id else [],
    )


def _selected_records(geometry: Any, indices: tuple[int, int, int]) -> list[dict[str, Any]]:
    lines = geometry.raw_bytes.decode("utf-8").splitlines()
    coordinate_lines = lines[2 : 2 + geometry.atom_count]
    return [
        {
            "atom_index": index,
            "element": geometry.symbols[index - 1],
            "x": geometry.coordinates[index - 1][0],
            "y": geometry.coordinates[index - 1][1],
            "z": geometry.coordinates[index - 1][2],
            "raw_line": coordinate_lines[index - 1],
        }
        for index in indices
    ]


def _records_csv(records: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["atom_index", "element", "x", "y", "z", "raw_line"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def _vector(
    point: tuple[float, float, float], vertex: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(float(left - right) for left, right in zip(point, vertex, strict=True))


__all__ = ["GeometryAngleParameters", "execute_geometry_angle", "make_geometry_angle_tool"]
