"""Strict scientific-matrix data contracts and finite case expansion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator

from .loader import BenchmarkConfigurationError, fixture_reference_path
from .models import BenchmarkAssertion, BenchmarkCase, BenchmarkModel

_OBJECT_MARKERS = {"xyz_text", "geometry", "charge", "multiplicity", "id", "label"}


class ScientificObject(BenchmarkModel):
    id: StrictStr
    label: StrictStr
    geometry: StrictStr
    charge: StrictInt
    multiplicity: StrictInt
    tags: list[StrictStr] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", value) is None:
            raise ValueError("scientific object id must be a lowercase simple identifier")
        return value

    @field_validator("label", "geometry")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scientific object label and geometry must not be blank")
        return value

    @field_validator("charge", "multiplicity")
    @classmethod
    def _closed_shell_singlet_v1(cls, value: int, info) -> int:
        expected = 0 if info.field_name == "charge" else 1
        if value != expected:
            raise ValueError(
                "Scientific Matrix v1 supports only neutral closed-shell singlets "
                "(charge=0, multiplicity=1)"
            )
        return value


class ScientificTaskTemplate(BenchmarkModel):
    id: StrictStr
    label: StrictStr
    fixture_template: StrictStr
    cost_class: Literal["low", "medium", "high"]
    assertions: list[BenchmarkAssertion] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if re.fullmatch(r"T[0-9]{3}", value) is None:
            raise ValueError("scientific task id must use the T### form")
        return value

    @field_validator("label", "fixture_template")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scientific task label and template path must not be blank")
        return value

    @field_validator("cost_class")
    @classmethod
    def _valid_cost_class(cls, value: str) -> str:
        if value not in {"low", "medium", "high"}:
            raise ValueError("cost_class must be low, medium, or high")
        return value


class ScientificMatrixCell(BenchmarkModel):
    object_id: StrictStr
    task_id: StrictStr
    enabled: StrictBool = True
    repeat: StrictInt = Field(default=1, ge=1, le=3)


class _ObjectCatalog(BenchmarkModel):
    objects: list[ScientificObject] = Field(min_length=1)


class _TaskCatalog(BenchmarkModel):
    tasks: list[ScientificTaskTemplate] = Field(min_length=1)


class _MatrixDefinition(BenchmarkModel):
    cells: list[ScientificMatrixCell] = Field(min_length=1)


@dataclass(frozen=True)
class ExpandedScientificCase:
    case: BenchmarkCase
    fixture_override: dict[str, Any]
    object_id: str
    object_label: str
    task_id: str
    task_label: str
    cost_class: str


def expand_matrix(matrix_root: str | Path) -> list[ExpandedScientificCase]:
    """Expand enabled cells into ordinary BenchmarkCases and fixture payloads."""

    root = Path(matrix_root).resolve()
    if not root.is_dir():
        raise BenchmarkConfigurationError(f"scientific matrix directory does not exist: {root}")
    objects = _load_model(_ObjectCatalog, root, "objects.json", "scientific objects")
    tasks = _load_model(_TaskCatalog, root, "tasks.json", "scientific tasks")
    definition = _load_model(_MatrixDefinition, root, "matrix.json", "scientific matrix")

    objects_by_id = _unique_by_id(objects.objects, "scientific object")
    tasks_by_id = _unique_by_id(tasks.tasks, "scientific task")
    seen_cells: set[tuple[str, str]] = set()
    expanded: list[ExpandedScientificCase] = []
    template_cache: dict[str, dict[str, Any]] = {}

    for cell in definition.cells:
        cell_key = (cell.task_id, cell.object_id)
        if cell_key in seen_cells:
            raise BenchmarkConfigurationError(
                f"duplicate scientific matrix cell: {cell.task_id} x {cell.object_id}"
            )
        seen_cells.add(cell_key)
        scientific_object = objects_by_id.get(cell.object_id)
        if scientific_object is None:
            raise BenchmarkConfigurationError(
                f"scientific matrix references unknown object: {cell.object_id}"
            )
        task = tasks_by_id.get(cell.task_id)
        if task is None:
            raise BenchmarkConfigurationError(
                f"scientific matrix references unknown task: {cell.task_id}"
            )
        if not cell.enabled:
            continue

        geometry_path = _safe_matrix_path(
            scientific_object.geometry,
            root,
            label=f"geometry for object {scientific_object.id}",
        )
        try:
            xyz_text = geometry_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise BenchmarkConfigurationError(
                f"cannot read UTF-8 geometry for object {scientific_object.id}: {error}"
            ) from error

        template = template_cache.get(task.fixture_template)
        if template is None:
            template_path = _safe_matrix_path(
                task.fixture_template,
                root,
                label=f"fixture template for task {task.id}",
            )
            try:
                raw_template = json.loads(template_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BenchmarkConfigurationError(
                    f"fixture template is invalid: {template_path}: {error}"
                ) from error
            if not isinstance(raw_template, dict):
                raise BenchmarkConfigurationError(
                    f"fixture template root must be a JSON object: {template_path}"
                )
            template = raw_template
            template_cache[task.fixture_template] = template

        marker_values: dict[str, Any] = {
            "xyz_text": xyz_text,
            "geometry": scientific_object.geometry,
            "charge": scientific_object.charge,
            "multiplicity": scientific_object.multiplicity,
            "id": scientific_object.id,
            "label": scientific_object.label,
        }
        fixture = _substitute_object_markers(template, marker_values)
        _validate_geometry_source(fixture, scientific_object, xyz_text)

        case_id = f"S_{task.id}__{scientific_object.id}"
        case = BenchmarkCase.model_validate(
            {
                "id": case_id,
                "category": "scientific",
                "support": "supported",
                "mode": "live_orca",
                "prompt": f"{task.label} for {scientific_object.label}",
                "repeat": cell.repeat,
                "fixture": task.fixture_template,
                "requires_llm": False,
                "requires_orca": True,
                "requires_pubchem": False,
                "assertions": [item.model_dump(mode="json") for item in task.assertions],
            },
            strict=True,
        )
        expanded.append(
            ExpandedScientificCase(
                case=case,
                fixture_override=fixture,
                object_id=scientific_object.id,
                object_label=scientific_object.label,
                task_id=task.id,
                task_label=task.label,
                cost_class=task.cost_class,
            )
        )

    if len({item.case.id for item in expanded}) != len(expanded):
        raise BenchmarkConfigurationError("expanded scientific case IDs are not unique")
    if not expanded:
        raise BenchmarkConfigurationError("scientific matrix has no enabled cells")
    return expanded


def _load_model(model_type, root: Path, filename: str, label: str):
    path = _safe_matrix_path(filename, root, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload, strict=True)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        raise BenchmarkConfigurationError(f"invalid {label} file {path}: {error}") from error


def _unique_by_id(items, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if item.id in result:
            raise BenchmarkConfigurationError(f"duplicate {label} id: {item.id}")
        result[item.id] = item
    return result


def _safe_matrix_path(reference: str, root: Path, *, label: str) -> Path:
    path = Path(reference)
    if path.is_absolute() or PureWindowsPath(reference).is_absolute():
        raise BenchmarkConfigurationError(f"{label} must be a relative path inside {root}")
    try:
        return fixture_reference_path(reference, root, label=label)
    except BenchmarkConfigurationError as error:
        raise BenchmarkConfigurationError(f"{label} must remain inside {root}: {error}") from error


def _substitute_object_markers(payload: Any, values: dict[str, Any]) -> Any:
    if isinstance(payload, dict):
        if "$object" in payload:
            if set(payload) != {"$object"}:
                raise BenchmarkConfigurationError("$object marker must be a one-key object")
            marker = payload["$object"]
            if not isinstance(marker, str) or marker not in _OBJECT_MARKERS:
                raise BenchmarkConfigurationError(f"unknown scientific object marker: {marker!r}")
            return values[marker]
        unknown = [key for key in payload if isinstance(key, str) and key.startswith("$")]
        if unknown:
            raise BenchmarkConfigurationError(f"unknown template marker: {unknown[0]}")
        return {key: _substitute_object_markers(value, values) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_substitute_object_markers(value, values) for value in payload]
    return payload


def _validate_geometry_source(
    fixture: dict[str, Any], scientific_object: ScientificObject, xyz_text: str
) -> None:
    if fixture.get("geometry") != scientific_object.geometry:
        raise BenchmarkConfigurationError(
            f"fixture geometry for {scientific_object.id} must use its object geometry file"
        )
    intake = fixture.get("intake")
    subjects = intake.get("subjects") if isinstance(intake, dict) else None
    if not isinstance(subjects, dict) or len(subjects) != 1:
        raise BenchmarkConfigurationError(
            "Scientific Matrix v1 templates must contain exactly one geometry subject"
        )
    subject = next(iter(subjects.values()))
    if not isinstance(subject, dict) or subject.get("inline_xyz") != xyz_text:
        raise BenchmarkConfigurationError(
            f"inline XYZ for {scientific_object.id} must come from its geometry file"
        )


__all__ = [
    "ExpandedScientificCase",
    "ScientificMatrixCell",
    "ScientificObject",
    "ScientificTaskTemplate",
    "expand_matrix",
]
