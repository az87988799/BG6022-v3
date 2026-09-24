"""Strict, frozen dataset expansion for the natural-language Benchmark 1.2."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator

from .loader import BenchmarkConfigurationError
from .models import BenchmarkModel

_ALLOWED_MARKERS = {"object_id", "label", "xyz_text", "charge", "multiplicity"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


class Benchmark12Object(BenchmarkModel):
    id: StrictStr
    label: StrictStr
    geometry: StrictStr
    charge: StrictInt
    multiplicity: StrictInt
    aliases: list[StrictStr] = Field(default_factory=list)
    tags: list[StrictStr] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if _ID_RE.fullmatch(value) is None:
            raise ValueError("object id must be a simple identifier")
        return value

    @field_validator("label", "geometry")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label and geometry must not be blank")
        return value

    @field_validator("charge", "multiplicity")
    @classmethod
    def _neutral_singlet(cls, value: int, info) -> int:
        expected = 0 if info.field_name == "charge" else 1
        if value != expected:
            raise ValueError("Benchmark 1.2 v1 objects must be neutral closed-shell singlets")
        return value


class PromptVariant(BenchmarkModel):
    id: StrictStr
    template: StrictStr
    style: Literal["canonical", "natural", "compact", "colloquial"]

    @field_validator("id", "template")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("variant id and template must not be blank")
        return value


class ExpectedTaskRole(BenchmarkModel):
    role: StrictStr
    capability: StrictStr
    method_profile: StrictStr | None = None


class ExpectedDependency(BenchmarkModel):
    source_role: StrictStr
    source_property: StrictStr
    target_role: StrictStr
    target_input_type: StrictStr


class ExpectedCheck(BenchmarkModel):
    role: StrictStr | None = None
    name: StrictStr
    status: Literal["passed", "failed", "unknown"]


class TaskGroundTruth(BenchmarkModel):
    route: Literal["compute", "context_query", "compute_then_query", "clarify", "unsupported"] = (
        "compute"
    )
    roles: list[ExpectedTaskRole] = Field(default_factory=list)
    dependencies: list[ExpectedDependency] = Field(default_factory=list)
    same_initial_geometry_groups: list[list[StrictStr]] = Field(default_factory=list)
    answer_property: StrictStr | None = None
    output_properties: dict[StrictStr, list[StrictStr]] = Field(default_factory=dict)
    comparison_mode: Literal["side_by_side", "numeric_difference"] | None = None
    scientific_checks: list[ExpectedCheck] = Field(default_factory=list)
    forbidden_tools: list[StrictStr] = Field(default_factory=list)
    max_orca_attempts: StrictInt | None = Field(default=None, ge=0)
    expected_repair_attempts: StrictInt | None = Field(default=None, ge=0)
    max_repair_attempts: StrictInt | None = Field(default=None, ge=0)
    require_semantic_path: StrictBool = True
    require_no_planner_calls: StrictBool = True
    require_no_intake_calls: StrictBool = True
    require_confirmation: StrictBool = True
    expect_confirmation_turn: StrictBool = True


class Benchmark12TaskTemplate(BenchmarkModel):
    id: StrictStr
    label: StrictStr
    input_mode: Literal["inline_xyz", "name"]
    prompt_variants: list[PromptVariant] = Field(min_length=1)
    ground_truth: TaskGroundTruth
    cost_class: Literal["low", "medium", "high"]

    @field_validator("id", "label")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task id and label must not be blank")
        return value

    @field_validator("prompt_variants")
    @classmethod
    def _unique_variants(cls, value: list[PromptVariant]) -> list[PromptVariant]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt variant ids must be unique within a task")
        return value


class Benchmark12Cell(BenchmarkModel):
    object_id: StrictStr
    task_id: StrictStr
    enabled: StrictBool = True
    variants: list[StrictStr] = Field(default_factory=lambda: ["canonical"])
    repeat: StrictInt = Field(default=1, ge=1, le=3)


class Benchmark12Suite(BenchmarkModel):
    name: StrictStr
    version: StrictStr
    schema_name: StrictStr = Field(alias="schema")
    official_profile: Literal["smoke", "core", "full"]
    prompt_generation: Literal["frozen_templates"]
    scientific_matrix_source: StrictStr
    identity_cases: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioTurn(BenchmarkModel):
    message_template: StrictStr
    confirm_after: StrictBool = False
    stop_after: StrictBool = False


class Benchmark12Scenario(BenchmarkModel):
    id: StrictStr
    label: StrictStr
    object_id: StrictStr | None = None
    turns: list[ScenarioTurn] = Field(min_length=1)
    ground_truth: TaskGroundTruth
    requires_llm: StrictBool = True
    requires_orca: StrictBool = False
    requires_pubchem: StrictBool = False


class _Objects(BenchmarkModel):
    objects: list[Benchmark12Object] = Field(min_length=1)


class _Matrix(BenchmarkModel):
    cells: list[Benchmark12Cell] = Field(min_length=1)


class _Scenarios(BenchmarkModel):
    scenarios: list[Benchmark12Scenario] = Field(default_factory=list)


class _HoldoutCase(BenchmarkModel):
    id: StrictStr
    task_id: StrictStr | None = None
    object_id: StrictStr | None = None
    style: Literal["canonical", "natural", "compact", "colloquial"]
    input_mode: Literal["inline_xyz", "name"] = "inline_xyz"
    prompt_template: StrictStr | None = None
    turns: list[ScenarioTurn] = Field(default_factory=list)
    ground_truth: TaskGroundTruth | None = None
    requires_llm: StrictBool = True
    requires_orca: StrictBool = True
    requires_pubchem: StrictBool = False


class _Holdout(BenchmarkModel):
    cases: list[_HoldoutCase] = Field(default_factory=list)


@dataclass(frozen=True)
class Benchmark12Item:
    id: str
    task_id: str
    object_id: str
    variant_id: str
    prompt: str
    ground_truth: TaskGroundTruth
    metadata: dict[str, Any]
    repeat: int = 1
    script: tuple[ScenarioTurn, ...] = ()
    requires_llm: bool = True
    requires_orca: bool = True
    requires_pubchem: bool = False


def collect_items(
    benchmark_root: str | Path,
    *,
    profile: Literal["smoke", "core", "full"],
    include_holdout: bool = False,
    sample_variants: int = 0,
    seed: int | None = None,
) -> list[Benchmark12Item]:
    """Expand a frozen profile. Sampling is available only as seeded dev behavior."""

    root = Path(benchmark_root).resolve()
    if not root.is_dir():
        raise BenchmarkConfigurationError(f"Benchmark 1.2 directory does not exist: {root}")
    suite = _load_model(Benchmark12Suite, root / "suite.json", "suite")
    objects_doc = _load_model(_Objects, root / "objects.json", "objects")
    task_files = sorted((root / "tasks").glob("*.json"))
    if not task_files:
        raise BenchmarkConfigurationError(f"no task definitions found in {root / 'tasks'}")
    tasks_doc = [_load_model(Benchmark12TaskTemplate, path, "task") for path in task_files]
    matrix_doc = _load_model(_Matrix, root / "matrix.json", "matrix")
    scenarios_doc = _load_model(_Scenarios, root / "scenarios.json", "scenarios")

    objects = _unique(objects_doc.objects, "object")
    tasks = _unique(tasks_doc, "task")
    scenarios = _unique(scenarios_doc.scenarios, "scenario")
    suite_hashes = dataset_hashes(root)
    geometry_cache: dict[str, tuple[str, str]] = {}
    task_variants = {
        task_id: {variant.id: variant for variant in task.prompt_variants}
        for task_id, task in tasks.items()
    }
    all_cells: set[tuple[str, str]] = set()
    for cell in matrix_doc.cells:
        key = (cell.task_id, cell.object_id)
        if key in all_cells:
            raise BenchmarkConfigurationError(
                f"duplicate benchmark matrix cell: {cell.task_id} x {cell.object_id}"
            )
        all_cells.add(key)
        if cell.object_id not in objects:
            raise BenchmarkConfigurationError(f"matrix references unknown object: {cell.object_id}")
        if cell.task_id not in tasks:
            raise BenchmarkConfigurationError(f"matrix references unknown task: {cell.task_id}")
        unknown_variants = set(cell.variants) - set(task_variants[cell.task_id])
        if unknown_variants:
            raise BenchmarkConfigurationError(
                f"matrix references unknown variant {sorted(unknown_variants)[0]!r} "
                f"for task {cell.task_id}"
            )
    seen_cells: set[tuple[str, str]] = set()
    items: list[Benchmark12Item] = []

    cells = matrix_doc.cells
    if profile == "smoke":
        cells = [cell for cell in cells if cell.enabled and cell.object_id == "water"]
        if not cells:
            raise BenchmarkConfigurationError("smoke profile needs enabled water matrix cells")
    for cell in cells:
        key = (cell.task_id, cell.object_id)
        if key in seen_cells:
            raise BenchmarkConfigurationError(
                f"duplicate benchmark matrix cell: {cell.task_id} x {cell.object_id}"
            )
        seen_cells.add(key)
        if not cell.enabled:
            continue
        obj = objects.get(cell.object_id)
        task = tasks.get(cell.task_id)
        if obj is None:
            raise BenchmarkConfigurationError(f"matrix references unknown object: {cell.object_id}")
        if task is None:
            raise BenchmarkConfigurationError(f"matrix references unknown task: {cell.task_id}")
        selected_variants = list(cell.variants)
        if profile == "full" and cell.object_id == "water":
            selected_variants.extend(
                name for name in ("natural", "compact") if name in task_variants[cell.task_id]
            )
        _append_task_items(
            items,
            root=root,
            obj=obj,
            task=task,
            variants=selected_variants,
            repeat=cell.repeat,
            metadata={"profile": profile, "kind": "scientific", "suite_hashes": suite_hashes},
            geometry_cache=geometry_cache,
        )

    if profile in {"core", "full"}:
        for scenario in scenarios.values():
            obj = objects.get(scenario.object_id) if scenario.object_id is not None else None
            marker_values = _marker_values(obj, root, geometry_cache) if obj is not None else {}
            rendered_turns = tuple(
                turn.model_copy(
                    update={
                        "message_template": render_prompt(
                            turn.message_template,
                            marker_values,
                        )
                    }
                )
                for turn in scenario.turns
            )
            if not scenario.ground_truth.roles and scenario.ground_truth.route == "compute":
                raise BenchmarkConfigurationError(
                    f"compute scenario {scenario.id} must declare expected roles"
                )
            first_prompt = rendered_turns[0].message_template
            items.append(
                Benchmark12Item(
                    id=scenario.id,
                    task_id=scenario.id,
                    object_id=scenario.object_id or "none",
                    variant_id="scenario",
                    prompt=first_prompt,
                    ground_truth=scenario.ground_truth,
                    metadata={
                        "profile": profile,
                        "kind": "scenario",
                        "label": scenario.label,
                        "input_mode": "inline_xyz" if obj is not None else "name",
                        "scenario_outcome": scenario.ground_truth.route,
                        "suite_hashes": suite_hashes,
                        "prompt_sha256": hashlib.sha256(first_prompt.encode("utf-8")).hexdigest(),
                    },
                    script=rendered_turns,
                    requires_llm=scenario.requires_llm,
                    requires_orca=scenario.requires_orca,
                    requires_pubchem=scenario.requires_pubchem,
                )
            )
        for raw_identity in suite.identity_cases:
            identity = _parse_identity_case(raw_identity)
            obj = objects.get(identity["object_id"])
            task = tasks.get(identity["task_id"])
            if obj is None:
                raise BenchmarkConfigurationError(
                    f"identity case references unknown object: {identity['object_id']}"
                )
            if task is None:
                raise BenchmarkConfigurationError(
                    f"identity case references unknown task: {identity['task_id']}"
                )
            prompt = render_prompt(
                identity["prompt_template"],
                {
                    "object_id": obj.id,
                    "label": obj.label,
                    "charge": obj.charge,
                    "multiplicity": obj.multiplicity,
                },
            )
            items.append(
                Benchmark12Item(
                    id=identity["id"],
                    task_id=task.id,
                    object_id=obj.id,
                    variant_id="name",
                    prompt=prompt,
                    ground_truth=task.ground_truth,
                    metadata={
                        "profile": profile,
                        "kind": "identity",
                        "label": task.label,
                        "input_mode": "name",
                        "cost_class": task.cost_class,
                        "suite_hashes": suite_hashes,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    },
                    requires_llm=True,
                    requires_orca=True,
                    requires_pubchem=True,
                )
            )

    if profile == "full" and include_holdout:
        holdout_path = _safe_file(root, "holdout.json", "holdout file")
        holdout_doc = _load_model(_Holdout, holdout_path, "holdout")
        for holdout in holdout_doc.cases:
            obj = objects.get(holdout.object_id or "")
            task = tasks.get(holdout.task_id or "")
            if holdout.object_id is not None and obj is None:
                raise BenchmarkConfigurationError(
                    f"holdout references unknown object: {holdout.object_id}"
                )
            if holdout.task_id is not None and task is None:
                raise BenchmarkConfigurationError(
                    f"holdout references unknown task: {holdout.task_id}"
                )
            if holdout.ground_truth is None and task is None:
                raise BenchmarkConfigurationError(
                    f"holdout {holdout.id} must reference a task or define ground_truth"
                )
            values = _marker_values(obj, root, geometry_cache) if obj is not None else {}
            rendered_script = tuple(
                turn.model_copy(
                    update={"message_template": render_prompt(turn.message_template, values)}
                )
                for turn in holdout.turns
            )
            prompt_template = holdout.prompt_template or (
                rendered_script[0].message_template if rendered_script else ""
            )
            prompt = (
                render_prompt(prompt_template, values)
                if holdout.prompt_template
                else prompt_template
            )
            if not prompt.strip():
                raise BenchmarkConfigurationError(f"holdout {holdout.id} has no prompt")
            items.append(
                Benchmark12Item(
                    id=holdout.id,
                    task_id=holdout.task_id or holdout.id,
                    object_id=holdout.object_id or "none",
                    variant_id=f"holdout_{holdout.style}",
                    prompt=prompt,
                    ground_truth=holdout.ground_truth or task.ground_truth,
                    metadata={
                        "profile": profile,
                        "kind": "holdout",
                        "style": holdout.style,
                        "input_mode": holdout.input_mode,
                        "suite_hashes": suite_hashes,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    },
                    script=rendered_script,
                    requires_llm=holdout.requires_llm,
                    requires_orca=holdout.requires_orca,
                    requires_pubchem=holdout.requires_pubchem,
                )
            )

    if sample_variants:
        if seed is None:
            raise BenchmarkConfigurationError("--sample-variants requires a fixed --seed")
        if sample_variants < 0:
            raise BenchmarkConfigurationError("sample_variants must be nonnegative")
        rng = random.Random(seed)
        sampled: list[Benchmark12Item] = []
        known_ids = {item.id for item in items}
        base_items = [
            item
            for item in items
            if item.metadata.get("kind") == "scientific" and item.variant_id == "canonical"
        ]
        for item in base_items:
            task = tasks.get(item.task_id)
            if task is None:
                continue
            bank = [
                variant
                for variant in task.prompt_variants
                if f"{task.id}__{item.object_id}__{variant.id}" not in known_ids
            ]
            for variant in rng.sample(bank, k=min(sample_variants, len(bank))):
                obj = objects[item.object_id]
                generated = _make_items(
                    root,
                    obj,
                    task,
                    [variant.id],
                    item.repeat,
                    item.metadata,
                    geometry_cache,
                )
                sampled.extend(generated)
                known_ids.update(candidate.id for candidate in generated)
        items.extend(sampled)

    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise BenchmarkConfigurationError("expanded Benchmark 1.2 item IDs are not unique")
    if not items:
        raise BenchmarkConfigurationError("Benchmark 1.2 profile has no enabled items")
    return items


def render_prompt(template: str, values: dict[str, Any]) -> str:
    """Interpolate the fixed allowlist only; no template engine or evaluation."""

    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise BenchmarkConfigurationError(f"invalid prompt template: {error}") from error
    markers: list[str] = []
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _ALLOWED_MARKERS or format_spec or conversion:
            raise BenchmarkConfigurationError(f"unknown or unsafe prompt marker: {{{field_name}}}")
        if field_name not in values:
            raise BenchmarkConfigurationError(f"prompt marker has no value: {{{field_name}}}")
        markers.append(field_name)
    try:
        result = template.format_map({key: str(value) for key, value in values.items()})
    except (KeyError, ValueError) as error:
        raise BenchmarkConfigurationError(f"cannot render prompt template: {error}") from error
    if not result.strip():
        raise BenchmarkConfigurationError("rendered prompt must not be blank")
    return result


def dataset_hashes(benchmark_root: str | Path) -> dict[str, str]:
    root = Path(benchmark_root).resolve()
    files = (
        "suite.json",
        "objects.json",
        "tasks",
        "matrix.json",
        "scenarios.json",
        "holdout.json",
        "geometries",
    )
    hashes: dict[str, str] = {}
    for entry in files:
        if entry == "tasks":
            paths = sorted((root / "tasks").glob("*.json")) if (root / "tasks").is_dir() else []
            for path in paths:
                _safe_file(root, f"tasks/{path.name}", "task file")
                hashes[f"tasks/{path.name}"] = _sha256(path)
        elif entry == "geometries":
            paths = (
                sorted((root / "geometries").glob("*.xyz"))
                if (root / "geometries").is_dir()
                else []
            )
            for path in paths:
                _safe_file(root, f"geometries/{path.name}", "geometry file")
                hashes[f"geometries/{path.name}"] = _sha256(path)
        else:
            path = _safe_file(root, entry, f"{entry} file")
            hashes[entry] = _sha256(path)
    return hashes


def _append_task_items(
    items: list[Benchmark12Item],
    *,
    root: Path,
    obj: Benchmark12Object,
    task: Benchmark12TaskTemplate,
    variants: list[str],
    repeat: int,
    metadata: dict[str, Any],
    geometry_cache: dict[str, tuple[str, str]],
) -> None:
    if len(variants) != len(set(variants)):
        raise BenchmarkConfigurationError(f"duplicate variant in {task.id} x {obj.id}")
    _append = _make_items(root, obj, task, variants, repeat, metadata, geometry_cache)
    items.extend(_append)


def _make_items(
    root: Path,
    obj: Benchmark12Object,
    task: Benchmark12TaskTemplate,
    variants: list[str],
    repeat: int,
    metadata: dict[str, Any],
    geometry_cache: dict[str, tuple[str, str]],
) -> list[Benchmark12Item]:
    values = (
        _marker_values(obj, root, geometry_cache)
        if task.input_mode == "inline_xyz"
        else {
            "object_id": obj.id,
            "label": obj.label,
            "charge": obj.charge,
            "multiplicity": obj.multiplicity,
        }
    )
    by_id = {variant.id: variant for variant in task.prompt_variants}
    expanded: list[Benchmark12Item] = []
    for variant_id in variants:
        variant = by_id.get(variant_id)
        if variant is None:
            raise BenchmarkConfigurationError(
                f"task {task.id} does not declare prompt variant {variant_id!r}"
            )
        prompt = render_prompt(variant.template, values)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expanded.append(
            Benchmark12Item(
                id=f"{task.id}__{obj.id}__{variant.id}",
                task_id=task.id,
                object_id=obj.id,
                variant_id=variant.id,
                prompt=prompt,
                ground_truth=task.ground_truth,
                metadata={
                    **metadata,
                    "label": task.label,
                    "cost_class": task.cost_class,
                    "input_mode": task.input_mode,
                    "style": variant.style,
                    "prompt_sha256": prompt_hash,
                    "geometry_sha256": geometry_cache.get(obj.geometry, ("", ""))[1],
                },
                repeat=repeat,
                requires_llm=True,
                requires_orca=True,
                requires_pubchem=task.input_mode == "name",
            )
        )
    return expanded


def _marker_values(
    obj: Benchmark12Object | None,
    root: Path,
    geometry_cache: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    if obj is None:
        return {}
    xyz_text: str | None = None
    geometry_hash: str | None = None
    if obj.geometry not in geometry_cache:
        path = _safe_file(root, obj.geometry, f"geometry for {obj.id}")
        try:
            data = path.read_bytes()
            xyz_text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise BenchmarkConfigurationError(
                f"cannot read UTF-8 XYZ for {obj.id}: {error}"
            ) from error
        geometry_hash = hashlib.sha256(data).hexdigest()
        geometry_cache[obj.geometry] = (xyz_text, geometry_hash)
    xyz_text, geometry_hash = geometry_cache[obj.geometry]
    return {
        "object_id": obj.id,
        "label": obj.label,
        "xyz_text": xyz_text,
        "charge": obj.charge,
        "multiplicity": obj.multiplicity,
    }


def _parse_identity_case(raw: dict[str, Any]) -> dict[str, str]:
    allowed = {"id", "object_id", "task_id", "prompt_template"}
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise BenchmarkConfigurationError(
            "identity cases need exactly id, object_id, task_id, and prompt_template"
        )
    if any(not isinstance(raw[key], str) or not raw[key].strip() for key in allowed):
        raise BenchmarkConfigurationError("identity case fields must be nonblank strings")
    if _ID_RE.fullmatch(raw["id"]) is None:
        raise BenchmarkConfigurationError("identity case id must be a simple identifier")
    return dict(raw)


def _load_model(model_type, path: Path, label: str):
    if not path.is_file():
        raise BenchmarkConfigurationError(f"missing {label} file: {path}")
    _reject_symlinks(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload, strict=True)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        raise BenchmarkConfigurationError(f"invalid {label} file {path}: {error}") from error


def _unique(items: list[Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if item.id in result:
            raise BenchmarkConfigurationError(f"duplicate {label} id: {item.id}")
        result[item.id] = item
    return result


def _safe_file(root: Path, reference: str, label: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise BenchmarkConfigurationError(f"{label} must be a nonblank relative path")
    windows = PureWindowsPath(reference)
    parts = re.split(r"[\\/]", reference)
    if (
        Path(reference).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise BenchmarkConfigurationError(f"{label} must remain inside the dataset directory")
    root = root.resolve()
    if root.is_symlink():
        raise BenchmarkConfigurationError("dataset directory cannot be a symlink")
    candidate_raw = root.joinpath(*parts)
    _reject_symlinks(candidate_raw)
    candidate = candidate_raw.resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise BenchmarkConfigurationError(f"{label} is missing or escapes the dataset directory")
    return candidate


def _reject_symlinks(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise BenchmarkConfigurationError("dataset paths cannot traverse symlinks")
        if current.parent == current:
            break
        current = current.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Benchmark12Cell",
    "Benchmark12Item",
    "Benchmark12Object",
    "Benchmark12Scenario",
    "Benchmark12Suite",
    "Benchmark12TaskTemplate",
    "ExpectedCheck",
    "ExpectedDependency",
    "ExpectedTaskRole",
    "PromptVariant",
    "ScenarioTurn",
    "TaskGroundTruth",
    "collect_items",
    "dataset_hashes",
    "render_prompt",
]
