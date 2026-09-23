from __future__ import annotations

import json
from pathlib import Path

import pytest

from bg6022.benchmark.loader import BenchmarkConfigurationError
from bg6022.benchmark.matrix import expand_matrix
from bg6022.benchmark.runner import BenchmarkModeDisabled, _build_contract, run_case
from bg6022.planner import validate_request_plan
from bg6022.tools.registry import build_registry


def _write_minimal_matrix(root: Path, *, geometry: str = "geometries/h2.xyz", marker=None) -> None:
    (root / "geometries").mkdir(parents=True)
    (root / "task_templates").mkdir()
    (root / "geometries" / "h2.xyz").write_text(
        "2\nH2 test geometry\nH 0 0 0\nH 0.74 0 0\n", encoding="utf-8"
    )
    (root / "objects.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "id": "h2",
                        "label": "H2",
                        "geometry": geometry,
                        "charge": 0,
                        "multiplicity": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "tasks.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "T001",
                        "label": "test task",
                        "fixture_template": "task_templates/task.json",
                        "cost_class": "low",
                        "assertions": [{"type": "no_new_attempt", "value": 0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "matrix.json").write_text(
        json.dumps({"cells": [{"object_id": "h2", "task_id": "T001"}]}),
        encoding="utf-8",
    )
    (root / "task_templates" / "task.json").write_text(
        json.dumps(
            {
                "geometry": {"$object": "geometry"},
                "intake": {
                    "subjects": {"object": {"inline_xyz": marker or {"$object": "xyz_text"}}}
                },
            }
        ),
        encoding="utf-8",
    )


def test_scientific_matrix_expands_exactly_the_eleven_enabled_cells() -> None:
    cases = expand_matrix("benchmarks/scientific_v1")

    assert len(cases) == 11
    assert len({item.case.id for item in cases}) == 11
    assert [item.case.id for item in cases] == [
        "S_T001__h2",
        "S_T001__water",
        "S_T002__water",
        "S_T003__water",
        "S_T004__water",
        "S_T005__water",
        "S_T002__ammonia",
        "S_T003__ammonia",
        "S_T003__co2",
        "S_T002__methane",
        "S_T002__ethanol",
    ]
    assert all(item.case.mode == "live_orca" for item in cases)
    assert all(item.case.requires_orca for item in cases)
    assert all(not item.case.requires_llm for item in cases)
    assert all(not item.case.requires_pubchem for item in cases)


def test_expanded_request_geometry_matches_the_execution_geometry_file() -> None:
    matrix_root = Path("benchmarks/scientific_v1")
    cases = expand_matrix(matrix_root)

    for expanded in cases:
        geometry_path = matrix_root / expanded.fixture_override["geometry"]
        xyz_text = geometry_path.read_bytes().decode("utf-8")
        assert expanded.fixture_override["intake"]["subjects"]["object"]["inline_xyz"] == xyz_text

        registry = build_registry()
        _, request, plan = _build_contract(
            expanded.fixture_override,
            prompt=expanded.case.prompt,
            case_id=expanded.case.id,
            run_index=1,
            registry=registry,
        )
        assert plan is not None
        validate_request_plan(request, plan, registry)


@pytest.mark.parametrize(
    ("matrix_change", "message"),
    [
        ({"object_id": "unknown", "task_id": "T001"}, "unknown object"),
        ({"object_id": "h2", "task_id": "T999"}, "unknown task"),
    ],
)
def test_unknown_matrix_object_or_task_is_rejected(tmp_path, matrix_change, message) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root)
    (root / "matrix.json").write_text(json.dumps({"cells": [matrix_change]}), encoding="utf-8")

    with pytest.raises(BenchmarkConfigurationError, match=message):
        expand_matrix(root)


def test_duplicate_cell_is_rejected(tmp_path) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root)
    cell = {"object_id": "h2", "task_id": "T001"}
    (root / "matrix.json").write_text(json.dumps({"cells": [cell, cell]}), encoding="utf-8")

    with pytest.raises(BenchmarkConfigurationError, match="duplicate scientific matrix cell"):
        expand_matrix(root)


def test_matrix_with_no_enabled_cells_is_rejected(tmp_path) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root)
    (root / "matrix.json").write_text(
        json.dumps({"cells": [{"object_id": "h2", "task_id": "T001", "enabled": False}]}),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigurationError, match="no enabled cells"):
        expand_matrix(root)


def test_geometry_path_escape_is_rejected(tmp_path) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root, geometry="../outside.xyz")
    (tmp_path / "outside.xyz").write_text("2\noutside\nH 0 0 0\nH 1 0 0\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigurationError, match="must remain inside"):
        expand_matrix(root)


def test_symlink_geometry_escape_is_rejected(tmp_path, monkeypatch) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root, geometry="geometries/linked.xyz")
    link_path = root / "geometries" / "linked.xyz"
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == link_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(BenchmarkConfigurationError, match="symlink"):
        expand_matrix(root)


def test_unknown_template_marker_is_rejected(tmp_path) -> None:
    root = tmp_path / "matrix"
    _write_minimal_matrix(root, marker={"$object": "xyz"})

    with pytest.raises(BenchmarkConfigurationError, match="unknown scientific object marker"):
        expand_matrix(root)


def test_live_orca_case_stays_blocked_without_explicit_permission() -> None:
    expanded = expand_matrix("benchmarks/scientific_v1")[0]

    with pytest.raises(BenchmarkModeDisabled, match="--live-orca"):
        run_case(
            expanded.case,
            config=None,
            benchmark_dir="benchmarks/scientific_v1",
            fixture_override=expanded.fixture_override,
        )
