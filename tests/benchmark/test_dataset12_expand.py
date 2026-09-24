from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bg6022.benchmark.dataset12 import collect_items
from bg6022.benchmark.loader import BenchmarkConfigurationError

DATASET = Path(__file__).parents[2] / "benchmarks" / "benchmark1.2"


def test_profiles_expand_expected_frozen_items_deterministically() -> None:
    smoke = collect_items(DATASET, profile="smoke")
    core = collect_items(DATASET, profile="core")
    full = collect_items(DATASET, profile="full")
    assert len(smoke) == 5
    assert len(core) == 18
    assert sum(item.metadata["kind"] == "scientific" for item in core) == 11
    assert len(full) == 28
    assert [item.id for item in core] == [
        item.id for item in collect_items(DATASET, profile="core")
    ]
    assert {item.id: item.metadata["prompt_sha256"] for item in core} == {
        item.id: item.metadata["prompt_sha256"] for item in collect_items(DATASET, profile="core")
    }


def test_full_profile_adds_holdout_only_when_requested() -> None:
    assert len(collect_items(DATASET, profile="full")) == 28
    with_holdout = collect_items(DATASET, profile="full", include_holdout=True)
    assert len(with_holdout) == 33
    assert {item.id for item in with_holdout if item.metadata["kind"] == "holdout"} == {
        "H001_water_sp_natural",
        "H002_water_opt_freq_compact",
        "H003_water_dual_opt",
        "H004_global_conformer",
        "H005_history_energy",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda matrix: matrix["cells"][0].update(object_id="unknown"), "unknown object"),
        (lambda matrix: matrix["cells"][0].update(task_id="T999"), "unknown task"),
        (lambda matrix: matrix["cells"][0].update(variants=["unknown"]), "unknown variant"),
    ],
)
def test_matrix_rejects_unknown_references(tmp_path, mutate, message) -> None:
    root = _copy_dataset(tmp_path)
    matrix_path = root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    mutate(matrix)
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(BenchmarkConfigurationError, match=message):
        collect_items(root, profile="smoke")


def test_matrix_rejects_duplicate_cells(tmp_path) -> None:
    root = _copy_dataset(tmp_path)
    path = root / "matrix.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    matrix["cells"].append(dict(matrix["cells"][0]))
    path.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(BenchmarkConfigurationError, match="duplicate benchmark matrix cell"):
        collect_items(root, profile="smoke")


def test_geometry_path_escape_is_rejected(tmp_path) -> None:
    root = _copy_dataset(tmp_path)
    path = root / "objects.json"
    objects = json.loads(path.read_text(encoding="utf-8"))
    objects["objects"][1]["geometry"] = "../water.xyz"
    path.write_text(json.dumps(objects), encoding="utf-8")
    with pytest.raises(BenchmarkConfigurationError, match="inside the dataset directory"):
        collect_items(root, profile="smoke")


def test_geometry_symlink_is_rejected(tmp_path) -> None:
    root = _copy_dataset(tmp_path)
    target = tmp_path / "outside.xyz"
    target.write_text("3\nwater\nO 0 0 0\nH 0 0 0\nH 0 0 0\n", encoding="utf-8")
    geometry = root / "geometries" / "water.xyz"
    geometry.unlink()
    try:
        geometry.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(BenchmarkConfigurationError, match="symlink"):
        collect_items(root, profile="smoke")


def test_sampling_requires_a_fixed_seed_and_is_reproducible() -> None:
    with pytest.raises(BenchmarkConfigurationError, match="fixed --seed"):
        collect_items(DATASET, profile="core", sample_variants=1)
    first = collect_items(DATASET, profile="core", sample_variants=1, seed=20260924)
    second = collect_items(DATASET, profile="core", sample_variants=1, seed=20260924)
    assert [item.id for item in first] == [item.id for item in second]


def test_full_variant_sampling_does_not_duplicate_frozen_profile_items() -> None:
    items = collect_items(DATASET, profile="full", sample_variants=2, seed=20260924)
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids))
    assert len(items) > len(collect_items(DATASET, profile="full"))


def _copy_dataset(tmp_path: Path) -> Path:
    destination = tmp_path / "dataset"
    shutil.copytree(DATASET, destination)
    return destination
