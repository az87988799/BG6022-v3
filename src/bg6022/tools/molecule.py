"""XYZ validation plus the real RDKit geometry-preparation Tool."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt

from bg6022.config import AppConfig
from bg6022.models import InputReference, Result, Run, Step, Tool
from bg6022.session import (
    artifact_path,
    find_artifact,
    register_bytes_artifact,
    run_directory,
    save_run,
)

SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


@dataclass(frozen=True)
class ParsedGeometry:
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    comment: str
    raw_bytes: bytes

    @property
    def atom_count(self) -> int:
        return len(self.symbols)


class GenerateGeometryParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seed: StrictInt | None = None


class GeometryEmbeddingError(ValueError):
    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


_MAX_EMBED_HELPER_OUTPUT_BYTES = 1024 * 1024
_EMBED_HELPER = r"""
import json
import sys


def main():
    request = json.load(sys.stdin)
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(request["smiles"])
    if mol is None:
        raise ValueError("SMILES cannot be rebuilt by RDKit")
    mol = Chem.AddHs(mol)
    for seed in request["seeds"][:2]:
        embedding = AllChem.ETKDGv3()
        embedding.randomSeed = int(seed)
        embedding.numThreads = 1
        conformer_id = AllChem.EmbedMolecule(mol, embedding)
        if conformer_id < 0:
            continue
        conformer = mol.GetConformer()
        coordinates = [
            [
                float(conformer.GetAtomPosition(index).x),
                float(conformer.GetAtomPosition(index).y),
                float(conformer.GetAtomPosition(index).z),
            ]
            for index in range(mol.GetNumAtoms())
        ]
        print(json.dumps({
            "ok": True,
            "seed": int(seed),
            "symbols": [atom.GetSymbol() for atom in mol.GetAtoms()],
            "coordinates": coordinates,
            "rdkit_version": getattr(rdBase, "rdkitVersion", "unknown"),
        }, separators=(",", ":")))
        return
    print(json.dumps({"ok": False, "error": "all embedding seeds failed"}))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
"""


def make_generate_geometry_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, run: Run, cancel: Event) -> Result:
        if config is None:
            raise RuntimeError("tool 'generate_geometry' is a description-only Tool")
        return execute_generate_geometry(config, step=step, run=run, cancel=cancel)

    return Tool(
        name="generate_geometry",
        description="Generate a bounded RDKit ETKDGv3 initial geometry from a molecule artifact.",
        parameter_model=GenerateGeometryParameters.__name__,
        parameter_schema=GenerateGeometryParameters.model_json_schema(),
        parameter_type=GenerateGeometryParameters,
        input_ports={"molecule": "molecule"},
        output_ports={"geometry": "molecular_geometry"},
        results={"geometry_atom_count": "integer"},
        result_properties={"geometry": "molecular_geometry", "geometry_atom_count": "atom_count"},
        result_metadata={
            "geometry": {
                "label": "初始 XYZ 结构文件",
                "description": "由已验证分子生成并通过 XYZ 解析校验的初始结构",
                "caveat": "这是初始猜测结构，不代表已完成几何优化",
            },
            "geometry_atom_count": {
                "label": "结构原子数",
                "description": "初始 XYZ 结构中的原子数量",
            },
        },
        success_conditions=["RDKit structure rebuilt", "XYZ parsed and atom order preserved"],
        repair_capabilities=[],
        requires_compute_permission=False,
        execute_function=execute if config is not None else None,
    )


def execute_generate_geometry(config: AppConfig, *, step: Step, run: Run, cancel: Event) -> Result:
    parameters = GenerateGeometryParameters.model_validate(step.parameters, strict=True)
    attempt = _next_attempt(run, step.id)
    relative = f"{step.id}/attempt-{attempt:02d}"
    (run_directory(config.data_root_path, run.id) / relative).mkdir(parents=True, exist_ok=True)
    try:
        reference = step.inputs.get("molecule")
        if reference is None:
            raise ValueError("generate_geometry requires a molecule input reference")
        molecule_artifact = resolve_artifact_reference(
            config, run, reference, expected_type="molecule"
        )
        molecule_path = artifact_path(config.data_root_path, run, molecule_artifact)
        payload = json.loads(molecule_path.read_text(encoding="utf-8"))
        facts = payload.get("facts", {})
        smiles = facts.get("isomeric_smiles") or facts.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("molecule artifact has no canonical SMILES")
        requested_seed = parameters.seed
        seeds = (
            [requested_seed]
            if requested_seed is not None
            else list(config.molecule.embedding_seeds)
        )
        seeds = seeds[:2]
        if not seeds:
            raise GeometryEmbeddingError(
                "no RDKit embedding seeds are configured", category="invalid_configuration"
            )
        embedded = _run_embedding_helper(
            smiles,
            [int(seed) for seed in seeds],
            timeout_seconds=min(
                float(config.molecule.embedding_timeout_seconds),
                _remaining_active_seconds(run, config),
            ),
            cancel=cancel,
        )
        if cancel.is_set():
            raise GeometryEmbeddingError("RDKit embedding cancelled", category="cancelled")
        try:
            used_seed = int(embedded["seed"])
            symbols = tuple(str(symbol) for symbol in embedded["symbols"])
            coordinates = tuple(
                tuple(float(value) for value in point) for point in embedded["coordinates"]
            )
            rdkit_version = str(embedded["rdkit_version"])
            rdkit_stderr = embedded.get("rdkit_stderr", "")
            if not isinstance(rdkit_stderr, str):
                rdkit_stderr = ""
        except (KeyError, TypeError, ValueError) as error:
            raise GeometryEmbeddingError(
                "RDKit helper returned malformed geometry", category="invalid_response"
            ) from error
        geometry_bytes = format_xyz(
            symbols,
            coordinates,
            comment=f"BG6022 v3 RDKit ETKDGv3 seed {used_seed}",
        )
        geometry = parse_xyz_bytes(geometry_bytes)
        expected_counts = _verified_molecule_counts(facts)
        actual_counts = dict(Counter(geometry.symbols))
        if actual_counts != expected_counts:
            raise ValueError("generated geometry element counts do not match the resolved molecule")
        diagnostics: dict[str, Any] = {}
        if rdkit_stderr:
            diagnostic_path = run_directory(config.data_root_path, run.id) / relative
            diagnostic_file = diagnostic_path / "rdkit.stderr.log"
            diagnostic_file.write_text(rdkit_stderr, encoding="utf-8")
            diagnostics["raw_paths"] = {"rdkit_stderr": str(diagnostic_file)}
        geometry_artifact = register_bytes_artifact(
            config.data_root_path,
            run,
            geometry_bytes,
            artifact_type="molecular_geometry",
            role="initial_geometry",
            source=f"rdkit:ETKDGv3:{used_seed}",
            extension=".xyz",
            step_id=step.id,
            attempt=attempt,
            metadata={
                "molecule_artifact_id": molecule_artifact.id,
                "structure_source": payload.get("source"),
                "rdkit_version": rdkit_version,
                "embedding": "ETKDGv3",
                "seed": used_seed,
                "atom_mapping": list(range(len(symbols))),
                "initial_guess_only": True,
            },
        )
        run.attempts.append(
            {
                "step_id": step.id,
                "attempt": attempt,
                "phase": "finished",
                "status": "succeeded",
                "artifact_ids": [geometry_artifact.id],
                "output_ports": {"geometry": geometry_artifact.id},
            }
        )
        save_run(config.data_root_path, run)
        return _molecule_result(
            run,
            step,
            attempt,
            "succeeded",
            values={"geometry_atom_count": geometry.atom_count},
            diagnostics=diagnostics,
            artifact_ids=[geometry_artifact.id],
            output_ports={"geometry": geometry_artifact.id},
            parameter_sources={"geometry": f"rdkit:ETKDGv3:{used_seed}"},
            relative=relative,
        )
    except GeometryEmbeddingError as error:
        return _molecule_result(
            run,
            step,
            attempt,
            "cancelled" if error.category == "cancelled" else "failed",
            diagnostics={"category": error.category, "reason": str(error)},
            relative=relative,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _molecule_result(
            run,
            step,
            attempt,
            "failed",
            diagnostics={"category": "geometry_generation_failed", "reason": str(error)},
            relative=relative,
        )


def _run_embedding_helper(
    smiles: str,
    seeds: list[int],
    *,
    timeout_seconds: float,
    cancel: Event,
) -> dict[str, Any]:
    """Run the bounded native RDKit call outside the Agent process."""

    if timeout_seconds <= 0:
        raise GeometryEmbeddingError("RDKit time budget is exhausted", category="timeout")
    payload = json.dumps({"smiles": smiles, "seeds": seeds}, separators=(",", ":")).encode("utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _EMBED_HELPER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=_embedding_environment(),
        )
    except OSError as error:
        raise GeometryEmbeddingError(
            f"cannot start the RDKit helper: {error}", category="helper_start_failed"
        ) from error

    completed: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def communicate() -> None:
        try:
            completed.put(("ok", process.communicate(input=payload)))
        except BaseException as error:  # pragma: no cover - defensive worker boundary
            completed.put(("error", error))

    reader = Thread(target=communicate, name="bg6022-rdkit-helper-reader", daemon=True)
    reader.start()
    deadline = time.monotonic() + float(timeout_seconds)
    while reader.is_alive():
        if cancel.is_set():
            _stop_embedding_helper(process)
            reader.join(timeout=2)
            raise GeometryEmbeddingError("RDKit embedding cancelled", category="cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_embedding_helper(process)
            reader.join(timeout=2)
            raise GeometryEmbeddingError(
                "RDKit embedding exceeded its wall-time bound", category="timeout"
            )
        reader.join(timeout=min(0.1, remaining))

    kind, value = completed.get()
    if kind == "error":
        raise GeometryEmbeddingError(
            f"RDKit helper communication failed: {value}", category="helper_failed"
        ) from value
    stdout, stderr = value
    if len(stdout) > _MAX_EMBED_HELPER_OUTPUT_BYTES or len(stderr) > _MAX_EMBED_HELPER_OUTPUT_BYTES:
        raise GeometryEmbeddingError(
            "RDKit helper output exceeded its size bound", category="output_too_large"
        )
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:2000]
        try:
            error_payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_payload = {}
        reason = error_payload.get("error") if isinstance(error_payload, dict) else None
        raise GeometryEmbeddingError(
            reason or detail or "RDKit embedding failed", category="embedding_failed"
        )
    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryEmbeddingError(
            "RDKit helper returned invalid JSON", category="invalid_response"
        ) from error
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise GeometryEmbeddingError(
            "RDKit helper returned no successful conformer", category="embedding_failed"
        )
    result["rdkit_stderr"] = stderr.decode("utf-8", errors="replace")[:8192]
    return result


def _stop_embedding_helper(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _embedding_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _remaining_active_seconds(run: Run, config: AppConfig) -> float:
    limit = float(
        run.resources.get("run_active_timeout_seconds", config.runtime.run_active_timeout_seconds)
    )
    return limit - run.current_active_seconds()


def _verified_molecule_counts(facts: Mapping[str, Any]) -> dict[str, int]:
    """Derive composition from retained SMILES and tolerate only legacy H:0."""

    smiles = facts.get("isomeric_smiles") or facts.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles:
        stored = facts.get("element_counts")
        if not isinstance(stored, dict) or not stored:
            raise ValueError("molecule artifact has no verifiable element counts")
        _validate_stored_counts(stored)
        return {str(element): int(count) for element, count in stored.items() if count > 0}
    # Keep one production RDKit fact implementation.  The import is local to
    # avoid coupling module registration to the PubChem adapter.
    from bg6022.tools.pubchem import _facts_from_smiles

    derived = _facts_from_smiles(smiles).get("element_counts")
    if not isinstance(derived, dict) or not derived:
        raise ValueError("retained molecule SMILES produced no element counts")
    stored = facts.get("element_counts")
    if stored is not None:
        if not isinstance(stored, dict):
            raise ValueError("molecule artifact element_counts is invalid")
        _validate_stored_counts(stored)
        for element, count in stored.items():
            if count == 0 and element == "H":
                # Older artifacts could serialize implicit hydrogen as H:0;
                # the retained structure is authoritative for this one repair.
                continue
            if count != derived.get(element):
                raise ValueError("molecule artifact element counts disagree with its SMILES")
        for element, count in derived.items():
            if stored.get(element) != count and not (element == "H" and stored.get(element) == 0):
                raise ValueError("molecule artifact element counts disagree with its SMILES")
    return {str(element): int(count) for element, count in derived.items() if count > 0}


def _validate_stored_counts(stored: Mapping[str, Any]) -> None:
    for element, count in stored.items():
        if element not in SUPPORTED_ELEMENTS or type(count) is not int or count < 0:
            raise ValueError("molecule artifact element counts contain an invalid value")


def resolve_artifact_reference(
    config: AppConfig, run: Run, reference: InputReference, *, expected_type: str
):
    if reference.artifact_id is not None:
        artifact = find_artifact(run, reference.artifact_id)
    else:
        if reference.step_id not in run.current_results:
            raise ValueError(
                f"upstream result {reference.step_id} is not the current successful result"
            )
        root = run_directory(config.data_root_path, run.id).resolve()
        result_path = (root / run.current_results[reference.step_id]).resolve()
        if root not in result_path.parents or result_path.name != "result.json":
            raise ValueError("upstream result path is outside the Run directory")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "succeeded":
            raise ValueError(f"upstream step {reference.step_id} did not succeed")
        upstream = next((item for item in run.plan.steps if item.id == reference.step_id), None)
        if upstream is None or payload.get("step_fingerprint") != _step_fingerprint(upstream):
            raise ValueError(f"upstream step {reference.step_id} result is stale")
        artifact_id = payload.get("output_ports", {}).get(reference.port)
        if not artifact_id or artifact_id not in payload.get("artifact_ids", []):
            raise ValueError(f"upstream port {reference.step_id}.{reference.port} has no artifact")
        artifact = find_artifact(run, artifact_id)
        if artifact.step_id != reference.step_id or artifact.attempt != payload.get("attempt"):
            raise ValueError("upstream output is not bound to its successful producing attempt")
    if artifact.artifact_type != expected_type:
        raise ValueError(f"artifact {artifact.id} is not a {expected_type} artifact")
    if artifact.role == "restart_candidate":
        raise ValueError("restart_candidate is not a normal molecule/geometry input")
    return artifact


def _molecule_result(
    run: Run,
    step: Step,
    attempt: int,
    status: str,
    *,
    values: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    artifact_ids: list[str] | None = None,
    output_ports: dict[str, str] | None = None,
    parameter_sources: dict[str, str] | None = None,
    relative: str,
) -> Result:
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        status=status,  # type: ignore[arg-type]
        values=values or {},
        diagnostics=diagnostics or {},
        artifact_ids=artifact_ids or [],
        output_ports=output_ports or {},
        parameter_sources=parameter_sources or {},
        attempt_relative_path=relative,
    )


def _next_attempt(run: Run, step_id: str) -> int:
    attempts = [
        int(item.get("attempt", 0)) for item in run.attempts if item.get("step_id") == step_id
    ]
    return max(attempts, default=0) + 1


def _step_fingerprint(step: Step) -> str:
    import hashlib

    payload = json.dumps(
        step.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_xyz_bytes(
    value: bytes, *, supported_elements: frozenset[str] = SUPPORTED_ELEMENTS
) -> ParsedGeometry:
    if not isinstance(value, bytes) or not value:
        raise ValueError("XYZ content must be non-empty bytes")
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("XYZ content is not valid UTF-8") from error
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ is truncated; an atom count and comment are required")
    count_text = lines[0].strip()
    try:
        count = int(count_text)
    except ValueError as error:
        raise ValueError("XYZ atom count is not an integer") from error
    if count <= 0:
        raise ValueError("XYZ atom count must be positive")
    if len(lines) < count + 2:
        raise ValueError("XYZ has fewer atom lines than its atom count")
    if any(line.strip() for line in lines[count + 2 :]):
        raise ValueError("XYZ contains non-empty lines after the declared atoms")
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for index, line in enumerate(lines[2 : count + 2], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"XYZ atom line {index} is missing a symbol or coordinate")
        symbol = _canonical_symbol(parts[0])
        if symbol not in supported_elements:
            raise ValueError(f"unsupported element {symbol!r} on XYZ atom line {index}")
        try:
            point = tuple(float(item) for item in parts[1:4])
        except ValueError as error:
            raise ValueError(f"invalid coordinate on XYZ atom line {index}") from error
        if any(not math.isfinite(item) for item in point):
            raise ValueError(f"non-finite coordinate on XYZ atom line {index}")
        symbols.append(symbol)
        coordinates.append(point)  # type: ignore[arg-type]
    if len(symbols) != count:
        raise ValueError("XYZ atom count does not match parsed atoms")
    return ParsedGeometry(tuple(symbols), tuple(coordinates), lines[1], value)


def parse_xyz_file(
    path: str | Path, *, supported_elements: frozenset[str] = SUPPORTED_ELEMENTS
) -> ParsedGeometry:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read XYZ file {source}: {error}") from error
    return parse_xyz_bytes(raw, supported_elements=supported_elements)


def validate_electronic_state(
    geometry: ParsedGeometry,
    *,
    charge: int,
    multiplicity: int,
) -> None:
    if type(charge) is not int:
        raise ValueError("charge must be an integer, not a boolean or float")
    if type(multiplicity) is not int or multiplicity <= 0:
        raise ValueError("multiplicity must be a positive integer")
    electrons = sum(ATOMIC_NUMBERS[symbol] for symbol in geometry.symbols) - charge
    if electrons <= 0:
        raise ValueError("electronic state has no positive electron count")
    unpaired = multiplicity - 1
    if unpaired > electrons:
        raise ValueError("multiplicity is incompatible with the electron count")
    if (electrons - unpaired) % 2:
        raise ValueError("charge and multiplicity have incompatible electron parity")


def format_xyz(
    symbols: tuple[str, ...],
    coordinates: tuple[tuple[float, float, float], ...],
    *,
    comment: str = "BG6022 v3 geometry",
    precision: int = 12,
) -> bytes:
    if len(symbols) != len(coordinates) or not symbols:
        raise ValueError("geometry symbols and coordinates must have the same non-zero length")
    rows = [str(len(symbols)), comment]
    rows.extend(
        f"{symbol} {x:.{precision}f} {y:.{precision}f} {z:.{precision}f}"
        for symbol, (x, y, z) in zip(symbols, coordinates, strict=True)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def compare_coordinates(
    left: ParsedGeometry,
    right: ParsedGeometry,
    *,
    tolerance: float = 2e-6,
) -> tuple[bool, str | None]:
    if left.symbols != right.symbols:
        return False, "atom symbols or order differ"
    for index, (left_point, right_point) in enumerate(
        zip(left.coordinates, right.coordinates, strict=True), start=1
    ):
        for axis, (left_value, right_value) in enumerate(
            zip(left_point, right_point, strict=True), start=1
        ):
            if abs(left_value - right_value) > tolerance:
                return False, f"atom {index} coordinate {axis} differs beyond {tolerance:g} Å"
    return True, None


def _canonical_symbol(value: str) -> str:
    if not value or len(value) > 2:
        raise ValueError(f"invalid element symbol {value!r}")
    return value[0].upper() + value[1:].lower()


__all__ = [
    "ATOMIC_NUMBERS",
    "GenerateGeometryParameters",
    "ParsedGeometry",
    "SUPPORTED_ELEMENTS",
    "compare_coordinates",
    "execute_generate_geometry",
    "format_xyz",
    "make_generate_geometry_tool",
    "parse_xyz_bytes",
    "parse_xyz_file",
    "resolve_artifact_reference",
    "validate_electronic_state",
]
