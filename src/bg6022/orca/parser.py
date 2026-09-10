"""Evidence-first ORCA output observation for both successful and failed attempts."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bg6022.tools.molecule import (
    ParsedGeometry,
    compare_coordinates,
    format_xyz,
    parse_xyz_bytes,
)


@dataclass(frozen=True)
class EnergyObservation:
    token: str
    value: float
    line: int
    offset: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptFacts:
    operation: str
    runner_status: str
    exit_code: int | None
    stop_reason: str | None
    process_tree_empty: bool | None
    normal_marker_count: int
    normal_termination: bool | None
    scf_converged: bool | None
    optimization_converged: bool | None
    orca_version: str | None
    energies: list[EnergyObservation]
    output_geometry: ParsedGeometry | None
    stdout_geometry: ParsedGeometry | None
    output_geometry_bytes: bytes | None
    stdout_geometry_bytes: bytes | None
    geometry_consistent: bool | None
    geometry_mismatch_reason: str | None
    stdout_valid_utf8: bool
    stderr_preview: str
    stdout_preview: str
    source_locations: dict[str, Any]
    diagnostics: list[str]
    error_category: str | None

    @property
    def final_energy(self) -> EnergyObservation | None:
        return self.energies[-1] if self.energies else None

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["energies"] = [item.to_dict() for item in self.energies]
        values["output_geometry"] = _geometry_to_dict(self.output_geometry)
        values["stdout_geometry"] = _geometry_to_dict(self.stdout_geometry)
        values.pop("output_geometry_bytes", None)
        values.pop("stdout_geometry_bytes", None)
        return values


def inspect_attempt(
    *,
    operation: str,
    stdout: bytes | str | Path,
    stderr: bytes | str | Path,
    exit_code: int | None,
    runner_status: str,
    input_geometry: ParsedGeometry,
    stop_reason: str | None = None,
    process_tree_empty: bool | None = True,
    output_xyz: bytes | str | Path | None = None,
) -> AttemptFacts:
    """Collect facts without throwing away a nonzero or truncated attempt."""

    stdout_bytes = _read_bytes(stdout)
    stderr_bytes = _read_bytes(stderr)
    try:
        stdout_text = stdout_bytes.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        valid_utf8 = False
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    normal_matches = list(re.finditer(r"\*+ORCA TERMINATED NORMALLY\*+", stdout_text, re.I))
    scf_matches = list(re.finditer(r"\bSCF\s+CONVERGED\b", stdout_text, re.I))
    scf_failure = bool(
        re.search(
            r"SCF\s+(?:NOT\s+CONVERGED|CONVERGENCE\s+FAILED|FAILED)|SCF\s+iterations?\s+exceeded",
            stdout_text,
            re.I,
        )
    )
    optimization_match = re.search(
        r"(?:THE\s+)?OPTIMIZATION\s+HAS\s+CONVERGED|OPTIMIZATION\s+CONVERGED",
        stdout_text,
        re.I,
    )
    opt_failure = bool(
        re.search(
            r"OPTIMIZATION\s+(?:DID\s+NOT\s+CONVERGE|FAILED)|MAX(?:IMUM)?\s+NUMBER\s+OF\s+OPTIMIZATION\s+STEPS",
            stdout_text,
            re.I,
        )
    )
    version_match = re.search(
        r"(?:Program Version|ORCA Version|Version)\s*[:=]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
        stdout_text,
        re.I,
    )
    if version_match is None:
        version_match = re.search(r"ORCA\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", stdout_text, re.I)
    energies = _extract_energies(stdout_text)
    normal_termination = _normal_termination_is_clean(stdout_text, normal_matches)
    scf_converged = _scf_is_bound(energies, scf_matches, scf_failure)
    output_bytes = _read_optional_bytes(output_xyz)
    output_geometry = None
    output_geometry_error = None
    if output_bytes:
        try:
            output_geometry = parse_xyz_bytes(output_bytes)
        except ValueError as error:
            output_geometry_error = str(error)
    stdout_geometry_bytes = _extract_final_stdout_geometry(
        stdout_text, input_geometry.symbols, operation=operation
    )
    stdout_geometry = None
    stdout_geometry_error = None
    if stdout_geometry_bytes is not None:
        try:
            stdout_geometry = parse_xyz_bytes(stdout_geometry_bytes)
        except ValueError as error:
            stdout_geometry_error = str(error)
    consistent: bool | None = None
    mismatch_reason: str | None = None
    if output_geometry is not None and stdout_geometry is not None:
        consistent, mismatch_reason = compare_coordinates(output_geometry, stdout_geometry)
    diagnostics: list[str] = []
    if not stdout_bytes:
        diagnostics.append("stdout.out is empty")
    if not valid_utf8:
        diagnostics.append("stdout.out contains invalid UTF-8 bytes")
    if normal_matches and not normal_termination:
        diagnostics.append("normal termination marker has contradictory trailing output")
    if output_geometry_error:
        diagnostics.append(f"input.xyz is invalid: {output_geometry_error}")
    if stdout_geometry_error:
        diagnostics.append(f"stdout final geometry is invalid: {stdout_geometry_error}")
    if mismatch_reason:
        diagnostics.append(f"output geometry mismatch: {mismatch_reason}")
    if stderr_text.strip():
        diagnostics.append("stderr.txt contains output")
    category = _classify_failure(
        runner_status=runner_status,
        exit_code=exit_code,
        stop_reason=stop_reason,
        operation=operation,
        stdout_text=stdout_text,
        stdout_bytes=stdout_bytes,
        normal_marker_count=len(normal_matches),
        scf_failure=scf_failure,
        optimization_match=optimization_match is not None,
        opt_failure=opt_failure,
        output_geometry=output_geometry,
        stdout_geometry=stdout_geometry,
        geometry_consistent=consistent,
    )
    if category is not None:
        diagnostics.append(f"classified failure: {category}")
    return AttemptFacts(
        operation=operation,
        runner_status=runner_status,
        exit_code=exit_code,
        stop_reason=stop_reason,
        process_tree_empty=process_tree_empty,
        normal_marker_count=len(normal_matches),
        normal_termination=normal_termination if stdout_bytes else None,
        scf_converged=scf_converged,
        optimization_converged=(optimization_match is not None) if operation == "Opt" else None,
        orca_version=None if version_match is None else version_match.group(1),
        energies=energies,
        output_geometry=output_geometry,
        stdout_geometry=stdout_geometry,
        output_geometry_bytes=output_bytes,
        stdout_geometry_bytes=stdout_geometry_bytes,
        geometry_consistent=consistent,
        geometry_mismatch_reason=mismatch_reason,
        stdout_valid_utf8=valid_utf8,
        stderr_preview=stderr_text[-2000:],
        stdout_preview=stdout_text[-4000:],
        source_locations={
            "normal_termination": _line_for_offset(stdout_text, normal_matches[0].start())
            if normal_matches
            else None,
            "energy": [_line_for_offset(stdout_text, item.offset) for item in energies],
            "optimization_converged": _line_for_offset(stdout_text, optimization_match.start())
            if optimization_match
            else None,
        },
        diagnostics=diagnostics,
        error_category=category,
    )


def _extract_energies(text: str) -> list[EnergyObservation]:
    matches = re.finditer(
        r"(?:FINAL SINGLE POINT ENERGY|FINAL ENERGY)\s+([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        text,
        re.I,
    )
    observations: list[EnergyObservation] = []
    for match in matches:
        token = match.group(1)
        try:
            value = float(token.replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        if math.isfinite(value):
            observations.append(
                EnergyObservation(
                    token, value, _line_for_offset(text, match.start()), match.start()
                )
            )
    return observations


def _normal_termination_is_clean(text: str, matches: list[re.Match[str]]) -> bool:
    if len(matches) != 1:
        return False
    tail = text[matches[0].end() :]
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped and not re.fullmatch(
            r"TOTAL RUN TIME:\s*\d+ days \d+ hours \d+ minutes \d+ seconds \d+ msec",
            stripped,
            re.I,
        ):
            return False
    return True


def _scf_is_bound(
    energies: list[EnergyObservation],
    matches: list[re.Match[str]],
    scf_failure: bool,
) -> bool | None:
    if scf_failure:
        return False
    if not energies:
        return None
    previous_energy = -1
    for energy in energies:
        if not any(previous_energy < match.start() < energy.offset for match in matches):
            return False
        previous_energy = energy.offset
    return True


def _extract_final_stdout_geometry(
    text: str, expected_symbols: tuple[str, ...], *, operation: str
) -> bytes | None:
    if operation != "Opt":
        return None
    converged = list(re.finditer(r"(?:THE\s+)?OPTIMIZATION\s+(?:HAS\s+)?CONVERGED", text, re.I))
    if converged:
        tail = text[converged[-1].end() :]
    else:
        cycles = list(re.finditer(r"GEOMETRY\s+OPTIMIZATION\s+CYCLE", text, re.I))
        tail = text[cycles[-1].start() :] if cycles else text
    stop = re.search(r"\*+ORCA TERMINATED", tail, re.I)
    if stop:
        tail = tail[: stop.start()]
    marker = re.compile(r"CARTESIAN\s+COORDINATES\s*\(ANGSTROEM\)", re.I)
    candidates: list[bytes] = []
    for match in marker.finditer(tail):
        section = tail[match.end() :]
        rows: list[tuple[str, tuple[float, float, float]]] = []
        for line in section.splitlines()[:80]:
            if re.search(r"CARTESIAN\s+COORDINATES\s*\(A\.?U\.?\)", line, re.I):
                break
            if rows and (not line.strip() or set(line.strip()) <= {"-"}):
                break
            parts = line.split()
            symbol_index = None
            for index in (0, 1):
                if index < len(parts) and parts[index] in expected_symbols:
                    symbol_index = index
                    break
            if symbol_index is None or len(parts) < symbol_index + 4:
                continue
            try:
                point = tuple(float(value) for value in parts[symbol_index + 1 : symbol_index + 4])
            except ValueError:
                continue
            if len(point) == 3 and all(math.isfinite(value) for value in point):
                rows.append((parts[symbol_index], point))  # type: ignore[arg-type]
                if len(rows) == len(expected_symbols):
                    break
        if len(rows) == len(expected_symbols):
            symbols = tuple(item[0] for item in rows)
            if symbols == expected_symbols:
                candidates.append(
                    format_xyz(
                        symbols, tuple(item[1] for item in rows), comment="ORCA final geometry"
                    )
                )
    return candidates[-1] if candidates else None


def _classify_failure(
    *,
    runner_status: str,
    exit_code: int | None,
    stop_reason: str | None,
    operation: str,
    stdout_text: str,
    stdout_bytes: bytes,
    normal_marker_count: int,
    scf_failure: bool,
    optimization_match: bool,
    opt_failure: bool,
    output_geometry: ParsedGeometry | None,
    stdout_geometry: ParsedGeometry | None,
    geometry_consistent: bool | None,
) -> str | None:
    reason = (stop_reason or "").casefold()
    status = runner_status.casefold()
    if "cancel" in reason or status == "cancelled":
        return "cancelled"
    if "deadline" in reason or "timeout" in reason or status == "timed_out":
        return "timeout"
    if "limit" in reason:
        return "resource_limit"
    if "cleanup" in reason or status == "interrupted":
        return "cleanup_unconfirmed"
    if "launch" in reason or status == "launch_failed":
        return "launch_failed"
    if scf_failure:
        return "scf_not_converged"
    if operation == "Opt" and opt_failure and not optimization_match:
        return "opt_not_converged"
    if (
        operation == "Opt"
        and not optimization_match
        and exit_code != 0
        and re.search(r"GEOMETRY\s+OPTIMIZATION\s+CYCLE", stdout_text, re.I)
    ):
        return "opt_not_converged"
    if (
        operation == "Opt"
        and optimization_match
        and output_geometry is not None
        and stdout_geometry is not None
        and geometry_consistent is False
    ):
        return "geometry_mismatch"
    if runner_status != "succeeded" or exit_code != 0:
        return "invalid_output" if stdout_bytes else "unknown_failure"
    if normal_marker_count != 1 or not stdout_bytes:
        return "invalid_output"
    return None


def _read_bytes(value: bytes | str | Path) -> bytes:
    if isinstance(value, bytes):
        return value
    return Path(value).read_bytes() if Path(value).exists() else b""


def _read_optional_bytes(value: bytes | str | Path | None) -> bytes | None:
    if value is None:
        return None
    data = _read_bytes(value)
    return data or None


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _geometry_to_dict(value: ParsedGeometry | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "symbols": list(value.symbols),
        "coordinates": [list(point) for point in value.coordinates],
        "comment": value.comment,
    }


__all__ = ["AttemptFacts", "EnergyObservation", "inspect_attempt"]
