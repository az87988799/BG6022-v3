"""Evidence-first ORCA output observation for successful and failed attempts."""

from __future__ import annotations

import math
import os
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

_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_ENERGY_LABEL_RE = re.compile(
    r"(?P<label>FINAL\s+SINGLE\s+POINT\s+ENERGY|FINAL\s+ENERGY)"
    r"(?!\s+EVALUATION)\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_STATIONARY_RE = re.compile(
    r"FINAL\s+ENERGY\s+EVALUATION\s+AT\s+THE\s+STATIONARY\s+POINT", re.IGNORECASE
)
_CYCLE_RE = re.compile(r"GEOMETRY\s+OPTIMIZATION\s+CYCLE", re.IGNORECASE)


@dataclass(frozen=True)
class EnergyObservation:
    token: str
    value: float | None
    line: int
    offset: int
    valid: bool = True
    error: str | None = None
    label: str = ""

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
    final_energy: EnergyObservation | None
    final_energy_error: str | None
    output_geometry: ParsedGeometry | None
    stdout_geometry: ParsedGeometry | None
    output_geometry_bytes: bytes | None
    stdout_geometry_bytes: bytes | None
    output_xyz_exists: bool
    output_geometry_error: str | None
    stdout_geometry_error: str | None
    geometry_consistent: bool | None
    geometry_mismatch_reason: str | None
    stdout_valid_utf8: bool
    stdout_size_exceeded: bool
    stderr_size_exceeded: bool
    input_hashes_match: bool
    input_hash_error: str | None
    stderr_preview: str
    stdout_preview: str
    source_locations: dict[str, Any]
    diagnostics: list[str]
    error_category: str | None
    opt_iteration_limit_reached: bool | None = None
    last_opt_cycle: int | None = None
    effective_geom_maxiter: int | None = None
    scf_iteration_limit_reached: bool | None = None
    effective_scf_maxiter: int | None = None
    effective_geom_maxiter_source: str | None = None
    effective_geom_maxiter_error: str | None = None
    scf_near_converged: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["energies"] = [item.to_dict() for item in self.energies]
        values["final_energy"] = None if self.final_energy is None else self.final_energy.to_dict()
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
    input_hashes_match: bool = True,
    input_hash_error: str | None = None,
    max_output_bytes: int = 64 * 1024 * 1024,
    effective_geom_maxiter: int | None = None,
    effective_scf_maxiter: int | None = None,
) -> AttemptFacts:
    """Collect facts without discarding a nonzero or truncated attempt."""

    stdout_bytes, stdout_size_exceeded = _read_bytes(stdout, max_output_bytes)
    stderr_bytes, stderr_size_exceeded = _read_bytes(stderr, max_output_bytes)
    try:
        stdout_text = stdout_bytes.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        valid_utf8 = False
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    (
        observed_geom_maxiter,
        geom_maxiter_line,
        geom_maxiter_error,
    ) = _extract_effective_geom_maxiter(stdout_text)
    if observed_geom_maxiter is not None:
        if effective_geom_maxiter is not None and effective_geom_maxiter != observed_geom_maxiter:
            geom_maxiter_error = (
                "input geom_maxiter "
                f"{effective_geom_maxiter} conflicts with ORCA geometry-settings value "
                f"{observed_geom_maxiter}"
            )
            effective_geom_maxiter = None
            geom_maxiter_source = None
        else:
            effective_geom_maxiter = observed_geom_maxiter
            geom_maxiter_source = "orca_output"
    elif effective_geom_maxiter is not None:
        # Explicit settings remain usable for older/synthetic output formats,
        # but are marked as input-derived.  A missing output setting can never
        # invent a default value for a None input.
        geom_maxiter_source = "input_parameter"
    else:
        geom_maxiter_source = None

    normal_matches = list(re.finditer(r"\*+ORCA TERMINATED NORMALLY\*+", stdout_text, re.I))
    scf_matches = list(re.finditer(r"\bSCF\s+CONVERGED\b", stdout_text, re.I))
    scf_failure = bool(
        re.search(
            r"SCF\s+(?:NOT\s+CONVERGED|CONVERGENCE\s+FAILED|FAILED)|"
            r"SCF\s+iterations?\s+exceeded",
            stdout_text,
            re.I,
        )
    )
    optimization_match = re.search(
        r"(?:THE\s+)?OPTIMIZATION\s+(?:HAS\s+)?CONVERGED|OPTIMIZATION\s+CONVERGED",
        stdout_text,
        re.I,
    )
    opt_failure = bool(
        re.search(
            r"OPTIMIZATION\s+(?:DID\s+NOT\s+CONVERGE|FAILED)|"
            r"MAX(?:IMUM)?\s+NUMBER\s+OF\s+OPTIMIZATION\s+STEPS",
            stdout_text,
            re.I,
        )
    )
    opt_limit_match = re.search(
        r"MAX(?:IMUM)?\s+NUMBER\s+OF\s+(?:OPTIMIZATION\s+STEPS|GEOMETRY\s+OPTIMIZATION\s+STEPS)"
        r"|(?:GEOM|OPTIMIZATION).*?(?:MAXITER|ITERATION\s+LIMIT).*?(?:REACHED|EXCEEDED|LIMIT)",
        stdout_text,
        re.I,
    )
    scf_limit_match = re.search(
        r"(?:SCF\s+)?(?:MAX(?:IMUM)?\s+NUMBER\s+OF\s+(?:SCF\s+)?ITERATIONS?|"
        r"SCF\s+ITERATIONS?\s+(?:EXCEEDED|REACHED)|SCF\s+MAXITER)",
        stdout_text,
        re.I,
    )
    version_match = re.search(
        r"(?:Program Version|ORCA Version|Version)\s*[:=]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
        stdout_text,
        re.I,
    )
    if version_match is None:
        version_match = re.search(r"ORCA\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", stdout_text, re.I)

    energies = _extract_energies(stdout_text)
    final_energy, final_energy_error, final_section = _select_final_energy(
        stdout_text, operation, energies
    )
    unbound_scf = None
    if final_energy is not None and final_energy_error is None:
        later_scfs = [match for match in scf_matches if match.start() > final_energy.offset]
        if later_scfs:
            unbound_scf = later_scfs[-1]
            final_energy_error = "later SCF convergence has no corresponding final energy record"
            final_energy = None
    normal_termination = _normal_termination_is_clean(stdout_text, normal_matches)
    scf_converged = _scf_is_bound(final_energy, energies, scf_matches, scf_failure)

    output_exists = _value_exists(output_xyz)
    output_bytes = _read_optional_bytes(output_xyz, max_output_bytes)
    output_geometry = None
    output_geometry_error = None
    if output_bytes is not None:
        try:
            output_geometry = parse_xyz_bytes(output_bytes)
        except ValueError as error:
            output_geometry_error = str(error)
    elif output_exists:
        output_geometry_error = "input.xyz exists but is empty or unreadable"

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
    if stdout_size_exceeded:
        diagnostics.append("stdout.out exceeds the parser size bound")
    if stderr_size_exceeded:
        diagnostics.append("stderr.txt exceeds the parser size bound")
    if not valid_utf8:
        diagnostics.append("stdout.out contains invalid UTF-8 bytes")
    if normal_matches and not normal_termination:
        diagnostics.append("normal termination marker has contradictory trailing output")
    if final_energy_error:
        diagnostics.append(f"final energy is invalid: {final_energy_error}")
    if opt_limit_match:
        diagnostics.append("optimization iteration limit was reported by ORCA output")
    if scf_limit_match:
        diagnostics.append("SCF iteration limit was reported by ORCA output")
    if geom_maxiter_error:
        diagnostics.append(f"effective geometry iteration limit is unusable: {geom_maxiter_error}")
    if output_geometry_error:
        diagnostics.append(f"input.xyz is invalid: {output_geometry_error}")
    if stdout_geometry_error:
        diagnostics.append(f"stdout final geometry is invalid: {stdout_geometry_error}")
    if mismatch_reason:
        diagnostics.append(f"output geometry mismatch: {mismatch_reason}")
    if not input_hashes_match:
        diagnostics.append(input_hash_error or "execution input hash changed after launch")
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
        final_energy_error=final_energy_error,
        stdout_valid_utf8=valid_utf8,
        output_geometry=output_geometry,
        stdout_geometry=stdout_geometry,
        geometry_consistent=consistent,
        input_hashes_match=input_hashes_match,
        output_size_exceeded=stdout_size_exceeded or stderr_size_exceeded,
    )
    if category is not None:
        diagnostics.append(f"classified failure: {category}")

    final_scf = _final_scf_location(final_energy, energies, scf_matches)
    source_locations = {
        "normal_termination": _line_for_offset(stdout_text, normal_matches[0].start())
        if normal_matches
        else None,
        "energy": [_line_for_offset(stdout_text, item.offset) for item in energies],
        "final_energy_section": None
        if final_section is None
        else _line_for_offset(stdout_text, final_section),
        "final_energy": None
        if final_energy is None
        else _line_for_offset(stdout_text, final_energy.offset),
        "final_scf": None
        if final_scf is None
        else _line_for_offset(stdout_text, final_scf.start()),
        "unbound_scf": None
        if unbound_scf is None
        else _line_for_offset(stdout_text, unbound_scf.start()),
        "optimization_converged": _line_for_offset(stdout_text, optimization_match.start())
        if optimization_match
        else None,
        "opt_iteration_limit": _line_for_offset(stdout_text, opt_limit_match.start())
        if opt_limit_match
        else None,
        "scf_iteration_limit": _line_for_offset(stdout_text, scf_limit_match.start())
        if scf_limit_match
        else None,
        "effective_geom_maxiter": geom_maxiter_line,
        "effective_geom_maxiter_source": geom_maxiter_source,
    }
    if geom_maxiter_error:
        source_locations["effective_geom_maxiter_error"] = geom_maxiter_error
    if input_hash_error:
        source_locations["input_hash_error"] = input_hash_error

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
        final_energy=final_energy,
        final_energy_error=final_energy_error,
        output_geometry=output_geometry,
        stdout_geometry=stdout_geometry,
        output_geometry_bytes=output_bytes,
        stdout_geometry_bytes=stdout_geometry_bytes,
        output_xyz_exists=output_exists,
        output_geometry_error=output_geometry_error,
        stdout_geometry_error=stdout_geometry_error,
        geometry_consistent=consistent,
        geometry_mismatch_reason=mismatch_reason,
        stdout_valid_utf8=valid_utf8,
        stdout_size_exceeded=stdout_size_exceeded,
        stderr_size_exceeded=stderr_size_exceeded,
        input_hashes_match=input_hashes_match,
        input_hash_error=input_hash_error,
        stderr_preview=stderr_text[-2000:],
        stdout_preview=stdout_text[-4000:],
        source_locations=source_locations,
        diagnostics=diagnostics,
        error_category=category,
        opt_iteration_limit_reached=(
            True
            if opt_limit_match
            else (False if operation == "Opt" and optimization_match is not None else None)
        ),
        last_opt_cycle=(len(list(_CYCLE_RE.finditer(stdout_text))) if operation == "Opt" else None),
        effective_geom_maxiter=effective_geom_maxiter,
        scf_iteration_limit_reached=(
            True if scf_limit_match else (False if not scf_failure else None)
        ),
        effective_scf_maxiter=effective_scf_maxiter,
        effective_geom_maxiter_source=geom_maxiter_source,
        effective_geom_maxiter_error=geom_maxiter_error,
        scf_near_converged=None,
    )


def _extract_effective_geom_maxiter(text: str) -> tuple[int | None, int | None, str | None]:
    """Read MaxIter only from ORCA's geometry-optimization settings block.

    ORCA prints another ``MaxIter`` in its SCF section.  Searching the whole
    output would therefore associate an electronic iteration limit with a
    geometry repair.  Unknown or conflicting geometry blocks return no value
    so the repair admission rule stops conservatively.
    """

    lines = text.splitlines()
    header = re.compile(r"^\s*Geometry\s+optimization\s+settings\s*:\s*$", re.I)
    row = re.compile(
        r"^\s*Max\.?\s+no\.?\s+of\s+cycles\s+MaxIter\s+\.\.\.\.\s*(\d+)\s*$",
        re.I,
    )
    values: list[tuple[int, int]] = []
    malformed_sections = 0
    for index, line in enumerate(lines):
        if not header.match(line):
            continue
        found: tuple[int, int] | None = None
        for offset in range(index + 1, min(len(lines), index + 80)):
            candidate = lines[offset]
            stripped = candidate.strip()
            if not stripped:
                break
            if re.match(r"^\s*(?:Convergence\s+Tolerances|SCF\s+Procedure)\s*:", candidate, re.I):
                break
            match = row.match(candidate)
            if match:
                found = (int(match.group(1)), offset + 1)
                break
        if found is None:
            malformed_sections += 1
        else:
            values.append(found)

    distinct = {value for value, _line in values}
    if len(distinct) > 1:
        return None, None, "conflicting geometry optimization MaxIter values in ORCA output"
    if malformed_sections and not values:
        return None, None, "geometry optimization settings block has no parseable MaxIter"
    if malformed_sections and values:
        return None, None, "one geometry optimization settings block has no parseable MaxIter"
    if not values:
        return None, None, None
    value, line = values[-1]
    return value, line, None


def _extract_energies(text: str) -> list[EnergyObservation]:
    """Return every labelled energy, including malformed final records."""

    observations: list[EnergyObservation] = []
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        match = _ENERGY_LABEL_RE.search(line)
        if match:
            raw = match.group("value").strip()
            label = re.sub(r"\s+", " ", match.group("label")).upper()
            if not raw:
                observations.append(
                    EnergyObservation(
                        token="",
                        value=None,
                        line=line_number,
                        offset=offset + match.start(),
                        valid=False,
                        error="missing numeric token",
                        label=label,
                    )
                )
            elif not _NUMBER_RE.fullmatch(raw):
                observations.append(
                    EnergyObservation(
                        token=raw,
                        value=None,
                        line=line_number,
                        offset=offset + match.start(),
                        valid=False,
                        error="numeric token is not a complete finite number",
                        label=label,
                    )
                )
            else:
                try:
                    value = float(raw.replace("D", "E").replace("d", "e"))
                except (OverflowError, ValueError):
                    value = None
                if value is None or not math.isfinite(value):
                    observations.append(
                        EnergyObservation(
                            token=raw,
                            value=None,
                            line=line_number,
                            offset=offset + match.start(),
                            valid=False,
                            error="numeric token is not finite",
                            label=label,
                        )
                    )
                else:
                    observations.append(
                        EnergyObservation(
                            token=raw,
                            value=value,
                            line=line_number,
                            offset=offset + match.start(),
                            label=label,
                        )
                    )
        offset += len(line)
    return observations


def _select_final_energy(
    text: str, operation: str, energies: list[EnergyObservation]
) -> tuple[EnergyObservation | None, str | None, int | None]:
    if not energies:
        return None, "missing final energy record", None

    if operation != "Opt":
        selected = energies[-1]
        return selected, None if selected.valid else selected.error, 0

    stationary = list(_STATIONARY_RE.finditer(text))
    cycles = list(_CYCLE_RE.finditer(text))
    last_stationary = stationary[-1] if stationary else None
    last_cycle = cycles[-1] if cycles else None

    if last_stationary is not None and (
        last_cycle is None or last_stationary.start() > last_cycle.start()
    ):
        section_start = last_stationary.start()
        candidates = [item for item in energies if item.offset > section_start]
        if not candidates:
            return None, "final stationary-point section has no energy record", section_start
        if len(candidates) != 1:
            return (
                None,
                "final stationary-point section has ambiguous energy records",
                section_start,
            )
        selected = candidates[0]
        return selected, None if selected.valid else selected.error, section_start

    section_start = last_cycle.start() if last_cycle is not None else 0
    candidates = [item for item in energies if item.offset > section_start]
    if not candidates:
        return None, "last optimization section has no energy record", section_start
    selected = candidates[-1]
    return selected, None if selected.valid else selected.error, section_start


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
    final_energy: EnergyObservation | None,
    energies: list[EnergyObservation],
    matches: list[re.Match[str]],
    scf_failure: bool,
) -> bool | None:
    if scf_failure:
        return False
    if final_energy is None or not final_energy.valid:
        return None
    previous = [item.offset for item in energies if item.offset < final_energy.offset]
    lower = previous[-1] if previous else -1
    return any(lower < match.start() < final_energy.offset for match in matches)


def _final_scf_location(
    final_energy: EnergyObservation | None,
    energies: list[EnergyObservation],
    matches: list[re.Match[str]],
) -> re.Match[str] | None:
    if final_energy is None:
        return None
    previous = [item.offset for item in energies if item.offset < final_energy.offset]
    lower = previous[-1] if previous else -1
    candidates = [match for match in matches if lower < match.start() < final_energy.offset]
    return candidates[-1] if candidates else None


def _extract_final_stdout_geometry(
    text: str, expected_symbols: tuple[str, ...], *, operation: str
) -> bytes | None:
    if operation != "Opt":
        return None
    converged = list(re.finditer(r"(?:THE\s+)?OPTIMIZATION\s+(?:HAS\s+)?CONVERGED", text, re.I))
    if converged:
        tail = text[converged[-1].end() :]
    else:
        cycles = list(_CYCLE_RE.finditer(text))
        tail = text[cycles[-1].start() :] if cycles else text
    stop = re.search(r"\*+ORCA TERMINATED", tail, re.I)
    if stop:
        tail = tail[: stop.start()]
    marker = re.compile(r"CARTESIAN\s+COORDINATES\s*\(ANGSTROEM\)", re.I)
    markers = list(marker.finditer(tail))
    if not markers:
        return None
    # The final marker is authoritative. A damaged final block must not fall
    # back to an earlier geometry that happened to parse.
    return _parse_geometry_block(tail[markers[-1].end() :], expected_symbols)


def _parse_geometry_block(section: str, expected_symbols: tuple[str, ...]) -> bytes | None:
    rows: list[tuple[str, tuple[float, float, float]]] = []
    for line in section.splitlines()[:120]:
        stripped = line.strip()
        if re.search(r"CARTESIAN\s+COORDINATES\s*\(A\.?U\.?\)", line, re.I):
            break
        if rows and (not stripped or set(stripped) <= {"-"}):
            break
        if not stripped or set(stripped) <= {"-"}:
            continue
        parts = stripped.split()
        symbol_index = None
        for index in (0, 1):
            if index < len(parts) and parts[index] in expected_symbols:
                symbol_index = index
                break
        if symbol_index is None:
            continue
        if len(parts) < symbol_index + 4:
            return None
        try:
            point = tuple(float(value) for value in parts[symbol_index + 1 : symbol_index + 4])
        except ValueError:
            return None
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        rows.append((parts[symbol_index], point))  # type: ignore[arg-type]
        if len(rows) == len(expected_symbols):
            symbols = tuple(item[0] for item in rows)
            if symbols != expected_symbols:
                return None
            return format_xyz(
                symbols, tuple(item[1] for item in rows), comment="ORCA final geometry"
            )
    return None


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
    final_energy_error: str | None,
    stdout_valid_utf8: bool,
    output_geometry: ParsedGeometry | None,
    stdout_geometry: ParsedGeometry | None,
    geometry_consistent: bool | None,
    input_hashes_match: bool,
    output_size_exceeded: bool,
) -> str | None:
    reason = (stop_reason or "").casefold()
    status = runner_status.casefold()
    if "cancel" in reason or status == "cancelled":
        return "cancelled"
    if "deadline" in reason or "timeout" in reason or status == "timed_out":
        return "timeout"
    if "limit" in reason or output_size_exceeded:
        return "resource_limit"
    if "cleanup" in reason or status == "interrupted":
        return "cleanup_unconfirmed"
    if "resource_lock" in reason:
        return "resource_lock"
    if "launch" in reason or status == "launch_failed":
        return "launch_failed"
    if not input_hashes_match:
        return "input_integrity_error"
    if not stdout_valid_utf8 or final_energy_error:
        if not (final_energy_error == "missing final energy record" and scf_failure):
            return "invalid_output"
    if scf_failure:
        return "scf_not_converged"
    if operation == "Opt" and opt_failure and not optimization_match:
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


def _read_bytes(value: bytes | str | Path, max_bytes: int) -> tuple[bytes, bool]:
    if isinstance(value, bytes):
        return value[: max_bytes + 1], len(value) > max_bytes
    path = Path(value)
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except (FileNotFoundError, IsADirectoryError, OSError):
        return b"", False
    return data[: max_bytes + 1], len(data) > max_bytes


def _read_optional_bytes(value: bytes | str | Path | None, max_bytes: int) -> bytes | None:
    if value is None:
        return None
    data, _exceeded = _read_bytes(value, max_bytes)
    return data or None


def _value_exists(value: bytes | str | Path | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        return True
    try:
        return os.path.lexists(Path(value))
    except OSError:
        return True


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
