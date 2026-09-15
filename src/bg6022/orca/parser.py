"""Evidence-first ORCA output observation for successful and failed attempts."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from bg6022.tools.molecule import (
    ParsedGeometry,
    compare_coordinates,
    format_xyz,
    parse_xyz_bytes,
)

from .frequency_parser import VibrationalFrequencySection, parse_vibrational_frequencies

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
_HESSIAN_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


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
    frequency_section: VibrationalFrequencySection | None = None
    hessian_present: bool = False
    hessian_valid: bool | None = None
    hessian_dimension: int | None = None
    hessian_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["energies"] = [item.to_dict() for item in self.energies]
        values["final_energy"] = None if self.final_energy is None else self.final_energy.to_dict()
        values["output_geometry"] = _geometry_to_dict(self.output_geometry)
        values["stdout_geometry"] = _geometry_to_dict(self.stdout_geometry)
        values["frequency_section"] = (
            None if self.frequency_section is None else asdict(self.frequency_section)
        )
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
    hessian: bytes | str | Path | None = None,
    expected_atom_count: int | None = None,
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
            r"MAX(?:IMUM)?\s+NUMBER\s+OF\s+(?:GEOMETRY\s+)?OPTIMIZATION\s+"
            r"(?:STEPS?|CYCLES?)",
            stdout_text,
            re.I,
        )
    )
    opt_limit_match = re.search(
        r"MAX(?:IMUM)?\s+NUMBER\s+OF\s+(?:GEOMETRY\s+)?OPTIMIZATION\s+"
        r"(?:STEPS?|CYCLES?)"
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
    if operation == "Freq" and final_energy_error == "missing final energy record":
        final_energy_error = None
    unbound_scf = None
    if final_energy is not None and final_energy_error is None:
        later_scfs = [match for match in scf_matches if match.start() > final_energy.offset]
        if later_scfs:
            unbound_scf = later_scfs[-1]
            final_energy_error = "later SCF convergence has no corresponding final energy record"
            final_energy = None
    normal_termination = _normal_termination_is_clean(stdout_text, normal_matches)
    scf_converged = _scf_is_bound(final_energy, energies, scf_matches, scf_failure)
    frequency_heading = re.search(r"^\s*VIBRATIONAL\s+FREQUENCIES\s*$", stdout_text, re.I | re.M)
    if operation == "Freq":
        scf_before_frequencies = bool(
            frequency_heading
            and any(match.start() < frequency_heading.start() for match in scf_matches)
        )
        scf_converged = False if scf_failure else (True if scf_before_frequencies else None)

    frequency_section = None
    hessian_present = False
    hessian_valid: bool | None = None
    hessian_dimension = None
    hessian_error = None
    if operation == "Freq":
        atom_count = input_geometry.atom_count
        frequency_section = parse_vibrational_frequencies(
            stdout_text, expected_atom_count=atom_count
        )
        if expected_atom_count is not None and (
            type(expected_atom_count) is not int or expected_atom_count != atom_count
        ):
            reason = (
                "expected_atom_count does not match the input geometry "
                f"({expected_atom_count!r} != {atom_count})"
            )
            frequency_section = replace(
                frequency_section,
                complete=False,
                error=f"{frequency_section.error}; {reason}" if frequency_section.error else reason,
            )
            hessian_error = reason

        hessian_present = _value_exists(hessian)
        if hessian is None or not hessian_present:
            hessian_valid = False
            hessian_error = hessian_error or "frequency Hessian file is missing"
        else:
            hessian_bytes, hessian_size_exceeded = _read_bytes(hessian, max_output_bytes)
            if hessian_size_exceeded:
                hessian_valid = False
                hessian_error = hessian_error or "frequency Hessian exceeds the parser size bound"
            elif not hessian_bytes:
                hessian_valid = False
                hessian_error = hessian_error or "frequency Hessian file is empty or unreadable"
            else:
                hessian_valid, hessian_dimension, parsed_hessian_error = _inspect_hessian(
                    hessian_bytes,
                    expected_atom_count=atom_count,
                    input_geometry=input_geometry,
                    frequency_section=frequency_section,
                )
                hessian_error = hessian_error or parsed_hessian_error

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
    if final_energy_error and operation != "Freq":
        diagnostics.append(f"final energy is invalid: {final_energy_error}")
    if frequency_section is not None and frequency_section.error:
        diagnostics.append(f"vibrational frequencies are incomplete: {frequency_section.error}")
    if hessian_error:
        diagnostics.append(f"frequency Hessian is invalid: {hessian_error}")
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
    if frequency_heading:
        source_locations["frequency_section"] = _line_for_offset(
            stdout_text, frequency_heading.start()
        )
    if operation == "Freq":
        source_locations["frequency_scf"] = (
            _line_for_offset(
                stdout_text,
                next(
                    match.start()
                    for match in reversed(scf_matches)
                    if frequency_heading and match.start() < frequency_heading.start()
                ),
            )
            if scf_converged is True
            else None
        )
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
        frequency_section=frequency_section,
        hessian_present=hessian_present,
        hessian_valid=hessian_valid,
        hessian_dimension=hessian_dimension,
        hessian_error=hessian_error,
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
    if not stdout_valid_utf8 or (operation != "Freq" and final_energy_error):
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


def _inspect_hessian(
    raw: bytes,
    *,
    expected_atom_count: int,
    input_geometry: ParsedGeometry,
    frequency_section: VibrationalFrequencySection,
) -> tuple[bool, int | None, str | None]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, None, "frequency Hessian is not valid UTF-8"

    hessian_lines, section_error = _named_section_lines(text, "$hessian")
    if section_error:
        return False, None, section_error
    assert hessian_lines is not None

    values = [line.strip() for line in hessian_lines if line.strip()]
    if not values or not re.fullmatch(r"\d+", values[0]):
        return False, None, "frequency Hessian dimension is missing or invalid"
    dimension = int(values[0])
    expected_dimension = 3 * expected_atom_count
    if dimension != expected_dimension:
        return (
            False,
            dimension,
            f"frequency Hessian dimension is {dimension}; expected {expected_dimension}",
        )

    atoms_lines, atoms_error = _named_section_lines(text, "$atoms")
    if atoms_error:
        return False, dimension, atoms_error
    assert atoms_lines is not None
    geometry_error = _validate_hessian_atoms(
        atoms_lines,
        expected_atom_count=expected_atom_count,
        input_geometry=input_geometry,
    )
    if geometry_error:
        return False, dimension, geometry_error

    matrix_error = _validate_hessian_matrix(values[1:], dimension=dimension)
    if matrix_error:
        return False, dimension, matrix_error

    frequency_error = _validate_hessian_frequency_data(text, frequency_section=frequency_section)
    if frequency_error:
        return False, dimension, frequency_error
    return True, dimension, None


def _validate_hessian_frequency_data(
    text: str, *, frequency_section: VibrationalFrequencySection
) -> str | None:
    if not frequency_section.complete or frequency_section.error:
        return "stdout frequency section is incomplete; Hessian frequencies cannot be bound"
    if frequency_section.scaling_factor is None:
        return "stdout frequency scaling factor is unavailable for Hessian binding"

    frequency_lines, frequency_error = _named_section_lines(text, "$vibrational_frequencies")
    if frequency_error:
        return frequency_error
    assert frequency_lines is not None
    values = [line.strip() for line in frequency_lines if line.strip()]
    if not values or not re.fullmatch(r"\d+", values[0]):
        return "frequency Hessian frequency count is missing or invalid"
    count = int(values[0])
    if count != len(frequency_section.modes):
        return (
            "frequency Hessian has "
            f"{count} frequencies; stdout reports {len(frequency_section.modes)}"
        )
    if len(values) != count + 1:
        return "frequency Hessian frequency block is truncated or has extra rows"

    for expected_index, (line, stdout_mode) in enumerate(
        zip(values[1:], frequency_section.modes, strict=True)
    ):
        row = line.split()
        if len(row) != 2 or not re.fullmatch(r"\d+", row[0]):
            return f"frequency Hessian frequency row {expected_index} has an unsupported layout"
        index = int(row[0])
        hessian_frequency = _finite_hessian_number(row[1])
        if hessian_frequency is None:
            return f"frequency Hessian frequency row {expected_index} contains an invalid value"
        if index != expected_index or stdout_mode.index != expected_index:
            return "frequency Hessian frequency indices do not match the stdout frequency list"
        # ORCA prints stdout frequencies rounded to 0.01 cm^-1, while the Hessian
        # stores substantially more digits. Allow only the stdout rounding error.
        if not math.isclose(hessian_frequency, stdout_mode.value, rel_tol=1e-8, abs_tol=0.0051):
            return (
                "frequency Hessian frequency values do not match stdout "
                f"at mode {expected_index} ({hessian_frequency} != {stdout_mode.value} cm^-1)"
            )

    scale_lines, scale_error = _named_section_lines(text, "$frequency_scale_factor")
    if scale_error:
        return scale_error
    assert scale_lines is not None
    scale_values = [line.strip() for line in scale_lines if line.strip()]
    if len(scale_values) != 1:
        return "frequency Hessian scaling factor is missing or has extra values"
    hessian_scale = _finite_hessian_number(scale_values[0])
    if hessian_scale is None or hessian_scale <= 0:
        return "frequency Hessian scaling factor is invalid"
    if not math.isclose(
        hessian_scale,
        frequency_section.scaling_factor,
        rel_tol=1e-9,
        abs_tol=5.1e-7,
    ):
        return (
            "frequency Hessian scaling factor does not match stdout "
            f"({hessian_scale} != {frequency_section.scaling_factor})"
        )
    return None


def _named_section_lines(text: str, name: str) -> tuple[list[str] | None, str | None]:
    lines = text.splitlines()
    marker = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
    matches = [index for index, line in enumerate(lines) if marker.fullmatch(line)]
    if not matches:
        return None, f"frequency Hessian section {name} is missing"
    if len(matches) != 1:
        return None, f"frequency Hessian section {name} occurs multiple times"
    start = matches[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].lstrip().startswith("$")),
        len(lines),
    )
    return lines[start:end], None


def _validate_hessian_atoms(
    lines: list[str], *, expected_atom_count: int, input_geometry: ParsedGeometry
) -> str | None:
    values = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not values or not re.fullmatch(r"\d+", values[0]):
        return "frequency Hessian atom count is missing or invalid"
    atom_count = int(values[0])
    if atom_count != expected_atom_count:
        return f"frequency Hessian has {atom_count} atoms; expected {expected_atom_count}"
    if len(values) != atom_count + 1:
        return "frequency Hessian atom block is truncated or has extra rows"

    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for atom_index, line in enumerate(values[1:]):
        parts = line.split()
        if len(parts) != 5:
            return f"frequency Hessian atom row {atom_index} has an unsupported layout"
        symbol = parts[0]
        mass = _finite_hessian_number(parts[1])
        point = tuple(_finite_hessian_number(item) for item in parts[2:])
        if mass is None or mass <= 0 or any(value is None for value in point):
            return f"frequency Hessian atom row {atom_index} contains invalid values"
        symbols.append(symbol)
        coordinates.append(point)  # type: ignore[arg-type]

    if tuple(symbols) != input_geometry.symbols:
        return "frequency Hessian atom symbols/order do not match the input geometry"
    for first_index in range(atom_count):
        for second_index in range(first_index + 1, atom_count):
            actual_distance = math.dist(coordinates[first_index], coordinates[second_index])
            expected_distance = (
                math.dist(
                    input_geometry.coordinates[first_index],
                    input_geometry.coordinates[second_index],
                )
                * _BOHR_PER_ANGSTROM
            )
            if abs(actual_distance - expected_distance) > 2e-5:
                return (
                    "frequency Hessian interatomic distances do not match input geometry "
                    f"for atoms {first_index} and {second_index}"
                )
    return None


def _validate_hessian_matrix(lines: list[str], *, dimension: int) -> str | None:
    values = [line.strip() for line in lines if line.strip()]
    cursor = 0
    next_column = 0
    while next_column < dimension:
        if cursor >= len(values):
            return "frequency Hessian matrix is truncated before all columns"
        header = values[cursor].split()
        if not header or any(not re.fullmatch(r"\d+", token) for token in header):
            return "frequency Hessian matrix has an unsupported column header"
        columns = [int(token) for token in header]
        if columns != list(range(next_column, next_column + len(columns))):
            return "frequency Hessian matrix column indices are not contiguous"
        if columns[-1] >= dimension:
            return "frequency Hessian matrix contains an out-of-range column"
        cursor += 1

        for expected_row in range(dimension):
            if cursor >= len(values):
                return "frequency Hessian matrix is truncated within a column block"
            row = values[cursor].split()
            cursor += 1
            if (
                len(row) != len(columns) + 1
                or not re.fullmatch(r"\d+", row[0])
                or int(row[0]) != expected_row
            ):
                return f"frequency Hessian matrix row {expected_row} has an invalid layout"
            if any(_finite_hessian_number(token) is None for token in row[1:]):
                return f"frequency Hessian matrix row {expected_row} contains non-finite values"
        next_column += len(columns)

    if cursor != len(values):
        return "frequency Hessian matrix contains unexpected trailing data"
    return None


def _finite_hessian_number(token: str) -> float | None:
    if not _HESSIAN_FLOAT_RE.fullmatch(token):
        return None
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


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
