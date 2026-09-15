"""Strict parsing for the ORCA vibrational-frequency output section."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^\s*VIBRATIONAL\s+FREQUENCIES\s*$", re.IGNORECASE)
_SCALE_RE = re.compile(
    r"^\s*Scaling\s+factor\s+for\s+frequencies\s*=\s*(\S+)"
    r"(?:\s*(\(.*\)))?\s*$",
    re.IGNORECASE,
)
_ROW_RE = re.compile(
    r"^\s*(\d+)\s*:\s*(\S+)\s+(\S+)"
    r"(?:\s+(\*{3}imaginary mode\*{3}))?\s*$",
    re.IGNORECASE,
)
_ROW_PREFIX_RE = re.compile(r"^\s*\d+\s*:")
_FLOAT_RE = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?$")
_KNOWN_SECTION_TITLES = {
    "NORMAL MODES",
    "IR SPECTRUM",
    "THERMOCHEMISTRY",
    "THERMOCHEMISTRY AT",
}


@dataclass(frozen=True)
class ParsedVibrationalMode:
    """One printed normal-mode frequency, preserving its sign and index."""

    index: int
    value: float
    unit: str = "cm^-1"


@dataclass(frozen=True)
class VibrationalFrequencySection:
    """Parsed frequency-section facts; incomplete parses retain valid rows."""

    modes: tuple[ParsedVibrationalMode, ...]
    scaling_factor: float | None
    scaling_applied: bool
    complete: bool
    error: str | None


def parse_vibrational_frequencies(
    stdout: bytes | str, expected_atom_count: int
) -> VibrationalFrequencySection:
    """Parse one complete ORCA 6.1 VIBRATIONAL FREQUENCIES section.

    The parser keeps every valid printed mode, including translation/rotation
    modes and negative frequencies. It does not infer thermochemistry or
    remove modes based on molecular size.
    """

    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
    lines = text.splitlines()
    headings = [index for index, line in enumerate(lines) if _HEADING_RE.fullmatch(line)]
    if len(headings) != 1:
        if not headings:
            return _incomplete("VIBRATIONAL FREQUENCIES section is missing")
        return _incomplete("multiple VIBRATIONAL FREQUENCIES sections were found")

    if (
        isinstance(expected_atom_count, bool)
        or not isinstance(expected_atom_count, int)
        or expected_atom_count <= 0
    ):
        return _incomplete("expected_atom_count must be a positive integer")

    start = headings[0] + 1
    end = _section_end(lines, start)
    section_lines = lines[start:end]
    errors: list[str] = []

    expected_mode_count = 3 * expected_atom_count
    modes: list[ParsedVibrationalMode] = []
    malformed_rows: list[str] = []
    scale_values: list[float] = []
    scale_errors: list[str] = []
    scale_markers: list[bool] = []

    for line_number, line in enumerate(section_lines, start=start + 1):
        scale_match = _SCALE_RE.fullmatch(line)
        if scale_match:
            scale_token, marker = scale_match.groups()
            scale_value, scale_error = _parse_number(scale_token, "scaling factor")
            if scale_error:
                scale_errors.append(f"line {line_number}: {scale_error}")
            elif scale_value is not None:
                if scale_value <= 0:
                    scale_errors.append(f"line {line_number}: scaling factor must be positive")
                else:
                    scale_values.append(scale_value)
            scale_markers.append(bool(marker and "already applied" in marker.lower()))
            continue

        row_match = _ROW_RE.fullmatch(line)
        if row_match:
            index_token, value_token, unit_token, imaginary_marker = row_match.groups()
            if unit_token.lower() != "cm**-1":
                malformed_rows.append(
                    f"line {line_number}: unsupported frequency unit {unit_token!r}"
                )
                continue
            value, value_error = _parse_number(value_token, "frequency value")
            if value_error:
                malformed_rows.append(f"line {line_number}: {value_error}")
                continue
            if value is not None:
                if imaginary_marker and value >= 0:
                    malformed_rows.append(
                        f"line {line_number}: imaginary-mode annotation requires a negative value"
                    )
                    continue
                modes.append(ParsedVibrationalMode(index=int(index_token), value=value))
            continue

        if _ROW_PREFIX_RE.match(line):
            malformed_rows.append(f"line {line_number}: unsupported frequency-row layout")

    errors.extend(scale_errors)
    errors.extend(malformed_rows)
    if not scale_values and not scale_errors:
        errors.append("scaling factor for frequencies is missing")
    elif len(scale_values) > 1:
        errors.append("multiple scaling factors for frequencies were found")
    elif scale_errors and not scale_values:
        errors.append("scaling factor for frequencies is invalid")
    if len(scale_markers) > 1:
        errors.append("multiple scaling factor lines were found")

    indices = [mode.index for mode in modes]
    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        errors.append(f"duplicate frequency mode indices: {duplicates}")

    expected_indices = set(range(expected_mode_count))
    actual_indices = set(indices)
    missing = sorted(expected_indices - actual_indices)
    unexpected = sorted(actual_indices - expected_indices)
    if missing:
        errors.append(f"frequency modes are missing indices: {missing}")
    if unexpected:
        errors.append(f"frequency modes contain unexpected indices: {unexpected}")
    if len(indices) != expected_mode_count:
        errors.append(
            f"expected {expected_mode_count} frequency modes for {expected_atom_count} "
            f"atoms, found {len(indices)}"
        )

    complete = not errors
    return VibrationalFrequencySection(
        modes=tuple(modes),
        scaling_factor=scale_values[0] if len(scale_values) == 1 else None,
        scaling_applied=scale_markers[0] if len(scale_markers) == 1 else False,
        complete=complete,
        error=None if complete else "; ".join(errors),
    )


def _section_end(lines: list[str], start: int) -> int:
    """Find the next ORCA-style uppercase heading and dashed underline."""

    for index in range(start, len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
        if not title or not re.fullmatch(r"[-=]{3,}", underline):
            continue
        if title.upper() == "VIBRATIONAL FREQUENCIES":
            continue
        if title.upper() in _KNOWN_SECTION_TITLES or _looks_like_section_title(title):
            return index
    return len(lines)


def _looks_like_section_title(title: str) -> bool:
    return title == title.upper() and any(char.isalpha() for char in title)


def _parse_number(token: str, label: str) -> tuple[float | None, str | None]:
    lowered = token.lower()
    if lowered in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return None, f"non-finite {label} {token!r}"
    if not _FLOAT_RE.fullmatch(token):
        return None, f"invalid {label} {token!r}"
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None, f"invalid {label} {token!r}"
    if not math.isfinite(value):
        return None, f"non-finite {label} {token!r}"
    return value, None


def _incomplete(error: str) -> VibrationalFrequencySection:
    return VibrationalFrequencySection(
        modes=(),
        scaling_factor=None,
        scaling_applied=False,
        complete=False,
        error=error,
    )
