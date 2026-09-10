"""Pure XYZ and electronic-state validation for explicitly supplied geometries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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
    "ParsedGeometry",
    "SUPPORTED_ELEMENTS",
    "compare_coordinates",
    "format_xyz",
    "parse_xyz_bytes",
    "parse_xyz_file",
    "validate_electronic_state",
]
