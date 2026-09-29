"""Shared, deliberately small contracts for Tool outputs.

The runtime keeps scientific output declarations on each Tool.  This module
only supplies the common type checks and public descriptors used by the
query/answer paths; it does not execute Tools or persist Run state.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

PROPERTY_ALIASES = {
    "energy": "electronic_energy",
    "geometry": "molecular_geometry",
    "frequencies": "frequency",
}

MAX_VALUE_BYTES = 8 * 1024
MAX_TURN_VALUE_BYTES = 16 * 1024
MAX_VIEW_ROWS = 50
MAX_VIEW_COLUMNS = 12


def bounded_verified_value(
    value: Any, view: Mapping[str, Any] | None = None, *, budget_bytes: int = MAX_VALUE_BYTES
):
    """Take a bounded view before serialization, never walking the entire table."""
    view = dict(view or {})
    offset, limit = view.get("offset", 0), view.get("limit", 30)
    columns = view.get("columns") or []
    if (
        type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= MAX_VIEW_ROWS
    ):
        raise ValueError("invalid page range")
    if len(columns) > MAX_VIEW_COLUMNS:
        raise ValueError("too many columns")
    table = isinstance(value, list)
    total = len(value) if table else 1
    if offset >= total and (total or offset):
        raise ValueError("page offset is outside the verified value")
    sample = value[:30] if table else [value]
    declared = []
    for row in sample:
        if isinstance(row, Mapping):
            for key in row:
                if key not in declared:
                    declared.append(key)
                if len(declared) >= MAX_VIEW_COLUMNS:
                    break
        if len(declared) >= MAX_VIEW_COLUMNS:
            break
    if columns and any(key not in declared for key in columns):
        raise ValueError("column is not in the bounded declared/sample view")
    remaining = [min(MAX_VALUE_BYTES, max(0, budget_bytes)) - 32]
    clipped = [False]

    def take(item, depth=0):
        if remaining[0] < 32 or depth > 6:
            clipped[0] = True
            return None
        if isinstance(item, Mapping):
            result = {}
            for index, (key, child) in enumerate(item.items()):
                if index >= MAX_VIEW_COLUMNS or remaining[0] < 32:
                    clipped[0] = True
                    break
                original_key = str(key)
                key = original_key[:160]
                clipped[0] |= len(key) < len(original_key)
                cost = len(json.dumps(key, ensure_ascii=False).encode()) + 4
                if cost + 4 > remaining[0]:
                    clipped[0] = True
                    break
                remaining[0] -= cost
                result[key] = take(child, depth + 1)
            return result
        if isinstance(item, list):
            result = []
            for child in item[:MAX_VIEW_ROWS]:
                if remaining[0] < 32:
                    break
                result.append(take(child, depth + 1))
            clipped[0] |= len(result) < len(item)
            return result
        if isinstance(item, str):
            # Bound the input to the encoder too: a single cell may be enormous.
            prefix = item[: max(0, remaining[0] // 6)]
            clipped[0] |= len(prefix) < len(item)
            item = prefix
        size = len(json.dumps(item, ensure_ascii=False).encode()) + 2
        if size > remaining[0]:
            clipped[0] = True
            return None
        remaining[0] -= size
        return item

    chosen = value[offset : offset + limit] if table else [value]
    preview = []
    for row in chosen:
        if remaining[0] < 32:
            clipped[0] = True
            break
        if columns and isinstance(row, Mapping):
            row = {key: row.get(key) for key in columns}
        preview.append(take(row))
    shown = len(preview)
    full = (
        offset == 0
        and shown == total
        and not clipped[0]
        and (not columns or set(columns) == set(declared))
    )
    meta = {
        "total_rows": total,
        "offset": offset,
        "shown_rows": shown,
        "has_more": offset + shown < total,
        "full_value_shown": full,
        "requested_scope_complete": (offset == 0 and shown == total and not clipped[0])
        if view.get("scope", "page") == "all"
        else shown == len(chosen) and not clipped[0],
    }
    return (preview if table else preview[0] if preview else None), meta


def resolve_output_selector(tool: Any, selector: str, *, kind: str | None = None) -> dict[str, Any]:
    """Resolve a field name or semantic property without directory-order guesses."""
    property_name = PROPERTY_ALIASES.get(selector, selector)
    matches = [
        item
        for item in tool.public_outputs()
        if (kind is None or item["kind"] == kind)
        and (item["name"] == selector or item["property"] == property_name)
    ]
    if not matches:
        raise ValueError(f"{tool.name} does not provide requested property {selector!r}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous output selector {selector!r} on {tool.name}")
    return matches[0]


def public_source_context(source: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist bounded, program-derived labels; never expose paths or values."""
    result: dict[str, Any] = {}
    for key in (
        "subject_label",
        "method_label",
        "operation_label",
        "geometry_label",
        "task_label",
        "source_status",
    ):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value[: 240 if key == "task_label" else 160]
    if type(source.get("attempt")) is int and source["attempt"] > 0:
        result["attempt"] = source["attempt"]
    if type(source.get("is_current_attempt")) is bool:
        result["is_current_attempt"] = source["is_current_attempt"]
    return result


# These are the value/port kinds currently understood by the public result
# path.  A future Tool may add a kind only after adding its local adapter and
# tests here; unknown strings must not silently become arbitrary science.
_KNOWN_TYPES = frozenset(
    {
        "Eh",
        "angstrom",
        "degree",
        "frequency",
        "integer",
        "text",
        "boolean",
        "molecule",
        "molecular_geometry",
        "energy_data",
        "orca_hessian",
        "file",
        "text_file",
        "scientific_check",
        "record",
        "record_list",
        "json",
    }
)


def validate_declared_output(
    name: str, expected_type: str, property_name: str, *, kind: str
) -> None:
    """Validate one public Tool output declaration."""

    validate_declared_type(name, expected_type, kind=kind)
    if not isinstance(property_name, str) or not _IDENTIFIER.fullmatch(property_name):
        raise ValueError(f"output {name!r} has an invalid result property: {property_name!r}")


def validate_declared_type(name: str, expected_type: str, *, kind: str) -> None:
    """Validate a Tool value/port type before it reaches any result path."""

    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"output name must be a simple identifier: {name!r}")
    if not isinstance(expected_type, str) or expected_type not in _KNOWN_TYPES:
        raise ValueError(
            f"output {name!r} uses unsupported public type {expected_type!r}; "
            "add a local output contract before exposing it"
        )
    if kind not in {"field", "port", "check"}:
        raise ValueError(f"unsupported output declaration kind: {kind!r}")
    if kind == "check" and expected_type != "scientific_check":
        raise ValueError("scientific checks must use the scientific_check public type")


def public_type_info(expected_type: str, *, kind: str) -> dict[str, Any]:
    """Return non-authoritative metadata safe to expose to the model."""

    unit = {
        "Eh": "Eh",
        "angstrom": "angstrom",
        "degree": "degree",
        "frequency": "cm^-1",
    }.get(expected_type)
    mime_type = None
    if expected_type == "molecular_geometry":
        mime_type = "chemical/x-xyz"
    elif expected_type == "molecule":
        mime_type = "application/json"
    elif expected_type == "energy_data":
        mime_type = "application/json"
    elif expected_type == "text_file":
        mime_type = "text/plain"
    elif kind == "port" or expected_type in {"file", "orca_hessian"}:
        mime_type = "application/octet-stream"
    if expected_type in {"Eh", "angstrom", "degree", "frequency", "integer", "text", "boolean"}:
        shape = "scalar"
    elif expected_type == "record":
        shape = "record"
    elif expected_type == "record_list":
        shape = "record_list"
    elif expected_type == "json":
        shape = "json"
    elif expected_type in {"molecular_geometry", "molecule", "text_file", "energy_data"}:
        shape = "text_file"
    elif expected_type in {"orca_hessian", "file"}:
        shape = "file"
    elif expected_type == "scientific_check":
        shape = "check"
    else:
        shape = "unknown"
    if kind == "port" or shape in {"file", "text_file"}:
        supported_views = ["auto", "code", "link"]
    elif shape in {"record", "record_list", "json"}:
        supported_views = ["auto", "plain", "table", "json", "code"]
    else:
        supported_views = ["auto", "plain", "table", "json"]
    return {
        "type": expected_type,
        "unit": unit,
        "mime_type": mime_type,
        "kind": kind,
        "shape": shape,
        "supported_views": supported_views,
    }


def is_compatible_value(value: Any, declared_type: str) -> bool:
    """Check persisted values without accepting NaN, wrong units, or junk."""

    if declared_type == "frequency":
        if not isinstance(value, Mapping) or value.get("complete") is not True:
            return False
        if value.get("unit") != "cm^-1" or type(value.get("scaling_applied")) is not bool:
            return False
        modes = value.get("modes")
        if not isinstance(modes, list) or not modes:
            return False
        indices: list[int] = []
        for mode in modes:
            if not isinstance(mode, Mapping):
                return False
            index, number = mode.get("index"), mode.get("value")
            if type(index) is not int or type(number) not in {int, float}:
                return False
            if mode.get("unit") != "cm^-1" or not math.isfinite(float(number)):
                return False
            indices.append(index)
        if indices != list(range(len(indices))):
            return False
        factor = value.get("scaling_factor")
        return factor is None or (
            type(factor) in {int, float} and math.isfinite(float(factor)) and factor > 0
        )
    if declared_type == "Eh":
        if not isinstance(value, Mapping):
            return False
        raw = value.get("value")
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            return False
        if value.get("unit") != "Eh":
            return False
        token = value.get("token")
        if token is not None:
            if not isinstance(token, str):
                return False
            try:
                if not math.isfinite(float(token)):
                    return False
            except ValueError:
                return False
        return True
    if declared_type in {"angstrom", "degree"}:
        if isinstance(value, Mapping):
            raw = value.get("value")
            unit = value.get("unit")
            if type(raw) not in {int, float} or not math.isfinite(float(raw)):
                return False
            if unit != declared_type:
                return False
            if declared_type == "angstrom":
                atom_indices = value.get("atom_indices")
                if atom_indices is not None and (
                    not isinstance(atom_indices, list)
                    or len(atom_indices) != 2
                    or any(type(item) is not int or item < 1 for item in atom_indices)
                    or atom_indices[0] == atom_indices[1]
                ):
                    return False
                atom_symbols = value.get("atom_symbols")
                if atom_symbols is not None and (
                    not isinstance(atom_symbols, list)
                    or len(atom_symbols) != 2
                    or any(not isinstance(item, str) or not item for item in atom_symbols)
                ):
                    return False
            return True
        return type(value) in {int, float} and math.isfinite(float(value))
    if declared_type == "integer":
        if isinstance(value, Mapping):
            value = value.get("value")
        return type(value) is int
    if declared_type == "text":
        if isinstance(value, Mapping):
            value = value.get("value")
        return type(value) is str
    if declared_type == "boolean":
        if isinstance(value, Mapping):
            value = value.get("value")
        return type(value) is bool
    if declared_type in {
        "molecule",
        "molecular_geometry",
        "energy_data",
        "orca_hessian",
        "file",
        "text_file",
    }:
        return isinstance(value, Mapping)
    if declared_type == "record":
        return isinstance(value, Mapping) and _finite_json(value)
    if declared_type == "record_list":
        return isinstance(value, list) and all(
            isinstance(row, Mapping) and _finite_json(row) for row in value
        )
    if declared_type == "json":
        return _finite_json(value)
    if declared_type == "scientific_check":
        return (
            isinstance(value, Mapping)
            and value.get("status") in {"passed", "not_met", "unverified"}
            and _finite_json(value)
        )
    return False


def canonical_public_outputs(tool: Any) -> list[dict[str, Any]]:
    """Return one canonical public descriptor for every Tool output.

    A port wins when an old Tool declares the same name in both ``results`` and
    ``output_ports``.  The descriptor deliberately falls back to the declared
    name when no semantic property mapping was supplied, so a valid output is
    never silently removed from the public directory.
    """

    declared: list[tuple[str, str, str]] = [
        ("port", name, expected_type) for name, expected_type in tool.output_ports.items()
    ]
    declared.extend(
        ("field", name, expected_type)
        for name, expected_type in tool.results.items()
        if name not in tool.output_ports
    )
    declared.extend(("check", name, "scientific_check") for name in tool.scientific_checks)
    seen: dict[str, tuple[str, str]] = {}
    outputs: list[dict[str, Any]] = []
    for kind, name, expected_type in declared:
        property_name = tool.result_properties.get(name, name)
        if property_name in seen:
            previous_kind, previous_name = seen[property_name]
            raise ValueError(
                "ambiguous public property: "
                f"{property_name!r} is declared by {previous_kind} {previous_name!r} "
                f"and {kind} {name!r}"
            )
        seen[property_name] = (kind, name)
        metadata = dict(tool.result_metadata.get(name, {}))
        descriptor = {
            "kind": kind,
            "name": name,
            "property": property_name,
            "type": expected_type,
            "metadata": metadata,
        }
        descriptor.update(public_type_info(expected_type, kind=kind))
        outputs.append(descriptor)
    return outputs


def property_evidence_matches(
    property_name: str, evidence: str, message: str, *, metadata: Mapping[str, Any] | None = None
) -> bool:
    """Match a dynamically declared property using its public description."""

    if not isinstance(evidence, str) or not evidence or evidence not in message:
        return False
    evidence_position = message.find(evidence)
    if _is_negated(evidence, 0) or _is_negated(message, evidence_position):
        return False
    details = metadata or {}
    phrases: list[str] = [property_name.replace("_", " ")]
    for key in ("label", "description"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            phrases.append(value.strip())
    evidence_folded = evidence.casefold()
    for phrase in phrases:
        for token in _meaningful_tokens(phrase):
            token_folded = token.casefold()
            if token_folded in evidence_folded and not _is_negated(
                evidence, evidence_folded.find(token_folded)
            ):
                return True
    evidence_terms = _semantic_terms(evidence)
    declared_terms = set().union(*(_semantic_terms(phrase) for phrase in phrases))
    if evidence_terms & declared_terms:
        return True
    return False


def _meaningful_tokens(value: str) -> list[str]:
    tokens = [item for item in re.split(r"[\s,，。；;:：()（）/]+", value) if item]
    return [item for item in tokens if len(item) >= 2] or [value]


# Generic public-output concepts, rather than Tool-specific vocabulary.  Tool
# labels and descriptions still provide the primary evidence; these small
# groups only bridge ordinary language to the public contract (for example
# ``角度`` and ``夹角`` or ``报告`` and ``文件``).
_SEMANTIC_CONCEPTS = (
    frozenset({"角", "角度", "夹角", "angle", "degree"}),
    frozenset({"报告", "report", "csv", "file", "文件", "表格"}),
    frozenset({"结构", "geometry", "structure", "xyz", "坐标"}),
    frozenset({"距离", "间距", "distance", "length", "长度"}),
)


def _semantic_terms(value: str) -> set[str]:
    folded = value.casefold()
    terms = {folded}
    for token in _meaningful_tokens(value):
        token_folded = token.casefold()
        terms.add(token_folded)
        for concept in _SEMANTIC_CONCEPTS:
            if token_folded in concept:
                terms.update(concept)
    for concept in _SEMANTIC_CONCEPTS:
        if any(token.casefold() in folded for token in concept):
            terms.update(concept)
    return terms


def _is_negated(value: str, position: int) -> bool:
    prefix = value[max(0, position - 8) : position].casefold()
    return any(token in prefix for token in ("不", "没有", "未", "not", "no", "without"))


def _finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_finite_json(item) for item in value)
    return False


__all__ = [
    "canonical_public_outputs",
    "is_compatible_value",
    "property_evidence_matches",
    "public_type_info",
    "validate_declared_output",
    "validate_declared_type",
]
