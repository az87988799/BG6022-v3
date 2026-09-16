"""Program-owned molecule identity constraints and ordinary formula parsing.

The intake model may suggest a molecule query, but this module constructs the
identity facts that are allowed to cross into a Request.  In particular, a
formula is a composition constraint; it is not permission to choose the first
structure returned by a remote service.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

MoleculeInputKind = Literal["name", "cas", "cid", "smiles", "formula"]

SUPPORTED_FORMULA_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_FORMULA_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z][a-z]?(?:[0-9₀-₉]+)?){1,16}(?![A-Za-z0-9_])"
)


def parse_formula_counts(raw: str) -> dict[str, int]:
    """Parse one complete ordinary molecular formula into element counts.

    This intentionally does not accept parentheses, dots, charges, isotope
    labels, or zero counts.  Those expressions need an explicit structure
    identifier in the first formula-input implementation.
    """

    if not isinstance(raw, str):
        raise ValueError("unsupported formula")
    text = raw.strip().translate(_SUBSCRIPTS)
    if not text or len(text) > 128:
        raise ValueError("unsupported formula")
    result: Counter[str] = Counter()
    position = 0
    for match in _TOKEN.finditer(text):
        if match.start() != position or match.group(1) not in SUPPORTED_FORMULA_ELEMENTS:
            raise ValueError("unsupported formula")
        result[match.group(1)] += int(match.group(2) or "1")
        position = match.end()
    if position != len(text) or not result:
        raise ValueError("unsupported formula")
    return dict(result)


def canonical_formula(counts: Mapping[str, int]) -> str:
    """Render counts in a stable Hill-like order for URLs and records."""

    ordered = []
    for element in ("C", "H"):
        if element in counts:
            ordered.append(element)
    ordered.extend(sorted(element for element in counts if element not in {"C", "H"}))
    return "".join(
        element + (str(int(counts[element])) if int(counts[element]) != 1 else "")
        for element in ordered
    )


def formula_token_from_text(message: str) -> str | None:
    """Return a complete formula token from a user message, if one is present.

    Explicit ``SMILES:`` text wins over formula detection.  Untyped one-letter
    or all-letter strings such as ``CO`` are deliberately not treated as a
    formula unless they contain a subscript/digit, avoiding a SMILES collision.
    """

    if not isinstance(message, str):
        return None
    tokens = _formula_tokens_from_text(message)
    return tokens[0] if tokens else None


def _formula_tokens_from_text(message: str) -> list[str]:
    explicit_smiles = re.search(r"(?i)\bsmiles\s*[:=]\s*[^\s,，。；;]+", message)
    explicit_formula_matches = list(
        re.finditer(
            r"(?i)\b(?:formula|molecular\s+formula|分子式)\s*[:=：]\s*([^\s,，。；;]+)",
            message,
        )
    )
    tokens: list[str] = []
    for match in explicit_formula_matches:
        candidate = match.group(1)
        parse_formula_counts(candidate)
        if candidate not in tokens:
            tokens.append(candidate)
    for match in _FORMULA_TOKEN.finditer(message):
        candidate = match.group(0)
        if (
            explicit_smiles is not None
            and explicit_smiles.start() <= match.start() < explicit_smiles.end()
        ):
            continue
        if not re.search(r"[0-9₀-₉]", candidate):
            continue
        try:
            parse_formula_counts(candidate)
        except ValueError:
            continue
        if candidate not in tokens:
            tokens.append(candidate)
    return tokens


def formula_constraint(raw: str) -> dict[str, Any]:
    """Build program-owned facts for a formula supplied by the user."""

    counts = parse_formula_counts(raw)
    return {
        "formula": canonical_formula(counts),
        "element_counts": counts,
        "scope": "neutral_single_component_nonisotopic",
    }


def build_identity_constraint(
    *,
    message: str,
    query: str | None,
    input_kind: str | None,
) -> dict[str, Any] | None:
    """Construct the Request identity record without trusting model facts."""

    formula_tokens = _formula_tokens_from_text(message)
    if len(formula_tokens) > 1:
        raise ValueError("multiple formula inputs require clarification")
    formula_raw = formula_tokens[0] if formula_tokens else None
    explicit_smiles = re.search(r"(?i)\bsmiles\s*[:=]\s*([^\s,，。；;]+)", message)
    if explicit_smiles is not None:
        # A typed SMILES is authoritative for input kind; formula-looking
        # fragments inside its token are not allowed to steal it. A separate
        # explicit formula remains an additional program-owned constraint.
        input_kind = "smiles"
        query = explicit_smiles.group(1)

    normalized_kind = input_kind
    if normalized_kind == "formula" and formula_raw is None:
        if not isinstance(query, str) or not query.strip() or query not in message:
            raise ValueError(
                "formula input must preserve the complete formula from the user message"
            )
        formula_raw = query
    if formula_raw is not None:
        formula = formula_constraint(formula_raw)
        lookup_query = query.strip() if isinstance(query, str) and query.strip() else formula_raw
        # If the model rewrote H20 as water, keep the user's formula as the
        # lookup query.  A named/CID lookup may still be retained when the
        # user explicitly supplied both a formula and a second identifier.
        if normalized_kind in {None, "formula"} or (
            lookup_query != formula_raw and lookup_query not in message
        ):
            lookup_query = formula_raw
        kind = normalized_kind or "formula"
        if normalized_kind in {None, "formula"} or lookup_query == formula_raw or (
            isinstance(query, str) and query.strip() and query not in message
        ):
            kind = "formula"
            lookup_query = formula_raw
        return {
            "input_kind": kind,
            "raw_query": formula_raw,
            "lookup_query": lookup_query,
            **formula,
            "selected_cid": None,
        }

    if not isinstance(query, str) or not query.strip() or normalized_kind is None:
        return None
    if normalized_kind not in {"name", "cas", "cid", "smiles", "formula"}:
        raise ValueError(f"unsupported molecule input kind: {normalized_kind!r}")
    return {
        "input_kind": normalized_kind,
        "raw_query": query,
        "lookup_query": query,
        "selected_cid": None,
    }


def validate_identity_constraint(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a persisted identity record, including old records."""

    if not isinstance(value, Mapping):
        raise ValueError("molecule_identity must be an object")
    allowed = {
        "input_kind",
        "raw_query",
        "lookup_query",
        "formula",
        "element_counts",
        "scope",
        "selected_cid",
        "selected_choice_id",
        "selected_smiles",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"molecule_identity contains unknown field(s): {unknown}")
    kind = value.get("input_kind")
    if kind not in {"name", "cas", "cid", "smiles", "formula"}:
        raise ValueError("molecule_identity.input_kind is invalid")
    for field in ("raw_query", "lookup_query"):
        if field in value and (not isinstance(value[field], str) or not value[field].strip()):
            raise ValueError(f"molecule_identity.{field} must be non-empty text")
    if "selected_cid" in value and value["selected_cid"] is not None:
        if type(value["selected_cid"]) is not int or value["selected_cid"] <= 0:
            raise ValueError("molecule_identity.selected_cid must be a positive integer")
    for field in ("selected_choice_id", "selected_smiles"):
        if field in value and value[field] is not None:
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"molecule_identity.{field} must be non-empty text")
    if "element_counts" in value:
        counts = value["element_counts"]
        if not isinstance(counts, dict) or not counts:
            raise ValueError("molecule_identity.element_counts must be a non-empty object")
        for element, count in counts.items():
            if element not in SUPPORTED_FORMULA_ELEMENTS or type(count) is not int or count <= 0:
                raise ValueError("molecule_identity.element_counts contains an invalid count")
        expected = canonical_formula(counts)
        if value.get("formula") != expected:
            raise ValueError("molecule_identity.formula does not match element_counts")
        if value.get("scope") != "neutral_single_component_nonisotopic":
            raise ValueError("molecule_identity.scope is invalid for a formula input")
    elif value.get("formula") is not None:
        raise ValueError("molecule_identity.formula requires element_counts")
    return dict(value)


def identity_matches_facts(
    identity: Mapping[str, Any] | None, facts: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Check a candidate's computed RDKit facts against the Request constraint."""

    if identity is None:
        return True, None
    expected_counts = identity.get("element_counts")
    if expected_counts is None:
        return True, None
    if facts.get("component_count") != 1:
        return False, "formula input requires one molecular component"
    if facts.get("isotopic"):
        return False, "formula input does not accept isotopic structures"
    if facts.get("formal_charge") != 0:
        return False, "formula input requires a neutral candidate structure"
    actual = facts.get("element_counts")
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected_counts):
        return False, (
            f"candidate composition {canonical_formula(actual or {})} does not match "
            f"requested composition {identity.get('formula')}"
        )
    return True, None


def normalize_identity_for_storage(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe identity copy after strict validation."""

    validated = validate_identity_constraint(value)
    if "element_counts" in validated:
        validated["element_counts"] = dict(validated["element_counts"])
    return validated


__all__ = [
    "MoleculeInputKind",
    "SUPPORTED_FORMULA_ELEMENTS",
    "build_identity_constraint",
    "canonical_formula",
    "formula_constraint",
    "formula_token_from_text",
    "identity_matches_facts",
    "normalize_identity_for_storage",
    "parse_formula_counts",
    "validate_identity_constraint",
]
