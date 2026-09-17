"""Program-owned molecule identity constraints and ordinary formula parsing."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

MoleculeInputKind = Literal["name", "cas", "cid", "smiles", "formula"]

SUPPORTED_FORMULA_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_FORMULA_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9₀-₉]*(?![A-Za-z0-9_])")
_FORMULA_LABEL = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:formula|molecular\s+formula|分子式)\s*[:=：]")
# Include ASCII chemistry punctuation in the labelled token.  This makes an
# unsupported suffix (charge, component separator, bracketed isotope, or
# parenthesized group) fail as a whole instead of being silently truncated to
# a valid-looking formula prefix.  Chinese sentence punctuation remains a
# delimiter, so ``formula:C6H14。`` still has the ordinary user-facing
# meaning while ``formula:C6H14. Cl`` is rejected.
_FORMULA_VALUE = re.compile(r"[A-Za-z0-9₀-₉.+()\-\[\]_^*/]+")
_EXPLICIT_SMILES_LABEL = re.compile(r"(?i)(?<![A-Za-z0-9_])smiles\s*[:=：]")
# Keep the structure token ASCII and narrow so Chinese text immediately after
# a SMILES label cannot be swallowed as part of the query.
_SMILES_VALUE = re.compile(r"[-A-Za-z0-9@+_=#\\/%().:\[\]]+")
_HORIZONTAL_WHITESPACE = frozenset({" ", "\t"})
_FORMULA_SUFFIX_CHARS = frozenset(".+-()[]_^*/")
_SIMPLE_CASE_FORMULA = re.compile(r"(?:[A-Za-z][1-9][0-9]*)+")
_SINGLE_LETTER_ELEMENTS = frozenset(
    element for element in SUPPORTED_FORMULA_ELEMENTS if len(element) == 1
)


def parse_formula_counts(raw: str) -> dict[str, int]:
    """Parse one complete ordinary molecular formula into element counts."""

    normalized = normalize_formula_token(raw)
    return _parse_standard_formula_counts(normalized)


def _parse_standard_formula_counts(text: str) -> dict[str, int]:
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


def normalize_formula_token(raw: str) -> str:
    """Return the only safe lookup spelling for one formula token.

    Normal formula spelling is parsed exactly.  A deliberately narrow
    fallback accepts only repeated one-letter-plus-positive-integer segments,
    which makes inputs such as ``c4h10`` unambiguous without turning ``Co``
    into ``CO`` or changing a SMILES token's case.
    """

    if not isinstance(raw, str):
        raise ValueError("unsupported formula")
    text = raw.strip().translate(_SUBSCRIPTS)
    if not text or len(text) > 128:
        raise ValueError("unsupported formula")
    try:
        counts = _parse_standard_formula_counts(text)
    except ValueError:
        if not _SIMPLE_CASE_FORMULA.fullmatch(text):
            raise ValueError("unsupported formula") from None
        counts: Counter[str] = Counter()
        position = 0
        for match in re.finditer(r"([A-Za-z])([1-9][0-9]*)", text):
            if match.start() != position:
                raise ValueError("unsupported formula") from None
            element = match.group(1).upper()
            if element not in _SINGLE_LETTER_ELEMENTS:
                raise ValueError("unsupported formula") from None
            counts[element] += int(match.group(2))
            position = match.end()
        if position != len(text) or not counts:
            raise ValueError("unsupported formula") from None
    return canonical_formula(counts)


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


def extract_explicit_smiles(message: str) -> tuple[dict[str, Any], ...]:
    """Extract every explicitly labelled SMILES token once, with its range."""

    if not isinstance(message, str):
        return ()
    extracted: list[dict[str, Any]] = []
    for label in _EXPLICIT_SMILES_LABEL.finditer(message):
        value = _match_label_value(message, label.end(), _SMILES_VALUE)
        if value is None:
            raise ValueError("SMILES input requires a complete structure after its label")
        extracted.append(
            {
                "input_kind": "smiles",
                "raw_query": value.group(0),
                "start": label.start(),
                "end": value.end(),
            }
        )
    return tuple(extracted)


def formula_token_from_text(message: str) -> str | None:
    """Return a complete formula token from a user message, if one is present."""

    if not isinstance(message, str):
        return None
    tokens = _formula_tokens_from_text(message)
    return tokens[0] if tokens else None


def _formula_tokens_from_text(message: str) -> list[str]:
    explicit_smiles = extract_explicit_smiles(message)
    tokens: list[str] = []
    normalized_tokens: set[str] = set()
    for label in _FORMULA_LABEL.finditer(message):
        value = _match_label_value(message, label.end(), _FORMULA_VALUE)
        if value is None:
            raise ValueError("formula input requires a complete formula after its label")
        candidate = value.group(0)
        normalized = normalize_formula_token(candidate)
        if normalized not in normalized_tokens:
            tokens.append(candidate)
            normalized_tokens.add(normalized)
    for match in _FORMULA_TOKEN.finditer(message):
        candidate = match.group(0)
        if any(item["start"] <= match.start() < item["end"] for item in explicit_smiles):
            continue
        if not re.search(r"[0-9₀-₉]", candidate):
            continue
        if match.end() < len(message) and message[match.end()] in _FORMULA_SUFFIX_CHARS:
            # The bare-token scanner must not turn ``C6H14+`` or
            # ``C6H14.Cl`` into the valid prefix ``C6H14``.
            continue
        if _looks_like_ring_smiles(candidate):
            continue
        try:
            normalized = normalize_formula_token(candidate)
        except ValueError:
            continue
        if normalized not in normalized_tokens:
            tokens.append(candidate)
            normalized_tokens.add(normalized)
    return tokens


def _match_label_value(
    message: str, position: int, pattern: re.Pattern[str]
) -> re.Match[str] | None:
    """Match a labelled value after horizontal, but not newline, whitespace."""

    while position < len(message) and message[position] in _HORIZONTAL_WHITESPACE:
        position += 1
    return pattern.match(message, position)


def _looks_like_ring_smiles(value: str) -> bool:
    letters = re.findall(r"[A-Za-z]", value)
    digits = re.findall(r"[0-9]", value)
    return (
        len(letters) >= 3
        and len(set(letter.upper() for letter in letters)) == 1
        and len(set(digits)) < len(digits)
    )


def formula_constraint(raw: str) -> dict[str, Any]:
    """Build program-owned facts for a formula supplied by the user."""

    normalized = normalize_formula_token(raw)
    counts = _parse_standard_formula_counts(normalized)
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
    name_evidence: str | None = None,
) -> dict[str, Any] | None:
    """Construct the Request identity record without trusting model facts."""

    if name_evidence is not None:
        if input_kind != "name":
            raise ValueError("molecule_name_evidence is only valid for name input")
        if not isinstance(name_evidence, str) or not name_evidence.strip():
            raise ValueError("molecule_name_evidence must be non-empty text")
        if name_evidence not in message:
            raise ValueError(
                "molecule_name_evidence must be an exact substring of the user message"
            )

    explicit_smiles = extract_explicit_smiles(message)
    if len(explicit_smiles) > 1:
        canonical_values = {canonical_structure(item["raw_query"]) for item in explicit_smiles}
        if len(canonical_values) != 1:
            raise ValueError("multiple conflicting SMILES inputs require clarification")
    formula_tokens = _formula_tokens_from_text(message)
    if len(formula_tokens) > 1:
        raise ValueError("multiple formula inputs require clarification")
    formula_raw = formula_tokens[0] if formula_tokens else None

    if explicit_smiles:
        # A typed SMILES is authoritative for input kind; formula-looking
        # fragments inside its token were excluded from formula scanning.  A
        # separate formula remains an additional program-owned constraint.
        smiles_query = explicit_smiles[0]["raw_query"]
        identity: dict[str, Any] = {
            "input_kind": "smiles",
            "raw_query": smiles_query,
            "lookup_query": smiles_query,
            "selected_cid": None,
        }
        if formula_raw is not None:
            identity.update(formula_constraint(formula_raw))
        return identity

    normalized_kind = input_kind
    if normalized_kind == "formula" and formula_raw is None:
        if not isinstance(query, str) or not query.strip() or query not in message:
            raise ValueError(
                "formula input must preserve the complete formula from the user message"
            )
        formula_raw = query
    if formula_raw is not None:
        formula = formula_constraint(formula_raw)
        normalized_formula = canonical_formula(formula["element_counts"])
        proposed_query = query.strip() if isinstance(query, str) and query.strip() else ""
        kind = normalized_kind if normalized_kind in {"name", "cas", "cid"} else None
        query_is_user_bound = proposed_query and proposed_query in message
        if kind == "name" and name_evidence is not None:
            query_is_user_bound = True
        if kind is not None and proposed_query and query_is_user_bound:
            lookup_query = proposed_query
        else:
            kind = "formula"
            lookup_query = normalized_formula
        selected_cid = None
        if kind == "cid":
            selected_cid = _positive_cid(lookup_query)
            if selected_cid is None:
                raise ValueError("CID input must be a positive integer")
            lookup_query = str(selected_cid)
        return {
            "input_kind": kind,
            "raw_query": formula_raw,
            "lookup_query": lookup_query,
            **formula,
            "selected_cid": selected_cid,
        }

    if not isinstance(query, str) or not query.strip() or normalized_kind is None:
        return None
    if normalized_kind not in {"name", "cas", "cid", "smiles", "formula"}:
        raise ValueError(f"unsupported molecule input kind: {normalized_kind!r}")
    raw_query = query
    if normalized_kind == "name":
        if name_evidence is not None:
            raw_query = name_evidence
    selected_cid = None
    if normalized_kind == "cid":
        selected_cid = _positive_cid(query)
        if selected_cid is None:
            raise ValueError("CID input must be a positive integer")
        query = str(selected_cid)
    return {
        "input_kind": normalized_kind,
        "raw_query": raw_query,
        "lookup_query": query,
        "selected_cid": selected_cid,
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
    """Check a candidate's computed facts against the Request constraint."""

    if identity is None:
        return True, None
    selected_cid = identity.get("selected_cid")
    if selected_cid is None and identity.get("input_kind") == "cid":
        selected_cid = _positive_cid(identity.get("lookup_query") or identity.get("raw_query"))
    if selected_cid is not None and _positive_cid(facts.get("cid")) != selected_cid:
        return False, "resolved structure CID does not match the selected CID"
    selected_smiles = identity.get("selected_smiles")
    if selected_smiles is not None:
        actual_smiles = facts.get("isomeric_smiles") or facts.get("canonical_smiles")
        if not isinstance(actual_smiles, str):
            return False, "resolved structure has no SMILES for selected-structure validation"
        try:
            if canonical_structure(actual_smiles) != canonical_structure(selected_smiles):
                return False, "resolved structure does not match the selected SMILES"
        except ValueError as error:
            return False, str(error)
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


def identity_mismatch_code(
    identity: Mapping[str, Any] | None, facts: Mapping[str, Any]
) -> str | None:
    """Return a stable reason code for a verified candidate mismatch."""

    if identity is None:
        return None
    selected_cid = identity.get("selected_cid")
    if selected_cid is None and identity.get("input_kind") == "cid":
        selected_cid = _positive_cid(identity.get("lookup_query") or identity.get("raw_query"))
    if selected_cid is not None and _positive_cid(facts.get("cid")) != selected_cid:
        return "selected_cid_mismatch"
    selected_smiles = identity.get("selected_smiles")
    if selected_smiles is not None:
        actual_smiles = facts.get("isomeric_smiles") or facts.get("canonical_smiles")
        if not isinstance(actual_smiles, str):
            return "selected_smiles_unavailable"
        try:
            if canonical_structure(actual_smiles) != canonical_structure(selected_smiles):
                return "selected_smiles_mismatch"
        except ValueError:
            return "selected_smiles_invalid"
    expected_counts = identity.get("element_counts")
    if expected_counts is None:
        return None
    if facts.get("component_count") != 1:
        return "excluded_multicomponent"
    if facts.get("isotopic"):
        return "excluded_isotopic"
    if facts.get("formal_charge") != 0:
        return "excluded_charged"
    actual = facts.get("element_counts")
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected_counts):
        return "excluded_composition"
    return None


def canonical_structure(smiles: str) -> str:
    """Canonicalize a structure without changing charge, isotope, or stereo."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be non-empty text")
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ValueError("RDKit is required for structure identity validation") from error
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES is not a valid RDKit structure")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def validate_resolve_binding(
    identity: Mapping[str, Any] | None, parameters: Mapping[str, Any]
) -> None:
    """Validate the query binding for a resolve step at the execution boundary."""

    if identity is None:
        return
    validate_identity_constraint(identity)
    input_kind = parameters.get("input_kind")
    query = parameters.get("query")
    if input_kind not in {"name", "cas", "cid", "smiles", "formula"}:
        raise ValueError("resolve step has an invalid molecule input kind")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("resolve step query must be non-empty text")
    if input_kind == "cid" and _positive_cid(query) is None:
        raise ValueError("CID input must be a positive integer")
    selected_cid = identity.get("selected_cid")
    if selected_cid is not None:
        if input_kind != "cid" or query != str(selected_cid):
            raise ValueError("selected CID must be the resolve step's exact CID query")
        return
    selected_smiles = identity.get("selected_smiles")
    if selected_smiles is not None:
        if input_kind != "smiles":
            raise ValueError("selected SMILES must be resolved through a SMILES query")
        if canonical_structure(query) != canonical_structure(selected_smiles):
            raise ValueError("resolve query does not match the selected SMILES")
        return
    expected_kind = identity.get("input_kind")
    expected_query = identity.get("lookup_query") or identity.get("raw_query")
    if input_kind != expected_kind or query != expected_query:
        raise ValueError("resolve step changed the user's molecule identity query")


def _positive_cid(value: Any) -> int | None:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        cid = int(value)
        return cid if cid > 0 else None
    return None


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
    "canonical_structure",
    "extract_explicit_smiles",
    "formula_constraint",
    "formula_token_from_text",
    "identity_matches_facts",
    "identity_mismatch_code",
    "normalize_formula_token",
    "normalize_identity_for_storage",
    "parse_formula_counts",
    "validate_identity_constraint",
    "validate_resolve_binding",
]
