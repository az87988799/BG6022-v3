"""Explicit method profiles used by the deterministic input renderer."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from bg6022.tools.molecule import SUPPORTED_ELEMENTS


@dataclass(frozen=True)
class MethodProfile:
    name: str
    orca_keyword: str
    supported_environments: frozenset[str]
    supported_elements: frozenset[str]
    supported_operations: frozenset[str]


@dataclass(frozen=True)
class ParameterResolution:
    effective_parameters: dict[str, Any]
    parameter_sources: dict[str, str]
    missing_fields: tuple[str, ...]


R2SCAN3C = MethodProfile(
    name="r2scan3c",
    orca_keyword="r2SCAN-3c",
    supported_environments=frozenset({"gas"}),
    supported_elements=SUPPORTED_ELEMENTS,
    supported_operations=frozenset({"SP", "Opt", "Freq"}),
)

PROFILES = {R2SCAN3C.name: R2SCAN3C}


def get_profile(name: str) -> MethodProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"method profile is not registered: {name}") from error


def list_profiles() -> tuple[MethodProfile, ...]:
    return tuple(PROFILES.values())


def resolve_parameters(
    request_parameters: Mapping[str, Any] | None = None,
    structure_facts: Mapping[str, Any] | None = None,
    tool_parameters: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | Any | None = None,
    *,
    user_modifications: Mapping[str, Any] | None = None,
    parameter_fields: Collection[str] | None = None,
) -> ParameterResolution:
    """Merge one effective scientific parameter set without inventing q/M.

    ``user_modifications`` has the highest priority, followed by parameters
    explicitly recorded on the Request, model/tool parameters, structure facts,
    and the configured method/environment policy.  ``None`` means "not known";
    it never erases a value from a lower-priority source.
    """

    request_parameters = request_parameters or {}
    structure_facts = structure_facts or {}
    tool_parameters = tool_parameters or {}
    user_modifications = user_modifications or {}
    supported_fields = None if parameter_fields is None else frozenset(parameter_fields)
    if supported_fields is not None:
        unsupported_tool_parameters = sorted(set(tool_parameters) - supported_fields)
        if unsupported_tool_parameters:
            raise ValueError(
                "step parameters are not accepted by the selected Tool: "
                + ", ".join(unsupported_tool_parameters)
            )
        request_parameters = {
            name: value for name, value in request_parameters.items() if name in supported_fields
        }
        user_modifications = {
            name: value for name, value in user_modifications.items() if name in supported_fields
        }
    if defaults is None:
        defaults_map: Mapping[str, Any] = {"method_profile": "r2scan3c", "environment": "gas"}
    elif isinstance(defaults, Mapping):
        defaults_map = defaults
    else:
        defaults_map = {
            "method_profile": getattr(defaults, "method_profile", "r2scan3c"),
            "environment": getattr(defaults, "environment", "gas"),
        }

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def choose(name: str, candidates: list[tuple[str, Mapping[str, Any]]]) -> None:
        if supported_fields is not None and name not in supported_fields:
            return
        for source, mapping in candidates:
            if name in mapping and mapping[name] is not None:
                value = mapping[name]
                if name == "method_profile":
                    value = _normalize_method_profile(value)
                values[name] = value
                sources[name] = source
                return

    choose(
        "method_profile",
        [
            ("user_modification", user_modifications),
            ("request_explicit", request_parameters),
            ("tool_request", tool_parameters),
            ("default_policy", defaults_map),
        ],
    )
    choose(
        "environment",
        [
            ("user_modification", user_modifications),
            ("request_explicit", request_parameters),
            ("tool_request", tool_parameters),
            ("default_policy", defaults_map),
        ],
    )
    choose(
        "charge",
        [
            ("user_modification", user_modifications),
            ("request_explicit", request_parameters),
            ("structure_facts", _structure_aliases(structure_facts, "charge", "formal_charge")),
        ],
    )
    choose(
        "multiplicity",
        [
            ("user_modification", user_modifications),
            ("request_explicit", request_parameters),
            ("structure_facts", _multiplicity_facts(structure_facts)),
        ],
    )
    for name in ("scf_maxiter", "geom_maxiter"):
        choose(
            name,
            [
                ("user_modification", user_modifications),
                ("request_explicit", request_parameters),
                ("tool_request", tool_parameters),
            ],
        )

    missing = tuple(
        name
        for name in ("charge", "multiplicity")
        if name not in values and (supported_fields is None or name in supported_fields)
    )
    return ParameterResolution(values, sources, missing)


def _normalize_method_profile(value: Any) -> Any:
    aliases = {
        "r2scan-3c": "r2scan3c",
        "r2scan_3c": "r2scan3c",
        "r2scan3c": "r2scan3c",
    }
    return aliases.get(str(value).casefold(), value)


def _structure_aliases(facts: Mapping[str, Any], wanted: str, *aliases: str) -> Mapping[str, Any]:
    for name in (wanted, *aliases):
        if name in facts and facts[name] is not None:
            return {wanted: facts[name]}
    return {}


def _multiplicity_facts(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    if facts.get("multiplicity") is not None:
        return {"multiplicity": facts["multiplicity"]}
    # A singlet suggestion is only made for structures explicitly known to have
    # no radical electrons and no unsupported/metal elements.  Unknown facts
    # remain missing rather than silently becoming a closed-shell calculation.
    atom_symbols = tuple(str(item) for item in facts.get("atom_symbols", ()))
    symbols = set(atom_symbols)
    # RDKit's neutral O=O representation has no atom-level radical flag, but
    # the ground electronic state is not safely inferable from that fact alone.
    # Keep this common open-shell case in clarification rather than silently
    # turning it into a singlet calculation.
    oxygen_dimer = len(atom_symbols) == 2 and atom_symbols.count("O") == 2
    if (
        not oxygen_dimer
        and facts.get("radical_electrons") == 0
        and symbols
        and symbols <= SUPPORTED_ELEMENTS
    ):
        return {"multiplicity": 1}
    return {}


__all__ = [
    "MethodProfile",
    "PROFILES",
    "ParameterResolution",
    "R2SCAN3C",
    "get_profile",
    "list_profiles",
    "resolve_parameters",
]
