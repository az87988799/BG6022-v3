"""Explicit method profiles used by the deterministic input renderer."""

from __future__ import annotations

from dataclasses import dataclass

from bg6022.tools.molecule import SUPPORTED_ELEMENTS


@dataclass(frozen=True)
class MethodProfile:
    name: str
    orca_keyword: str
    supported_environments: frozenset[str]
    supported_elements: frozenset[str]


R2SCAN3C = MethodProfile(
    name="r2scan3c",
    orca_keyword="r2SCAN-3c",
    supported_environments=frozenset({"gas"}),
    supported_elements=SUPPORTED_ELEMENTS,
)

PROFILES = {R2SCAN3C.name: R2SCAN3C}


def get_profile(name: str) -> MethodProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"method profile is not registered: {name}") from error


def list_profiles() -> tuple[MethodProfile, ...]:
    return tuple(PROFILES.values())


__all__ = ["MethodProfile", "PROFILES", "R2SCAN3C", "get_profile", "list_profiles"]
