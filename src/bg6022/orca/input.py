"""Closed, deterministic ORCA input rendering."""

from __future__ import annotations

from dataclasses import dataclass

from .profiles import get_profile


@dataclass(frozen=True)
class OrcaInputSpec:
    operation: str
    method_profile: str
    environment: str
    charge: int
    multiplicity: int
    cores: int
    maxcore_mb: int
    scf_maxiter: int | None = None
    geom_maxiter: int | None = None


def render_input(spec: OrcaInputSpec) -> bytes:
    profile = get_profile(spec.method_profile)
    if spec.operation not in profile.supported_operations:
        raise ValueError(f"unsupported ORCA operation: {spec.operation}")
    if spec.environment not in profile.supported_environments:
        raise ValueError(
            f"environment {spec.environment!r} is not supported by profile {profile.name!r}"
        )
    for name, value in {
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "cores": spec.cores,
        "maxcore_mb": spec.maxcore_mb,
    }.items():
        if type(value) is not int or value <= 0 and name not in {"charge"}:
            raise ValueError(f"{name} must be an integer in the supported range")
    if type(spec.charge) is not int:
        raise ValueError("charge must be an integer")
    if spec.multiplicity <= 0 or spec.cores <= 0 or spec.maxcore_mb <= 0:
        raise ValueError("multiplicity, cores, and maxcore_mb must be positive")
    _validate_iteration("scf_maxiter", spec.scf_maxiter)
    _validate_iteration("geom_maxiter", spec.geom_maxiter)
    if spec.operation != "Opt" and spec.geom_maxiter is not None:
        raise ValueError("geom_maxiter is only valid for optimize_geometry")
    lines = [
        f"! {profile.orca_keyword} TightSCF {spec.operation}",
        "%pal",
        f"  nprocs {spec.cores}",
        "end",
        f"%maxcore {spec.maxcore_mb}",
    ]
    if spec.scf_maxiter is not None:
        lines.extend(["%scf", f"  MaxIter {spec.scf_maxiter}", "end"])
    if spec.geom_maxiter is not None:
        lines.extend(["%geom", f"  MaxIter {spec.geom_maxiter}", "end"])
    lines.extend([f"* xyzfile {spec.charge} {spec.multiplicity} geometry.xyz", "", ""])
    rendered = "\n".join(lines).encode("ascii")
    if not rendered.endswith(b"\n\n"):
        raise AssertionError("ORCA input must end with a blank line")
    return rendered


def _validate_iteration(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or not 1 <= value <= 1000):
        raise ValueError(f"{name} must be an integer between 1 and 1000")


__all__ = ["OrcaInputSpec", "render_input"]
