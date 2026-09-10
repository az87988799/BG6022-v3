"""The single source of truth for SP and Opt scientific success."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .parser import AttemptFacts


@dataclass(frozen=True)
class CheckOutcome:
    success: bool
    checks: dict[str, Any]
    failure_category: str | None


def evaluate_success(facts: AttemptFacts, *, operation: str) -> CheckOutcome:
    final_energy = facts.final_energy
    checks: dict[str, Any] = {
        "runner_succeeded": facts.runner_status == "succeeded",
        "exit_code_zero": facts.exit_code == 0,
        "process_tree_empty": facts.process_tree_empty is True,
        "normal_termination": facts.normal_termination is True,
        "stdout_valid_utf8": facts.stdout_valid_utf8,
        "scf_converged": facts.scf_converged is True,
        "final_energy_selected": final_energy is not None and facts.final_energy_error is None,
        "finite_final_energy": (
            final_energy is not None
            and final_energy.value is not None
            and math.isfinite(final_energy.value)
        ),
        "input_hashes_match": facts.input_hashes_match,
    }
    if operation == "Opt":
        checks.update(
            {
                "optimization_converged": facts.optimization_converged is True,
                "output_geometry_present": facts.output_geometry is not None,
                "stdout_geometry_present": facts.stdout_geometry is not None,
                "geometry_consistent": facts.geometry_consistent is True,
            }
        )
    required = list(checks.values())
    success = all(value is True for value in required)
    category = None if success else _failure_category(facts, operation, checks)
    return CheckOutcome(success, checks, category)


def _failure_category(facts: AttemptFacts, operation: str, checks: dict[str, Any]) -> str:
    if facts.error_category:
        return facts.error_category
    if not checks["input_hashes_match"]:
        return "input_integrity_error"
    if not checks["process_tree_empty"]:
        return "cleanup_unconfirmed"
    if facts.final_energy_error is not None or not checks["stdout_valid_utf8"]:
        return "invalid_output"
    if not checks["scf_converged"]:
        return "scf_not_converged"
    if operation == "Opt" and not checks.get("optimization_converged"):
        return "opt_not_converged"
    if operation == "Opt" and not checks.get("geometry_consistent"):
        return "geometry_mismatch"
    if facts.runner_status == "cancelled":
        return "cancelled"
    if facts.runner_status == "timed_out":
        return "timeout"
    if facts.runner_status == "interrupted":
        return "cleanup_unconfirmed"
    return "invalid_output"


__all__ = ["CheckOutcome", "evaluate_success"]
