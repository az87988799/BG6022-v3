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
    if operation not in {"SP", "Opt", "Freq"}:
        raise ValueError(f"unsupported ORCA operation: {operation}")

    final_energy = facts.final_energy
    checks: dict[str, Any] = {
        "runner_succeeded": facts.runner_status == "succeeded",
        "exit_code_zero": facts.exit_code == 0,
        "process_tree_empty": facts.process_tree_empty is True,
        "normal_termination": facts.normal_termination is True,
        "stdout_valid_utf8": facts.stdout_valid_utf8,
        "stdout_within_size_limit": not facts.stdout_size_exceeded,
        "stderr_within_size_limit": not facts.stderr_size_exceeded,
        "scf_converged": facts.scf_converged is True,
        "input_hashes_match": facts.input_hashes_match,
    }
    if operation != "Freq":
        checks.update(
            {
                "final_energy_selected": (
                    final_energy is not None and facts.final_energy_error is None
                ),
                "finite_final_energy": (
                    final_energy is not None
                    and final_energy.value is not None
                    and math.isfinite(final_energy.value)
                ),
            }
        )
    if operation == "Opt":
        checks.update(
            {
                "optimization_converged": facts.optimization_converged is True,
                "output_geometry_present": facts.output_geometry is not None,
                "stdout_geometry_present": facts.stdout_geometry is not None,
                "geometry_consistent": facts.geometry_consistent is True,
            }
        )
    elif operation == "Freq":
        frequency_section = facts.frequency_section
        modes = () if frequency_section is None else frequency_section.modes
        expected_indices = list(range(facts.hessian_dimension or 0))
        checks.update(
            {
                "frequency_section_complete": (
                    frequency_section is not None and frequency_section.complete
                ),
                "frequency_values_finite": bool(modes)
                and all(math.isfinite(mode.value) for mode in modes),
                "frequency_mode_indices_match_hessian": (
                    facts.hessian_dimension is not None
                    and [mode.index for mode in modes] == expected_indices
                ),
                "hessian_present": facts.hessian_present,
                "hessian_valid": facts.hessian_valid is True,
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
    if not checks["stdout_within_size_limit"] or not checks["stderr_within_size_limit"]:
        return "resource_limit"
    if not checks["stdout_valid_utf8"] or (
        operation != "Freq" and facts.final_energy_error is not None
    ):
        return "invalid_output"
    if operation == "Freq" and (
        not checks["frequency_section_complete"]
        or not checks["hessian_present"]
        or not checks["hessian_valid"]
        or not checks["frequency_mode_indices_match_hessian"]
    ):
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
