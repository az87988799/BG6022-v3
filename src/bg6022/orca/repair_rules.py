"""Deterministic admission rules for bounded ORCA repair actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bg6022.models import Artifact, RepairOption, Result, Run, Step
from bg6022.orca.profiles import get_profile

MAX_ITERATION = 1000
DEFAULT_RESTART_GEOM_MAXITER = 100


def applicable_repairs(run: Run, step: Step, result: Result) -> list[RepairOption]:
    """Return only actions supported by concrete facts from one failed attempt."""

    if (
        result.status != "failed"
        or step.tool != "optimize_geometry"
        or not _profile_allows(step, "restart_optimization")
    ):
        return []
    diagnostics = result.diagnostics
    facts = diagnostics.get("facts") if isinstance(diagnostics.get("facts"), dict) else {}
    process = diagnostics.get("process") if isinstance(diagnostics.get("process"), dict) else {}
    if not _cleanly_stopped(process) or not _input_integrity_ok(diagnostics):
        return []
    if facts.get("opt_iteration_limit_reached") is not True:
        return []
    if facts.get("effective_geom_maxiter") is None:
        return []
    if facts.get("effective_geom_maxiter_error") is not None:
        return []
    candidate = _candidate_for_result(run, result)
    if candidate is None or not _candidate_is_eligible(candidate, step, result):
        return []
    old = int(facts["effective_geom_maxiter"])
    new = min(MAX_ITERATION, max(old + 1, old * 2, DEFAULT_RESTART_GEOM_MAXITER))
    if new <= old:
        return []
    if not _scope_allows_patch(
        run, step, action="restart_optimization", patch={"geom_maxiter": new}
    ):
        return []
    evidence = (
        "opt_iteration_limit_reached",
        "candidate_geometry_valid",
        "process_cleanup_confirmed",
        "input_hashes_match",
    )
    return [
        RepairOption(
            action="restart_optimization",
            failed_step_id=step.id,
            candidate_artifact_id=candidate.id,
            parameter_patch={"geom_maxiter": new},
            evidence_refs=evidence,
            reason="ORCA reported the optimization iteration limit with a valid final geometry",
        )
    ]


def applicable_scf_repair(run: Run, step: Step, result: Result) -> list[RepairOption]:
    """Expose SCF repair only when an explicit verified near-convergence fact exists."""

    if (
        step.tool not in {"single_point", "optimize_geometry"}
        or result.status != "failed"
        or not _profile_allows(step, "increase_scf_maxiter")
    ):
        return []
    facts = result.diagnostics.get("facts", {})
    process = result.diagnostics.get("process", {})
    if (
        not isinstance(facts, dict)
        or facts.get("scf_iteration_limit_reached") is not True
        or facts.get("effective_scf_maxiter") is None
        or facts.get("scf_near_converged") is not True
        or not _cleanly_stopped(process)
        or not _input_integrity_ok(result.diagnostics)
    ):
        return []
    old = int(facts["effective_scf_maxiter"])
    new = min(MAX_ITERATION, max(old + 1, old * 2))
    if new <= old:
        return []
    if not _scope_allows_patch(
        run, step, action="increase_scf_maxiter", patch={"scf_maxiter": new}
    ):
        return []
    return [
        RepairOption(
            action="increase_scf_maxiter",
            failed_step_id=step.id,
            candidate_artifact_id=None,
            parameter_patch={"scf_maxiter": new},
            evidence_refs=(
                "scf_iteration_limit_reached",
                "scf_near_converged",
                "process_cleanup_confirmed",
                "input_hashes_match",
            ),
            reason="SCF reached an explicit limit inside the verified near-convergence range",
        )
    ]


def validate_repair_option(
    option: RepairOption,
    *,
    run: Run,
    step: Step,
    result: Result,
    requested_action: str,
    requested_patch: Mapping[str, Any],
    requested_candidate_id: str | None,
    evidence_refs: list[str],
) -> tuple[Step, dict[str, Any]]:
    """Validate an advisory proposal and derive a new Step deterministically."""

    if requested_action != option.action:
        raise ValueError("repair action is not one of the currently applicable actions")
    if requested_candidate_id != option.candidate_artifact_id:
        raise ValueError("repair candidate is not the validated candidate for this attempt")
    if set(evidence_refs) != set(option.evidence_refs):
        raise ValueError("repair evidence references do not match the validated facts")
    normalized_patch = {str(key): value for key, value in requested_patch.items()}
    if normalized_patch != option.parameter_patch:
        raise ValueError("repair parameter patch differs from the deterministic allowed patch")
    if any(type(value) is not int for value in normalized_patch.values()):
        raise ValueError("repair iteration limits must be integers")
    if set(normalized_patch) - {"geom_maxiter", "scf_maxiter"}:
        raise ValueError("repair may change only a bounded iteration parameter")
    if not _scope_allows_patch(run, step, action=option.action, patch=normalized_patch):
        raise ValueError("repair patch exceeds the accepted repair scope")
    old_parameters = dict(step.parameters)
    new_parameters = dict(old_parameters)
    new_parameters.update(normalized_patch)
    if option.action == "restart_optimization":
        if step.tool != "optimize_geometry" or "geom_maxiter" not in normalized_patch:
            raise ValueError("restart_optimization applies only to geometry optimization")
        candidate = _candidate_for_result(run, result)
        if candidate is None or candidate.id != requested_candidate_id:
            raise ValueError("restart candidate is missing from the failed Result")
        new_inputs = dict(step.inputs)
        geometry_name = "geometry"
        from bg6022.models import InputReference

        new_inputs[geometry_name] = InputReference(artifact_id=candidate.id)
    elif option.action == "increase_scf_maxiter":
        if "scf_maxiter" not in normalized_patch:
            raise ValueError("increase_scf_maxiter needs an scf_maxiter patch")
        new_inputs = dict(step.inputs)
    else:
        raise ValueError(f"unsupported repair action: {option.action}")

    new_step = Step.model_validate(
        {
            **step.model_dump(mode="python"),
            "parameters": new_parameters,
            "inputs": new_inputs,
        },
        strict=True,
    )
    record = {
        "action": option.action,
        "failed_step_id": step.id,
        "failed_attempt": result.attempt,
        "failed_result_path": result.attempt_relative_path + "/result.json",
        "candidate_artifact_id": requested_candidate_id,
        "candidate_sha256": _artifact_sha(run, requested_candidate_id),
        "old_parameters": old_parameters,
        "new_parameters": new_parameters,
        "parameter_patch": normalized_patch,
        "evidence_refs": list(evidence_refs),
        "reason": option.reason,
        "validated": True,
    }
    return new_step, record


def _candidate_for_result(run: Run, result: Result) -> Artifact | None:
    for artifact_id in result.artifact_ids:
        artifact = next((item for item in run.artifact_index if item.id == artifact_id), None)
        if artifact is not None and artifact.role == "restart_candidate":
            return artifact
    return None


def _candidate_is_eligible(candidate: Artifact, step: Step, result: Result) -> bool:
    return (
        candidate.step_id == step.id
        and candidate.attempt == result.attempt
        and candidate.metadata.get("eligible_for") == "optimization_restart_only"
        and candidate.id in result.artifact_ids
    )


def _artifact_sha(run: Run, artifact_id: str | None) -> str | None:
    if artifact_id is None:
        return None
    for artifact in run.artifact_index:
        if artifact.id == artifact_id:
            return artifact.sha256
    return None


def _cleanly_stopped(process: Any) -> bool:
    return (
        isinstance(process, dict)
        and process.get("process_tree_empty") is True
        and process.get("stop_confirmed") is True
        and process.get("status") not in {"cancelled", "timed_out", "interrupted"}
    )


def _input_integrity_ok(diagnostics: Mapping[str, Any]) -> bool:
    hashes = diagnostics.get("input_hashes")
    return isinstance(hashes, dict) and hashes.get("match") is True


def _scope_allows_patch(run: Run, step: Step, *, action: str, patch: Mapping[str, Any]) -> bool:
    """Check the accepted per-Run repair scope when one is available.

    A few M0 unit callers construct an in-memory Run without an acceptance
    snapshot; those compatibility tests retain the old local rule.  Any real
    Agent Run has a snapshot before a repair can be proposed and therefore
    must pass this narrower check.
    """

    snapshot = run.accepted_snapshot
    if not snapshot:
        return True
    repair_scope = snapshot.get("repair_scope")
    if not isinstance(repair_scope, dict):
        return False
    scopes = repair_scope.get("steps")
    if not isinstance(scopes, dict):
        return False
    scope = scopes.get(step.id)
    if not isinstance(scope, dict):
        return False
    actions = scope.get("actions")
    action_scope = actions.get(action) if isinstance(actions, dict) else None
    if not isinstance(action_scope, dict):
        return False
    fields = action_scope.get("fields")
    maximum = action_scope.get("maximum")
    if not isinstance(fields, list) or set(patch) != set(fields):
        return False
    if type(maximum) is not int:
        return False
    return all(type(value) is int and 1 <= value <= maximum for value in patch.values())


def _profile_allows(step: Step, action: str) -> bool:
    operation = {
        "single_point": "SP",
        "optimize_geometry": "Opt",
    }.get(step.tool)
    if operation is None:
        return False
    method_profile = step.parameters.get("method_profile", "r2scan3c")
    if not isinstance(method_profile, str):
        return False
    try:
        profile = get_profile(method_profile)
    except ValueError:
        return False
    return action in profile.repair_options_by_operation.get(operation, ())


__all__ = [
    "MAX_ITERATION",
    "RepairOption",
    "applicable_repairs",
    "applicable_scf_repair",
    "validate_repair_option",
]
