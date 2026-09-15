"""LLM repair proposals plus deterministic local application."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bg6022.llm import LlmClient
from bg6022.models import Result, Run, Step
from bg6022.orca.repair_rules import RepairOption, validate_repair_option
from bg6022.planner import load_prompt


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: StrictStr
    failed_step_key: StrictStr
    candidate_alias: StrictStr | None = None
    parameter_patch: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _candidate_for_action(self) -> RepairProposal:
        if (
            self.action == "restart_optimization"
            and self.candidate_alias != "last_complete_geometry"
        ):
            raise ValueError(
                "restart_optimization requires the program-provided last_complete_geometry alias"
            )
        if self.action != "restart_optimization" and self.candidate_alias is not None:
            raise ValueError("candidate_alias is only valid for restart_optimization")
        if self.action == "none" and (self.parameter_patch or self.evidence_refs):
            raise ValueError("none repair proposal cannot contain a patch or evidence")
        return self


def propose_repair(
    client: LlmClient,
    *,
    run: Run,
    step: Step,
    result: Result,
    options: list[RepairOption],
    cancel: Any = None,
    remaining_timeout_seconds: float | None = None,
) -> RepairProposal | None:
    if not options:
        return None
    response = client.complete_json(
        [
            {"role": "system", "content": load_prompt("repair")},
            {
                "role": "user",
                "content": _context(
                    run,
                    step,
                    result,
                    options,
                    remaining_timeout_seconds=remaining_timeout_seconds,
                ),
            },
        ],
        RepairProposal,
        purpose="repair",
        example={
            "action": options[0].action,
            "failed_step_key": step.id,
            "candidate_alias": "last_complete_geometry"
            if options[0].candidate_artifact_id
            else None,
            "parameter_patch": options[0].parameter_patch,
            "evidence_refs": list(options[0].evidence_refs),
        },
        cancel=cancel,
        remaining_timeout_seconds=remaining_timeout_seconds,
    )
    return response


def apply_repair_proposal(
    proposal: RepairProposal,
    *,
    option: RepairOption,
    run: Run,
    step: Step,
    result: Result,
) -> tuple[Step, dict[str, Any]]:
    if proposal.failed_step_key != step.id:
        raise ValueError("repair proposal targets a different step")
    candidate_id = option.candidate_artifact_id if proposal.candidate_alias else None
    return validate_repair_option(
        option,
        run=run,
        step=step,
        result=result,
        requested_action=proposal.action,
        requested_patch=proposal.parameter_patch,
        requested_candidate_id=candidate_id,
        evidence_refs=list(proposal.evidence_refs),
    )


def _context(
    run: Run,
    step: Step,
    result: Result,
    options: list[RepairOption],
    *,
    remaining_timeout_seconds: float | None,
) -> str:
    import json

    origin = run.origin_step_map.get(step.id, step.origin_step_id or step.id)
    max_attempts = int(run.budget.get("max_attempts_per_science_step", 3))
    used_attempts = int(run.attempt_counts.get(origin, 0))
    attempts = [
        {
            name: item[name]
            for name in ("attempt", "phase", "status", "result_category", "operation")
            if name in item
        }
        for item in run.attempts
        if item.get("step_id") == step.id
    ][-max_attempts:]
    snapshot = run.accepted_snapshot if isinstance(run.accepted_snapshot, dict) else {}
    original_request = snapshot.get("request")
    if not isinstance(original_request, dict):
        original_request = run.request.model_dump(mode="json")
    accepted_plan = snapshot.get("plan")
    original_step = None
    if isinstance(accepted_plan, dict) and isinstance(accepted_plan.get("steps"), list):
        original_step = next(
            (
                item
                for item in accepted_plan["steps"]
                if isinstance(item, dict) and item.get("id") == step.id
            ),
            None,
        )
    return json.dumps(
        {
            "failed_step": step.id,
            "tool": step.tool,
            "prior_attempts": attempts,
            "attempt_budget": {
                "origin_step_id": origin,
                "used": used_attempts,
                "maximum": max_attempts,
                "remaining": max(0, max_attempts - used_attempts),
            },
            "run_budget": {
                "extra_orca_executions_used": run.extra_orca_executions,
                "extra_orca_executions_maximum": int(
                    run.budget.get("max_extra_orca_executions", 3)
                ),
                "remaining_active_seconds": remaining_timeout_seconds,
            },
            "original_scientific_constraints": {
                "request": {
                    name: original_request.get(name)
                    for name in (
                        "description",
                        "operations",
                        "requested_results",
                        "explicit_parameters",
                        "user_modifications",
                    )
                    if name in original_request
                },
                "step": original_step
                or {
                    "tool": step.tool,
                    "parameters": dict(step.parameters),
                    "inputs": {
                        name: reference.model_dump(mode="json")
                        for name, reference in step.inputs.items()
                    },
                    "goal_checks": [goal.model_dump(mode="json") for goal in step.goal_checks],
                },
            },
            "result": {
                "status": result.status,
                "attempt": result.attempt,
                "diagnostics": result.diagnostics,
            },
            "applicable_actions": [option.to_dict() for option in options],
            "budget": run.budget,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


__all__ = ["RepairProposal", "apply_repair_proposal", "propose_repair"]
