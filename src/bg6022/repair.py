"""LLM repair proposals plus deterministic local application."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bg6022.llm import LlmClient
from bg6022.models import Plan, RepairOption, Result, Run, Step, Tool
from bg6022.planner import load_prompt


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    option_id: StrictStr | None = None
    # Read-only compatibility with earlier saved repair prompts.
    action: StrictStr | None = None
    failed_step_key: StrictStr
    candidate_alias: StrictStr | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_patch: dict[str, Any] = Field(default_factory=dict)
    input_aliases: list[StrictStr] = Field(default_factory=list)
    evidence_refs: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _proposal_contract(self) -> RepairProposal:
        if self.option_id is None and self.action is None:
            raise ValueError("repair proposal must select an option_id")
        if self.option_id is not None and self.action is not None and self.option_id != self.action:
            raise ValueError("legacy action conflicts with option_id")
        if self.parameters and self.parameter_patch and self.parameters != self.parameter_patch:
            raise ValueError("legacy parameter_patch conflicts with parameters")
        if self.candidate_alias is not None and self.input_aliases:
            if self.input_aliases != [self.candidate_alias]:
                raise ValueError("candidate_alias conflicts with input_aliases")
        if self.selected_option_id == "none" and (
            self.selected_parameters or self.evidence_refs or self.selected_input_aliases
        ):
            raise ValueError("none repair proposal cannot contain a patch or evidence")
        return self

    @property
    def selected_option_id(self) -> str:
        return self.option_id or self.action or "none"

    @property
    def selected_parameters(self) -> dict[str, Any]:
        return self.parameters or self.parameter_patch

    @property
    def selected_input_aliases(self) -> list[str]:
        return list(self.input_aliases or ([self.candidate_alias] if self.candidate_alias else []))


def propose_repair(
    client: LlmClient,
    *,
    run: Run,
    step: Step,
    result: Result,
    options: list[RepairOption],
    budget_category: str = "none",
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
                    budget_category=budget_category,
                    remaining_timeout_seconds=remaining_timeout_seconds,
                ),
            },
        ],
        RepairProposal,
        purpose="repair",
        example={
            "option_id": options[0].option_id,
            "failed_step_key": step.id,
            "parameters": dict(options[0].parameters),
            "input_aliases": list(options[0].input_aliases),
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
    tool: Tool,
) -> tuple[Plan, dict[str, Any]]:
    if proposal.failed_step_key != step.id:
        raise ValueError("repair proposal targets a different step")
    if proposal.selected_option_id != option.option_id:
        raise ValueError("repair option is not one of the currently applicable options")
    if proposal.selected_parameters != dict(option.parameters):
        raise ValueError("repair parameters differ from the selected Tool option")
    if set(proposal.selected_input_aliases) != set(option.input_aliases):
        raise ValueError("repair input aliases differ from the selected Tool option")
    if set(proposal.evidence_refs) != set(option.evidence_refs):
        raise ValueError("repair evidence references differ from the selected Tool option")
    return tool.apply_repair(
        option,
        run,
        step,
        result,
        {
            "option_id": option.option_id,
            "parameters": proposal.selected_parameters,
            "input_aliases": proposal.selected_input_aliases,
            "evidence_refs": list(proposal.evidence_refs),
        },
    )


def _context(
    run: Run,
    step: Step,
    result: Result,
    options: list[RepairOption],
    *,
    budget_category: str,
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
                "execution_category": budget_category,
                "extra_executions_used": run.extra_executions_by_category.get(budget_category, 0),
                "extra_executions_maximum": int(
                    run.budget.get("max_extra_executions_by_category", {}).get(budget_category, 0)
                ),
                "remaining_active_seconds": remaining_timeout_seconds,
            },
            "original_scientific_constraints": {
                "request": {
                    name: original_request.get(name)
                    for name in (
                        "description",
                        "subjects",
                        "requirements",
                        "answer_goals",
                        "missing_fields",
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
