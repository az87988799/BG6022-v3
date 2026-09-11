"""Natural-language intake and Tool-directory-driven plan construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from bg6022.llm import LlmClient
from bg6022.models import InputReference, Plan, Request, ResultTarget, Step
from bg6022.tools.registry import ToolRegistry

Intent = Literal["chemistry_compute", "chemistry_qa", "daily_qa", "context_query"]
Operation = Literal["SP", "Opt"]


class IntakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: Intent
    answer: StrictStr | None = None
    operation: Operation | None = None
    molecule_query: StrictStr | None = None
    molecule_input_kind: Literal["name", "cas", "cid", "smiles"] | None = None
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    structure_input: dict[str, Any] = Field(default_factory=dict)
    requested_results: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    context_reference: StrictStr | None = None

    @model_validator(mode="after")
    def _intent_fields(self) -> IntakeOutput:
        if self.intent == "context_query" and self.explicit_parameters:
            raise ValueError("context_query cannot contain a parameter patch")
        return self


class InputBindingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step_key: StrictStr | None = None
    port: StrictStr | None = None
    artifact_alias: StrictStr | None = None

    @model_validator(mode="after")
    def _one_binding(self) -> InputBindingProposal:
        if self.artifact_alias is not None:
            if self.step_key is not None or self.port is not None:
                raise ValueError("artifact_alias cannot be combined with step_key/port")
            return self
        if (self.step_key is None) != (self.port is None):
            raise ValueError("step_key and port must be provided together")
        if self.step_key is None:
            raise ValueError("input binding needs a step_key/port or artifact_alias")
        return self


class PlanStepProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: StrictStr
    tool: StrictStr
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputBindingProposal] = Field(default_factory=dict)
    goal_checks: list[str] = Field(default_factory=list)


class PlanTargetProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step_key: StrictStr
    field: StrictStr | None = None
    port: StrictStr | None = None

    @model_validator(mode="after")
    def _one_target(self) -> PlanTargetProposal:
        if (self.field is None) == (self.port is None):
            raise ValueError("plan target needs exactly one field or port")
        return self


class PlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    steps: list[PlanStepProposal]
    requested_results: list[PlanTargetProposal] = Field(default_factory=list)
    explanation: StrictStr | None = None

    @field_validator("steps")
    @classmethod
    def _steps_nonempty(cls, value: list[PlanStepProposal]) -> list[PlanStepProposal]:
        if not value:
            raise ValueError("plan proposal must contain at least one step")
        return value


def load_prompt(name: str) -> str:
    """Read prompts from the installed package, independent of the cwd."""

    from importlib.resources import files

    try:
        return files("bg6022.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as error:
        raise RuntimeError(f"packaged prompt is unavailable: {name}") from error


def intake_message(
    client: LlmClient,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> IntakeOutput:
    if not message.strip():
        raise ValueError("message must not be empty")
    prompt = load_prompt("intake")
    return client.complete_json(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_context(
                    {"message": message, "recent_context": _bounded_context(context)}
                ),
            },
        ],
        IntakeOutput,
        purpose="intake",
        example={
            "intent": "chemistry_compute",
            "operation": "Opt",
            "molecule_query": "water",
            "molecule_input_kind": "name",
            "structure_input": {},
            "explicit_parameters": {},
            "requested_results": ["energy"],
            "missing_fields": [],
        },
        cancel=cancel,
    )


def plan_message(
    client: LlmClient,
    request: Request,
    *,
    registry: ToolRegistry,
    context: Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> PlanProposal:
    prompt = load_prompt("planner")
    directory = registry.describe()
    return client.complete_json(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_context(
                    {
                        "request": request.model_dump(mode="json"),
                        "tool_directory": directory,
                        "context": _bounded_context(context),
                    }
                ),
            },
        ],
        PlanProposal,
        purpose="planner",
        example={
            "steps": [
                {
                    "key": "molecule",
                    "tool": "resolve_molecule",
                    "parameters": {"query": "water", "input_kind": "name"},
                    "inputs": {},
                },
                {
                    "key": "geometry",
                    "tool": "generate_geometry",
                    "parameters": {},
                    "inputs": {"molecule": {"step_key": "molecule", "port": "molecule"}},
                },
                {
                    "key": "opt",
                    "tool": "optimize_geometry",
                    "parameters": {"method_profile": "r2scan3c", "environment": "gas"},
                    "inputs": {"geometry": {"step_key": "geometry", "port": "geometry"}},
                },
            ],
            "requested_results": [
                {"step_key": "opt", "field": "opt_final_electronic_energy"},
                {"step_key": "opt", "port": "optimized_geometry"},
            ],
        },
        cancel=cancel,
    )


def proposal_to_plan(
    request: Request, proposal: PlanProposal, registry: ToolRegistry, *, plan_id: str
) -> Plan:
    """Map model-only keys to stable domain IDs and run both plan validations."""

    keys = [item.key for item in proposal.steps]
    if len(set(keys)) != len(keys):
        raise ValueError("planner returned duplicate step keys")
    step_ids = {key: _step_id(index, key) for index, key in enumerate(keys, start=1)}
    steps: list[Step] = []
    for item in proposal.steps:
        inputs: dict[str, InputReference] = {}
        for name, binding in item.inputs.items():
            if binding.artifact_alias is not None:
                inputs[name] = InputReference(artifact_id=binding.artifact_alias)
            else:
                assert binding.step_key is not None and binding.port is not None
                if binding.step_key not in step_ids:
                    raise ValueError(
                        f"planner input references unknown step key {binding.step_key}"
                    )
                inputs[name] = InputReference(step_id=step_ids[binding.step_key], port=binding.port)
        steps.append(
            Step(
                id=step_ids[item.key],
                origin_step_id=step_ids[item.key],
                tool=item.tool,
                parameters=dict(item.parameters),
                inputs=inputs,
                goal_checks=list(item.goal_checks),
            )
        )
    targets = [
        ResultTarget(step_id=step_ids[target.step_key], field=target.field, port=target.port)
        for target in proposal.requested_results
    ]
    plan = Plan(
        id=plan_id,
        request_id=request.id,
        steps=steps,
        requested_results=targets,
    )
    return registry.validate_plan(plan)


def request_from_intake(message: str, intake: IntakeOutput, *, request_id: str) -> Request:
    if any(key in intake.structure_input for key in {"path", "local_path", "file"}):
        raise ValueError("model intake cannot authorize a local file path")
    return Request(
        id=request_id,
        description=message,
        original_text=message,
        requested_results=[ResultTarget(field=value) for value in intake.requested_results],
        explicit_parameters=dict(intake.explicit_parameters),
        structure_input=dict(intake.structure_input),
        source="chat",
        missing_fields=list(intake.missing_fields),
    )


def _step_id(index: int, key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", key).strip("_") or "step"
    return f"s{index:02d}_{cleaned}"


def _bounded_context(context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not context:
        return {}
    recent = context.get("recent_messages")
    if isinstance(recent, list):
        recent = recent[-12:]
    results = context.get("recent_results")
    if isinstance(results, list):
        results = results[-3:]
    return {"recent_messages": recent or [], "recent_results": results or []}


def _json_context(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "InputBindingProposal",
    "IntakeOutput",
    "PlanProposal",
    "PlanStepProposal",
    "PlanTargetProposal",
    "intake_message",
    "load_prompt",
    "plan_message",
    "proposal_to_plan",
    "request_from_intake",
]
