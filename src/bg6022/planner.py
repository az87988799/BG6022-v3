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
        _proposal_target_to_result_target(target, step_ids) for target in proposal.requested_results
    ]
    plan = Plan(
        id=plan_id,
        request_id=request.id,
        steps=steps,
        requested_results=targets,
    )
    return validate_request_plan(request, plan, registry)


def validate_request_plan(request: Request, plan: Plan, registry: ToolRegistry) -> Plan:
    """Validate a model Plan against the user's original semantic request.

    ``ToolRegistry`` can prove that a Plan is internally well-formed, but it
    cannot know whether the model silently dropped the requested scientific
    operation or changed the meaning of a result such as ``energy``.  Keep
    that check at the boundary where both Request and Plan are available.
    """

    plan = registry.validate_plan(plan)
    science_tools = {
        step.tool for step in plan.steps if step.tool in {"single_point", "optimize_geometry"}
    }
    if request.operation is not None:
        expected_tool = {"SP": "single_point", "Opt": "optimize_geometry"}[request.operation]
        if expected_tool not in science_tools:
            raise ValueError(f"Plan does not cover the requested {request.operation} calculation")
        unexpected = sorted(science_tools - {expected_tool})
        if unexpected:
            raise ValueError(
                f"Plan adds unrequested scientific operation(s): {', '.join(unexpected)}"
            )

    plan_targets = plan.requested_results
    for request_target in request.requested_results:
        matches = [
            target
            for target in plan_targets
            if _request_target_matches(request_target, target, request)
        ]
        if not matches:
            name = request_target.port or request_target.field or "<unknown>"
            raise ValueError(f"Plan does not cover the requested result: {name}")
        if len(matches) > 1:
            name = request_target.port or request_target.field or "<unknown>"
            raise ValueError(f"Requested result is ambiguous in the Plan: {name}")
    return plan


def request_from_intake(message: str, intake: IntakeOutput, *, request_id: str) -> Request:
    if any(key in intake.structure_input for key in {"path", "local_path", "file"}):
        raise ValueError("model intake cannot authorize a local file path")
    return Request(
        id=request_id,
        description=message,
        original_text=message,
        requested_results=[ResultTarget(field=value) for value in intake.requested_results],
        explicit_parameters=filter_user_explicit_parameters(message, intake.explicit_parameters),
        structure_input=dict(intake.structure_input),
        source="chat",
        operation=intake.operation,
        missing_fields=list(intake.missing_fields),
    )


def filter_user_explicit_parameters(message: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-extracted q/M only when the user actually stated them.

    Method and iteration options are ordinary planner hints and remain
    available for later deterministic validation.  Charge and multiplicity
    are different: a model's guessed values must never become scientific
    evidence merely because they appeared in an intake JSON object.
    """

    filtered = dict(parameters)
    for name in ("charge", "multiplicity"):
        if name in filtered and not _message_declares_parameter(message, name, filtered[name]):
            filtered.pop(name)
    return filtered


def _request_target_matches(
    request_target: ResultTarget, plan_target: ResultTarget, request: Request
) -> bool:
    if request_target.step_id is not None and request_target.step_id != plan_target.step_id:
        return False

    request_kind, request_name = _semantic_target(request_target, request)
    plan_kind = "port" if plan_target.port is not None else "field"
    plan_name = plan_target.port or plan_target.field
    if request_target.field == "energy" and request.operation is None:
        return plan_kind == "field" and plan_name in {
            "sp_electronic_energy",
            "opt_final_electronic_energy",
        }
    return request_kind == plan_kind and request_name == plan_name


def _proposal_target_to_result_target(
    target: PlanTargetProposal, step_ids: Mapping[str, str]
) -> ResultTarget:
    try:
        step_id = step_ids[target.step_key]
    except KeyError as error:
        raise ValueError(
            f"planner requested result references unknown step key {target.step_key!r}"
        ) from error
    return ResultTarget(step_id=step_id, field=target.field, port=target.port)


def _semantic_target(target: ResultTarget, request: Request) -> tuple[str, str | None]:
    if target.port is not None:
        return "port", target.port
    name = target.field
    aliases = {
        "optimized_geometry": ("port", "optimized_geometry"),
        "geometry": ("port", "geometry"),
        "sp_energy": ("field", "sp_electronic_energy"),
        "opt_energy": ("field", "opt_final_electronic_energy"),
    }
    if name in aliases:
        return aliases[name]
    if name == "energy":
        if request.operation == "SP":
            return "field", "sp_electronic_energy"
        if request.operation == "Opt":
            return "field", "opt_final_electronic_energy"
    return "field", name


def _message_declares_parameter(message: str, name: str, value: Any) -> bool:
    if type(value) is not int:
        return False
    integer = rf"\+?{value}" if value >= 0 else rf"{value}"
    if name == "charge":
        patterns = [
            rf"(?:\bcharge\b|\bq\b|电荷)\s*(?:is\s*|=\s*|:\s*|为\s*)?{integer}\b",
            rf"{integer}\s*(?:\bcharge\b|\bq\b|电荷)",
        ]
        if value == 0:
            patterns.append(r"\b(?:neutral|中性)\b")
    else:
        patterns = [
            rf"(?:\bmultiplicity\b|\bspin\s*multiplicity\b|\bmult\b|\bM\b|多重度|自旋多重度|自旋)\s*(?:is\s*|=\s*|:\s*|为\s*)?{integer}\b",
            rf"{integer}\s*(?:\bmultiplicity\b|\bmult\b|\bM\b|多重度|自旋多重度)",
        ]
        if value == 1:
            patterns.append(r"\b(?:singlet|单重态)\b")
        if value == 3:
            patterns.append(r"\b(?:triplet|三重态)\b")
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)


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
    "filter_user_explicit_parameters",
    "intake_message",
    "load_prompt",
    "plan_message",
    "proposal_to_plan",
    "request_from_intake",
    "validate_request_plan",
]
