"""Natural-language intake and Tool-directory-driven plan construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    create_model,
    field_validator,
    model_validator,
)

from bg6022.llm import LlmClient
from bg6022.models import (
    GoalCheckRequirement,
    InputReference,
    Operation,
    Plan,
    Request,
    ResultProperty,
    ResultTarget,
    Step,
)
from bg6022.tools.registry import ToolRegistry

Intent = Literal["chemistry_compute", "chemistry_qa", "daily_qa", "context_query"]
QuerySelectionStatus = Literal["selected", "clarify", "unavailable"]
ElectronicStateField = Literal["charge", "multiplicity"]
ElectronicStateStatus = Literal["absent", "set", "ambiguous", "invalid"]


class ElectronicStateCandidate(BaseModel):
    """Untrusted model extraction accompanied by an exact quote from this turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    field: ElectronicStateField
    raw_value: StrictStr
    evidence: StrictStr


class QueryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject_ref: StrictStr
    property: ResultProperty
    evidence: StrictStr


class QuerySelection(BaseModel):
    """Ephemeral model output for choosing facts from this intake round.

    The short references are created by the Agent and are never persisted as
    domain objects.  A dynamic Intake model adds the current candidate-set
    validator before the model call is made.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    status: QuerySelectionStatus
    targets: list[QueryTarget] = Field(default_factory=list)
    clarification: StrictStr | None = None
    missing_description: StrictStr | None = None

    @field_validator("targets")
    @classmethod
    def _bounded_targets(cls, value: list[QueryTarget]) -> list[QueryTarget]:
        if len(value) > 3:
            raise ValueError("query selection may contain at most three targets")
        identities = [(item.subject_ref, item.property) for item in value]
        if len(set(identities)) != len(identities):
            raise ValueError("query targets must be unique per subject and property")
        return value

    @model_validator(mode="after")
    def _status_matches_refs(self) -> QuerySelection:
        if self.status == "selected" and not self.targets:
            raise ValueError("selected query result must contain at least one target")
        if self.status != "selected" and self.targets:
            raise ValueError("clarify/unavailable query results cannot contain targets")
        return self


@dataclass(frozen=True)
class ElectronicStateInput:
    status: ElectronicStateStatus
    value: int | None = None
    evidence: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ParameterNormalization:
    explicit_parameters: dict[str, Any]
    states: dict[str, ElectronicStateInput]

    @property
    def clarification_fields(self) -> tuple[str, ...]:
        return tuple(
            name for name, state in self.states.items() if state.status in {"ambiguous", "invalid"}
        )


class IntakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: Intent
    answer: StrictStr | None = None
    operations: list[Operation] = Field(default_factory=list)
    molecule_query: StrictStr | None = None
    molecule_input_kind: Literal["name", "cas", "cid", "smiles"] | None = None
    history_geometry_alias: StrictStr | None = None
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    electronic_state_candidates: list[ElectronicStateCandidate] = Field(default_factory=list)
    structure_input: dict[str, Any] = Field(default_factory=dict)
    requested_results: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    query_selection: QuerySelection | None = None

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_operation(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "operation" not in value:
            return value
        data = dict(value)
        legacy = data.pop("operation")
        if legacy is not None:
            if "operations" in data and data["operations"] != [legacy]:
                raise ValueError("legacy operation conflicts with operations")
            data["operations"] = [legacy]
        return data

    @model_validator(mode="after")
    def _intent_fields(self) -> IntakeOutput:
        if self.intent == "context_query" and (
            self.explicit_parameters
            or self.electronic_state_candidates
            or self.history_geometry_alias
        ):
            raise ValueError("context_query cannot contain a parameter patch")
        if self.intent == "context_query" and self.query_selection is None:
            raise ValueError("context_query must contain query_selection")
        if self.intent != "context_query" and self.query_selection is not None:
            raise ValueError("query_selection is only valid for context_query")
        return self

    @field_validator("operations")
    @classmethod
    def _unique_operations(cls, value: list[Operation]) -> list[Operation]:
        if len(set(value)) != len(value):
            raise ValueError("requested operations must be unique")
        return value

    @property
    def operation(self) -> Operation | None:
        """Read-only compatibility with pre-M2 intake handlers."""

        return self.operations[0] if len(self.operations) == 1 else None


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


class GoalCheckProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_step_key: StrictStr
    check: StrictStr
    required_status: Literal["passed", "not_met", "unverified"] = "passed"


class PlanStepProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: StrictStr
    tool: StrictStr
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputBindingProposal] = Field(default_factory=dict)
    goal_checks: list[GoalCheckProposal] = Field(default_factory=list)


class PlanTargetProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step_key: StrictStr
    field: StrictStr | None = None
    port: StrictStr | None = None
    check: StrictStr | None = None

    @model_validator(mode="after")
    def _one_target(self) -> PlanTargetProposal:
        if sum(value is not None for value in (self.field, self.port, self.check)) != 1:
            raise ValueError("plan target needs exactly one field, port, or check")
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
    result_catalog: list[Mapping[str, Any]] | None = None,
    geometry_catalog: list[Mapping[str, Any]] | None = None,
    cancel: Any = None,
) -> IntakeOutput:
    if not message.strip():
        raise ValueError("message must not be empty")
    prompt = load_prompt("intake")
    catalog = [dict(item) for item in (result_catalog or [])]
    candidate_refs = tuple(
        str(item["subject_ref"])
        for item in catalog
        if isinstance(item, Mapping) and isinstance(item.get("subject_ref"), str)
    )
    schema = _intake_schema(candidate_refs)
    value = client.complete_json(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_context(
                    {
                        "message": message,
                        "recent_context": _bounded_context(context),
                        "result_catalog": catalog,
                        "geometry_catalog": [dict(item) for item in (geometry_catalog or [])],
                    }
                ),
            },
        ],
        schema,
        purpose="intake",
        example={
            "intent": "chemistry_compute",
            "operations": ["Opt"],
            "molecule_query": "water",
            "molecule_input_kind": "name",
            "history_geometry_alias": None,
            "structure_input": {},
            "explicit_parameters": {},
            "electronic_state_candidates": [],
            "requested_results": ["energy"],
            "missing_fields": [],
        },
        cancel=cancel,
    )
    return _coerce_intake_output(value, schema, candidate_refs, message=message)


def plan_message(
    client: LlmClient,
    request: Request,
    *,
    registry: ToolRegistry,
    context: Mapping[str, Any] | None = None,
    validation_feedback: str | None = None,
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
                        "validation_feedback": validation_feedback,
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
    request: Request,
    proposal: PlanProposal,
    registry: ToolRegistry,
    *,
    plan_id: str,
    artifact_aliases: Mapping[str, str] | None = None,
) -> Plan:
    """Map model-only keys to stable domain IDs and run both plan validations."""

    keys = [item.key for item in proposal.steps]
    if len(set(keys)) != len(keys):
        raise ValueError("planner returned duplicate step keys")
    step_ids = {key: _step_id(index, key) for index, key in enumerate(keys, start=1)}
    allowed_artifact_aliases = dict(artifact_aliases or {})
    steps: list[Step] = []
    for item in proposal.steps:
        inputs: dict[str, InputReference] = {}
        for name, binding in item.inputs.items():
            if binding.artifact_alias is not None:
                resolved_alias = allowed_artifact_aliases.get(binding.artifact_alias)
                if resolved_alias is None:
                    raise ValueError(
                        "planner input uses an unknown or unselected artifact alias "
                        f"{binding.artifact_alias!r}"
                    )
                inputs[name] = InputReference(artifact_id=resolved_alias)
            else:
                assert binding.step_key is not None and binding.port is not None
                if binding.step_key not in step_ids:
                    raise ValueError(
                        f"planner input references unknown step key {binding.step_key}"
                    )
                inputs[name] = InputReference(step_id=step_ids[binding.step_key], port=binding.port)
        unknown_goal_sources = sorted(
            {goal.source_step_key for goal in item.goal_checks} - set(step_ids)
        )
        if unknown_goal_sources:
            raise ValueError(
                "planner goal check references unknown step key(s): "
                + ", ".join(unknown_goal_sources)
            )
        steps.append(
            Step(
                id=step_ids[item.key],
                origin_step_id=step_ids[item.key],
                tool=item.tool,
                parameters=dict(item.parameters),
                inputs=inputs,
                goal_checks=[
                    GoalCheckRequirement(
                        source_step_id=step_ids[goal.source_step_key],
                        check=goal.check,
                        required_status=goal.required_status,
                    )
                    for goal in item.goal_checks
                ],
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
    proposed_operations = [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ]
    # A missing operation list is retained only for pre-composition explicit
    # CLI/test Requests. Chat intake is rejected unless it declares an
    # operation, so new user Plans always receive this full coverage check.
    if request.operations and proposed_operations != request.operations:
        missing = [item for item in request.operations if item not in proposed_operations]
        extra = [item for item in proposed_operations if item not in request.operations]
        if missing:
            if len(missing) == 1:
                label = {"SP": "SP", "Opt": "Opt", "Freq": "frequency"}[missing[0]]
                raise ValueError(f"Plan does not cover the requested {label} calculation")
            raise ValueError(f"Plan does not cover requested operation(s): {', '.join(missing)}")
        if extra:
            raise ValueError(f"Plan adds unrequested operation(s): {', '.join(extra)}")
        raise ValueError("Plan operation order does not match the user's requested order")

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
            name = (
                request_target.check or request_target.port or request_target.field or "<unknown>"
            )
            raise ValueError(f"Requested result is ambiguous in the Plan: {name}")

    # A requested local-minimum goal paired with SP is a prerequisite, not
    # merely a report target. Enforce the edge here so a model cannot omit the
    # runtime gate and still launch the downstream calculation.
    if "SP" in request.operations and any(
        target.check == "local_minimum_supported" for target in request.requested_results
    ):
        check_targets = [
            target for target in plan_targets if target.check == "local_minimum_supported"
        ]
        if len(check_targets) != 1 or check_targets[0].step_id is None:
            raise ValueError("the local-minimum check must have one explicit producer before SP")
        check_step_id = check_targets[0].step_id
        for step in plan.steps:
            if "SP" not in registry.get(step.tool).operations:
                continue
            prerequisite = GoalCheckRequirement(
                source_step_id=check_step_id,
                check="local_minimum_supported",
                required_status="passed",
            )
            if prerequisite not in step.goal_checks:
                raise ValueError(
                    f"SP step {step.id!r} must require local_minimum_supported "
                    "to be passed before execution"
                )
    return plan


def request_from_intake(
    message: str,
    intake: IntakeOutput,
    *,
    request_id: str,
    normalized_parameters: ParameterNormalization | None = None,
) -> Request:
    if intake.intent == "chemistry_compute" and not intake.operations:
        raise ValueError(
            "the requested scientific operation is missing; please specify SP, Opt, or Freq"
        )
    if {"SP", "Opt"}.issubset(intake.operations) and any(
        value in {"energy", "electronic_energy"} for value in intake.requested_results
    ):
        raise ValueError(
            "the request includes both Opt and SP, so ‘energy’ is ambiguous; "
            "specify Opt energy or SP energy"
        )
    if any(key in intake.structure_input for key in {"path", "local_path", "file"}):
        raise ValueError("model intake cannot authorize a local file path")
    normalized = normalized_parameters or normalize_user_explicit_parameters(
        message, intake.explicit_parameters, intake.electronic_state_candidates
    )
    if normalized.clarification_fields:
        raise ValueError(
            "electronic-state parameters need clarification: "
            + ", ".join(normalized.clarification_fields)
        )
    return Request(
        id=request_id,
        description=message,
        original_text=message,
        requested_results=[
            ResultTarget(check=value)
            if value in {"frequency_complete", "local_minimum_supported"}
            else ResultTarget(field=value)
            for value in intake.requested_results
        ],
        explicit_parameters=dict(normalized.explicit_parameters),
        structure_input={
            **dict(intake.structure_input),
            **(
                {"history_geometry_alias": intake.history_geometry_alias}
                if intake.history_geometry_alias is not None
                else {}
            ),
        },
        source="chat",
        operations=list(intake.operations),
        missing_fields=list(intake.missing_fields),
    )


def filter_user_explicit_parameters(message: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility projection of the one authoritative parameter normalizer."""

    return normalize_user_explicit_parameters(message, parameters).explicit_parameters


def normalize_user_explicit_parameters(
    message: str,
    parameters: Mapping[str, Any],
    candidates: list[ElectronicStateCandidate] | tuple[ElectronicStateCandidate, ...] = (),
) -> ParameterNormalization:
    """Normalize one intake turn, retaining absent/ambiguous/invalid q/M states.

    Model values are suggestions only.  A q/M claim can enter the patch only
    when the user's own message contains one unambiguous, syntactically valid
    statement.  Candidate quotes are accepted as provenance only when they
    are exact substrings and independently parse to the same field/value.
    """

    patch = {
        name: value for name, value in parameters.items() if name not in {"charge", "multiplicity"}
    }
    states: dict[str, ElectronicStateInput] = {}
    for name in ("charge", "multiplicity"):
        state = _parse_electronic_state(message, name)
        verified_evidence = _verified_candidate_evidence(message, name, state, candidates)
        if verified_evidence is not None and state.status == "set":
            state = ElectronicStateInput(
                status="set", value=state.value, evidence=verified_evidence
            )
        states[name] = state
        if state.status == "set":
            patch[name] = state.value
    return ParameterNormalization(explicit_parameters=patch, states=states)


def electronic_state_clarification(normalized: ParameterNormalization) -> str:
    """Return a concise clarification for q/M statements that cannot be applied."""

    labels = {"charge": "总电荷", "multiplicity": "自旋多重度"}
    invalid = [
        name
        for name in normalized.clarification_fields
        if normalized.states[name].status == "invalid"
    ]
    if invalid:
        names = "、".join(labels[name] for name in invalid)
        return (
            f"本轮指定的{names}不是受支持的整数值；我没有改写成其他数值，"
            "也没有更新或启动计算。请给出明确的整数。"
        )
    names = "、".join(labels[name] for name in normalized.clarification_fields)
    return (
        f"本轮关于{names}的表达存在冲突、否定或歧义；我没有更新或启动计算。"
        "请明确给出要采用的整数值。"
    )


def _request_target_matches(
    request_target: ResultTarget, plan_target: ResultTarget, request: Request
) -> bool:
    if request_target.step_id is not None and request_target.step_id != plan_target.step_id:
        return False

    request_kind, request_name = _semantic_target(request_target, request)
    plan_kind = (
        "check"
        if plan_target.check is not None
        else "port"
        if plan_target.port is not None
        else "field"
    )
    plan_name = plan_target.check or plan_target.port or plan_target.field
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
    return ResultTarget(
        step_id=step_id,
        field=target.field,
        port=target.port,
        check=target.check,
    )


def _semantic_target(target: ResultTarget, request: Request) -> tuple[str, str | None]:
    if target.check is not None:
        return "check", target.check
    if target.port is not None:
        return "port", target.port
    name = target.field
    if name in {"geometry", "molecular_geometry"} and "Opt" in request.operations:
        return "port", "optimized_geometry"
    aliases = {
        "optimized_geometry": ("port", "optimized_geometry"),
        "geometry": ("port", "geometry"),
        "molecular_geometry": ("port", "geometry"),
        "sp_energy": ("field", "sp_electronic_energy"),
        "opt_energy": ("field", "opt_final_electronic_energy"),
        "frequency": ("field", "vibrational_frequencies"),
        "frequencies": ("field", "vibrational_frequencies"),
    }
    if name in aliases:
        return aliases[name]
    if name in {"energy", "electronic_energy"}:
        operations = set(request.operations)
        if "SP" in operations and "Opt" not in operations:
            return "field", "sp_electronic_energy"
        if "Opt" in operations and "SP" not in operations:
            return "field", "opt_final_electronic_energy"
    return "field", name


_STATE_TOKEN = (
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"true|false|null|none|nan|infinity|inf|"
    r"负[零〇一二两三四五六七八九十]|正[零〇一二两三四五六七八九十]|"
    r"[零〇一二两三四五六七八九十]"
)
_STATE_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<token>{_STATE_TOKEN})(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)
_STATE_LABELS = {
    "charge": r"(?:总电荷|电荷|(?<![A-Za-z])charge(?![A-Za-z])|(?<![A-Za-z0-9])q(?![A-Za-z0-9]))",
    "multiplicity": (
        r"(?:自旋多重度|多重度|(?<![A-Za-z])spin\s*multiplicity(?![A-Za-z])|"
        r"(?<![A-Za-z])multiplicity(?![A-Za-z])|(?<![A-Za-z])mult(?![A-Za-z])|"
        r"(?<![A-Za-z0-9])M(?![A-Za-z0-9])|自旋)"
    ),
}
_ANY_STATE_LABEL_RE = re.compile(
    rf"(?:{_STATE_LABELS['charge']}|{_STATE_LABELS['multiplicity']})", re.IGNORECASE
)
_FIELD_CORRECTION_RE = re.compile(
    rf"^\s*(?:(?:(?:changed?|change)\s+)?(?:from|从|由)\s*)?"
    rf"(?P<old>{_STATE_TOKEN})\s*"
    rf"(?:改\s*(?:为|成)|变为|change(?:d)?\s+to|switch\s+to|to|->|→)\s*"
    rf"(?P<new>{_STATE_TOKEN})",
    re.IGNORECASE,
)
_FIELD_ASSIGNMENT_PREFIX_RE = re.compile(
    r"^\s*(?:(?:is|equals?\s*(?:to)?|set(?:ting)?\s+(?:to|as)|to)\s*|"
    r"(?:设(?:置)?(?:为|成)|指定(?:为|成)|为|是|等于)|[:=：])\s*",
    re.IGNORECASE,
)
_UNCHANGED_STATE_RE = re.compile(
    r"^\s*(?:(?:is|remains?|stays?)\s+)?(?:"
    r"不变|原值|当前值|现值|原样|不(?:再)?(?:修改|改变|变更|改动)|"
    r"不要(?:再)?(?:修改|改(?:变)?|变更|改动)|无需(?:修改|改变|变更|改动)|"
    r"unchanged|same(?:\s+as\s+before)?|the\s+same(?:\s+as\s+before)?|as\s+is|"
    r"(?:do\s+not|don't)\s+change|"
    r"(?:keep|leave)(?:\s+it)?\s+(?:unchanged|as\s+is|the\s+same))",
    re.IGNORECASE,
)


def _parse_electronic_state(message: str, name: str) -> ElectronicStateInput:
    if name not in _STATE_LABELS:
        raise ValueError(f"unsupported electronic-state field: {name}")
    values: set[int] = set()
    invalid: list[str] = []
    ambiguous = False
    mentioned = False
    unchanged_only = False
    evidence: str | None = None

    phrase_values = (
        [(r"中性|(?<![A-Za-z])neutral(?![A-Za-z])", 0)]
        if name == "charge"
        else [
            (r"单重态|(?<![A-Za-z])singlet(?![A-Za-z])", 1),
            (r"三重态|(?<![A-Za-z])triplet(?![A-Za-z])", 3),
        ]
    )
    for pattern, value in phrase_values:
        for match in re.finditer(pattern, message, re.IGNORECASE):
            mentioned = True
            phrase = match.group(0)
            if _is_negated_statement(message, match.start()):
                continue
            values.add(value)
            evidence = evidence or phrase

    all_labels = list(_ANY_STATE_LABEL_RE.finditer(message))
    for label_index, label in enumerate(all_labels):
        field = next(
            (
                candidate
                for candidate, pattern in _STATE_LABELS.items()
                if re.fullmatch(pattern, label.group(0), re.IGNORECASE)
            ),
            None,
        )
        if field != name:
            continue
        mentioned = True
        if _is_negated_statement(message, label.start()):
            continue
        start = label.end()
        end = len(message)
        if label_index + 1 < len(all_labels):
            end = min(end, all_labels[label_index + 1].start())
        sentence_break = re.search(r"[\n。！？?!;；,，]", message[start:end])
        if sentence_break is not None:
            end = start + sentence_break.start()
        segment = message[start:end]
        prior_breaks = [
            message.rfind(mark, 0, label.start()) + 1
            for mark in ("\n", "。", "！", "？", "!", "?", ";", "；", ",", "，")
        ]
        local_prefix = message[max(prior_breaks) : label.start()]
        unchanged_match = _UNCHANGED_STATE_RE.match(segment)
        unchanged_before_label = bool(
            re.search(
                r"(?:保持|维持|保留|不要(?:再)?(?:修改|改(?:变)?|变更)|"
                r"不(?:再)?(?:修改|改变|变更|改动)|无需(?:修改|改变|变更)|"
                r"(?:do\s+not|don't)\s+change)\s*$",
                local_prefix,
                re.IGNORECASE,
            )
        )
        if unchanged_match or unchanged_before_label:
            # An explicit request to retain this field is not an assignment.
            # In particular, numbers belonging to a later parameter must not
            # flow through the electronic-state label.
            if values:
                ambiguous = True
            unchanged_only = True
            continue

        token = _bound_electronic_state_token(segment)
        if token is None:
            segment_has_value_phrase = any(
                re.search(pattern, local_prefix + segment, re.IGNORECASE) is not None
                for pattern, _value in phrase_values
            )
            if _STATE_TOKEN_RE.search(segment) is not None or not segment_has_value_phrase:
                ambiguous = True
            continue
        token_end = segment.find(token) + len(token)
        if re.match(
            rf"\s*(?:或(?:者)?|还是|or)\s*{_STATE_TOKEN}(?![A-Za-z0-9_.])",
            segment[token_end:],
            re.IGNORECASE,
        ):
            ambiguous = True
            continue
        parsed = _parse_signed_integer(token)
        if parsed is None or (name == "multiplicity" and parsed <= 0):
            invalid.append(token)
            continue
        if _is_negated_statement(segment, segment.find(token)):
            continue
        values.add(parsed)
        evidence = evidence or segment.strip()

    if invalid:
        return ElectronicStateInput(status="invalid", evidence=evidence, detail=", ".join(invalid))
    if len(values) > 1 or ambiguous or (unchanged_only and values):
        return ElectronicStateInput(status="ambiguous", evidence=evidence)
    if len(values) == 1:
        return ElectronicStateInput(status="set", value=next(iter(values)), evidence=evidence)
    if mentioned:
        if unchanged_only:
            return ElectronicStateInput(status="absent")
        return ElectronicStateInput(status="ambiguous", evidence=evidence)
    return ElectronicStateInput(status="absent")


def _bound_electronic_state_token(segment: str) -> str | None:
    """Extract only the value directly bound to this q/M label."""

    correction = _FIELD_CORRECTION_RE.match(segment)
    if correction is not None:
        return correction.group("new")

    value_text = segment
    assignment_prefix = _FIELD_ASSIGNMENT_PREFIX_RE.match(value_text)
    if assignment_prefix is not None:
        value_text = value_text[assignment_prefix.end() :]
    value_text = value_text.lstrip()
    direct_value = _STATE_TOKEN_RE.match(value_text)
    return direct_value.group("token") if direct_value is not None else None


def _verified_candidate_evidence(
    message: str,
    name: str,
    state: ElectronicStateInput,
    candidates: list[ElectronicStateCandidate] | tuple[ElectronicStateCandidate, ...],
) -> str | None:
    if state.status != "set" or state.value is None:
        return None
    for candidate in candidates:
        if candidate.field != name or not candidate.evidence or candidate.evidence not in message:
            continue
        cited_state = _parse_electronic_state(candidate.evidence, name)
        candidate_value = _parse_signed_integer(candidate.raw_value.strip())
        if candidate_value is None:
            phrase_values = (
                [(r"中性|neutral", 0)]
                if name == "charge"
                else [
                    (r"单重态|singlet", 1),
                    (r"三重态|triplet", 3),
                ]
            )
            candidate_value = next(
                (
                    value
                    for pattern, value in phrase_values
                    if re.fullmatch(pattern, candidate.raw_value.strip(), re.IGNORECASE)
                ),
                None,
            )
        raw_matches_evidence = _candidate_raw_value_matches_evidence(
            candidate.raw_value.strip(), candidate.evidence
        )
        if (
            cited_state.status == "set"
            and cited_state.value == state.value == candidate_value
            and raw_matches_evidence
        ):
            return candidate.evidence
    return None


def _candidate_raw_value_matches_evidence(raw_value: str, evidence: str) -> bool:
    if re.fullmatch(_STATE_TOKEN, raw_value, re.IGNORECASE):
        return (
            re.search(
                rf"(?<![A-Za-z0-9_.]){re.escape(raw_value)}(?![A-Za-z0-9_.])",
                evidence,
                re.IGNORECASE,
            )
            is not None
        )
    return (
        re.search(
            r"单重态|三重态|singlet|triplet|neutral|中性",
            raw_value,
            re.IGNORECASE,
        )
        is not None
        and raw_value.casefold() in evidence.casefold()
    )


def _is_negated_statement(message: str, start: int) -> bool:
    prefix = message[max(0, start - 16) : start]
    return (
        re.search(
            r"(?:不要(?:用|使用|采用|设为|设置为|看|给|把|将|提供|展示|返回|输出)?|"
            r"不用|不采用|不使用|不需要|不想要|不含|并非|不是|非|不|"
            r"do\s+not(?:\s+(?:use|set|choose|include|request|give|provide|show|return|want|need|me|the|a|an))*|"
            r"don't(?:\s+(?:use|set|choose|include|want|need|give|provide|show|return|me|the|a|an))*|"
            r"not(?:\s+(?:a|the|use|an|want|need|request|include))?|"
            r"without|excluding|exclude|non[-\s]?)\s*$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _parse_signed_integer(value: str) -> int | None:
    if value.startswith("负"):
        magnitude = _chinese_integer(value[1:])
        return -magnitude if magnitude is not None else None
    if value.startswith("正"):
        return _chinese_integer(value[1:])
    if value.isdecimal() or (value[:1] in {"+", "-"} and value[1:].isdecimal()):
        return int(value)
    return _chinese_integer(value)


def _chinese_integer(value: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if value.startswith("十") and len(value) == 2 and value[1] in digits:
        return 10 + digits[value[1]]
    if value.endswith("十") and len(value) == 2 and value[0] in digits:
        return digits[value[0]] * 10
    return digits.get(value)


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
        # The durable session keeps IDs as a locator for the Agent, but the
        # intake model receives only a bounded status hint.  The actual
        # result_catalog is the sole model-visible source for fact selection.
        results = [
            {"status": item.get("status")} for item in results[-3:] if isinstance(item, Mapping)
        ]
    geometry_catalog = context.get("geometry_catalog")
    if isinstance(geometry_catalog, list):
        geometry_catalog = [
            dict(item) for item in geometry_catalog[:8] if isinstance(item, Mapping)
        ]
    else:
        geometry_catalog = []
    return {
        "recent_messages": recent or [],
        "recent_results": results or [],
        "geometry_catalog": geometry_catalog,
    }


def _intake_schema(candidate_refs: tuple[str, ...]) -> type[BaseModel]:
    """Build the local subject-reference-constrained schema for one intake round."""

    if not candidate_refs:
        # Keep the base type for the empty-catalog case so existing lightweight
        # clients can still classify ordinary compute requests.  The
        # cross-field and empty-candidate checks below remain authoritative.
        return IntakeOutput

    allowed = frozenset(candidate_refs)

    def _targets_are_candidates(value: list[QueryTarget]) -> list[QueryTarget]:
        unknown = sorted({target.subject_ref for target in value} - allowed)
        if unknown:
            raise ValueError(f"query subject references are outside this catalog: {unknown}")
        return value

    targets_validator = field_validator("targets")(_targets_are_candidates)
    selection_model = create_model(
        "QuerySelectionForCatalog",
        __base__=QuerySelection,
        __validators__={"_targets_are_candidates": targets_validator},
    )
    return create_model(
        "IntakeOutputForCatalog",
        __base__=IntakeOutput,
        query_selection=(selection_model | None, None),
    )


def _coerce_intake_output(
    value: Any,
    schema: type[BaseModel],
    candidate_refs: tuple[str, ...],
    *,
    message: str,
) -> IntakeOutput:
    payload = value.model_dump(mode="python") if isinstance(value, BaseModel) else value
    try:
        output = schema.model_validate(payload, strict=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"intake output failed local validation: {error}") from error
    if not isinstance(output, IntakeOutput):
        raise ValueError("intake output has an unexpected model type")
    return _validate_query_selection(output, candidate_refs, message=message)


def _validate_query_selection(
    output: IntakeOutput, candidate_refs: tuple[str, ...], *, message: str
) -> IntakeOutput:
    selection = output.query_selection
    if output.intent != "context_query":
        if selection is not None:
            raise ValueError("non-query intake output cannot select saved results")
        return output
    if selection is None:
        raise ValueError("context_query must contain query_selection")
    candidate_set = set(candidate_refs)
    unknown = sorted({target.subject_ref for target in selection.targets} - candidate_set)
    if unknown:
        raise ValueError(f"query subject references are outside this catalog: {unknown}")
    if not candidate_set and selection.status == "selected":
        raise ValueError("cannot select a result from an empty catalog")
    if selection.status == "selected" and any(
        not _query_property_evidence_matches(target.property, target.evidence, message)
        for target in selection.targets
    ):
        replacement = QuerySelection(
            status="clarify",
            clarification="请明确说明要查询的科学量；我不会用其他性质的结果代替。",
        )
        return output.model_copy(update={"query_selection": replacement})
    return output


def _query_property_evidence_matches(property_name: str, evidence: str, message: str) -> bool:
    if not evidence or evidence not in message:
        return False
    text = evidence.casefold().replace("电子结构", "")
    if property_name == "electronic_energy":
        specific = (
            "零点",
            "zero-point",
            "zero point",
            "zpe",
            "自由能",
            "free energy",
            "free_energy",
        )
        if any(token in text for token in specific):
            return False
        return _electronic_energy_evidence_matches(evidence, message)
    if property_name == "molecular_geometry":
        return _has_affirmed_property_phrase(
            ("结构", "几何", "geometry", "structure"), evidence, message
        ) and _geometry_output_requested(evidence)
    if property_name == "zero_point_energy":
        return _has_affirmed_property_phrase(
            ("零点", "zero-point", "zero point", "zpe"), evidence, message
        )
    if property_name == "free_energy":
        return _has_affirmed_property_phrase(
            ("自由能", "free energy", "free_energy"), evidence, message
        )
    if property_name == "frequency":
        return _has_affirmed_property_phrase(
            ("频率", "frequency", "frequencies"), evidence, message
        )
    if property_name == "atom_count":
        return _has_affirmed_property_phrase(
            ("原子数", "atom count", "number of atoms"), evidence, message
        )
    return False


def _has_affirmed_property_phrase(phrases: tuple[str, ...], evidence: str, message: str) -> bool:
    for phrase in phrases:
        for phrase_match in re.finditer(re.escape(phrase), evidence, re.IGNORECASE):
            for evidence_match in re.finditer(re.escape(evidence), message, re.IGNORECASE):
                position = evidence_match.start() + phrase_match.start()
                if not _is_negated_statement(message, position):
                    return True
    return False


def _electronic_energy_evidence_matches(evidence: str, message: str) -> bool:
    for phrase in ("能量", "电子能", "electronic energy", "energy"):
        for phrase_match in re.finditer(re.escape(phrase), evidence, re.IGNORECASE):
            for evidence_match in re.finditer(re.escape(evidence), message, re.IGNORECASE):
                position = evidence_match.start() + phrase_match.start()
                if _is_negated_statement(message, position):
                    continue
                prefix = message[max(0, position - 24) : position]
                if phrase.casefold() == "energy" and re.search(
                    r"(?:zero[-\s]?point|free)\s*$", prefix, re.IGNORECASE
                ):
                    continue
                return True
    return False


def _geometry_output_requested(evidence: str) -> bool:
    """Separate a structure mentioned as the subject from a requested output."""

    other_property = (
        r"电子能|能量|zero[-\s]?point|zpe|自由能|free\s+energy|频率|frequency|原子数|atom\s+count"
    )
    action_before = (
        r"给我|给出|输出|提供|返回|展示|显示|列出|(?<!不)需要|想要|希望|要求|"
        r"(?<!不)(?:同时|也|还)?要|查看|看看|show|return|output|provide|give|"
        r"include|need|want|display|list|fetch"
    )
    action_after = (
        r"给我|给出|输出|提供|返回|展示|显示|列出|也要|还要|同时要|要看|看看|"
        r"show|return|output|provide|give|include|need|want|display|list|fetch"
    )
    geometry = r"结构|几何|geometry|structure"
    joint = re.search(
        rf"(?:{other_property}).{{0,8}}(?:和|及|以及|、|and|&)\s*(?:{geometry})|"
        rf"(?:{geometry}).{{0,8}}(?:和|及|以及|、|and|&)\s*(?:{other_property})",
        evidence,
        re.IGNORECASE,
    )
    if joint is not None and re.search(action_before, evidence, re.IGNORECASE):
        return True
    for match in re.finditer(geometry, evidence, re.IGNORECASE):
        prefix = evidence[: match.start()]
        suffix = evidence[match.end() :]
        followed_by_property = re.match(
            rf"(?:的|['’]s\s+).{{0,8}}(?:{other_property})", suffix, re.IGNORECASE
        )
        has_action_before = re.search(action_before, prefix[-32:], re.IGNORECASE) is not None
        has_other_property_before = re.search(other_property, prefix, re.IGNORECASE) is not None
        has_action_after = re.search(action_after, suffix[:24], re.IGNORECASE) is not None
        if (has_action_before and not has_other_property_before and not followed_by_property) or (
            has_action_after and not followed_by_property
        ):
            return True
    return False


def _json_context(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "InputBindingProposal",
    "IntakeOutput",
    "QuerySelection",
    "QueryTarget",
    "PlanProposal",
    "PlanStepProposal",
    "PlanTargetProposal",
    "ElectronicStateCandidate",
    "ElectronicStateInput",
    "ParameterNormalization",
    "filter_user_explicit_parameters",
    "normalize_user_explicit_parameters",
    "electronic_state_clarification",
    "intake_message",
    "load_prompt",
    "plan_message",
    "proposal_to_plan",
    "request_from_intake",
    "validate_request_plan",
]
