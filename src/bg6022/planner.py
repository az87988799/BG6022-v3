"""Natural-language intake and Tool-directory-driven plan construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationInfo,
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
    RequiredGeometryBinding,
    ResultProperty,
    ResultTarget,
    Step,
    validate_output_preferences,
)
from bg6022.molecule_identity import (
    MoleculeInputKind,
    build_identity_constraint,
    formula_token_from_text,
    normalize_identity_for_storage,
    validate_resolve_binding,
)
from bg6022.output_contracts import property_evidence_matches
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.registry import ToolRegistry, merge_explicit_step_parameters

Intent = Literal["chemistry_compute", "chemistry_qa", "daily_qa", "context_query"]
PendingAction = Literal["none", "supplement_identity", "replace_identity", "clarify"]
QuerySelectionStatus = Literal["selected", "clarify", "unavailable"]
QuerySelectionReason = Literal[
    "ambiguous_subject",
    "ambiguous_property",
    "unavailable_source",
    "invalid_binding",
    "missing_requirement",
]
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
    reference_mode: Literal["explicit", "followup"] = "explicit"


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
    reason: QuerySelectionReason | None = None

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
    parameter_issues: dict[str, str] = dataclass_field(default_factory=dict)

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
    molecule_input_kind: MoleculeInputKind | None = None
    molecule_name_evidence: StrictStr | None = None
    pending_action: PendingAction = "none"
    pending_action_evidence: StrictStr | None = Field(default=None, max_length=512)
    history_geometry_alias: StrictStr | None = None
    explicit_parameters: dict[str, Any] = Field(default_factory=dict)
    electronic_state_candidates: list[ElectronicStateCandidate] = Field(default_factory=list)
    structure_input: dict[str, Any] = Field(default_factory=dict)
    requested_results: list[str] = Field(default_factory=list)
    unresolved_results: list[StrictStr] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    query_selection: QuerySelection | None = None
    output_preferences: dict[str, str] = Field(default_factory=dict)

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

    @field_validator("output_preferences", mode="before")
    @classmethod
    def _validate_output_preferences(cls, value: Any) -> dict[str, str]:
        return validate_output_preferences(value)

    @property
    def operation(self) -> Operation | None:
        """Read-only compatibility with pre-M2 intake handlers."""

        return self.operations[0] if len(self.operations) == 1 else None


class IntakeStructureInput(BaseModel):
    """Intake-only geometry schema that preserves inline XYZ and other keys."""

    model_config = ConfigDict(extra="allow", strict=True)

    required_bindings: list[RequiredGeometryBinding] = Field(default_factory=list)


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
    capability_catalog: list[Mapping[str, Any]] | None = None,
    registry: ToolRegistry | None = None,
    validation_feedback: str | None = None,
    pending_context: Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> IntakeOutput:
    if not message.strip():
        raise ValueError("message must not be empty")
    inline_xyz = _extract_single_inline_xyz(message)
    model_message = message
    if inline_xyz is not None and _mentions_computation(message):
        xyz_text, atom_count = inline_xyz
        replacement = (
            f"[Valid inline XYZ for {atom_count} atoms is present; use it as the input "
            "geometry. The application preserves the exact coordinate text separately. "
            "Do not resolve or regenerate a geometry.]"
        )
        model_message = message.replace(xyz_text, replacement, 1)
    prompt = load_prompt("intake")
    catalog = [dict(item) for item in (result_catalog or [])]
    candidate_refs = tuple(
        str(item["subject_ref"])
        for item in catalog
        if isinstance(item, Mapping) and isinstance(item.get("subject_ref"), str)
    )
    capabilities = [dict(item) for item in (capability_catalog or [])]
    schema = _intake_schema(
        candidate_refs,
        capabilities,
        result_catalog=catalog,
        registry=registry,
        message=message,
        pending_context=pending_context,
    )
    value = client.complete_json(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_context(
                    {
                        "message": model_message,
                        "recent_context": _bounded_context(context),
                        "pending_context": dict(pending_context or {}),
                        "result_catalog": catalog,
                        "geometry_catalog": [dict(item) for item in (geometry_catalog or [])],
                        "capability_catalog": capabilities,
                        "parameter_capability_catalog": (
                            registry.request_parameter_capabilities()
                            if registry is not None
                            else []
                        ),
                        "method_capability_catalog": (
                            registry.method_capability_catalog() if registry is not None else []
                        ),
                        "validation_feedback": validation_feedback,
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
            "molecule_name_evidence": "water",
            "pending_action": "none",
            "pending_action_evidence": None,
            "history_geometry_alias": None,
            "structure_input": {},
            "explicit_parameters": {},
            "electronic_state_candidates": [],
            "requested_results": ["opt_final_electronic_energy"],
            "missing_fields": [],
            "unresolved_results": [],
        },
        cancel=cancel,
    )
    output = _coerce_intake_output(
        value,
        schema,
        candidate_refs,
        message=message,
        result_catalog=catalog,
        capability_catalog=capabilities,
    )
    if inline_xyz is not None and output.intent == "chemistry_compute":
        xyz_text, _atom_count = inline_xyz
        structure = dict(output.structure_input)
        if output.history_geometry_alias is not None or structure.get("history_geometry_alias"):
            raise ValueError("inline XYZ cannot be combined with a historical geometry selection")
        structure["xyz_text"] = xyz_text
        output = output.model_copy(
            update={
                "molecule_query": None,
                "molecule_input_kind": None,
                "molecule_name_evidence": None,
                "structure_input": structure,
            }
        )
    return output


def _extract_single_inline_xyz(message: str) -> tuple[str, int] | None:
    """Find one complete XYZ block and retain its exact original text."""

    lines = message.splitlines(keepends=True)
    matches: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        count_text = line.strip()
        if not count_text.isdecimal():
            continue
        count = int(count_text)
        if count <= 0 or count > 10000 or index + count + 2 > len(lines):
            continue
        block = "".join(lines[index : index + count + 2])
        try:
            parsed = parse_xyz_bytes(block.encode("utf-8"))
        except (TypeError, ValueError):
            continue
        matches.append((block, parsed.atom_count))
        if len(matches) > 1:
            return None
    return matches[0] if matches else None


def _mentions_computation(message: str) -> bool:
    return (
        re.search(
            r"(?i)(?:\b(?:opt|sp|freq)\b|geometry\s+optimization|optimiz|"
            r"single[ -]?point|frequency|frequencies|几何优化|优化|单点|频率|计算)",
            message,
        )
        is not None
    )


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
                        "capability_catalog": registry.result_capabilities(),
                        "method_capability_catalog": registry.method_capability_catalog(),
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

    plan = Plan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "steps": [
                merge_explicit_step_parameters(registry.get(step.tool), step, request)
                for step in plan.steps
            ],
        },
        strict=True,
    )
    plan = registry.validate_plan(plan)
    _validate_molecule_identity_contract(request, plan, registry)
    _validate_required_geometry_bindings(request, plan, registry)
    operation_free_distance_steps = [
        step
        for step in plan.steps
        if not registry.get(step.tool).operations
        and "distance" in set(registry.get(step.tool).result_properties.values())
    ]
    if len(operation_free_distance_steps) > 1:
        raise ValueError("one Request may contain only one distance-measurement Step")
    proposed_operations = [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ]
    # Chat requests always receive this full coverage check. An empty list is
    # meaningful for operation-free Tools such as geometry_distance: an ORCA
    # Tool cannot be smuggled into that request because it would add an item.
    if request.source == "chat" or request.operations:
        if proposed_operations != request.operations:
            missing = [item for item in request.operations if item not in proposed_operations]
            extra = [item for item in proposed_operations if item not in request.operations]
            if missing:
                if len(missing) == 1:
                    label = {"SP": "SP", "Opt": "Opt", "Freq": "frequency"}[missing[0]]
                    raise ValueError(f"Plan does not cover the requested {label} calculation")
                raise ValueError(
                    f"Plan does not cover requested operation(s): {', '.join(missing)}"
                )
            if extra:
                raise ValueError(f"Plan adds unrequested operation(s): {', '.join(extra)}")
            raise ValueError("Plan operation order does not match the user's requested order")

    plan_targets = plan.requested_results
    for request_target in request.requested_results:
        matches = [
            target
            for target in plan_targets
            if _request_target_matches(request_target, target, request, registry)
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


def _validate_molecule_identity_contract(
    request: Request, plan: Plan, registry: ToolRegistry
) -> None:
    """Prevent a Planner from changing or bypassing a user molecule constraint."""

    identity = request.structure_input.get("molecule_identity")
    if not isinstance(identity, Mapping):
        return
    expected_kind = identity.get("input_kind")
    selected_cid = identity.get("selected_cid")
    selected_smiles = identity.get("selected_smiles")
    resolve_steps = [step for step in plan.steps if step.tool == "resolve_molecule"]
    if expected_kind == "formula" and selected_cid is None and selected_smiles is None:
        has_inline_xyz = (
            request.structure_input.get("xyz_text") is not None
            or request.structure_input.get("xyz") is not None
        )
        if has_inline_xyz:
            if resolve_steps:
                raise ValueError(
                    "an inline XYZ formula request cannot also resolve another structure"
                )
            return
        if len(resolve_steps) != 1:
            raise ValueError("a formula request must have exactly one formula-resolve Step")
    if selected_cid is not None:
        if len(resolve_steps) != 1:
            raise ValueError("a selected molecule identity must have one resolve Step")
    if selected_smiles is not None:
        if len(resolve_steps) != 1:
            raise ValueError("a selected SMILES identity must have one resolve Step")
    if resolve_steps:
        if len(resolve_steps) != 1:
            raise ValueError("a molecule identity must have at most one resolve Step")
        try:
            validate_resolve_binding(identity, resolve_steps[0].parameters)
        except ValueError as error:
            raise ValueError(str(error)) from error


def request_from_intake(
    message: str,
    intake: IntakeOutput,
    *,
    request_id: str,
    registry: ToolRegistry,
    normalized_parameters: ParameterNormalization | None = None,
) -> Request:
    blocking = intake_blocking_requirements(intake, registry)
    if blocking:
        raise ValueError(
            "本次请求还有尚未支持或尚未明确的要求："
            + "；".join(blocking)
            + "。请明确这些要求，或重新指定只计算已支持的部分。"
        )
    if intake.intent == "chemistry_compute" and not intake.operations:
        if not intake.requested_results:
            raise ValueError(
                "an operation-free request must name a registered operation-free result"
            )
        operation_free_tools = registry.tools_for_request([], intake.requested_results)
        if not operation_free_tools or any(tool.operations for tool in operation_free_tools):
            raise ValueError(
                "an operation-free request must target a registered operation-free Tool"
            )
    if any(key in intake.structure_input for key in {"path", "local_path", "file"}):
        raise ValueError("model intake cannot authorize a local file path")
    if "molecule_identity" in intake.structure_input:
        raise ValueError("model intake cannot provide program-owned molecule identity facts")
    normalized = normalized_parameters or normalize_user_explicit_parameters(
        message,
        intake.explicit_parameters,
        intake.electronic_state_candidates,
        parameter_names=tuple(
            registry.request_index_parameter_fields(intake.operations, intake.requested_results)
        ),
    )
    if normalized.clarification_fields:
        raise ValueError(
            "electronic-state parameters need clarification: "
            + ", ".join(normalized.clarification_fields)
        )
    structure_input = dict(intake.structure_input)
    identity = build_identity_constraint(
        message=message,
        query=intake.molecule_query,
        input_kind=intake.molecule_input_kind,
        name_evidence=intake.molecule_name_evidence,
    )
    if identity is not None:
        structure_input["molecule_identity"] = normalize_identity_for_storage(identity)
    raw_bindings = structure_input.get("required_bindings")
    if raw_bindings is not None:
        normalized_bindings = []
        for item in raw_bindings:
            binding = RequiredGeometryBinding.model_validate(item, strict=True)
            normalized_binding = binding.model_dump(mode="json")
            if binding.consumer_operation is None:
                normalized_binding.pop("consumer_operation", None)
            else:
                normalized_binding.pop("consumer_tool", None)
            normalized_bindings.append(normalized_binding)
        structure_input["required_bindings"] = normalized_bindings
    request = Request(
        id=request_id,
        description=message,
        original_text=message,
        requested_results=[
            registry.resolve_result_target(value, intake.operations, canonical_only=True)
            for value in intake.requested_results
        ],
        explicit_parameters=dict(normalized.explicit_parameters),
        structure_input={
            **structure_input,
            **(
                {"history_geometry_alias": intake.history_geometry_alias}
                if intake.history_geometry_alias is not None
                else {}
            ),
        },
        source="chat",
        operations=list(intake.operations),
        missing_fields=list(intake.missing_fields),
        output_preferences=_request_output_preferences(message, intake.output_preferences),
    )
    _validate_required_geometry_contract(request, registry)
    _require_composite_geometry_sources(request, registry)
    return request


def _request_output_preferences(message: str, value: Mapping[str, Any] | None) -> dict[str, str]:
    """Keep presentation preferences user-authorized and non-scientific.

    ``link_only`` changes whether verified file content is shown.  It is
    therefore accepted only when the user's message explicitly asks for a
    link/path-only response; a model suggestion cannot silently hide content.
    """

    preferences = validate_output_preferences(value)
    # Keep negation and the requested view inside one punctuation-bounded
    # clause.  A broad ``.{0,n}`` expression can incorrectly let a negated
    # phrase in one clause suppress an explicit request in the next clause,
    # e.g. “不要链接，把正文给我”.
    clauses = [part for part in re.split(r"[,，。；;！？!?\n]+", message) if part.strip()]
    link_only_pattern = re.compile(
        r"(?:仅|只|只需|只要|只给|仅需|仅要)\s*(?:文件)?(?:链接|链结|路径|文件入口)|"
        r"(?:link\s*only|path\s*only|only\s+(?:the\s+)?(?:link|path))",
        re.IGNORECASE,
    )
    body_pattern = re.compile(
        r"正文|文件内容|内容|file\s+(?:content|body)|show\s+(?:the\s+)?content|"
        r"display\s+(?:the\s+)?content|include\s+(?:the\s+)?content",
        re.IGNORECASE,
    )
    body_negative_pattern = re.compile(
        r"(?:不要|不显示|不展示|不提供|do\s+not|don't|without)\s*"
        r"(?:文件的?\s*)?(?:正文|文件内容|内容|file\s+(?:content|body))",
        re.IGNORECASE,
    )
    explicit_link_only = any(
        link_only_pattern.search(clause) or body_negative_pattern.search(clause)
        for clause in clauses
    )
    explicit_body_request = any(
        body_pattern.search(clause) and not body_negative_pattern.search(clause)
        for clause in clauses
    )
    if explicit_body_request:
        # An explicit body request wins over a model-proposed link-only view,
        # including “正文和文件路径都给我”.
        preferences["file_content"] = "show"
    elif explicit_link_only:
        preferences["file_content"] = "link_only"
    elif preferences.get("file_content") == "link_only" and not explicit_link_only:
        preferences["file_content"] = "auto"
    return preferences


def intake_blocking_requirements(intake: IntakeOutput, registry: ToolRegistry) -> tuple[str, ...]:
    """Return unresolved compute requirements that are not deferred Tool inputs."""

    if intake.intent != "chemistry_compute":
        return ()

    requested_operations = set(intake.operations)
    deferred: set[str] = set()
    involved = registry.tools_for_request(requested_operations, intake.requested_results)
    required_request_parameters: set[str] = set()
    supplied = set(intake.explicit_parameters)
    for tool in involved:
        deferred.update(tool.deferred_parameters)
        if tool.parameter_type is not None:
            required_request_parameters.update(
                name
                for name in tool.request_parameters
                if tool.parameter_type.model_fields[name].is_required()
                and name not in tool.deferred_parameters
            )

    unclassified = [item for item in intake.missing_fields if item not in deferred]
    missing_declared = sorted(required_request_parameters - supplied)
    return tuple(dict.fromkeys([*intake.unresolved_results, *unclassified, *missing_declared]))


def filter_user_explicit_parameters(message: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility projection of the one authoritative parameter normalizer."""

    return normalize_user_explicit_parameters(message, parameters).explicit_parameters


def normalize_user_explicit_parameters(
    message: str,
    parameters: Mapping[str, Any],
    candidates: list[ElectronicStateCandidate] | tuple[ElectronicStateCandidate, ...] = (),
    parameter_names: tuple[str, ...] | list[str] | None = None,
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
    index_parameters = ("atom_i", "atom_j") if parameter_names is None else tuple(parameter_names)
    explicit_atoms, atom_issues = _parse_explicit_index_assignments(message, index_parameters)
    for name in atom_issues:
        patch.pop(name, None)
    patch.update(explicit_atoms)
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
    return ParameterNormalization(
        explicit_parameters=patch,
        states=states,
        parameter_issues=atom_issues,
    )


def _parse_explicit_atom_parameters(message: str) -> dict[str, int]:
    """Parse only unambiguous atom-index assignments made by the user."""

    parsed, _issues = _parse_explicit_index_assignments(message, ("atom_i", "atom_j"))
    return parsed


def strict_positive_index(raw: str) -> int:
    """Parse one complete 1-based integer token without prefix coercion."""

    token = raw.strip()
    if re.fullmatch(r"[+]?[0-9]+", token) is None:
        raise ValueError("atom index must be a complete integer token")
    value = int(token)
    if value < 1:
        raise ValueError("atom index must be 1-based")
    return value


def _parse_explicit_index_assignments(
    message: str, parameter_names: tuple[str, ...]
) -> tuple[dict[str, int], dict[str, str]]:
    """Parse complete assignment tokens for a configurable index set.

    The right-hand side is captured up to a natural separator before the
    integer parser is called.  Thus ``2.5``, ``2e0`` and ``2/3`` are rejected
    as whole tokens instead of being truncated to ``2``.
    """

    parsed: dict[str, int] = {}
    issues: dict[str, str] = {}
    assignment = "=|:|改成|改为|设为|设置为"
    for name in parameter_names:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_]).{{0,16}}?"
            rf"(?:{assignment})\s*(?P<raw>[^\s,，。；;:：]+)",
            re.IGNORECASE,
        )
        values: set[int] = set()
        saw_assignment = False
        for match in pattern.finditer(message):
            saw_assignment = True
            if _is_negated_statement(message, match.start()):
                issues[name] = "assignment is negated"
                continue
            raw = match.group("raw")
            try:
                values.add(strict_positive_index(raw))
            except ValueError as error:
                issues[name] = str(error)
        if len(values) > 1:
            issues[name] = "conflicting atom-index assignments"
        elif values and name not in issues:
            parsed[name] = values.pop()
        elif saw_assignment and name not in issues:
            issues[name] = "atom index assignment is not usable"
    return parsed, issues


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
    request_target: ResultTarget,
    plan_target: ResultTarget,
    request: Request,
    registry: ToolRegistry,
) -> bool:
    if request_target.step_id is not None and request_target.step_id != plan_target.step_id:
        return False

    if request_target.check is not None:
        request_kind, request_name = "check", request_target.check
    elif request_target.port is not None:
        request_kind, request_name = "port", request_target.port
    else:
        canonical = registry.resolve_result_target(request_target.field or "", request.operations)
        request_kind, request_name = _target_identity(canonical)
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


def _target_identity(target: ResultTarget) -> tuple[str, str]:
    if target.check is not None:
        return "check", target.check
    if target.port is not None:
        return "port", target.port
    assert target.field is not None
    return "field", target.field


def _required_geometry_bindings(request: Request) -> list[RequiredGeometryBinding]:
    raw = request.structure_input.get("required_bindings", [])
    if not isinstance(raw, list):
        raise ValueError("Request.structure_input.required_bindings must be a list")
    return [RequiredGeometryBinding.model_validate(item, strict=True) for item in raw]


def _tool_for_operation(registry: ToolRegistry, operation: str):
    matches = [
        registry.get(name)
        for name in registry.names()
        if operation in registry.get(name).operations
    ]
    if len(matches) != 1:
        raise ValueError(
            f"operation {operation!r} must have exactly one registered production Tool"
        )
    return matches[0]


def _validate_required_geometry_contract(request: Request, registry: ToolRegistry) -> None:
    for binding in _required_geometry_bindings(request):
        if binding.consumer_operation is not None:
            if binding.consumer_operation not in request.operations:
                raise ValueError(
                    f"geometry binding consumer {binding.consumer_operation!r} "
                    "is not a requested operation"
                )
            consumer = _tool_for_operation(registry, binding.consumer_operation)
        else:
            consumer_name = binding.consumer_tool
            if consumer_name is None:
                raise ValueError("geometry binding has no consumer selector")
            try:
                consumer = registry.get(consumer_name)
            except ValueError as error:
                raise ValueError(
                    f"geometry binding consumer Tool {consumer_name!r} is not registered"
                ) from error
            if not consumer.available:
                raise ValueError(f"geometry binding consumer Tool {consumer_name!r} is unavailable")
            if consumer not in registry.tools_for_request(
                request.operations, request.requested_results
            ):
                raise ValueError(
                    f"geometry binding consumer Tool {consumer_name!r} is not involved "
                    "in this Request"
                )
        input_type = consumer.input_ports.get(binding.input_port)
        if input_type is None:
            raise ValueError(f"Tool {consumer.name!r} has no input port {binding.input_port!r}")
        if input_type != "molecular_geometry":
            raise ValueError("required geometry bindings must target a molecular-geometry input")
        if binding.source_operation is None:
            if binding.source_port != "initial_geometry":
                raise ValueError(
                    "an initial-geometry binding must use source_port='initial_geometry'"
                )
            continue
        if binding.source_operation not in request.operations:
            raise ValueError(
                f"geometry binding source {binding.source_operation!r} is not a requested operation"
            )
        producer = _tool_for_operation(registry, binding.source_operation)
        output_type = producer.output_ports.get(binding.source_port)
        if output_type is None:
            raise ValueError(f"Tool {producer.name!r} has no output port {binding.source_port!r}")
        if output_type != input_type:
            raise ValueError(
                f"geometry binding type mismatch: {producer.name}.{binding.source_port} "
                f"cannot feed {consumer.name}.{binding.input_port}"
            )


def _require_composite_geometry_sources(request: Request, registry: ToolRegistry) -> None:
    operations = set(request.operations)
    if "Opt" not in operations:
        return
    bindings = _required_geometry_bindings(request)
    bound_operations = {
        item.consumer_operation for item in bindings if item.consumer_operation is not None
    }
    bound_tools = {item.consumer_tool for item in bindings if item.consumer_tool is not None}
    consumers: list[tuple[str, str]] = []
    for tool in registry.tools_for_request(request.operations, request.requested_results):
        if "geometry" not in tool.input_ports:
            continue
        if "Opt" in tool.operations:
            continue
        selector = tool.name
        if tool.operations:
            for operation in tool.operations:
                if operation in operations:
                    consumers.append(("operation", operation))
        else:
            consumers.append(("tool", selector))
    missing = [
        (kind, name)
        for kind, name in consumers
        if (name not in bound_operations if kind == "operation" else name not in bound_tools)
    ]
    if missing:
        names = "、".join(name for _kind, name in missing)
        raise ValueError(
            f"请明确 {names} 使用优化后的结构还是初始结构；我没有让 Planner 自行选择几何来源。"
        )


def _validate_required_geometry_bindings(
    request: Request, plan: Plan, registry: ToolRegistry
) -> None:
    _validate_required_geometry_contract(request, registry)
    bindings = _required_geometry_bindings(request)
    if not bindings:
        return
    steps_by_operation: dict[str, list[Step]] = {}
    steps_by_tool: dict[str, list[Step]] = {}
    for step in plan.steps:
        tool = registry.get(step.tool)
        steps_by_tool.setdefault(tool.name, []).append(step)
        for operation in tool.operations:
            steps_by_operation.setdefault(operation, []).append(step)
    for binding in bindings:
        consumers = (
            steps_by_operation.get(binding.consumer_operation, [])
            if binding.consumer_operation is not None
            else steps_by_tool.get(binding.consumer_tool or "", [])
        )
        if len(consumers) != 1:
            selector = binding.consumer_operation or binding.consumer_tool
            raise ValueError(f"geometry binding cannot identify one {selector} Step")
        consumer = consumers[0]
        actual = consumer.inputs.get(binding.input_port)
        if actual is None:
            raise ValueError(f"Step {consumer.id!r} omits required input {binding.input_port!r}")
        if binding.source_operation is not None:
            producers = steps_by_operation.get(binding.source_operation, [])
            if len(producers) != 1:
                raise ValueError(
                    f"geometry binding cannot identify one {binding.source_operation} Step"
                )
            expected = InputReference(
                step_id=producers[0].id,
                port=binding.source_port,
            )
            if actual != expected:
                raise ValueError(
                    f"Step {consumer.id!r}.{binding.input_port} must reference "
                    f"{producers[0].id}.{binding.source_port} as requested; "
                    f"received {actual.model_dump(exclude_none=True)}"
                )
            continue

        if binding.source_port != "initial_geometry":
            raise ValueError("unsupported initial geometry source contract")
        optimizers = steps_by_operation.get("Opt", [])
        if optimizers:
            if len(optimizers) != 1:
                raise ValueError("initial geometry binding cannot identify one Opt Step")
            initial_reference = optimizers[0].inputs.get(binding.input_port)
            if actual != initial_reference:
                raise ValueError(
                    f"Step {consumer.id!r}.{binding.input_port} must reuse the original "
                    f"geometry consumed by Opt {optimizers[0].id!r}; it must not use the "
                    "optimized output or another geometry"
                )


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
    delivery = context.get("last_delivery")
    if isinstance(delivery, list):
        # Private locators (run IDs, paths, hashes, and artifact IDs) never
        # enter the intake prompt. The public result catalog is regenerated
        # and revalidated for the current turn instead.
        delivery = [
            {key: item[key] for key in ("output_ref", "property", "kind") if key in item}
            for item in delivery[-8:]
            if isinstance(item, Mapping)
        ]
    else:
        delivery = []
    return {
        "recent_messages": recent or [],
        "recent_results": results or [],
        "geometry_catalog": geometry_catalog,
        "last_delivery": delivery,
    }


_HAN_NAME = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _structure_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="python", exclude_unset=True)
    else:
        payload = dict(value or {})
    if payload.get("required_bindings") == []:
        payload.pop("required_bindings")
    return payload


def _validate_lookup_name(value: Any, *, message: str) -> Any:
    """Keep a name's source evidence separate from its bounded lookup spelling."""

    if (
        value.intent != "chemistry_compute"
        or value.molecule_input_kind != "name"
        or value.molecule_query is None
    ):
        return value
    query = value.molecule_query
    if not query.strip() or len(query) > 128 or "\n" in query or "\r" in query:
        raise ValueError("name lookup must be a bounded single-line name")

    evidence = value.molecule_name_evidence
    if evidence is None and query in message:
        evidence = query
    if not evidence or not evidence.strip() or evidence not in message:
        raise ValueError(
            "molecule_name_evidence must quote the complete original name "
            "from this user message; keep lookup spelling separate"
        )

    formula = formula_token_from_text(evidence)
    if formula is not None and formula == evidence.strip():
        raise ValueError(
            "a formula is not name evidence; preserve the formula input "
            "instead of translating it into one selected molecule"
        )

    if _HAN_NAME.search(query):
        raise ValueError(
            "NAME_LOOKUP_NOT_NORMALIZED: return a reliable English lookup "
            "name and preserve the original Chinese name as exact evidence. "
            "Do not invent CID/SMILES or remove chemical qualifiers. "
            "If no reliable lookup name is available, leave molecule_query "
            "and molecule_input_kind null and explicitly report the missing "
            "identity in missing_fields; preserve all requested operations."
        )
    return value


def _validate_pending_action(
    value: IntakeOutput,
    *,
    message: str,
    pending_context: Mapping[str, Any] | None,
) -> IntakeOutput:
    """Validate whether this Intake round may mutate a waiting Run's identity."""

    action = value.pending_action
    if action == "none":
        if value.pending_action_evidence is not None:
            raise ValueError("pending_action=none cannot contain action evidence")
        return value

    pending = pending_context or {}
    if pending.get("waiting_for") not in {"clarification", "confirmation"}:
        raise ValueError("there is no waiting task for a pending action")
    if value.intent != "chemistry_compute":
        raise ValueError("question answering cannot mutate a waiting task")
    evidence = value.pending_action_evidence
    if not evidence or not evidence.strip() or evidence not in message:
        raise ValueError("pending action needs exact evidence from this message")

    if (
        value.operations
        or value.requested_results
        or value.unresolved_results
        or value.missing_fields
        or value.explicit_parameters
        or value.electronic_state_candidates
        or value.history_geometry_alias
        or _structure_payload(value.structure_input)
    ):
        raise ValueError(
            "identity-only actions cannot contain new calculations, outputs, "
            "parameters, or geometry bindings; use pending_action=none and "
            "preserve the complete request"
        )

    if action == "clarify":
        if value.molecule_query or value.molecule_input_kind:
            raise ValueError("clarify must not commit a molecule selection")
        return value

    if not value.molecule_query or not value.molecule_input_kind:
        raise ValueError("an identity action needs a query and input kind")
    if action == "supplement_identity" and not pending.get("identity_required"):
        raise ValueError("the current task is not waiting for molecule identity")
    if action == "replace_identity" and not pending.get("can_replace_identity"):
        raise ValueError("the molecule replacement target is not unique")
    return value


def _intake_schema(
    candidate_refs: tuple[str, ...],
    capability_catalog: list[Mapping[str, Any]],
    *,
    result_catalog: list[Mapping[str, Any]] | None = None,
    registry: ToolRegistry | None = None,
    message: str | None = None,
    pending_context: Mapping[str, Any] | None = None,
) -> type[BaseModel]:
    """Constrain query subjects and calculation targets for one intake round."""

    allowed_subjects = frozenset(candidate_refs)
    capabilities = [dict(item) for item in capability_catalog]
    allowed_targets = tuple(sorted({str(item["name"]) for item in capabilities}))
    allowed_query_pairs = _allowed_query_pairs(result_catalog)

    def _targets_are_candidates(value: list[QueryTarget]) -> list[QueryTarget]:
        unknown = sorted({target.subject_ref for target in value} - allowed_subjects)
        if unknown:
            raise ValueError(f"query subject references are outside this catalog: {unknown}")
        if allowed_query_pairs:
            invalid = sorted(
                {
                    (target.subject_ref, target.property)
                    for target in value
                    if (target.subject_ref, target.property) not in allowed_query_pairs
                }
            )
            if invalid:
                raise ValueError(
                    f"query subject/property pairs are outside this catalog: {invalid}"
                )
        return value

    def _requested_results_are_available(value: list[str], info: ValidationInfo) -> list[str]:
        unknown = sorted(set(value) - set(allowed_targets))
        if unknown:
            raise ValueError(
                f"requested results are outside the Tool capability catalog: {unknown}"
            )
        intent = info.data.get("intent")
        if value and intent != "chemistry_compute":
            raise ValueError("only chemistry_compute may request calculation results")
        operations = set(info.data.get("operations", []))
        for target in value:
            candidates = [item for item in capabilities if item["name"] == target]
            if not any(
                not item.get("operations") or bool(set(item["operations"]) & operations)
                for item in candidates
            ):
                raise ValueError(
                    f"requested result {target!r} is not produced by the requested operation(s)"
                )
        return value

    def _request_contract(value: IntakeOutput) -> IntakeOutput:
        if value.intent != "chemistry_compute":
            return value
        if registry is None:
            if value.explicit_parameters or not value.operations:
                raise ValueError("parameter capability catalog is required")
            return value
        involved = registry.tools_for_request(value.operations, value.requested_results)
        allowed = {name for tool in involved for name in tool.request_parameters}
        if not value.operations and not value.requested_results:
            # A parameter-only continuation is intentionally parsed before the
            # Agent decides whether a waiting Run can consume it.  It is not a
            # standalone operation-free Request; request_from_intake rejects
            # that case unless a real operation-free result is named.
            allowed = {
                name
                for item in registry.request_parameter_capabilities()
                for name in item["request_parameters"]
            }
        unknown = sorted(set(value.explicit_parameters) - allowed)
        if unknown:
            raise ValueError(
                "explicit parameter names are outside the Tool catalog: "
                f"{unknown}; allowed names: {sorted(allowed)}"
            )
        if not value.operations and not value.requested_results:
            return value
        if not value.operations:
            if not value.requested_results or not involved:
                raise ValueError(
                    "an operation-free chemistry request must target an available result Tool"
                )
            if any(tool.operations for tool in involved):
                raise ValueError(
                    "an operation-free chemistry request cannot target an ORCA operation"
                )
        return value

    def _required_geometry_bindings_are_valid(value: IntakeOutput) -> IntakeOutput:
        if value.intent != "chemistry_compute":
            return value
        structure_input = value.structure_input
        if isinstance(structure_input, IntakeStructureInput):
            structure_value = structure_input.model_dump(mode="python", exclude_unset=True)
        else:
            structure_value = structure_input
        request = Request(
            id="intake_geometry_validation",
            description="validate intake geometry bindings",
            operations=list(value.operations),
            requested_results=[
                registry.resolve_result_target(target, value.operations, canonical_only=True)
                if registry is not None
                else ResultTarget(field=target)
                for target in value.requested_results
            ],
            structure_input=structure_value,
        )
        bindings = _required_geometry_bindings(request)
        operations = set(request.operations)
        for binding in bindings:
            if (
                binding.consumer_operation is not None
                and binding.consumer_operation not in operations
            ):
                raise ValueError(
                    f"geometry binding consumer {binding.consumer_operation!r} "
                    "is not a requested operation"
                )
            if binding.source_operation is not None and binding.source_operation not in operations:
                raise ValueError(
                    f"geometry binding source {binding.source_operation!r} "
                    "is not a requested operation"
                )
        if registry is not None:
            _validate_required_geometry_contract(request, registry)
        return value

    query_selection_type: Any = QuerySelection | None
    validators: dict[str, Any] = {
        "_requested_results_are_available": field_validator("requested_results")(
            _requested_results_are_available
        ),
        "_request_contract": model_validator(mode="after")(_request_contract),
        "_required_geometry_bindings_are_valid": model_validator(mode="after")(
            _required_geometry_bindings_are_valid
        ),
    }

    def _dialogue_contract(value: IntakeOutput) -> IntakeOutput:
        if message is None:
            return value
        _validate_pending_action(
            value,
            message=message,
            pending_context=pending_context,
        )
        _validate_lookup_name(value, message=message)
        return value

    validators["_dialogue_contract"] = model_validator(mode="after")(_dialogue_contract)
    if allowed_targets:
        target_literal = Literal.__getitem__(allowed_targets)
        requested_results_type: Any = list[target_literal]
    else:
        # An absent registry cannot authorize arbitrary result strings.
        requested_results_type = list[StrictStr]

    if allowed_subjects:
        targets_validator = field_validator("targets")(_targets_are_candidates)
        selection_type = create_model(
            "QuerySelectionForCatalog",
            __base__=QuerySelection,
            __validators__={"_targets_are_candidates": targets_validator},
        )
        query_selection_type = selection_type | None

    structure_input_type = create_model(
        "IntakeStructureInputForCatalog",
        __base__=IntakeStructureInput,
    )

    return create_model(
        "IntakeOutputForCatalog",
        __base__=IntakeOutput,
        __validators__=validators,
        requested_results=(requested_results_type, Field(default_factory=list)),
        structure_input=(structure_input_type, Field(default_factory=structure_input_type)),
        query_selection=(query_selection_type, None),
    )


def _coerce_intake_output(
    value: Any,
    schema: type[BaseModel],
    candidate_refs: tuple[str, ...],
    *,
    message: str,
    result_catalog: list[Mapping[str, Any]] | None = None,
    capability_catalog: list[Mapping[str, Any]] | None = None,
) -> IntakeOutput:
    payload = (
        value.model_dump(mode="python", exclude_unset=True)
        if isinstance(value, BaseModel)
        else value
    )
    try:
        output = schema.model_validate(payload, strict=True)
    except ValueError as error:
        # Test doubles and older clients may return a raw dict without running
        # the dynamic schema first.  Keep the user-facing path safe and
        # actionable for an invalid subject/property binding while strict
        # clients still reject the payload at schema validation time.
        if "subject/property pairs are outside this catalog" in str(error):
            output = IntakeOutput(
                intent="context_query",
                query_selection=QuerySelection(
                    status="clarify",
                    reason="invalid_binding",
                    clarification="所选任务没有该性质；请从当前任务实际提供的结果中选择。",
                ),
            )
        else:
            raise ValueError(f"intake output failed local validation: {error}") from error
    except TypeError as error:
        raise ValueError(f"intake output failed local validation: {error}") from error
    if not isinstance(output, IntakeOutput):
        raise ValueError("intake output has an unexpected model type")
    if isinstance(output.structure_input, IntakeStructureInput):
        structure = output.structure_input.model_dump(mode="python", exclude_unset=True)
        if structure.get("required_bindings") == []:
            structure.pop("required_bindings")
        output = output.model_copy(update={"structure_input": structure})
    return _validate_query_selection(
        output,
        candidate_refs,
        message=message,
        result_catalog=result_catalog,
        capability_catalog=capability_catalog,
    )


def _validate_query_selection(
    output: IntakeOutput,
    candidate_refs: tuple[str, ...],
    *,
    message: str,
    result_catalog: list[Mapping[str, Any]] | None = None,
    capability_catalog: list[Mapping[str, Any]] | None = None,
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
    allowed_pairs = _allowed_query_pairs(result_catalog)
    invalid_pairs = sorted(
        {
            (target.subject_ref, target.property)
            for target in selection.targets
            if allowed_pairs and (target.subject_ref, target.property) not in allowed_pairs
        }
    )
    if invalid_pairs:
        replacement = QuerySelection(
            status="clarify",
            reason="invalid_binding",
            clarification="所选任务没有该性质；请从当前任务实际提供的结果中选择。",
        )
        return output.model_copy(update={"query_selection": replacement})
    recent_pairs: set[tuple[str, str]] = set()
    for item in result_catalog or []:
        if not isinstance(item, Mapping) or item.get("recently_delivered") is not True:
            continue
        result = item.get("result")
        if (
            isinstance(item.get("subject_ref"), str)
            and isinstance(result, Mapping)
            and isinstance(result.get("property"), str)
        ):
            recent_pairs.add((item["subject_ref"], result["property"]))
    invalid_evidence = False
    if selection.status == "selected":
        for target in selection.targets:
            if target.reference_mode == "followup":
                followup = (
                    target.subject_ref,
                    target.property,
                ) in recent_pairs and _followup_reference_is_safe(
                    target,
                    message,
                    result_catalog,
                    capability_catalog=capability_catalog,
                )
                if not followup:
                    invalid_evidence = True
                    break
                continue
            if not _query_property_evidence_matches(
                target.property,
                target.evidence,
                message,
                metadata=_query_property_metadata(target.property, result_catalog),
            ):
                invalid_evidence = True
                break
    if invalid_evidence:
        replacement = QuerySelection(
            status="clarify",
            clarification="请明确说明要查询的科学量；我不会用其他性质的结果代替。",
        )
        return output.model_copy(update={"query_selection": replacement})
    return output


def _followup_reference_is_safe(
    target: QueryTarget,
    message: str,
    result_catalog: list[Mapping[str, Any]] | None,
    *,
    capability_catalog: list[Mapping[str, Any]] | None = None,
) -> bool:
    """Allow a follow-up only for one recent result without a new request.

    A recent delivery is necessary but not sufficient: a new molecule, file,
    negation, or scientific property in the current message must be handled
    as an explicit query instead of inheriting the old result.
    """

    recent_pairs = {
        (str(item.get("subject_ref")), str(result.get("property")))
        for item in result_catalog or []
        if isinstance(item, Mapping)
        and item.get("recently_delivered") is True
        and isinstance(result := item.get("result"), Mapping)
        and isinstance(item.get("subject_ref"), str)
        and isinstance(result.get("property"), str)
    }
    if (target.subject_ref, target.property) not in recent_pairs:
        return False
    if not _followup_subject_is_safe(target, message, result_catalog):
        return False
    if _followup_has_new_request(
        target,
        message,
        result_catalog,
        capability_catalog=capability_catalog,
    ):
        return False

    target_metadata = _query_property_metadata(target.property, result_catalog)
    if _query_property_evidence_matches(
        target.property,
        target.evidence,
        message,
        metadata=target_metadata,
    ):
        return True

    # A unique recent delivery is a sufficient conversational focus for an
    # implicit question such as “是什么告诉我”.  If several recent
    # subject/property pairs exist, the model must provide an explicit
    # property reference instead of guessing which one “what” means.
    has_reference_marker = re.search(
        r"刚才|上次|上一条|前面|刚生成|刚输出|刚才的|同一|这个结果|该结果|"
        r"previous|earlier|above|that result|same result|again",
        message,
        re.IGNORECASE,
    )
    return has_reference_marker is not None or len(recent_pairs) == 1


def _followup_subject_is_safe(
    target: QueryTarget,
    message: str,
    result_catalog: list[Mapping[str, Any]] | None,
) -> bool:
    """Do not inherit a result when the current message names another subject."""

    recent_items = [
        item
        for item in result_catalog or []
        if isinstance(item, Mapping) and item.get("recently_delivered") is True
    ]
    recent_subjects = {
        str(item.get("subject_ref"))
        for item in recent_items
        if isinstance(item.get("subject_ref"), str)
    }
    subject = _explicit_followup_subject(message)
    target_items = [item for item in recent_items if item.get("subject_ref") == target.subject_ref]
    if subject is not None:
        return bool(target_items) and any(
            _catalog_subject_matches(subject, item) for item in target_items
        )
    # “把刚才那个给我” is not a safe selector once more than one subject was
    # delivered.  A property-only follow-up may remain implicit for one
    # subject, but it must not choose among molecule/task identities.
    return len(recent_subjects) <= 1


def _explicit_followup_subject(message: str) -> str | None:
    property_words = (
        r"xyz|结构|几何|坐标|文件|能量|自由能|频率|角度|距离|"
        r"geometry|structure|coordinates?|file|energy|frequency|angle|distance"
    )
    patterns = (
        rf"(?P<subject>[\u3400-\u9fffA-Za-z0-9]"
        rf"[\u3400-\u9fffA-Za-z0-9 .+()_\-]{{0,48}}?)\s*(?:的|之)\s*"
        rf"(?:{property_words})",
        rf"(?:{property_words})\s+(?:for|of)\s+(?P<subject>[A-Za-z0-9][A-Za-z0-9 .+()_\-]{{0,48}})",
    )
    generic = re.compile(
        r"^(?:刚才|上次|上一条|前面|刚生成|刚输出|这个|那个|该|同一|"
        r"previous|earlier|above|this|that|same)(?:的|那个|个|结果|内容|文件)?$",
        re.IGNORECASE,
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match is None:
            continue
        subject = re.sub(
            r"^(?:请|把|将|给我|帮我|查询|查一下|告诉我|显示|提供|返回|我要|我想要)\s*",
            "",
            match.group("subject").strip(),
            flags=re.IGNORECASE,
        ).strip(" \t，,。；;:：")
        if subject and not generic.fullmatch(subject):
            return subject
    return None


def _catalog_subject_matches(subject: str, item: Mapping[str, Any]) -> bool:
    normalized_subject = _normalize_subject_text(subject)
    if not normalized_subject:
        return False
    values: list[str] = []
    system = item.get("system")
    if isinstance(system, Mapping):
        values.extend(
            str(system[key]) for key in ("formula", "title", "query", "cid") if key in system
        )
    task = item.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("description"), str):
        values.append(task["description"])
    return any(
        normalized_subject == _normalize_subject_text(value)
        or normalized_subject in _normalize_subject_text(value).split()
        for value in values
        if value
    )


def _normalize_subject_text(value: str) -> str:
    translated = value.strip().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return re.sub(r"[\s_\-()（）]+", "", translated.casefold())


def _followup_has_new_request(
    target: QueryTarget,
    message: str,
    result_catalog: list[Mapping[str, Any]] | None,
    *,
    capability_catalog: list[Mapping[str, Any]] | None = None,
) -> bool:
    """Detect a new subject/property before a follow-up can inherit focus."""

    # These are request-level transitions, not molecule or Tool names.  The
    # bounded suffix check avoids treating a harmless view change such as
    # “改成表格” as a new scientific request.
    replacement = re.search(
        r"(?:改成|改为|换成|换为|变成|替换为|instead\s+of|rather\s+than|"
        r"switch\s+to)\s*(?P<suffix>[^，,。！？!?;；\n]{1,64})",
        message,
        re.IGNORECASE,
    )
    if replacement is not None:
        suffix = replacement.group("suffix")
        if not re.search(
            r"表格|链接|链结|路径|文件入口|正文|内容|link|path|table|content|body|"
            r"json|csv",
            suffix,
            re.IGNORECASE,
        ):
            return True

    if _message_mentions_other_property(
        target.property,
        message,
        result_catalog=result_catalog,
        capability_catalog=capability_catalog,
    ):
        return True

    # A negated old result followed by a new request must never inherit the
    # old binding, even when the new property is not in the saved catalog.
    if re.search(
        r"不要|不用|不需要|不想要|不是|并非|do\s+not|don't|without|instead",
        message,
        re.IGNORECASE,
    ) and re.search(
        r"我要|我需要|请给|需要|想要|want|need|give|show|provide|return",
        message,
        re.IGNORECASE,
    ):
        return True
    return False


def _message_mentions_other_property(
    target_property: str,
    message: str,
    *,
    result_catalog: list[Mapping[str, Any]] | None,
    capability_catalog: list[Mapping[str, Any]] | None,
) -> bool:
    descriptors: list[tuple[str, Mapping[str, Any]]] = []
    for item in [*(result_catalog or []), *(capability_catalog or [])]:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result") if isinstance(item.get("result"), Mapping) else item
        if not isinstance(result, Mapping):
            continue
        property_name = result.get("property")
        if not isinstance(property_name, str) or not property_name:
            continue
        metadata = {
            key: result[key] for key in ("label", "description") if isinstance(result.get(key), str)
        }
        descriptors.append((property_name, metadata))

    for property_name, metadata in descriptors:
        if property_name == target_property:
            continue
        for evidence in (
            property_name.replace("_", " "),
            str(metadata.get("label", "")),
            str(metadata.get("description", "")),
        ):
            if evidence and _query_property_evidence_matches(
                property_name, evidence, message, metadata=metadata
            ):
                return True

    # Keep the fallback vocabulary bounded to scientific property concepts
    # already handled by the public query contract.  Future Tool-specific
    # labels still flow through the metadata path above.
    generic_cues = (
        ("free_energy", r"自由能|free[\s_-]*energy"),
        ("zero_point_energy", r"零点(?:能)?|zero[\s_-]*point|zpe"),
        ("frequency", r"频率|frequenc(?:y|ies)"),
        ("electronic_energy", r"电子能|electronic[\s_-]*energy|(?<!自由)能量|(?<!free )energy"),
        ("distance", r"距离|间距|distance|bond[ -]?length"),
        ("molecular_geometry", r"结构|几何|xyz|坐标|geometry|structure"),
        ("angle", r"角度|夹角|张角|angle|degree"),
        ("atom_count", r"原子数|atom[ _-]?count|number of atoms"),
    )
    target_metadata = next(
        (metadata for property_name, metadata in descriptors if property_name == target_property),
        {},
    )
    for property_name, pattern in generic_cues:
        if property_name == target_property:
            continue
        for match in re.finditer(pattern, message, re.IGNORECASE):
            evidence = match.group(0)
            if not _query_property_evidence_matches(
                target_property,
                evidence,
                message,
                metadata=target_metadata,
            ):
                return True
    return False


def _query_property_metadata(
    property_name: str, result_catalog: list[Mapping[str, Any]] | None
) -> Mapping[str, Any] | None:
    for item in result_catalog or []:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result")
        if not isinstance(result, Mapping) or result.get("property") != property_name:
            continue
        return {
            key: result[key] for key in ("label", "description") if isinstance(result.get(key), str)
        }
    return None


def _allowed_query_pairs(
    result_catalog: list[Mapping[str, Any]] | None,
) -> frozenset[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in result_catalog or []:
        if not isinstance(item, Mapping):
            continue
        subject_ref = item.get("subject_ref")
        result = item.get("result")
        if not isinstance(subject_ref, str) or not isinstance(result, Mapping):
            continue
        property_name = result.get("property")
        if isinstance(property_name, str) and property_name:
            pairs.add((subject_ref, property_name))
    return frozenset(pairs)


def _query_property_evidence_matches(
    property_name: str,
    evidence: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
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
        affirmed = _has_affirmed_property_phrase(
            ("结构", "几何", "geometry", "structure", "xyz", "坐标"), evidence, message
        )
        requested = _geometry_output_requested(evidence) or bool(
            re.search(
                r"(?:xyz|坐标).{0,8}(?:文件|结构)|(?:文件|结构).{0,8}(?:xyz|坐标)",
                evidence,
                re.I,
            )
        )
        return affirmed and requested
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
    if property_name == "distance":
        return _has_affirmed_property_phrase(
            ("距离", "间距", "distance", "bond length", "bond-length"),
            evidence,
            message,
        )
    # New Tool properties are accepted only when their declared public label or
    # property identifier is present in the user's exact evidence phrase.  The
    # catalog still decides whether the property exists and belongs to the
    # selected subject.
    return property_evidence_matches(property_name, evidence, message, metadata=metadata)


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
        r"电子能|能量|zero[-\s]?point|zpe|自由能|free\s+energy|频率|frequency|"
        r"原子数|atom\s+count|距离|间距|distance|bond\s+length"
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
    geometry = r"结构|几何|geometry|structure|xyz|坐标"
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
    if re.search(r"(?i)xyz|坐标", evidence) and not re.search(other_property, evidence, re.I):
        # A short property-only request such as ``xyz`` or ``水分子xyz`` is
        # already an explicit request for the geometry output.  Do not apply
        # this shortcut when the geometry is only the subject of another
        # property, e.g. ``这个XYZ的能量``.
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
    "strict_positive_index",
    "electronic_state_clarification",
    "intake_message",
    "load_prompt",
    "plan_message",
    "proposal_to_plan",
    "request_from_intake",
    "validate_request_plan",
]
