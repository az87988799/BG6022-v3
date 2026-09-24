"""Minimal model contract for translating a user turn into scientific intent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    create_model,
    field_validator,
    model_validator,
)

from bg6022.intake_utils import extract_single_inline_xyz, mentions_computation
from bg6022.llm import LlmClient
from bg6022.planner import QuerySelection, QueryTarget, _bounded_context, load_prompt
from bg6022.tools.registry import ToolRegistry

SemanticMode = Literal[
    "compute",
    "modify",
    "context_query",
    "qa",
    "clarify",
    "unsupported",
]
SemanticProperty = Literal["energy", "geometry", "frequencies", "distance", "angle"]

_FORBIDDEN_PARAMETER_KEYS = {
    "subject_id",
    "requirement_id",
    "step_id",
    "artifact_id",
    "method_profile",
}
_PROGRAM_DERIVED_CAPABILITIES = frozenset({"same_geometry_method_energy_difference"})
_PROPERTY_NAMES = {
    "energy": "electronic_energy",
    "geometry": "molecular_geometry",
    "frequencies": "frequency",
    "distance": "distance",
    "angle": "angle",
}


class SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SemanticSubject(SemanticModel):
    key: StrictStr
    query: StrictStr
    input_kind: Literal["name", "cid", "smiles", "formula"] = "name"
    evidence: StrictStr

    @field_validator("key", "query", "evidence")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("semantic subject fields must not be blank")
        return value

    @field_validator("key")
    @classmethod
    def _safe_local_key(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is None:
            raise ValueError("semantic subject keys must be simple local identifiers")
        return value


class SemanticTask(SemanticModel):
    key: StrictStr
    subject_key: StrictStr
    capability: StrictStr
    method_request: StrictStr | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    requested_properties: list[SemanticProperty] = Field(default_factory=list)

    @field_validator("key", "subject_key", "capability")
    @classmethod
    def _nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("semantic task identity fields must not be blank")
        return value

    @field_validator("key", "subject_key")
    @classmethod
    def _safe_local_key(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is None:
            raise ValueError("semantic task and subject keys must be simple local identifiers")
        return value

    @field_validator("parameters")
    @classmethod
    def _forbid_internal_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        invalid = sorted(set(value) & _FORBIDDEN_PARAMETER_KEYS)
        if invalid:
            raise ValueError(f"semantic task contains internal fields: {invalid}")
        return value

    @field_validator("requested_properties")
    @classmethod
    def _unique_properties(cls, value: list[SemanticProperty]) -> list[SemanticProperty]:
        if len(value) != len(set(value)):
            raise ValueError("requested_properties must not contain duplicates")
        return value


class UseOutputRelation(SemanticModel):
    type: Literal["use_output"]
    source_task: StrictStr
    target_task: StrictStr
    property: Literal["geometry", "energy"]


class CompareRelation(SemanticModel):
    type: Literal["compare"]
    tasks: list[StrictStr]
    property: Literal["energy", "geometry"] = "energy"

    @field_validator("tasks")
    @classmethod
    def _validate_tasks(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("compare requires at least two tasks")
        if len(value) != len(set(value)):
            raise ValueError("compare task refs must be unique")
        return value


class DifferenceRelation(SemanticModel):
    type: Literal["difference"]
    tasks: list[StrictStr]
    property: Literal["energy"] = "energy"
    direction: Literal["second_minus_first"] = "second_minus_first"

    @field_validator("tasks")
    @classmethod
    def _two_tasks(cls, value: list[str]) -> list[str]:
        if len(value) != 2:
            raise ValueError("energy difference requires exactly two tasks")
        if len(value) != len(set(value)):
            raise ValueError("energy difference task refs must be unique")
        return value


SemanticRelation = Annotated[
    UseOutputRelation | CompareRelation | DifferenceRelation,
    Field(discriminator="type"),
]


class TaskModification(SemanticModel):
    target_task_ref: StrictStr
    method_request: StrictStr | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_task_ref")
    @classmethod
    def _nonblank_ref(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target_task_ref must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def _forbid_internal_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        invalid = sorted(set(value) & _FORBIDDEN_PARAMETER_KEYS)
        if invalid:
            raise ValueError(f"semantic modification contains internal fields: {invalid}")
        return value


class SemanticProposal(SemanticModel):
    mode: SemanticMode
    subjects: list[SemanticSubject] = Field(default_factory=list)
    tasks: list[SemanticTask] = Field(default_factory=list)
    relations: list[SemanticRelation] = Field(default_factory=list)
    modification: TaskModification | None = None
    query_selection: QuerySelection | None = None
    clarification: StrictStr | None = None
    unsupported_requirements: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mode_contract(self) -> SemanticProposal:
        if self.mode == "compute" and not self.tasks:
            raise ValueError("compute requires at least one task")
        if self.mode == "modify" and self.modification is None:
            raise ValueError("modify requires modification")
        if self.mode == "context_query" and self.query_selection is None:
            raise ValueError("context_query requires query_selection")
        if self.mode == "clarify" and not (self.clarification or "").strip():
            raise ValueError("clarify requires clarification")
        if self.mode == "unsupported" and not self.unsupported_requirements:
            raise ValueError("unsupported requires unresolved requirements")
        if self.mode != "compute" and (self.subjects or self.tasks or self.relations):
            raise ValueError(f"{self.mode} cannot contain compute tasks or subjects")
        if self.mode != "modify" and self.modification is not None:
            raise ValueError("modification is only valid for modify mode")
        if self.mode != "context_query" and self.query_selection is not None:
            raise ValueError("query_selection is only valid for context_query mode")
        if self.mode != "clarify" and self.clarification is not None:
            raise ValueError("clarification is only valid for clarify mode")
        if self.mode != "unsupported" and self.unsupported_requirements:
            raise ValueError("unsupported_requirements is only valid for unsupported mode")
        task_keys = [item.key for item in self.tasks]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("semantic task keys must be unique")
        subject_keys = [item.key for item in self.subjects]
        if len(subject_keys) != len(set(subject_keys)):
            raise ValueError("semantic subject keys must be unique")
        if set(item.subject_key for item in self.tasks) - set(subject_keys):
            raise ValueError("semantic tasks must refer to declared subjects")
        known = set(task_keys)
        for relation in self.relations:
            refs = (
                (relation.source_task, relation.target_task)
                if isinstance(relation, UseOutputRelation)
                else tuple(relation.tasks)
            )
            unknown = sorted(set(refs) - known)
            if unknown:
                raise ValueError(f"semantic relation refers to unknown tasks: {unknown}")
            if (
                isinstance(relation, UseOutputRelation)
                and relation.source_task == relation.target_task
            ):
                raise ValueError("use_output cannot reference the same task as source and target")
        return self


def compact_tool_catalog(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Expose only task intent and public typed inputs/defaults to Semantic LLM."""

    result = []
    for name in registry.names():
        tool = registry.get(name)
        if (
            not tool.available
            or tool.planning_role != "task"
            or tool.name in _PROGRAM_DERIVED_CAPABILITIES
        ):
            continue
        result.append(
            {
                "name": tool.name,
                "description": tool.description,
                "accepts_method": "method_profile" in tool.request_parameters,
                "parameters": [
                    item for item in tool.request_parameters if item != "method_profile"
                ],
                "input_types": sorted(set(tool.input_ports.values())),
                "default_outputs": list(tool.default_outputs),
            }
        )
    return result


def compact_method_catalog(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [
        {
            "name": item["display_name"],
            "family": (
                "composite"
                if item["family"].casefold() == item["name"].casefold()
                else item["family"]
            ),
            "display_name": item["display_name"],
            "aliases": [alias for alias in item["aliases"] if alias != item["name"]],
            "operations": item["operations"],
        }
        for item in registry.method_capability_catalog()
    ]


def compact_result_catalog(catalog: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Reduce stored-result context to ephemeral refs and semantic properties."""

    compact: list[dict[str, str]] = []
    for item in catalog:
        subject_ref = item.get("subject_ref")
        raw_result = item.get("result")
        result = raw_result if isinstance(raw_result, Mapping) else item
        property_name = result.get("property")
        if not isinstance(subject_ref, str) or not isinstance(property_name, str):
            continue
        label = result.get("label")
        compact.append(
            {
                "subject_ref": subject_ref,
                "property": property_name,
                "label": label if isinstance(label, str) else property_name,
            }
        )
    return compact


def semantic_schema(
    registry: ToolRegistry,
    *,
    result_catalog: list[Mapping[str, Any]] | None = None,
    pending_tasks: list[Mapping[str, Any]] | None = None,
) -> type[BaseModel]:
    """Create a strict response type whose task names match current Tools."""

    task_capabilities = tuple(
        name
        for name in registry.names()
        if registry.get(name).available
        and registry.get(name).planning_role == "task"
        and name not in _PROGRAM_DERIVED_CAPABILITIES
    )
    if task_capabilities:
        task_type: Any = create_model(
            "SemanticTaskForCatalog",
            __base__=SemanticTask,
            capability=(Literal.__getitem__(task_capabilities), ...),
        )
    else:
        task_type = SemanticTask

    query_items = compact_result_catalog(result_catalog or [])
    allowed_query_pairs = {
        (str(item["subject_ref"]), str(item["property"]))
        for item in query_items
        if isinstance(item.get("subject_ref"), str)
        and isinstance(item.get("property"), str)
    }
    candidate_refs = {subject_ref for subject_ref, _property in allowed_query_pairs}
    pending_by_ref = {
        str(item["task_ref"]): dict(item)
        for item in (pending_tasks or [])
        if isinstance(item, Mapping) and isinstance(item.get("task_ref"), str)
    }

    def _query_targets_are_catalogued(value: list[QueryTarget]) -> list[QueryTarget]:
        unknown = sorted({item.subject_ref for item in value} - candidate_refs)
        if unknown:
            raise ValueError(f"query subject references are outside this catalog: {unknown}")
        invalid = sorted(
            {
                (item.subject_ref, item.property)
                for item in value
                if (item.subject_ref, item.property) not in allowed_query_pairs
            }
        )
        if invalid:
            raise ValueError(f"query subject/property pairs are outside this catalog: {invalid}")
        return value

    selection_type = create_model(
        "SemanticQuerySelectionForCatalog",
        __base__=QuerySelection,
        __validators__={
            "_targets_are_catalogued": field_validator("targets")(
                _query_targets_are_catalogued
            )
        },
    )

    def _catalog_contract(value: SemanticProposal) -> SemanticProposal:
        requirements_by_key = {item.key: item for item in value.tasks}
        for task in value.tasks:
            tool = registry.get(task.capability)
            if not tool.available or tool.planning_role != "task":
                raise ValueError(f"task capability is not selectable: {task.capability!r}")
            unknown_parameters = sorted(set(task.parameters) - set(tool.request_parameters))
            if unknown_parameters:
                raise ValueError(
                    f"task parameters are outside the Tool contract: {unknown_parameters}"
                )
            if task.parameters:
                tool.validate_parameter_patch(dict(task.parameters))
            if task.method_request is not None:
                if not task.method_request.strip():
                    raise ValueError("method_request must be nonempty when supplied")
                if "method_profile" not in tool.request_parameters:
                    raise ValueError(f"{tool.name} does not accept methods")
            for requested_property in task.requested_properties:
                wanted = _PROPERTY_NAMES[requested_property]
                if not any(item["property"] == wanted for item in tool.public_outputs()):
                    raise ValueError(
                        f"{tool.name} does not provide requested property "
                        f"{requested_property!r}"
                    )
        if value.modification is not None:
            pending = pending_by_ref.get(value.modification.target_task_ref)
            if pending is None:
                raise ValueError("modify target_task_ref is not present in the waiting task")
            try:
                target_tool = registry.get(str(pending.get("capability", "")))
            except ValueError as error:
                raise ValueError("modify target capability is not registered") from error
            unknown_modifications = sorted(
                set(value.modification.parameters) - set(target_tool.request_parameters)
            )
            if unknown_modifications:
                raise ValueError(
                    "modification fields are outside the target Tool contract: "
                    f"{unknown_modifications}"
                )
            if value.modification.parameters:
                target_tool.validate_parameter_patch(value.modification.parameters)
            if value.modification.method_request is not None:
                if not value.modification.method_request.strip():
                    raise ValueError("method_request must be nonempty when supplied")
                if "method_profile" not in target_tool.request_parameters:
                    raise ValueError("selected Tool does not accept methods")
        if value.query_selection is not None:
            if value.query_selection.status == "selected" and not candidate_refs:
                raise ValueError("cannot select a result from an empty result catalog")
        # Relation-specific type derivation is completed by the program Canonicalizer.
        if set(requirements_by_key) != {item.key for item in value.tasks}:
            raise ValueError("semantic task refs are not unique")
        return value

    return create_model(
        "SemanticProposalForCatalog",
        __base__=SemanticProposal,
        __validators__={"_catalog_contract": model_validator(mode="after")(_catalog_contract)},
        tasks=(list[task_type], Field(default_factory=list)),
        query_selection=(selection_type | None, None),
    )


def semantic_message(
    client: LlmClient,
    message: str,
    *,
    registry: ToolRegistry,
    recent_context: Mapping[str, Any] | None = None,
    pending_tasks: list[Mapping[str, Any]] | None = None,
    result_catalog: list[Mapping[str, Any]] | None = None,
    validation_feedback: str | None = None,
    cancel: Any = None,
) -> SemanticProposal:
    if not message.strip():
        raise ValueError("message must not be empty")
    unsupported_requirement = _known_unsupported_requirement(message)
    if unsupported_requirement is not None:
        return SemanticProposal(
            mode="unsupported",
            unsupported_requirements=[unsupported_requirement],
        )
    inline_xyz = extract_single_inline_xyz(message)
    model_message = message
    if inline_xyz is not None and mentions_computation(message):
        model_message = message.replace(
            inline_xyz[0],
            "[Valid inline XYZ is present. The application preserves its exact coordinate text. "
            "Do not resolve or regenerate a geometry.]",
            1,
        )
    results = compact_result_catalog(result_catalog or [])
    tasks = [
        {
            "task_ref": str(item.get("task_ref", "")),
            "capability": str(item.get("capability", "")),
            "parameters": dict(item.get("parameters", {})),
            **(
                {"method_request": str(item["method_request"])}
                if isinstance(item.get("method_request"), str)
                else {}
            ),
        }
        for item in (pending_tasks or [])
        if isinstance(item, Mapping)
    ]
    context = _semantic_context(recent_context)
    schema = semantic_schema(
        registry,
        result_catalog=results,
        pending_tasks=tasks,
    )
    example = {
        "mode": "compute",
        "subjects": [
            {
                "key": "subject_1",
                "query": "water",
                "input_kind": "name",
                "evidence": "water",
            }
        ],
        "tasks": [
            {
                "key": "t1",
                "subject_key": "subject_1",
                "capability": "optimize_geometry",
                "method_request": None,
                "parameters": {},
                "requested_properties": ["energy"],
            }
        ],
        "relations": [],
        "modification": None,
        "query_selection": None,
        "clarification": None,
        "unsupported_requirements": [],
    }
    value = client.complete_json(
        [
            {"role": "system", "content": load_prompt("semantic")},
            {
                "role": "user",
                "content": _semantic_json(
                    {
                        "message": model_message,
                        "recent_context": context,
                        "pending_tasks": tasks,
                        "result_catalog": results,
                        "tool_catalog": compact_tool_catalog(registry),
                        "method_catalog": compact_method_catalog(registry),
                        "validation_feedback": validation_feedback,
                        "inline_xyz_present": inline_xyz is not None,
                    }
                ),
            },
        ],
        schema,
        purpose="semantic",
        example=example,
        cancel=cancel,
    )
    if not isinstance(value, SemanticProposal):
        value = schema.model_validate(value, strict=True)
    return value


def _known_unsupported_requirement(message: str) -> str | None:
    asks_for_gibbs = re.search(
        r"(?:gibbs(?:\s*[- ]?free\s*[- ]?energy|\s*自由能)?|自由能|Δ\s*g|delta\s*g)",
        message,
        flags=re.IGNORECASE,
    )
    asks_to_calculate = re.search(
        r"(?:计算|求取|算出|calculate|compute|determine|evaluate)",
        message,
        flags=re.IGNORECASE,
    )
    if asks_for_gibbs is not None and asks_to_calculate is not None:
        return "Gibbs free energy"
    asks_for_global_conformer_search = re.search(
        r"(?:全局|全球|global|systematic).{0,12}(?:构象|conformer)|"
        r"(?:构象|conformer).{0,12}(?:全局|全球|global|lowest[ -]?energy|最低能)",
        message,
        flags=re.IGNORECASE,
    )
    if asks_for_global_conformer_search is not None:
        return "global conformer search"
    return None


def _semantic_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    bounded = _bounded_context(context)
    recent_messages = bounded.get("recent_messages", [])
    recent_results = bounded.get("recent_results", [])
    delivery = bounded.get("last_delivery", [])
    return {
        "recent_messages": list(recent_messages)[-8:] if isinstance(recent_messages, list) else [],
        "recent_results": list(recent_results)[-3:] if isinstance(recent_results, list) else [],
        "last_delivery": list(delivery)[-8:] if isinstance(delivery, list) else [],
    }


def _semantic_json(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CompareRelation",
    "DifferenceRelation",
    "SemanticMode",
    "SemanticProposal",
    "SemanticSubject",
    "SemanticTask",
    "TaskModification",
    "UseOutputRelation",
    "compact_method_catalog",
    "compact_result_catalog",
    "compact_tool_catalog",
    "semantic_message",
    "semantic_schema",
]
