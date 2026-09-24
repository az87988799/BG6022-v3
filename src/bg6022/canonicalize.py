"""Translate user-facing semantic proposals into the existing Request contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from bg6022.intake_utils import extract_single_inline_xyz, mentions_computation
from bg6022.models import Request
from bg6022.orca.profiles import resolve_method_request
from bg6022.planner import (
    AnswerGoalProposal,
    IntakeOutput,
    IntakeSubjectProposal,
    RequirementInputBindingProposal,
    RequirementProposal,
    request_from_intake,
)
from bg6022.semantic import (
    CompareRelation,
    DifferenceRelation,
    SemanticProposal,
    SemanticTask,
    UseOutputRelation,
)
from bg6022.tools.registry import ToolRegistry

_SEMANTIC_PROPERTY = {
    "energy": "electronic_energy",
    "geometry": "molecular_geometry",
    "frequencies": "frequency",
    "distance": "distance",
    "angle": "angle",
}
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def semantic_to_intake(
    message: str,
    semantic: SemanticProposal,
    *,
    registry: ToolRegistry,
    history_geometry_alias: str | None = None,
) -> IntakeOutput:
    """Canonicalize semantic compute intent while reusing current Request checks."""

    if semantic.mode == "qa":
        return IntakeOutput(intent="chemistry_qa")
    if semantic.mode == "context_query":
        return IntakeOutput(intent="context_query", query_selection=semantic.query_selection)
    if semantic.mode != "compute":
        raise ValueError(f"semantic mode {semantic.mode!r} is not a Request")

    task_by_key = {item.key: item for item in semantic.tasks}
    subject_proposals: dict[str, IntakeSubjectProposal] = {}
    inline = extract_single_inline_xyz(message) if mentions_computation(message) else None
    if history_geometry_alias is not None and inline is not None:
        raise ValueError("a request cannot combine historical geometry and inline XYZ")
    if inline is not None and len(semantic.subjects) > 1:
        raise ValueError("inline XYZ must identify exactly one calculation subject")
    if history_geometry_alias is not None and len(semantic.subjects) > 1:
        raise ValueError("historical geometry must identify exactly one calculation subject")
    for subject in semantic.subjects:
        if _ID_RE.fullmatch(subject.key) is None:
            raise ValueError("semantic subject keys must be simple local identifiers")
        if subject.evidence not in message:
            raise ValueError(
                f"subject {subject.key!r} evidence must quote the original user message"
            )
        subject_proposals[subject.key] = IntakeSubjectProposal(
            key=subject.key,
            molecule_query=None if inline is not None else subject.query,
            molecule_input_kind=None if inline is not None else subject.input_kind,
            molecule_name_evidence=None if inline is not None else subject.evidence,
            inline_xyz=inline[0] if inline is not None else None,
            history_geometry_alias=history_geometry_alias,
        )
    if inline is not None and not subject_proposals:
        subject_proposals["subject_1"] = IntakeSubjectProposal(
            key="subject_1", inline_xyz=inline[0]
        )
    if history_geometry_alias is not None and not subject_proposals:
        raise ValueError("historical geometry needs one explicitly identified calculation subject")

    proposals: dict[str, dict[str, Any]] = {}
    outputs_by_task: dict[str, list[str]] = {key: [] for key in task_by_key}
    for task in semantic.tasks:
        if _ID_RE.fullmatch(task.key) is None or _ID_RE.fullmatch(task.subject_key) is None:
            raise ValueError("semantic task and subject keys must be simple local identifiers")
        tool = registry.get(task.capability)
        if not tool.available or tool.planning_role != "task":
            raise ValueError(f"task capability is not selectable: {tool.name!r}")
        if tool.name == "same_geometry_method_energy_difference":
            raise ValueError(
                "energy-difference Tools are derived from the semantic difference relation"
            )
        if task.subject_key not in subject_proposals:
            raise ValueError(f"task {task.key!r} refers to an undeclared subject")
        parameters = dict(task.parameters)
        if task.method_request is not None:
            if "method_profile" not in tool.request_parameters:
                raise ValueError(f"{tool.name} does not accept methods")
            parameters["method_request"] = task.method_request
        unknown_parameters = sorted(
            set(parameters) - set(tool.request_parameters) - {"method_request"}
        )
        if unknown_parameters:
            raise ValueError(
                f"task {task.key!r} has parameters outside the Tool contract: {unknown_parameters}"
            )
        if parameters:
            validation_values = {
                name: value for name, value in parameters.items() if name != "method_request"
            }
            if validation_values:
                tool.validate_parameter_patch(validation_values)
        proposals[task.key] = {
            "key": task.key,
            "subject_key": task.subject_key,
            "capability": tool.name,
            "parameters": parameters,
            "outputs": outputs_by_task[task.key],
            "input_bindings": {},
            "parameter_evidence": [],
            "constraints": {},
        }
        for requested_property in task.requested_properties:
            outputs_by_task[task.key].append(
                _output_for_property(tool, _SEMANTIC_PROPERTY[requested_property])
            )

    consumed: set[str] = set()
    answer_goals: list[AnswerGoalProposal] = []
    for relation in semantic.relations:
        if isinstance(relation, UseOutputRelation):
            source = task_by_key[relation.source_task]
            target = task_by_key[relation.target_task]
            if source.subject_key != target.subject_key:
                raise ValueError("use_output requires tasks for the same subject")
            _apply_use_output(relation, proposals, task_by_key, registry)
            consumed.add(relation.source_task)
        elif isinstance(relation, CompareRelation):
            common_output = _common_output(
                relation.tasks,
                _SEMANTIC_PROPERTY[relation.property],
                task_by_key,
                registry,
            )
            for task_key in relation.tasks:
                _append_unique(outputs_by_task[task_key], common_output)
            answer_goals.append(
                AnswerGoalProposal(
                    kind="compare",
                    requirement_keys=list(relation.tasks),
                    output=common_output,
                    mode="side_by_side",
                )
            )
        elif isinstance(relation, DifferenceRelation):
            _apply_difference(relation, proposals, task_by_key, registry)
            consumed.update(relation.tasks)
            energy_output = _common_output(
                relation.tasks,
                _SEMANTIC_PROPERTY[relation.property],
                task_by_key,
                registry,
            )
            answer_goals.append(
                AnswerGoalProposal(
                    kind="compare",
                    requirement_keys=list(relation.tasks),
                    output=energy_output,
                    mode="numeric_difference",
                )
            )

    derived_keys = {key for key in proposals if key not in task_by_key}
    for task_key, proposal in proposals.items():
        if task_key in derived_keys:
            continue
        requested = outputs_by_task[task_key]
        if not requested and task_key not in consumed:
            tool = registry.get(str(proposal["capability"]))
            requested.extend(tool.default_outputs)
        proposal["outputs"] = list(dict.fromkeys(requested))

    requirements = [
        RequirementProposal.model_validate(item, strict=True) for item in proposals.values()
    ]
    return IntakeOutput(
        intent="chemistry_compute",
        requirements=requirements,
        subjects=subject_proposals,
        answer_goals=answer_goals,
    )


def canonicalize_semantic_request(
    message: str,
    semantic: SemanticProposal,
    *,
    request_id: str,
    registry: ToolRegistry,
    history_geometry_alias: str | None = None,
) -> Request:
    intake = semantic_to_intake(
        message,
        semantic,
        registry=registry,
        history_geometry_alias=history_geometry_alias,
    )
    return request_from_intake(
        message,
        intake,
        request_id=request_id,
        registry=registry,
    )


def canonicalize_modification(
    modification: Any,
    *,
    pending_ref_to_requirement: Mapping[str, str],
    run: Any,
    registry: ToolRegistry,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve a short task ref and method wording to a scoped typed patch."""

    requirement_id = pending_ref_to_requirement.get(str(modification.target_task_ref))
    if requirement_id is None:
        raise ValueError("modify target_task_ref is not present in the waiting task")
    requirement = next(
        (item for item in run.request.requirements if item.id == requirement_id), None
    )
    if requirement is None:
        raise ValueError("modify target does not belong to the current Request")
    tool = registry.get(requirement.capability)
    patch = dict(modification.parameters)
    constraint_patch: dict[str, Any] = {}
    if modification.method_request is not None:
        if "method_profile" not in tool.request_parameters:
            raise ValueError("selected Tool does not accept methods")
        resolution = resolve_method_request(modification.method_request)
        if resolution.status not in {"resolved", "proposed"} or resolution.profile is None:
            if resolution.status == "ambiguous":
                raise ValueError("方法存在多个已注册候选：" + ", ".join(resolution.candidates))
            raise ValueError("方法未注册：" + modification.method_request)
        patch["method_profile"] = resolution.profile
        constraint_patch["method_resolution"] = {
            "request": modification.method_request,
            "profile": resolution.profile,
            "status": resolution.status,
            "source": resolution.source,
        }
    unknown = sorted(set(patch) - set(tool.request_parameters))
    if unknown:
        raise ValueError(f"update fields outside Tool contract: {unknown}")
    if patch:
        tool.validate_parameter_patch(patch)
    return requirement_id, patch, constraint_patch


def _output_for_property(tool: Any, property_name: str) -> str:
    matches = [
        str(item["name"]) for item in tool.public_outputs() if item["property"] == property_name
    ]
    if len(matches) != 1:
        raise ValueError(
            "requested property is not uniquely derivable from the Tool contract: "
            f"{tool.name}.{property_name} -> {sorted(matches)}"
        )
    return matches[0]


def _common_output(
    task_keys: list[str],
    property_name: str,
    tasks: Mapping[str, SemanticTask],
    registry: ToolRegistry,
) -> str:
    candidate_sets = [
        {
            str(item["name"])
            for item in registry.get(tasks[key].capability).public_outputs()
            if item["property"] == property_name
        }
        for key in task_keys
    ]
    common = set.intersection(*candidate_sets) if candidate_sets else set()
    if len(common) != 1:
        raise ValueError(
            f"comparison output is not uniquely derivable from Tool contracts: {sorted(common)}"
        )
    return next(iter(common))


def _apply_use_output(
    relation: UseOutputRelation,
    proposals: dict[str, dict[str, Any]],
    tasks: Mapping[str, SemanticTask],
    registry: ToolRegistry,
) -> None:
    source_task = tasks[relation.source_task]
    target_task = tasks[relation.target_task]
    source_tool = registry.get(source_task.capability)
    target_tool = registry.get(target_task.capability)
    wanted_type = {"geometry": "molecular_geometry", "energy": "energy_data"}[relation.property]
    source_ports = [
        name for name, value in source_tool.output_ports.items() if value == wanted_type
    ]
    target_ports = [name for name, value in target_tool.input_ports.items() if value == wanted_type]
    if len(source_ports) != 1 or len(target_ports) != 1:
        raise ValueError("typed dependency is not uniquely derivable")
    target_bindings = proposals[relation.target_task]["input_bindings"]
    input_name = target_ports[0]
    if input_name in target_bindings:
        raise ValueError(f"target input {input_name!r} has conflicting semantic relations")
    target_bindings[input_name] = RequirementInputBindingProposal(
        requirement_key=relation.source_task,
        port=source_ports[0],
    )


def _apply_difference(
    relation: DifferenceRelation,
    proposals: dict[str, dict[str, Any]],
    tasks: Mapping[str, SemanticTask],
    registry: ToolRegistry,
) -> None:
    left_key, right_key = relation.tasks
    left = tasks[left_key]
    right = tasks[right_key]
    if left.capability != "single_point" or right.capability != "single_point":
        raise ValueError("electronic-energy difference requires two single-point tasks")
    if left.subject_key != right.subject_key:
        raise ValueError("energy difference requires the same subject")
    difference_tool = registry.get("same_geometry_method_energy_difference")
    if not difference_tool.available or difference_tool.planning_role != "task":
        raise ValueError("same-geometry energy difference Tool is unavailable")
    for key in (left_key, right_key):
        inputs = proposals[key]["input_bindings"]
        current = inputs.get("geometry")
        if current is not None and current.artifact_alias != "initial_geometry":
            raise ValueError("energy difference tasks must share the same initial geometry")
        inputs["geometry"] = RequirementInputBindingProposal(artifact_alias="initial_geometry")
    derived_key = f"difference_{left_key}_{right_key}"
    if _ID_RE.fullmatch(derived_key) is None:
        raise ValueError("derived energy-difference key is too long")
    if derived_key in proposals:
        raise ValueError("energy difference task key collides with a derived Requirement")
    proposals[derived_key] = {
        "key": derived_key,
        "subject_key": left.subject_key,
        "capability": difference_tool.name,
        "parameters": {},
        "outputs": ["method_energy_difference"],
        "input_bindings": {
            "energy_a": RequirementInputBindingProposal(
                requirement_key=left_key, port="energy_data"
            ),
            "energy_b": RequirementInputBindingProposal(
                requirement_key=right_key, port="energy_data"
            ),
        },
        "parameter_evidence": [],
        "constraints": {},
    }


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = [
    "canonicalize_modification",
    "canonicalize_semantic_request",
    "semantic_to_intake",
]
