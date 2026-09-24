"""Deterministically materialize a Plan from a canonical Request."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from bg6022.models import InputReference, Plan, Request, ResultTarget, Step
from bg6022.planner import validate_request_plan
from bg6022.tools.registry import ToolRegistry


class PlanBuildError(ValueError):
    """A canonical Request does not determine one valid typed Plan."""


def build_plan(request: Request, *, registry: ToolRegistry, plan_id: str) -> Plan:
    """Build preparation and requested-task Steps without an LLM Planner."""

    ordered_requirements = _topological_requirements(request)
    steps: list[Step] = []
    requirement_step_ids: dict[str, str] = {}
    subject_geometry: dict[str, InputReference] = {}

    subjects = request.subjects
    for subject_id, subject in subjects.items():
        if _subject_needs_initial_geometry(request, subject_id, registry):
            geometry_ref, preparation = _subject_geometry_source(
                subject_id,
                subject,
                offset=len(steps),
            )
            steps.extend(preparation)
            subject_geometry[subject_id] = geometry_ref

    for requirement in ordered_requirements:
        tool = registry.get(requirement.capability)
        step_id = _stable_step_id(len(steps) + 1, tool.name)
        inputs: dict[str, InputReference] = {}
        unknown_bindings = sorted(set(requirement.input_bindings) - set(tool.input_ports))
        if unknown_bindings:
            raise PlanBuildError(
                f"Requirement {requirement.id!r} binds undeclared Tool inputs: {unknown_bindings}"
            )
        for input_name, input_type in tool.input_ports.items():
            binding = requirement.input_bindings.get(input_name)
            if binding is not None:
                if binding.artifact_alias is not None:
                    if binding.artifact_alias != "initial_geometry":
                        raise PlanBuildError(
                            f"unregistered artifact alias {binding.artifact_alias!r}"
                        )
                    if input_type != "molecular_geometry":
                        raise PlanBuildError(
                            f"initial_geometry cannot bind {tool.name}.{input_name}"
                        )
                    reference = subject_geometry.get(requirement.subject_id)
                    if reference is None:
                        raise PlanBuildError(
                            f"no initial geometry source for Requirement {requirement.id!r}"
                        )
                    inputs[input_name] = reference
                    continue

                source_id = binding.source_requirement_id
                source_step_id = requirement_step_ids.get(str(source_id))
                if source_step_id is None:
                    raise PlanBuildError(
                        f"source Requirement {source_id!r} has not been materialized"
                    )
                source_requirement = next(
                    (item for item in request.requirements if item.id == source_id), None
                )
                if source_requirement is None:
                    raise PlanBuildError(f"unknown source Requirement {source_id!r}")
                source_tool = registry.get(source_requirement.capability)
                source_port = str(binding.source_port)
                if source_tool.output_ports.get(source_port) != input_type:
                    raise PlanBuildError(
                        f"typed dependency mismatch for {tool.name}.{input_name}: "
                        f"{source_tool.name}.{source_port} does not provide {input_type!r}"
                    )
                inputs[input_name] = InputReference(step_id=source_step_id, port=source_port)
                continue

            if input_type == "molecular_geometry":
                reference = subject_geometry.get(requirement.subject_id)
                if reference is None:
                    raise PlanBuildError(
                        f"no geometry source for Requirement {requirement.id!r}"
                    )
                inputs[input_name] = reference
                continue
            raise PlanBuildError(f"unbound non-geometry input: {tool.name}.{input_name}")

        step = Step(
            id=step_id,
            origin_step_id=step_id,
            tool=tool.name,
            parameters=dict(requirement.parameters),
            inputs=inputs,
            requirement_id=requirement.id,
            subject_id=requirement.subject_id,
        )
        steps.append(step)
        requirement_step_ids[requirement.id] = step_id

    requested_results = _build_result_targets(request, requirement_step_ids, registry)
    plan = Plan(
        id=plan_id,
        request_id=request.id,
        steps=steps,
        requested_results=requested_results,
    )
    try:
        return validate_request_plan(request, plan, registry)
    except (TypeError, ValueError) as error:
        raise PlanBuildError(str(error)) from error


def _topological_requirements(request: Request) -> list[Any]:
    requirements = list(request.requirements)
    by_id = {item.id: item for item in requirements}
    if len(by_id) != len(requirements):
        raise PlanBuildError("Request Requirement IDs must be unique")
    remaining = {item.id: item for item in requirements}
    ordered: list[Any] = []
    while remaining:
        ready = [
            item
            for item in requirements
            if item.id in remaining
            and all(
                binding.source_requirement_id not in remaining
                for binding in item.input_bindings.values()
            )
        ]
        if not ready:
            raise PlanBuildError("Requirement dependency graph contains a cycle")
        for item in ready:
            for binding in item.input_bindings.values():
                source_id = binding.source_requirement_id
                if source_id is not None and source_id not in by_id:
                    raise PlanBuildError(
                        f"Requirement {item.id!r} references unknown source {source_id!r}"
                    )
            ordered.append(item)
            remaining.pop(item.id)
    return ordered


def _subject_needs_initial_geometry(
    request: Request,
    subject_id: str,
    registry: ToolRegistry,
) -> bool:
    for requirement in request.requirements:
        if requirement.subject_id != subject_id:
            continue
        tool = registry.get(requirement.capability)
        for input_name, input_type in tool.input_ports.items():
            if input_type != "molecular_geometry":
                continue
            binding = requirement.input_bindings.get(input_name)
            if binding is None or binding.artifact_alias == "initial_geometry":
                return True
    return False


def _subject_geometry_source(
    subject_id: str,
    subject: Any,
    *,
    offset: int,
) -> tuple[InputReference, list[Step]]:
    structure = dict(subject.structure_input)
    history_alias = structure.get("history_geometry_alias")
    if isinstance(history_alias, str) and history_alias:
        return InputReference(artifact_id=history_alias), []
    if structure.get("xyz_text") is not None or structure.get("xyz") is not None:
        key = str(subject.key)
        return InputReference(artifact_id=f"request_geometry_{key}"), []
    if not subject.molecule_query:
        raise PlanBuildError("subject has neither geometry nor molecule identity")
    if subject.molecule_input_kind is None:
        raise PlanBuildError("subject molecule identity has no declared input kind")

    resolve_id = _stable_step_id(offset + 1, f"resolve_molecule_{subject.key}")
    geometry_id = _stable_step_id(offset + 2, f"generate_geometry_{subject.key}")
    resolve_step = Step(
        id=resolve_id,
        origin_step_id=resolve_id,
        tool="resolve_molecule",
        parameters={
            "query": subject.molecule_query,
            "input_kind": subject.molecule_input_kind,
        },
        inputs={},
        subject_id=subject_id,
    )
    geometry_step = Step(
        id=geometry_id,
        origin_step_id=geometry_id,
        tool="generate_geometry",
        parameters={},
        inputs={"molecule": InputReference(step_id=resolve_id, port="molecule")},
        subject_id=subject_id,
    )
    return InputReference(step_id=geometry_id, port="geometry"), [resolve_step, geometry_step]


def _build_result_targets(
    request: Request,
    requirement_step_ids: Mapping[str, str],
    registry: ToolRegistry,
) -> list[ResultTarget]:
    targets: list[ResultTarget] = []
    for requirement in request.requirements:
        step_id = requirement_step_ids.get(requirement.id)
        if step_id is None:
            raise PlanBuildError(f"Requirement {requirement.id!r} has no Plan Step")
        tool = registry.get(requirement.capability)
        public = {str(item["name"]): item for item in tool.public_outputs()}
        for output in requirement.outputs:
            descriptor = public.get(output)
            if descriptor is None:
                raise PlanBuildError(f"undeclared output {output!r} on {tool.name}")
            kind = str(descriptor["kind"])
            targets.append(
                ResultTarget(
                    step_id=step_id,
                    requirement_id=requirement.id,
                    **{kind: output},
                )
            )
    return targets


def _stable_step_id(index: int, value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "step"
    return f"s{index:02d}_{cleaned}"


__all__ = ["PlanBuildError", "build_plan"]
