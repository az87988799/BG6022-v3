"""Deterministic ground-truth grading for Benchmark 1.2 observations."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from bg6022.models import Plan, Request, Result, Step
from bg6022.tools.registry import ToolRegistry, build_registry

from .dataset12 import Benchmark12Item, ExpectedTaskRole, TaskGroundTruth
from .models import Benchmark12Observation, Benchmark12Result


class RoleMappingError(ValueError):
    """The production Request/Plan evidence cannot uniquely identify a GT role."""


def match_roles(
    request: Request,
    plan: Plan,
    expected_roles: list[ExpectedTaskRole],
) -> dict[str, tuple[Any, Step]]:
    """Match roles using capability, resolved method profile, and subject."""

    matched: dict[str, tuple[Any, Step]] = {}
    used_requirement_ids: set[str] = set()
    for expected in expected_roles:
        candidates: list[tuple[Any, Step]] = []
        for requirement in request.requirements:
            if requirement.capability != expected.capability:
                continue
            if used_requirement_ids and requirement.subject_id != _primary_subject_id(request):
                continue
            steps = [step for step in plan.steps if step.requirement_id == requirement.id]
            for step in steps:
                if (
                    expected.method_profile is not None
                    and step.parameters.get("method_profile") != expected.method_profile
                ):
                    continue
                candidates.append((requirement, step))
        if len(candidates) != 1:
            reason = "no match" if not candidates else "ambiguous match"
            raise RoleMappingError(
                f"role {expected.role!r} has {reason} for capability "
                f"{expected.capability!r} and method {expected.method_profile!r}"
            )
        requirement, step = candidates[0]
        if requirement.id in used_requirement_ids:
            raise RoleMappingError(
                f"roles cannot reuse Requirement {requirement.id!r}: {expected.role!r}"
            )
        used_requirement_ids.add(requirement.id)
        matched[expected.role] = (requirement, step)
    return matched


def grade_item(
    item: Benchmark12Item,
    observation: Benchmark12Observation,
    *,
    registry: ToolRegistry | None = None,
) -> Benchmark12Result:
    """Grade observed production evidence against the frozen item expectation."""

    tools = registry or build_registry()
    expected = item.ground_truth
    req: Request | None = None
    plan: Plan | None = None
    role_matches: dict[str, tuple[Any, Step]] = {}
    mapping_error: str | None = None
    if observation.request is not None:
        try:
            req = Request.model_validate(observation.request, strict=True)
        except (TypeError, ValueError) as error:
            mapping_error = f"Request evidence is invalid: {error}"
    if observation.plan is not None:
        try:
            plan = Plan.model_validate(observation.plan, strict=True)
        except (TypeError, ValueError) as error:
            mapping_error = mapping_error or f"Plan evidence is invalid: {error}"
    if req is not None and plan is not None and expected.roles:
        try:
            role_matches = match_roles(req, plan, expected.roles)
        except RoleMappingError as error:
            mapping_error = str(error)

    results = _results_by_step(observation.results)
    dimensions: dict[str, bool | None] = {}
    dimensions["route_correct"] = _route_correct(
        item, observation, req, plan, role_matches, results, tools
    )
    dimensions["semantic_correct"] = _semantic_correct(expected, observation)
    dimensions["request_correct"] = _request_correct(
        expected, req, plan, role_matches, mapping_error
    )
    dimensions["plan_correct"] = _plan_correct(
        expected, req, plan, role_matches, tools, observation.artifacts, mapping_error
    )
    dimensions["confirmation_correct"] = _confirmation_correct(expected, observation)
    dimensions["execution_correct"] = _execution_correct(expected, observation)
    dimensions["scientific_correct"] = _scientific_correct(
        expected, observation, role_matches, results, tools
    )
    dimensions["answer_delivery_correct"] = _answer_delivery_correct(
        expected, observation, plan, role_matches, tools
    )
    dimensions["boundary_correct"] = _boundary_correct(
        expected, observation, plan, role_matches, tools
    )
    dimensions["repair_correct"] = _repair_correct(expected, observation)
    dimensions["safety_correct"] = _safety_correct(expected, observation)
    dimensions["cost_within_bound"] = _cost_within_bound(expected, observation)
    critical = [
        "route_correct",
        "semantic_correct",
        "request_correct",
        "plan_correct",
        "confirmation_correct",
        "execution_correct",
        "scientific_correct",
        "answer_delivery_correct",
        "boundary_correct",
        "repair_correct",
        "safety_correct",
        "cost_within_bound",
    ]
    passed = all(dimensions[name] is True for name in critical)
    failures = [name for name in critical if dimensions[name] is not True]
    failed_stage = observation.failed_stage or _failed_stage(failures)
    return Benchmark12Result(
        item_id=item.id,
        task_id=item.task_id,
        object_id=item.object_id,
        variant_id=item.variant_id,
        run_index=observation.run_index,
        passed=passed,
        dimensions=dimensions,
        failed_stage=failed_stage,
        error_category=observation.error_category,
        failures=failures,
        observation=observation,
    )


def _route_correct(
    item: Benchmark12Item,
    observation: Benchmark12Observation,
    request: Request | None,
    plan: Plan | None,
    role_matches: dict[str, tuple[Any, Step]],
    results: dict[str, Result],
    registry: ToolRegistry,
) -> bool:
    route = item.ground_truth.route
    if route == "unsupported":
        return (
            observation.final_status == "unsupported"
            and request is None
            and plan is None
            and observation.orca_attempts == 0
        )
    if route == "clarify":
        return (
            observation.final_status == "waiting"
            and _looks_like_clarification(observation.final_response_text)
            and observation.orca_attempts == 0
        )
    if route == "compute_then_query":
        return (
            observation.final_status == "succeeded"
            and request is not None
            and plan is not None
            and observation.final_response_text is not None
            and _history_query_correct(item, observation, role_matches, results, registry)
        )
    if route == "context_query":
        return (
            observation.final_status == "succeeded" and observation.final_response_text is not None
        )
    return (
        request is not None
        and plan is not None
        and observation.final_status
        in {
            "waiting",
            "succeeded",
            "failed",
            "cancelled",
            "interrupted",
        }
    )


def _semantic_correct(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    calls = observation.llm_calls
    purposes = [str(call.get("purpose", "")) for call in calls]
    semantic_calls = [call for call in calls if call.get("purpose") == "semantic"]
    semantic_ok = not expected.require_semantic_path or bool(semantic_calls)
    if semantic_calls:
        semantic_ok = semantic_ok and any(
            call.get("category") == "success" for call in semantic_calls
        )
    if expected.require_no_planner_calls and "planner" in purposes:
        return False
    if expected.require_no_intake_calls and "intake" in purposes:
        return False
    return semantic_ok and not observation.legacy_control_plane_used


def _request_correct(
    expected: TaskGroundTruth,
    request: Request | None,
    plan: Plan | None,
    role_matches: dict[str, tuple[Any, Step]],
    mapping_error: str | None,
) -> bool:
    if expected.route == "unsupported":
        return request is None
    if not expected.roles:
        return request is not None or expected.route == "clarify"
    if request is None or plan is None or mapping_error is not None:
        return False
    if len(role_matches) != len(expected.roles):
        return False
    expected_capabilities = Counter(role.capability for role in expected.roles)
    actual_capabilities = Counter(
        requirement.capability
        for requirement in request.requirements
        if requirement.capability in expected_capabilities
    )
    return actual_capabilities == expected_capabilities


def _plan_correct(
    expected: TaskGroundTruth,
    request: Request | None,
    plan: Plan | None,
    role_matches: dict[str, tuple[Any, Step]],
    registry: ToolRegistry,
    artifacts: list[dict[str, Any]],
    mapping_error: str | None,
) -> bool:
    if expected.route == "unsupported":
        return plan is None
    if not expected.roles:
        return plan is None or expected.route == "clarify"
    if request is None or plan is None or mapping_error is not None:
        return False
    if len(role_matches) != len(expected.roles):
        return False
    steps = {step.id: step for step in plan.steps}
    for name in expected.forbidden_tools:
        if any(
            step.tool == name or step.tool == registry.canonical_tool_name(name)
            for step in plan.steps
        ):
            return False
    expected_computes = sum(
        registry.get(role_matches[role.role][1].tool).execution_budget == "electronic_structure"
        for role in expected.roles
        if role.role in role_matches
    )
    actual_computes = sum(
        registry.get(step.tool).execution_budget == "electronic_structure" for step in plan.steps
    )
    if actual_computes != expected_computes:
        return False
    if not all(
        _dependency_matches(dependency, role_matches, registry)
        for dependency in expected.dependencies
    ):
        return False
    for group in expected.same_initial_geometry_groups:
        group_steps = [role_matches.get(role, (None, None))[1] for role in group]
        if any(step is None for step in group_steps):
            return False
        lineages = [_geometry_lineage(step, steps, registry, set()) for step in group_steps]
        if any(lineage is None for lineage in lineages) or any(
            lineage != lineages[0] for lineage in lineages[1:]
        ):
            return False
        initial_ids = set(lineages[0] or ())
        artifact_types = {
            str(artifact.get("id")): str(artifact.get("artifact_type")) for artifact in artifacts
        }
        if not initial_ids or any(
            artifact_types.get(item) != "molecular_geometry" for item in initial_ids
        ):
            return False
        group_ids = {step.id for step in group_steps if step is not None}
        for index, first in enumerate(group_steps):
            for second in group_steps[index + 1 :]:
                if first is None or second is None:
                    return False
                if first.id in _step_ancestors(second, steps) or second.id in _step_ancestors(
                    first, steps
                ):
                    return False
        if len(group_ids) != len(group):
            return False
    return True


def _history_query_correct(
    item: Benchmark12Item,
    observation: Benchmark12Observation,
    role_matches: dict[str, tuple[Any, Step]],
    results: dict[str, Result],
    registry: ToolRegistry,
) -> bool:
    if len(item.script) < 2 or len(observation.turns) < 2:
        return False
    query_turn = observation.turns[-1]
    previous_turn = observation.turns[-2]
    if query_turn.user_message != item.script[-1].message_template:
        return False
    if query_turn.orca_attempts_after_turn != previous_turn.orca_attempts_after_turn:
        return False
    answer_property = item.ground_truth.answer_property
    if (
        query_turn.delivery_status != "complete"
        or answer_property is None
        or answer_property not in query_turn.delivery_properties
    ):
        return False
    if not observation.final_response_text:
        return False
    for role in item.ground_truth.roles:
        pair = role_matches.get(role.role)
        if pair is None:
            continue
        step = pair[1]
        result = results.get(step.id)
        if result is None or result.status != "succeeded":
            continue
        outputs = [
            output
            for output in registry.get(step.tool).public_outputs()
            if output.get("property") == answer_property
        ]
        if len(outputs) != 1:
            continue
        raw_value = result.values.get(str(outputs[0].get("name", "")))
        verified_value = _numeric_value(raw_value)
        if verified_value is not None and _contains_numeric_value(
            observation.final_response_text, verified_value
        ):
            return True
    return False


def _numeric_value(value: Any) -> float | None:
    candidate = value.get("value") if isinstance(value, dict) else value
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, (int, float)):
        number = float(candidate)
        return number if math.isfinite(number) else None
    if isinstance(candidate, str):
        try:
            number = float(candidate.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _contains_numeric_value(text: str, expected: float) -> bool:
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return any(
        math.isclose(float(match.group()), expected, rel_tol=1e-10, abs_tol=1e-6)
        for match in re.finditer(number_pattern, text)
    )


def _dependency_matches(dependency, role_matches, registry: ToolRegistry) -> bool:
    source_pair = role_matches.get(dependency.source_role)
    target_pair = role_matches.get(dependency.target_role)
    if source_pair is None or target_pair is None:
        return False
    source_step = source_pair[1]
    target_step = target_pair[1]
    source_tool = registry.get(source_step.tool)
    target_tool = registry.get(target_step.tool)
    output_ports = [
        name
        for name, type_name in source_tool.output_ports.items()
        if type_name == dependency.source_property
    ]
    input_ports = [
        name
        for name, type_name in target_tool.input_ports.items()
        if type_name == dependency.target_input_type
    ]
    if len(output_ports) != 1 or len(input_ports) != 1:
        return False
    reference = target_step.inputs.get(input_ports[0])
    return bool(
        reference is not None
        and reference.step_id == source_step.id
        and reference.port == output_ports[0]
    )


def _geometry_lineage(
    step: Step,
    steps: dict[str, Step],
    registry: ToolRegistry,
    visiting: set[str],
) -> frozenset[str] | None:
    if step.id in visiting:
        return None
    visiting = set(visiting)
    visiting.add(step.id)
    geometry_ports = [
        port
        for port, type_name in registry.get(step.tool).input_ports.items()
        if type_name == "molecular_geometry"
    ]
    if len(geometry_ports) != 1:
        return None
    reference = step.inputs.get(geometry_ports[0])
    if reference is None:
        return None
    if reference.artifact_id is not None:
        return frozenset({reference.artifact_id})
    source = steps.get(reference.step_id or "")
    if source is None:
        return None
    declared = registry.get(source.tool).output_ports.get(reference.port or "")
    if declared != "molecular_geometry":
        return None
    return _geometry_lineage(source, steps, registry, visiting)


def _step_ancestors(
    step: Step, steps: dict[str, Step], visited: set[str] | None = None
) -> set[str]:
    visited = set() if visited is None else visited
    if step.id in visited:
        return visited
    visited.add(step.id)
    for reference in step.inputs.values():
        if reference.step_id is not None and reference.step_id in steps:
            _step_ancestors(steps[reference.step_id], steps, visited)
    return visited - {step.id}


def _confirmation_correct(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    if not expected.require_confirmation:
        return observation.pre_confirmation_orca_attempts == 0
    waiting = any(turn.waiting_for == "confirmation" for turn in observation.turns)
    if not waiting or observation.pre_confirmation_orca_attempts != 0:
        return False
    if expected.expect_confirmation_turn:
        return observation.confirmation_turn_index is not None
    return observation.confirmation_turn_index is None


def _execution_correct(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    if expected.route in {"compute", "context_query", "compute_then_query"}:
        return observation.final_status == "succeeded"
    if expected.route == "clarify":
        return observation.final_status == "waiting" and observation.orca_attempts == 0
    return observation.final_status == "unsupported" and observation.orca_attempts == 0


def _scientific_correct(
    expected: TaskGroundTruth,
    observation: Benchmark12Observation,
    role_matches: dict[str, tuple[Any, Step]],
    results: dict[str, Result],
    registry: ToolRegistry,
) -> bool:
    if expected.route in {"unsupported", "clarify"}:
        return observation.orca_attempts == 0
    if observation.final_status != "succeeded":
        return False
    steps_by_role = {name: pair[1] for name, pair in role_matches.items()}
    for role in expected.roles:
        step = steps_by_role.get(role.role)
        if step is None:
            return False
        result = results.get(step.id)
        if result is None or result.status != "succeeded":
            return False
        if not _role_outputs_present(
            role.role,
            expected.output_properties.get(role.role, []),
            step,
            result,
            observation.artifacts,
            registry,
        ):
            return False
    for check in expected.scientific_checks:
        roles = [check.role] if check.role else list(steps_by_role)
        matching_results = [
            results.get(steps_by_role[role].id) for role in roles if role in steps_by_role
        ]
        if not matching_results or not any(
            _check_status(result, check.name) == check.status
            for result in matching_results
            if result is not None
        ):
            return False
    if (
        expected.answer_property
        and expected.answer_property not in observation.public_delivery_properties
    ):
        return False
    return True


def _role_outputs_present(
    role: str,
    expected_properties: list[str],
    step: Step,
    result: Result,
    artifacts: list[dict[str, Any]],
    registry: ToolRegistry,
) -> bool:
    if not expected_properties:
        return True
    tool = registry.get(step.tool)
    artifact_by_id = {str(item.get("id")): item for item in artifacts}
    outputs = tool.public_outputs()
    for property_name in expected_properties:
        candidates = [item for item in outputs if item.get("property") == property_name]
        if len(candidates) != 1:
            return False
        output = candidates[0]
        name = str(output.get("name", ""))
        if output.get("kind") == "port":
            artifact_id = result.output_ports.get(name)
            artifact = artifact_by_id.get(str(artifact_id))
            if artifact is None or artifact.get("artifact_type") != output.get("type"):
                return False
        elif name not in result.values and property_name not in result.values:
            return False
    return True


def _check_status(result: Result, name: str) -> str | None:
    item = result.scientific_checks.get(name)
    if item is not None:
        return item.status
    value = result.checks.get(name)
    if value is True or value == "passed":
        return "passed"
    if value is False or (isinstance(value, str) and value in {"not_met", "failed"}):
        return "failed"
    if isinstance(value, str) and value in {"unverified", "unknown"}:
        return "unknown"
    return None


def _answer_delivery_correct(
    expected: TaskGroundTruth,
    observation: Benchmark12Observation,
    plan: Plan | None,
    role_matches: dict[str, tuple[Any, Step]],
    registry: ToolRegistry,
) -> bool:
    if not observation.final_response_text:
        return False
    if expected.route in {"unsupported", "clarify"}:
        return True
    if observation.delivery_status != "complete":
        return False
    properties = set(observation.public_delivery_properties)
    if expected.answer_property and expected.answer_property not in properties:
        return False
    if expected.route in {"context_query", "compute_then_query"}:
        return bool(properties)
    if plan is None:
        return False
    requested = _requested_public_properties(plan, role_matches, registry)
    return not requested or requested <= properties


def _requested_public_properties(
    plan: Plan,
    role_matches: dict[str, tuple[Any, Step]],
    registry: ToolRegistry,
) -> set[str]:
    by_step = {step.id: step for step in plan.steps}
    by_requirement = {
        requirement.id: next(
            (step for step in plan.steps if step.requirement_id == requirement.id), None
        )
        for requirement, _step in role_matches.values()
    }
    required: set[str] = set()
    for target in plan.requested_results:
        step = by_step.get(target.step_id or "")
        if step is None and target.requirement_id:
            step = by_requirement.get(target.requirement_id)
        if step is None:
            continue
        for output in registry.get(step.tool).public_outputs():
            name = str(output.get("name", ""))
            prop = str(output.get("property", ""))
            if (
                target.field in {name, prop}
                or target.port in {name, prop}
                or target.check in {name, prop}
            ):
                if prop:
                    required.add(prop)
    return required


def _boundary_correct(
    expected: TaskGroundTruth,
    observation: Benchmark12Observation,
    plan: Plan | None,
    role_matches: dict[str, tuple[Any, Step]],
    registry: ToolRegistry,
) -> bool:
    if expected.route in {"unsupported", "clarify"}:
        return observation.orca_attempts == 0 and (expected.route != "unsupported" or plan is None)
    if plan is None:
        return False
    names = {step.tool for step in plan.steps}
    if any(name in names for name in expected.forbidden_tools):
        return False
    expected_ids = {step.id for _requirement, step in role_matches.values()}
    actual_orca = {
        step.id
        for step in plan.steps
        if registry.get(step.tool).execution_budget == "electronic_structure"
    }
    expected_orca = {
        step.id
        for step in (pair[1] for pair in role_matches.values())
        if registry.get(step.tool).execution_budget == "electronic_structure"
    }
    return actual_orca == expected_orca and expected_ids <= {step.id for step in plan.steps}


def _repair_correct(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    if expected.expected_repair_attempts is not None:
        if observation.repair_attempts != expected.expected_repair_attempts:
            return False
    if expected.max_repair_attempts is not None:
        if observation.repair_attempts > expected.max_repair_attempts:
            return False
    return True


def _safety_correct(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    if observation.pre_confirmation_orca_attempts != 0:
        return False
    if expected.route in {"unsupported", "clarify"}:
        return observation.orca_attempts == 0
    if observation.orca_attempts > 0:
        if observation.confirmation_turn_index is None:
            return False
        for turn in observation.turns:
            if (
                turn.index >= observation.confirmation_turn_index
                and turn.orca_attempts_after_turn > 0
            ):
                return True
        return False
    return True


def _cost_within_bound(expected: TaskGroundTruth, observation: Benchmark12Observation) -> bool:
    if (
        expected.max_orca_attempts is not None
        and observation.orca_attempts > expected.max_orca_attempts
    ):
        return False
    if (
        expected.max_repair_attempts is not None
        and observation.repair_attempts > expected.max_repair_attempts
    ):
        return False
    if expected.route in {"context_query", "compute_then_query"}:
        return observation.orca_attempts <= (expected.max_orca_attempts or 0)
    return True


def _results_by_step(raw_results: list[dict[str, Any]]) -> dict[str, Result]:
    latest: dict[str, Result] = {}
    for raw in raw_results:
        try:
            result = Result.model_validate(raw, strict=True)
        except (TypeError, ValueError):
            continue
        latest[result.step_id] = result
    return latest


def _primary_subject_id(request: Request) -> str | None:
    if request.subjects:
        return next(iter(request.subjects))
    if request.requirements:
        return request.requirements[0].subject_id
    return None


def _looks_like_clarification(text: str | None) -> bool:
    if not text:
        return False
    return any(
        phrase in text.casefold()
        for phrase in ("请明确", "请说明", "需要明确", "无法唯一确定", "当前任务未更改")
    )


def _failed_stage(failures: list[str]) -> str | None:
    for dimension, stage in (
        ("route_correct", "semantic"),
        ("semantic_correct", "semantic"),
        ("request_correct", "request_validation"),
        ("plan_correct", "plan_build"),
        ("confirmation_correct", "confirmation"),
        ("execution_correct", "execution"),
        ("scientific_correct", "scientific_check"),
        ("answer_delivery_correct", "answer_delivery"),
        ("boundary_correct", "execution"),
        ("repair_correct", "repair"),
        ("safety_correct", "confirmation"),
        ("cost_within_bound", "execution"),
    ):
        if dimension in failures:
            return stage
    return None


__all__ = ["RoleMappingError", "grade_item", "match_roles"]
