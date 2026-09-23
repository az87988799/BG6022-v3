"""Deterministic graders over Request, Plan, Run, Result, and Artifact facts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .models import AssertionResult, BenchmarkAssertion, BenchmarkCase, CaseObservation, CaseResult

_DIMENSION_BY_ASSERTION = {
    "intent_equals": "intent_correct",
    "subject_count": "intent_correct",
    "requirement_count": "intent_correct",
    "requirement_capabilities": "intent_correct",
    "requirement_method_profiles": "intent_correct",
    "requirement_outputs": "intent_correct",
    "answer_goal": "answer_correct",
    "tool_present": "plan_correct",
    "forbid_tool": "plan_correct",
    "step_count": "plan_correct",
    "same_input_geometry": "plan_correct",
    "input_from_requirement": "plan_correct",
    "input_from_port": "plan_correct",
    "input_from_tool_port": "plan_correct",
    "result_target_present": "plan_correct",
    "result_status": "execution_correct",
    "scientific_check": "scientific_correct",
    "artifact_type_present": "scientific_correct",
    "no_new_attempt": "no_extra_compute",
    "max_orca_attempts": "no_extra_compute",
    "repair_attempt_count": "repair_success",
    "repair_action": "repair_success",
    "error_category": "boundary_correct",
    "boundary_blocked": "boundary_correct",
    "schema_corrections_at_most": "cost_within_bound",
}


def grade_case(case: BenchmarkCase, observation: CaseObservation) -> CaseResult:
    if case.id != observation.case_id:
        raise ValueError("benchmark case and observation ids do not match")
    assertions = [grade_assertion(item, observation) for item in case.assertions]
    dimensions: dict[str, bool | None] = {
        "intent_correct": None,
        "plan_correct": None,
        "scientific_correct": None,
        "execution_correct": None,
        "boundary_correct": None,
        "repair_success": None,
        "no_extra_compute": None,
        "answer_correct": None,
        "cost_within_bound": None,
    }
    by_dimension: dict[str, list[bool]] = {}
    for assertion, result in zip(case.assertions, assertions, strict=True):
        dimension = _DIMENSION_BY_ASSERTION[assertion.type]
        by_dimension.setdefault(dimension, []).append(result.passed)
    for dimension, values in by_dimension.items():
        dimensions[dimension] = all(values)
    failed_critical = [
        f"assertion:{item.assertion_type}"
        for item in assertions
        if item.critical and not item.passed
    ]
    no_new_attempt_failed = any(
        item.type == "no_new_attempt" and not result.passed
        for item, result in zip(case.assertions, assertions, strict=True)
    )
    observation = observation.model_copy(
        update={
            "critical_violations": [*observation.critical_violations, *failed_critical],
            "unrequested_compute": observation.unrequested_compute
            or (observation.orca_attempts > 0 and not case.requires_orca),
            "unexpected_recomputation": observation.unexpected_recomputation
            or no_new_attempt_failed,
        }
    )
    passed = all(item.passed for item in assertions)
    failed_stage = None if passed else _failed_stage(case, assertions, observation)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        support=case.support,
        mode=case.mode,
        run_index=observation.run_index,
        passed=passed,
        dimensions=dimensions,
        failed_stage=failed_stage,
        assertions=assertions,
        observation=observation,
    )


def grade_assertion(assertion: BenchmarkAssertion, observation: CaseObservation) -> AssertionResult:
    request = observation.request or {}
    plan = observation.plan or {}
    run = observation.run or {}
    expected = assertion.value
    observed: Any = None
    passed = False
    detail: str | None = None

    if assertion.type == "intent_equals":
        observed = (observation.intake or {}).get("intent")
        passed = observed == expected
    elif assertion.type == "subject_count":
        observed = len(request.get("subjects") or {})
        passed = observed == expected
    elif assertion.type == "requirement_count":
        observed = len(request.get("requirements") or [])
        passed = observed == expected
    elif assertion.type == "requirement_capabilities":
        capabilities = [item.get("capability") for item in request.get("requirements", [])]
        observed = dict(Counter(item for item in capabilities if item is not None))
        if isinstance(expected, list):
            passed = sorted(capabilities) == sorted(expected)
        elif isinstance(expected, dict):
            passed = observed == expected
    elif assertion.type == "requirement_method_profiles":
        requirements = request.get("requirements", [])
        profiles = [item.get("parameters", {}).get("method_profile") for item in requirements]
        statuses = [
            (item.get("constraints", {}).get("method_resolution") or {}).get("status")
            for item in requirements
        ]
        observed = {"profiles": profiles, "resolution_statuses": statuses}
        if isinstance(expected, list):
            passed = profiles == expected
        elif isinstance(expected, dict):
            profiles_match = "profiles" not in expected or profiles == expected["profiles"]
            statuses_match = (
                "resolution_statuses" not in expected or statuses == expected["resolution_statuses"]
            )
            passed = profiles_match and statuses_match
    elif assertion.type == "requirement_outputs":
        requirements = request.get("requirements", [])
        selected = [
            item
            for item in requirements
            if assertion.requirement_id is None or item.get("id") == assertion.requirement_id
        ]
        observed = [item.get("outputs", []) for item in selected]
        if isinstance(expected, list):
            passed = len(selected) == 1 and observed[0] == expected
        elif isinstance(expected, dict):
            observed_map = {item.get("id"): item.get("outputs", []) for item in selected}
            passed = observed_map == expected
    elif assertion.type == "answer_goal":
        observed = request.get("answer_goals", [])
        passed = _matches_expected_list(observed, expected)
    elif assertion.type == "tool_present":
        tools = [item.get("tool") for item in plan.get("steps", [])]
        observed = tools
        wanted = expected if isinstance(expected, list) else [expected]
        passed = all(item in tools for item in wanted)
    elif assertion.type == "forbid_tool":
        tools = [item.get("tool") for item in plan.get("steps", [])]
        observed = tools
        forbidden = expected if isinstance(expected, list) else [expected]
        passed = not any(item in tools for item in forbidden)
    elif assertion.type == "step_count":
        observed = len(plan.get("steps", []))
        passed = observed == expected
    elif assertion.type == "same_input_geometry":
        if assertion.step_id is not None and assertion.other_step_id is not None:
            first = _find_step(plan, assertion.step_id)
            second = _find_step(plan, assertion.other_step_id)
            observed = [_geometry_binding(first), _geometry_binding(second)]
            passed = first is not None and second is not None and observed[0] == observed[1]
        else:
            compute_tools = {"single_point", "optimize_geometry", "frequency"}
            bindings = [
                _geometry_binding(item)
                for item in plan.get("steps", [])
                if item.get("tool") in compute_tools and _geometry_binding(item) is not None
            ]
            observed = bindings
            passed = len(bindings) >= 2 and all(item == bindings[0] for item in bindings[1:])
    elif assertion.type == "input_from_requirement":
        consumer = _find_step(plan, assertion.step_id)
        refs = _input_references(consumer)
        source_steps = {
            item.get("id")
            for item in plan.get("steps", [])
            if item.get("requirement_id") == expected
        }
        observed = [ref for ref in refs if ref.get("step_id") in source_steps]
        passed = bool(observed)
    elif assertion.type == "input_from_port":
        consumer = _find_step(plan, assertion.step_id)
        refs = _input_references(consumer)
        observed = [
            ref
            for ref in refs
            if ref.get("step_id") == assertion.other_step_id and ref.get("port") == expected
        ]
        passed = bool(observed)
    elif assertion.type == "input_from_tool_port":
        consumer, consumer_count = _unique_step_by_tool(plan, assertion.tool)
        source, source_count = _unique_step_by_tool(plan, assertion.other_tool)
        if consumer is None or source is None:
            observed = {
                "consumer_tool": assertion.tool,
                "consumer_matches": consumer_count,
                "source_tool": assertion.other_tool,
                "source_matches": source_count,
            }
            if consumer_count > 1 or source_count > 1:
                detail = "ambiguous semantic selector"
        else:
            observed = [
                ref
                for ref in _input_references(consumer)
                if ref.get("step_id") == source.get("id") and ref.get("port") == expected
            ]
            passed = bool(observed)
    elif assertion.type == "result_target_present":
        targets = plan.get("requested_results", [])
        observed = targets
        passed = _target_is_present(targets, expected)
    elif assertion.type == "result_status":
        results = observation.results
        if assertion.step_id is not None:
            results = [item for item in results if item.get("step_id") == assertion.step_id]
        latest_by_step: dict[str, dict[str, Any]] = {}
        for item in results:
            step_id = str(item.get("step_id", ""))
            current = latest_by_step.get(step_id)
            if current is None or int(item.get("attempt", 0)) >= int(current.get("attempt", 0)):
                latest_by_step[step_id] = item
        observed = [item.get("status") for item in latest_by_step.values()]
        passed = bool(observed) and all(item == expected for item in observed)
    elif assertion.type == "scientific_check":
        check_name = expected.get("name") if isinstance(expected, dict) else None
        expected_status = expected.get("status") if isinstance(expected, dict) else None
        matched: list[dict[str, Any]] = []
        for result in observation.results:
            if assertion.step_id is not None and result.get("step_id") != assertion.step_id:
                continue
            checks = result.get("scientific_checks", {})
            for name, value in checks.items():
                if check_name is None or name == check_name:
                    item = value if isinstance(value, dict) else {"status": value}
                    matched.append({"name": name, **item})
        observed = matched
        passed = any(
            item.get("name") == check_name and item.get("status") == expected_status
            for item in matched
        )
    elif assertion.type == "artifact_type_present":
        artifact_types = [item.get("artifact_type") for item in observation.artifacts]
        observed = artifact_types
        expected_types = expected if isinstance(expected, list) else [expected]
        passed = all(item in artifact_types for item in expected_types)
    elif assertion.type == "no_new_attempt":
        observed = observation.orca_attempts
        expected = 0 if expected is None else expected
        passed = observed == expected
    elif assertion.type == "max_orca_attempts":
        observed = observation.orca_attempts
        passed = type(expected) is int and observed <= expected
    elif assertion.type == "repair_attempt_count":
        records = run.get("repair_records", [])
        observed = len(records)
        passed = observed == expected
    elif assertion.type == "repair_action":
        records = run.get("repair_records", [])
        actions = [item.get("action") for item in records]
        observed = actions
        wanted = expected if isinstance(expected, list) else [expected]
        passed = all(item in actions for item in wanted)
    elif assertion.type == "error_category":
        observed = observation.error_category
        if observed is None:
            observed = _latest_result_diagnostic(observation.results, "category")
        passed = observed == expected
    elif assertion.type == "boundary_blocked":
        observed = observation.status == "blocked"
        expected = True if expected is None else expected
        passed = observed is expected
    elif assertion.type == "schema_corrections_at_most":
        observed = sum(
            int(item.get("structured_correction_count", 0))
            for item in observation.llm_calls
            if type(item.get("structured_correction_count", 0)) is int
        )
        passed = type(expected) is int and observed <= expected
    else:  # pragma: no cover - the Literal and loader reject this path
        detail = f"no handler registered for {assertion.type}"

    if not passed and detail is None:
        detail = f"expected {expected!r}, observed {observed!r}"
    return AssertionResult(
        assertion_type=assertion.type,
        passed=passed,
        expected=expected,
        observed=observed,
        detail=detail,
        critical=assertion.critical,
    )


def _matches_expected_list(observed: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return observed == expected
    if isinstance(expected, dict):
        return any(
            all(item.get(key) == value for key, value in expected.items()) for item in observed
        )
    return expected in observed


def _find_step(plan: dict[str, Any], step_id: str | None) -> dict[str, Any] | None:
    if step_id is None:
        return None
    return next((item for item in plan.get("steps", []) if item.get("id") == step_id), None)


def _unique_step_by_tool(
    plan: dict[str, Any], tool: str | None
) -> tuple[dict[str, Any] | None, int]:
    if tool is None:
        return None, 0
    matches = [item for item in plan.get("steps", []) if item.get("tool") == tool]
    return (matches[0] if len(matches) == 1 else None), len(matches)


def _geometry_binding(step: dict[str, Any] | None) -> Any:
    if step is None:
        return None
    return (step.get("inputs") or {}).get("geometry")


def _input_references(step: dict[str, Any] | None) -> list[dict[str, Any]]:
    if step is None:
        return []
    return [item for item in (step.get("inputs") or {}).values() if isinstance(item, dict)]


def _target_is_present(targets: list[dict[str, Any]], expected: Any) -> bool:
    if isinstance(expected, str):
        return any(
            expected in {item.get("field"), item.get("port"), item.get("check")} for item in targets
        )
    if isinstance(expected, dict):
        return any(
            all(target.get(key) == value for key, value in expected.items()) for target in targets
        )
    return False


def _latest_result_diagnostic(results: list[dict[str, Any]], key: str) -> Any:
    for result in reversed(results):
        diagnostics = result.get("diagnostics", {})
        if key in diagnostics:
            return diagnostics[key]
    return None


def _failed_stage(
    case: BenchmarkCase, results: list[AssertionResult], observation: CaseObservation
):
    if observation.status in {"failed", "blocked", "exception"}:
        return observation.stage
    failed_types = {item.assertion_type for item in results if not item.passed}
    if "intent_equals" in failed_types:
        return "intake"
    if failed_types & {
        "subject_count",
        "requirement_count",
        "requirement_capabilities",
        "requirement_method_profiles",
        "requirement_outputs",
    }:
        return "request_normalization"
    if "answer_goal" in failed_types:
        return "answer"
    if failed_types & {"repair_attempt_count", "repair_action"}:
        return "repair"
    if failed_types & {"result_status", "scientific_check", "artifact_type_present"}:
        return "result_publish" if observation.results else "execution"
    if failed_types & {"no_new_attempt", "max_orca_attempts"}:
        return "query" if case.category == "query" else "execution"
    if failed_types & {"error_category", "boundary_blocked"}:
        return "request_normalization" if observation.request is not None else "intake"
    if observation.plan is None:
        return "planner"
    return "plan_validation"


__all__ = ["grade_assertion", "grade_case"]
