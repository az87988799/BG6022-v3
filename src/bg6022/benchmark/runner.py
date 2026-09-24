"""Execute offline, replay, and explicitly enabled live benchmark cases."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bg6022.agent import INPUT_GEOMETRY_PLACEHOLDER, Agent, AgentResponse
from bg6022.config import AppConfig
from bg6022.llm import LlmClient, LlmError
from bg6022.models import InputReference, Plan, Request, Result, Run
from bg6022.orca.profiles import get_profile
from bg6022.planner import (
    IntakeOutput,
    PlanProposal,
    proposal_to_plan,
    request_from_intake,
    validate_request_plan,
)
from bg6022.session import (
    artifact_path,
    create_run,
    publish_step_result,
    register_bytes_artifact,
    register_file_artifact,
    run_directory,
    save_run,
    sha256_file,
)
from bg6022.tools.registry import ToolRegistry, build_registry

from .loader import (
    BenchmarkConfigurationError,
    load_fixture,
    load_fixture_reference,
)
from .models import BenchmarkCase, CaseObservation, LiveResultFixture, LiveSetup
from .observers import count_orca_attempts, observation_from_runtime


class BenchmarkModeDisabled(RuntimeError):
    """Raised when a case requires a live flag that the caller did not enable."""


def run_case(
    case: BenchmarkCase,
    *,
    config: AppConfig | None,
    allow_live_llm: bool = False,
    allow_live_orca: bool = False,
    allow_live_pubchem: bool = False,
    benchmark_dir: str | Path,
    data_root: str | Path | None = None,
    fixture_override: dict[str, Any] | None = None,
) -> list[CaseObservation]:
    """Run the requested repetitions; live work is impossible without explicit flags."""

    if fixture_override is not None and case.mode != "live_orca":
        raise ValueError("fixture_override is only supported for live ORCA benchmark cases")
    _check_live_permissions(
        case,
        allow_live_llm=allow_live_llm,
        allow_live_orca=allow_live_orca,
        allow_live_pubchem=allow_live_pubchem,
        config=config,
    )
    if config is not None and (case.requires_orca or case.mode == "live_llm"):
        data_root = data_root or _default_data_root(benchmark_dir)
    observations: list[CaseObservation] = []
    for run_index in range(1, case.repeat + 1):
        if case.mode in {"offline", "replay"}:
            observation = _run_fixture_case(
                case,
                run_index=run_index,
                benchmark_dir=benchmark_dir,
                config=config,
                data_root=data_root,
            )
        elif case.mode == "live_llm":
            observation = _run_live_llm(
                case,
                run_index=run_index,
                benchmark_dir=benchmark_dir,
                config=_required_config(config),
                data_root=data_root,
            )
        elif case.mode == "live_orca":
            observation = _run_live_orca(
                case,
                run_index=run_index,
                benchmark_dir=benchmark_dir,
                config=_required_config(config),
                data_root=data_root,
                fixture_override=fixture_override,
            )
        else:
            observation = _run_live_e2e(
                case,
                run_index=run_index,
                benchmark_dir=benchmark_dir,
                config=_required_config(config),
                data_root=data_root,
            )
        observations.append(observation)
    return observations


def _check_live_permissions(
    case: BenchmarkCase,
    *,
    allow_live_llm: bool,
    allow_live_orca: bool,
    allow_live_pubchem: bool,
    config: AppConfig | None,
) -> None:
    if case.requires_llm and not allow_live_llm:
        raise BenchmarkModeDisabled(f"case {case.id} requires --live-llm")
    if case.requires_orca and not allow_live_orca:
        raise BenchmarkModeDisabled(f"case {case.id} requires --live-orca")
    if case.requires_pubchem and not allow_live_pubchem:
        raise BenchmarkModeDisabled(f"case {case.id} requires --live-pubchem")
    if case.mode in {"live_llm", "live_orca", "live_e2e"} and config is None:
        raise BenchmarkModeDisabled(f"case {case.id} requires --config")


def _run_fixture_case(
    case: BenchmarkCase,
    *,
    run_index: int,
    benchmark_dir: str | Path,
    config: AppConfig | None,
    data_root: str | Path | None,
) -> CaseObservation:
    started = time.monotonic()
    try:
        fixture = load_fixture(case, benchmark_dir)
    except BenchmarkConfigurationError:
        raise
    if "observation" in fixture:
        observation = CaseObservation.model_validate(
            {
                **fixture["observation"],
                "case_id": case.id,
                "run_index": run_index,
            },
            strict=True,
        )
        return observation.model_copy(
            update={"elapsed_seconds": max(time.monotonic() - started, 0.0)}
        )

    registry = build_registry(config)
    stage = "request_normalization"
    intake: IntakeOutput | None = None
    request: Request | None = None
    plan: Plan | None = None
    try:
        intake, request, plan = _build_contract(
            fixture,
            prompt=case.prompt,
            case_id=case.id,
            run_index=run_index,
            registry=registry,
        )
        stage = "plan_validation"
        if plan is not None:
            plan = validate_request_plan(request, plan, registry)
        run: Run | None = None
        results: list[Result] = []
        if case.mode == "replay" and fixture.get("replay_artifact") is not None:
            stage = "execution"
            run, plan = _replay_artifact_input(
                case=case,
                run_index=run_index,
                fixture=fixture,
                request=request,
                plan=plan,
                config=config,
                benchmark_dir=benchmark_dir,
                data_root=data_root,
            )
        if case.mode == "replay" and fixture.get("result") is not None:
            stage = "result_publish"
            run, result = _replay_result(
                case=case,
                run_index=run_index,
                fixture=fixture,
                request=request,
                plan=plan,
                registry=registry,
                config=config,
                benchmark_dir=benchmark_dir,
                data_root=data_root,
            )
            results.append(result)
        if fixture.get("answer_fixture") is not None:
            stage = "answer"
            from bg6022.answer import AnswerOutput, render_answer_output, validate_result_answer

            answer = AnswerOutput.model_validate(fixture["answer_fixture"]["answer"], strict=True)
            outputs = fixture["answer_fixture"].get("outputs_by_ref", {})
            required_refs = fixture["answer_fixture"].get("required_refs", [])
            validate_result_answer(answer, outputs, required_refs)
            response_text = render_answer_output(
                answer,
                outputs_by_ref=outputs,
                required_refs=required_refs,
            )
        else:
            response_text = fixture.get("response_text")
        requested_status = fixture.get("status", "completed")
        requested_error = fixture.get("error_category")
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status=requested_status,
            stage="complete" if requested_status == "completed" else stage,
            intake=intake,
            request=request,
            plan=plan,
            run=run,
            results=results,
            response_text=response_text,
            elapsed_seconds=time.monotonic() - started,
            error_category=requested_error,
            error_message=fixture.get("error_message"),
        )
    except (TypeError, ValueError, KeyError, OSError) as error:
        expected_error = fixture.get("expected_error")
        is_expected = expected_error is not None
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="blocked" if is_expected else "exception",
            stage=str(fixture.get("error_stage") or stage),
            intake=intake,
            request=request,
            plan=plan,
            elapsed_seconds=time.monotonic() - started,
            error_category=(
                str(fixture.get("error_category") or "validation_error")
                if is_expected
                else "fixture_error"
            ),
            error_message=str(error),
        )


def _build_contract(
    fixture: dict[str, Any],
    *,
    prompt: str,
    case_id: str,
    run_index: int,
    registry: ToolRegistry,
) -> tuple[IntakeOutput | None, Request, Plan | None]:
    request_id = f"benchmark_{case_id}_{run_index}"
    intake: IntakeOutput | None = None
    if "intake" in fixture:
        intake = IntakeOutput.model_validate(fixture["intake"], strict=True)
        request = request_from_intake(
            str(fixture.get("request_text") or prompt),
            intake,
            request_id=request_id,
            registry=registry,
        )
    elif "request" in fixture:
        request = Request.model_validate({**fixture["request"], "id": request_id}, strict=True)
    else:
        raise BenchmarkConfigurationError("fixture needs either intake or request")

    plan: Plan | None = None
    if "plan_proposal" in fixture:
        raw_proposal = _resolve_fixture_references(fixture["plan_proposal"], fixture, request)
        proposal = PlanProposal.model_validate(raw_proposal, strict=True)
        plan = proposal_to_plan(
            request,
            proposal,
            registry,
            plan_id=f"plan_{case_id}_{run_index}",
            artifact_aliases=_artifact_aliases(fixture),
        )
    elif "plan" in fixture:
        plan = Plan.model_validate(
            {**fixture["plan"], "id": f"plan_{case_id}_{run_index}", "request_id": request.id},
            strict=True,
        )
        plan = validate_request_plan(request, plan, registry)
    return intake, request, plan


def _resolve_fixture_references(payload: Any, fixture: dict[str, Any], request: Request) -> Any:
    if not isinstance(payload, dict):
        return payload
    result = json.loads(json.dumps(payload))
    intake_raw = fixture.get("intake", {})
    subject_keys: dict[str, str] = {}
    subject_proposals = intake_raw.get("subjects", {}) if isinstance(intake_raw, dict) else {}
    request_subjects = {item.key: subject_id for subject_id, item in request.subjects.items()}
    for key in subject_proposals:
        if key in request_subjects:
            subject_keys[key] = request_subjects[key]
    requirement_keys: dict[str, str] = {}
    intake_requirements = intake_raw.get("requirements", []) if isinstance(intake_raw, dict) else []
    by_key = {item.get("key"): item for item in intake_requirements if isinstance(item, dict)}
    requirements_by_capability: dict[str, list[str]] = {}
    for requirement in request.requirements:
        requirements_by_capability.setdefault(requirement.capability, []).append(requirement.id)
    for key, raw in by_key.items():
        capability = raw.get("capability")
        candidates = requirements_by_capability.get(capability, [])
        if len(candidates) == 1:
            requirement_keys[key] = candidates[0]
        elif candidates:
            # Repeated capabilities retain the intake ordering used by Request.
            ordered = [item.id for item in request.requirements if item.capability == capability]
            same_capability = [
                item.get("key")
                for item in intake_requirements
                if item.get("capability") == capability
            ]
            if key in same_capability and len(ordered) == len(same_capability):
                requirement_keys[key] = ordered[same_capability.index(key)]

    for step in result.get("steps", []):
        if "requirement_key" in step:
            key = step.pop("requirement_key")
            if key not in requirement_keys:
                raise BenchmarkConfigurationError(
                    f"fixture refers to unknown requirement key: {key}"
                )
            step["requirement_id"] = requirement_keys[key]
        if "subject_key" in step:
            key = step.pop("subject_key")
            if key not in subject_keys:
                raise BenchmarkConfigurationError(f"fixture refers to unknown subject key: {key}")
            step["subject_id"] = subject_keys[key]
    return result


def _artifact_aliases(fixture: dict[str, Any]) -> dict[str, str]:
    aliases = {
        "initial_geometry": INPUT_GEOMETRY_PLACEHOLDER,
        "input_geometry": INPUT_GEOMETRY_PLACEHOLDER,
        "provided_geometry": INPUT_GEOMETRY_PLACEHOLDER,
    }
    custom = fixture.get("artifact_aliases", {})
    if isinstance(custom, dict):
        aliases.update({str(key): str(value) for key, value in custom.items()})
    return aliases


class _RecordingLlmClient(LlmClient):
    """Record validated model outputs and bounded errors without changing calls."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.structured_outputs: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

    def complete_json(
        self,
        messages,
        schema,
        *,
        purpose: str = "json",
        example=None,
        cancel=None,
        remaining_timeout_seconds=None,
    ):
        try:
            value = super().complete_json(
                messages,
                schema,
                purpose=purpose,
                example=example,
                cancel=cancel,
                remaining_timeout_seconds=remaining_timeout_seconds,
            )
        except LlmError as error:
            self.errors.append(
                {
                    "purpose": error.purpose or purpose,
                    "category": error.category,
                    "message": _redact_api_key(str(error), self.settings.api_key_env),
                    "diagnostics": [
                        {
                            str(key): str(item)
                            for key, item in dict(diagnostic).items()
                            if key in {"path", "message"}
                        }
                        for diagnostic in error.diagnostics
                    ],
                }
            )
            raise
        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        if isinstance(payload, dict):
            self.structured_outputs.append({"purpose": purpose, "value": payload})
        return value


class _PlanningBarrierAgent(Agent):
    """Run the production chat route but stop at the Tool execution boundary."""

    def __init__(self, *args, **kwargs) -> None:
        self.barrier_calls = 0
        super().__init__(*args, **kwargs)

    def advance(self, run: Run, *, cancel=None) -> Result | None:
        self.barrier_calls += 1
        if run.status not in {"succeeded", "failed", "cancelled", "interrupted"}:
            run.status = "waiting"
            if run.waiting_for is None:
                run.waiting_for = "confirmation"
            run.pending_data = {
                **dict(run.pending_data),
                "benchmark_barrier": True,
            }
            save_run(self.config.data_root_path, run)
        return None


def _run_live_llm(
    case: BenchmarkCase,
    *,
    run_index: int,
    benchmark_dir: str | Path,
    config: AppConfig,
    data_root: str | Path | None,
) -> CaseObservation:
    started = time.monotonic()
    stage = "intake"
    llm: _RecordingLlmClient | None = None
    agent: _PlanningBarrierAgent | None = None
    seeded_run: Run | None = None
    try:
        fixture = load_fixture(case, benchmark_dir) if case.fixture is not None else {}
        try:
            live_setup = LiveSetup.model_validate(fixture.get("live_setup", {}), strict=True)
        except (TypeError, ValueError) as error:
            raise BenchmarkConfigurationError(
                f"invalid live_setup for case {case.id}: {error}"
            ) from error

        isolated_root = _case_data_root(data_root, case.id, run_index)
        isolated = _planning_only_config(config, isolated_root)
        registry = build_registry(isolated)
        llm = _RecordingLlmClient(isolated)
        agent = _PlanningBarrierAgent(
            isolated,
            registry,
            llm=llm,
            session_id=f"bench_live_{case.id}_{run_index}",
        )
        seeded_run = _seed_live_scenario(
            agent,
            live_setup,
            run_index=run_index,
            case_id=case.id,
            benchmark_dir=benchmark_dir,
            registry=registry,
        )
        response = agent.handle_message(case.prompt)
        return _observation_from_live_agent(
            case,
            run_index=run_index,
            response=response,
            agent=agent,
            llm=llm,
            seeded_run=seeded_run,
            started=started,
            registry=registry,
        )
    except (BenchmarkConfigurationError, LlmError, TypeError, ValueError, OSError) as error:
        if llm is None:
            return observation_from_runtime(
                case_id=case.id,
                run_index=run_index,
                status="exception",
                stage=stage,
                elapsed_seconds=time.monotonic() - started,
                error_category=(
                    error.category if isinstance(error, LlmError) else "benchmark_harness_error"
                ),
                error_message=str(error),
                registry=None,
            )
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage=stage,
            run=seeded_run,
            request=seeded_run.request if seeded_run is not None else None,
            plan=seeded_run.plan if seeded_run is not None else None,
            llm_calls=llm.calls,
            elapsed_seconds=time.monotonic() - started,
            error_category=(
                error.category if isinstance(error, LlmError) else "benchmark_harness_error"
            ),
            error_message=_redact_api_key(str(error), config.llm.api_key_env),
            error_diagnostics=_flatten_llm_errors(llm.errors),
            llm_structured_outputs=llm.structured_outputs,
        )


def _planning_only_config(config: AppConfig, data_root: Path) -> AppConfig:
    isolated = _isolated_config(config, data_root)
    runtime = isolated.runtime.model_copy(update={"confirm_before_compute": True})
    return isolated.model_copy(update={"runtime": runtime}, deep=True)


def _seed_live_scenario(
    agent: _PlanningBarrierAgent,
    setup: LiveSetup,
    *,
    run_index: int,
    case_id: str,
    benchmark_dir: str | Path,
    registry: ToolRegistry,
) -> Run | None:
    agent._session["recent_messages"] = [dict(item) for item in setup.recent_messages]
    agent._session["recent_results"] = [dict(item) for item in setup.recent_results]
    agent._session["last_delivery"] = [dict(item) for item in setup.last_delivery]
    if setup.active_run_fixture is None:
        agent._save_session()
        return None

    seed_fixture = load_fixture_reference(setup.active_run_fixture, benchmark_dir)
    _intake, request, plan = _build_contract(
        seed_fixture,
        prompt=f"benchmark live scenario seed: {case_id}",
        case_id=f"{case_id}_seed",
        run_index=run_index,
        registry=registry,
    )
    if plan is None:
        raise BenchmarkConfigurationError(
            f"active_run_fixture for {case_id} must define a validated Plan"
        )
    run = agent._create_chat_run(request, validate_request_plan(request, plan, registry))

    if setup.active_run_status == "waiting":
        step = next(
            (
                item
                for item in run.plan.steps
                if registry.get(item.tool).requires_compute_permission
            ),
            None,
        )
        if step is None:
            raise BenchmarkConfigurationError(
                f"waiting live scenario {case_id} needs a compute Tool in its Plan"
            )
        agent._prepare_confirmation(run, step)
        run.waiting_for = setup.waiting_for
        run.pending_data.update(setup.pending_data)
        save_run(agent.config.data_root_path, run)
    elif setup.active_run_status == "succeeded":
        assert setup.published_result_fixture is not None
        result_fixture_payload = load_fixture_reference(
            setup.published_result_fixture, benchmark_dir
        )
        try:
            result_fixture = LiveResultFixture.model_validate(result_fixture_payload, strict=True)
        except (TypeError, ValueError) as error:
            raise BenchmarkConfigurationError(
                f"invalid published result fixture for {case_id}: {error}"
            ) from error
        _publish_live_result_fixture(agent, run, result_fixture, benchmark_dir, registry)
    else:
        raise BenchmarkConfigurationError(f"active Run for {case_id} needs a supported status")

    agent._session["active_run_id"] = run.id
    if setup.active_run_status == "succeeded":
        result = _latest_result_for_run(agent.config.data_root_path, run)
        if result is None:
            raise BenchmarkConfigurationError(
                f"succeeded live scenario {case_id} did not publish a Result"
            )
        agent._record_result_summary(run, result)
        _seed_live_result_delivery(agent, run)
    else:
        agent._save_session()
    return run


def _seed_live_result_delivery(agent: _PlanningBarrierAgent, run: Run) -> None:
    """Mark the fixture's verified public outputs as the seeded prior delivery."""

    catalog = agent._build_query_catalog()
    locators: list[dict[str, Any]] = []
    for item in catalog:
        result = item.get("result")
        subject_ref = item.get("subject_ref")
        if not isinstance(result, dict) or not isinstance(subject_ref, str):
            continue
        property_name = result.get("property")
        binding = agent._query_bindings.get((subject_ref, property_name))
        if not isinstance(binding, dict):
            continue
        locator = {
            key: binding[key]
            for key in (
                "run_id",
                "step_id",
                "property",
                "kind",
                "attempt",
                "step_fingerprint",
            )
        }
        artifact_id = binding.get("artifact_id")
        artifact_hash = binding.get("artifact_sha256")
        if isinstance(artifact_id, str) and isinstance(artifact_hash, str):
            locator["artifact_id"] = artifact_id
            locator["sha256"] = artifact_hash
        locators.append(locator)
    if not locators:
        raise BenchmarkConfigurationError(
            f"succeeded live scenario {run.id} has no verified public outputs to deliver"
        )
    prior = agent._session.get("last_delivery", [])
    existing = (
        [dict(item) for item in prior if isinstance(item, dict)] if isinstance(prior, list) else []
    )
    agent._session["last_delivery"] = [*existing, *locators][-8:]
    agent._save_session()


def _publish_live_result_fixture(
    agent: _PlanningBarrierAgent,
    run: Run,
    fixture: LiveResultFixture,
    benchmark_dir: str | Path,
    registry: ToolRegistry,
) -> None:
    matches = [step for step in run.plan.steps if step.tool == fixture.tool]
    if len(matches) != 1:
        raise BenchmarkConfigurationError(
            "published result fixture must select exactly one Step by Tool"
        )
    step = matches[0]
    tool = registry.get(step.tool)
    if step.tool != "optimize_geometry":
        raise BenchmarkConfigurationError(
            "live benchmark saved-result fixtures currently support verified Opt Results"
        )
    geometry_path = _safe_fixture_file(fixture.output_geometry, benchmark_dir)
    bindings = {
        name: reference.artifact_id
        for name, reference in step.inputs.items()
        if reference.artifact_id is not None
    }
    if "geometry" not in bindings:
        raise BenchmarkConfigurationError("saved Opt Result fixture needs a bound geometry input")
    input_artifacts = {item.id: item for item in run.artifact_index}
    input_hashes = {
        artifact_id: sha256_file(
            artifact_path(agent.config.data_root_path, run, input_artifacts[artifact_id])
        )
        for artifact_id in bindings.values()
        if artifact_id in input_artifacts
    }
    if len(input_hashes) != len(bindings):
        raise BenchmarkConfigurationError("saved Result has an unknown input Artifact")

    output_geometry = register_file_artifact(
        agent.config.data_root_path,
        run,
        geometry_path,
        artifact_type="molecular_geometry",
        role="optimized_geometry",
        source="benchmark fixture: previously verified real ORCA Opt output",
        step_id=step.id,
        attempt=1,
        metadata={"evidence_class": fixture.evidence_class},
    )
    value = fixture.values.get("opt_final_electronic_energy")
    if not isinstance(value, dict) or type(value.get("value")) not in {int, float}:
        raise BenchmarkConfigurationError(
            "saved Opt Result fixture needs an electronic energy value"
        )
    profile = get_profile(str(step.parameters.get("method_profile", "")))
    energy_payload = {
        "schema": "bg6022.energy_data.v1",
        "property": "electronic_energy",
        "value": value["value"],
        "unit": "Eh",
        "method_profile": profile.name,
        "method_keyword": profile.orca_keyword,
        "operation": "Opt",
        "charge": step.parameters.get("charge"),
        "multiplicity": step.parameters.get("multiplicity"),
        "source": {"step_id": step.id, "attempt": 1},
        "geometry": {
            "artifact_id": output_geometry.id,
            "sha256": output_geometry.sha256,
        },
        "observation": value,
    }
    energy_artifact = register_bytes_artifact(
        agent.config.data_root_path,
        run,
        json.dumps(energy_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        artifact_type="energy_data",
        role="verified_energy_data",
        source="benchmark fixture: previously verified real ORCA energy evidence",
        extension=".json",
        step_id=step.id,
        attempt=1,
        metadata={
            "evidence_class": fixture.evidence_class,
            "geometry_sha256": output_geometry.sha256,
            "method_profile": profile.name,
            "operation": "Opt",
            "charge": step.parameters.get("charge"),
            "multiplicity": step.parameters.get("multiplicity"),
            "unit": "Eh",
        },
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        values=fixture.values,
        checks=fixture.checks,
        artifact_ids=[output_geometry.id, energy_artifact.id],
        output_ports={
            "optimized_geometry": output_geometry.id,
            "energy_data": energy_artifact.id,
        },
        input_artifact_ids=list(bindings.values()),
        input_bindings=bindings,
        attempt_relative_path=f"{step.id}/attempt-01",
    )
    run.attempts.append(
        {
            "step_id": step.id,
            "attempt": 1,
            "phase": "finished",
            "status": "succeeded",
            "artifact_ids": list(result.artifact_ids),
            "evidence_class": fixture.evidence_class,
        }
    )
    run.attempt_counts[run.origin_step_map.get(step.id, step.id)] = 1
    run.execution_permission = True
    publish_step_result(
        agent.config.data_root_path,
        run,
        step,
        tool,
        result,
        expected_input_bindings=bindings,
        expected_input_hashes=input_hashes,
    )
    run.status = "succeeded"
    run.waiting_for = None
    run.pending_data = {}
    save_run(agent.config.data_root_path, run)


def _latest_result_for_run(data_root: str | Path, run: Run) -> Result | None:
    if not run.result_index:
        return None
    relative = Path(run.result_index[-1])
    root = run_directory(data_root, run.id).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return Result.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)


def _observation_from_live_agent(
    case: BenchmarkCase,
    *,
    run_index: int,
    response: AgentResponse,
    agent: _PlanningBarrierAgent,
    llm: _RecordingLlmClient,
    seeded_run: Run | None,
    started: float,
    registry: ToolRegistry,
) -> CaseObservation:
    intake = next(
        (
            item["value"]
            for item in reversed(llm.structured_outputs)
            if item.get("purpose") == "intake" and isinstance(item.get("value"), dict)
        ),
        None,
    )
    if intake is None:
        semantic = next(
            (
                item["value"]
                for item in reversed(llm.structured_outputs)
                if item.get("purpose") == "semantic"
                and isinstance(item.get("value"), dict)
            ),
            None,
        )
        if semantic is not None:
            semantic_intents = {
                "compute": "chemistry_compute",
                "qa": "chemistry_qa",
                "context_query": "context_query",
                "modify": "modify",
                "clarify": "clarify",
                "unsupported": "unsupported",
            }
            intent = semantic_intents.get(str(semantic.get("mode", "")))
            intake = {"intent": intent} if intent is not None else None
    run = response.run or agent._coerce_run(None)
    unchanged_pending_run = bool(
        seeded_run is not None
        and run is not None
        and run.id == seeded_run.id
        and run.plan.model_dump(mode="json") == seeded_run.plan.model_dump(mode="json")
        and run.status == "waiting"
        and run.waiting_for is not None
    )
    latest_error = llm.errors[-1] if llm.errors else None
    intent = intake.get("intent") if intake is not None else None
    unresolved = (
        intake.get("unresolved_requirements") or intake.get("missing_fields")
        if intake is not None
        else None
    )
    if latest_error is not None:
        error_purpose = str(latest_error.get("purpose") or "intake")
        stage = _stage_for_llm_purpose(error_purpose)
        status = "failed"
        error_category = str(latest_error.get("category") or "llm_error")
        error_message = str(latest_error.get("message") or "language-model call failed")
    elif intent == "context_query":
        stage = "query"
        status = "completed"
        error_category = None
        error_message = None
    elif unchanged_pending_run:
        stage = "intake"
        status = "blocked"
        error_category = "clarification_required"
        error_message = response.text
    elif run is not None and run.plan is not None:
        stage = "complete"
        status = "completed"
        error_category = None
        error_message = None
    elif intent == "chemistry_compute" and unresolved:
        stage = "intake"
        status = "blocked"
        error_category = "unsupported_or_unresolved"
        error_message = response.text
    elif intent in {"chemistry_qa", "daily_qa"}:
        stage = "answer"
        status = "completed"
        error_category = None
        error_message = None
    elif intent is None:
        stage = "intake"
        status = "blocked"
        error_category = "agent_route_short_circuit"
        error_message = response.text
    else:
        stage = "planner" if intent == "chemistry_compute" else "intake"
        status = "failed" if intent == "chemistry_compute" else "blocked"
        error_category = "agent_planning_failed" if intent == "chemistry_compute" else None
        error_message = response.text if status == "failed" else None

    before_attempts = count_orca_attempts(seeded_run, registry)
    after_attempts = count_orca_attempts(run, registry)
    return observation_from_runtime(
        case_id=case.id,
        run_index=run_index,
        status=status,
        stage=stage,
        intake=intake,
        request=run.request if run is not None else None,
        plan=run.plan if run is not None else None,
        run=run,
        response_text=response.text,
        llm_calls=llm.calls,
        elapsed_seconds=time.monotonic() - started,
        error_category=error_category,
        error_message=error_message,
        error_diagnostics=_flatten_llm_errors(llm.errors),
        llm_structured_outputs=llm.structured_outputs,
        registry=registry,
    ).model_copy(
        update={
            # Historical attempts in a seeded saved-result Run are context,
            # not work performed by this live planning-only invocation.
            "orca_attempts": max(0, after_attempts - before_attempts),
            "orca_successes": 0,
            "orca_failures": 0,
        }
    )


def _flatten_llm_errors(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for error in errors:
        context = {
            "purpose": str(error.get("purpose", "json")),
            "category": str(error.get("category", "llm_error")),
        }
        diagnostics = error.get("diagnostics", [])
        if isinstance(diagnostics, list) and diagnostics:
            for item in diagnostics:
                if isinstance(item, dict):
                    flattened.append(
                        context
                        | {key: str(item[key]) for key in ("path", "message") if key in item}
                    )
        else:
            flattened.append(context | {"message": str(error.get("message", ""))})
    return flattened


def _stage_for_llm_purpose(purpose: str) -> str:
    return {
        "answer": "answer",
        "planner": "planner",
        "intake": "intake",
    }.get(purpose, "intake")


def _redact_api_key(message: str, env_name: str) -> str:
    secret = os.environ.get(env_name)
    return message.replace(secret, "[REDACTED]") if secret else message


def _run_live_orca(
    case: BenchmarkCase,
    *,
    run_index: int,
    benchmark_dir: str | Path,
    config: AppConfig,
    data_root: str | Path | None,
    fixture_override: dict[str, Any] | None = None,
) -> CaseObservation:
    started = time.monotonic()
    fixture = (
        fixture_override if fixture_override is not None else load_fixture(case, benchmark_dir)
    )
    registry = build_registry(config)
    try:
        _validate_compute_budget(config)
        isolated_root = _case_data_root(data_root, case.id, run_index)
        isolated = _isolated_config(config, isolated_root)
        if not case.requires_llm:
            isolated = isolated.model_copy(
                update={"repair": isolated.repair.model_copy(update={"enabled": False})},
                deep=True,
            )
        registry = build_registry(isolated)
        _intake, request, plan = _build_contract(
            fixture,
            prompt=case.prompt,
            case_id=case.id,
            run_index=run_index,
            registry=registry,
        )
        if plan is None:
            raise BenchmarkConfigurationError(f"live ORCA case {case.id} has no Plan fixture")
        geometry = _geometry_file(fixture, benchmark_dir)
        if fixture.get("force_missing_executable") is True:
            isolated = isolated.model_copy(
                update={"executable_path": str(isolated_root / "missing" / "orca.exe")}
            )
            registry = build_registry(isolated)
        agent = Agent(
            isolated, registry, llm=LlmClient(isolated), session_id=f"bench_{case.id}_{run_index}"
        )
        run, result = agent.execute_plan(request, plan, xyz_path=geometry, execute=True)
        status = "completed" if result.status == "succeeded" else "failed"
        stage = "complete" if status == "completed" else "execution"
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status=status,
            stage=stage,
            request=request,
            plan=run.plan,
            run=run,
            results=_load_published_results(isolated_root, run, result),
            llm_calls=agent.llm.calls,
            elapsed_seconds=time.monotonic() - started,
            error_category=result.diagnostics.get("category"),
            error_message=result.diagnostics.get("reason"),
        ).model_copy(update={"orca_attempts": count_orca_attempts(run, registry)})
    except LlmError as error:
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage="repair",
            elapsed_seconds=time.monotonic() - started,
            error_category=error.category,
            error_message=str(error),
        )
    except (OSError, ValueError, PermissionError, BenchmarkConfigurationError) as error:
        category = (
            "executable_missing"
            if isinstance(error, FileNotFoundError) or "executable" in str(error).casefold()
            else "environment_error"
        )
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="blocked" if fixture.get("force_missing_executable") else "failed",
            stage="execution",
            elapsed_seconds=time.monotonic() - started,
            error_category=category,
            error_message=str(error),
        )


def _run_live_e2e(
    case: BenchmarkCase,
    *,
    run_index: int,
    benchmark_dir: str | Path,
    config: AppConfig,
    data_root: str | Path | None,
) -> CaseObservation:
    started = time.monotonic()
    fixture = load_fixture(case, benchmark_dir)
    registry = build_registry(config)
    try:
        _validate_compute_budget(config)
        if not os.environ.get(config.llm.api_key_env):
            return observation_from_runtime(
                case_id=case.id,
                run_index=run_index,
                status="failed",
                stage="repair",
                elapsed_seconds=time.monotonic() - started,
                error_category="missing_api_key",
                error_message=f"set {config.llm.api_key_env} before starting a repair benchmark",
            )
        isolated_root = _case_data_root(data_root, case.id, run_index)
        isolated = _isolated_config(config, isolated_root)
        registry = build_registry(isolated)
        _intake, request, plan = _build_contract(
            fixture,
            prompt=case.prompt,
            case_id=case.id,
            run_index=run_index,
            registry=registry,
        )
        if plan is None:
            raise BenchmarkConfigurationError(f"live E2E case {case.id} has no Plan fixture")
        geometry = _geometry_file(fixture, benchmark_dir)
        llm = LlmClient(isolated)
        agent = Agent(isolated, registry, llm=llm, session_id=f"bench_{case.id}_{run_index}")
        run, result = agent.execute_plan(request, plan, xyz_path=geometry, execute=True)
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="completed" if result.status == "succeeded" else "failed",
            stage="complete" if result.status == "succeeded" else "repair",
            request=request,
            plan=run.plan,
            run=run,
            results=_load_published_results(isolated_root, run, result),
            response_text=None,
            llm_calls=llm.calls,
            elapsed_seconds=time.monotonic() - started,
            error_category=result.diagnostics.get("category"),
            error_message=result.diagnostics.get("reason"),
        ).model_copy(update={"orca_attempts": count_orca_attempts(run, registry)})
    except LlmError as error:
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage="repair",
            elapsed_seconds=time.monotonic() - started,
            error_category=error.category,
            error_message=str(error),
        )
    except (OSError, ValueError, PermissionError, BenchmarkConfigurationError) as error:
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage="execution",
            elapsed_seconds=time.monotonic() - started,
            error_category="environment_error",
            error_message=str(error),
        )


def _replay_result(
    *,
    case: BenchmarkCase,
    run_index: int,
    fixture: dict[str, Any],
    request: Request,
    plan: Plan | None,
    registry: ToolRegistry,
    config: AppConfig | None,
    benchmark_dir: str | Path,
    data_root: str | Path | None,
) -> tuple[Run, Result]:
    if plan is None:
        raise BenchmarkConfigurationError("result replay requires a validated Plan")
    if data_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="bg6022-benchmark-replay-")
        root = Path(temporary_root.name)
    else:
        root = _case_data_root(data_root, case.id, run_index)
    root.mkdir(parents=True, exist_ok=True)
    geometry_path = _geometry_file(fixture, benchmark_dir)
    run = Run(
        id=f"benchmark_{case.id}_{run_index}",
        request=request,
        plan=plan,
        resources=config.resources if config is not None else {},
        status="planned",
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    create_run(root, run)
    input_artifact = register_file_artifact(
        root,
        run,
        geometry_path,
        artifact_type="molecular_geometry",
        role="input_geometry",
        source=f"benchmark fixture: {geometry_path.name}",
    )
    steps = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        for name, ref in list(inputs.items()):
            if ref.artifact_id == INPUT_GEOMETRY_PLACEHOLDER:
                from bg6022.models import InputReference

                inputs[name] = InputReference(artifact_id=input_artifact.id)
        steps.append(step.model_copy(update={"inputs": inputs}))
    run.plan = plan.model_copy(update={"steps": steps})
    step_key = str(fixture["result"].get("step_key", ""))
    step = next(
        (item for item in run.plan.steps if item.id == step_key or item.origin_step_id == step_key),
        None,
    )
    if step is None:
        raise BenchmarkConfigurationError(f"result fixture names an unknown Step: {step_key}")
    tool = registry.get(step.tool)
    raw_result = dict(fixture["result"])
    for field in ("step_key", "evidence_class"):
        raw_result.pop(field, None)
    raw_result.update(
        {
            "run_id": run.id,
            "step_id": step.id,
            "attempt": int(raw_result.get("attempt", 1)),
            "attempt_relative_path": f"{step.id}/attempt-{int(raw_result.get('attempt', 1)):02d}",
        }
    )
    if (
        raw_result.get("status") == "succeeded"
        and fixture["result"].get("evidence_class") != "real_orca_fixture"
    ):
        raise BenchmarkConfigurationError(
            "a replayed successful Result needs real ORCA fixture provenance"
        )
    bindings = {
        name: ref.artifact_id for name, ref in step.inputs.items() if ref.artifact_id is not None
    }
    raw_result["input_bindings"] = bindings
    raw_result["input_artifact_ids"] = list(bindings.values())
    result = Result.model_validate(raw_result, strict=True)
    artifacts = {item.id: item for item in run.artifact_index}
    input_hashes = {
        artifact_id: sha256_file(artifact_path(root, run, artifacts[artifact_id]))
        for artifact_id in bindings.values()
    }
    publish_step_result(
        root,
        run,
        step,
        tool,
        result,
        expected_input_bindings=bindings,
        expected_input_hashes=input_hashes,
    )
    return run, result


def _load_published_results(data_root: str | Path, run: Run, fallback: Result) -> list[Result]:
    """Read the complete published Result history from the isolated Run."""

    run_root = run_directory(data_root, run.id).resolve()
    results: list[Result] = []
    for relative in run.result_index:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError("published Result paths must stay inside the Run directory")
        candidate = run_root / relative_path
        path = candidate.resolve()
        if run_root not in path.parents or not path.is_file():
            raise ValueError("published Result path escapes the Run directory")
        current = candidate
        while current != run_root:
            if current.is_symlink():
                raise ValueError("published Result paths cannot traverse a symlink")
            current = current.parent
        result = Result.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)
        if result.run_id != run.id:
            raise ValueError("published Result belongs to a different Run")
        results.append(result)
    if not results:
        results.append(fallback)
    return results


def _replay_artifact_input(
    *,
    case: BenchmarkCase,
    run_index: int,
    fixture: dict[str, Any],
    request: Request,
    plan: Plan | None,
    config: AppConfig | None,
    benchmark_dir: str | Path,
    data_root: str | Path | None,
) -> tuple[Run, Plan]:
    if plan is None:
        raise BenchmarkConfigurationError("Artifact replay requires a validated Plan")
    specification = fixture["replay_artifact"]
    if not isinstance(specification, dict):
        raise BenchmarkConfigurationError("replay_artifact must be an object")
    source_path = _safe_fixture_file(str(specification.get("path", "")), benchmark_dir)
    temporary_root = None
    if data_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="bg6022-benchmark-artifact-")
        root = Path(temporary_root.name)
    else:
        root = _case_data_root(data_root, case.id, run_index)
    root.mkdir(parents=True, exist_ok=True)
    run = Run(
        id=f"benchmark_{case.id}_{run_index}",
        request=request,
        plan=plan,
        resources=config.resources if config is not None else {},
        status="planned",
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    create_run(root, run)
    artifact = register_file_artifact(
        root,
        run,
        source_path,
        artifact_type=str(specification.get("artifact_type", "molecular_geometry")),
        role=str(specification.get("role", "input_geometry")),
        source=str(specification.get("source", f"benchmark replay fixture: {source_path.name}")),
        metadata={"evidence_class": "replay_fixture", "no_calculation_performed": True},
    )
    alias = str(specification.get("alias", "initial_geometry"))
    declared_alias = _artifact_aliases(fixture).get(alias)
    replacement_ids = {declared_alias, alias, INPUT_GEOMETRY_PLACEHOLDER}
    steps = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        for name, reference in inputs.items():
            if reference.artifact_id in replacement_ids:
                inputs[name] = InputReference(artifact_id=artifact.id)
        steps.append(step.model_copy(update={"inputs": inputs}))
    plan = plan.model_copy(update={"steps": steps})
    run.plan = plan
    save_run(root, run)
    return run, plan


def _geometry_file(fixture: dict[str, Any], benchmark_dir: str | Path) -> Path:
    relative = fixture.get("geometry")
    if not isinstance(relative, str):
        raise BenchmarkConfigurationError("fixture must declare a geometry file for execution")
    return _safe_fixture_file(relative, benchmark_dir)


def _safe_fixture_file(relative: str, benchmark_dir: str | Path) -> Path:
    root = Path(benchmark_dir).resolve()
    raw_candidate = root / relative
    current = raw_candidate
    while current != current.parent:
        if current.is_symlink():
            raise BenchmarkConfigurationError("fixture paths cannot traverse a symlink")
        current = current.parent
    candidate = raw_candidate.resolve()
    if root not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
        raise BenchmarkConfigurationError(
            "fixture file must be a regular file inside benchmark data"
        )
    return candidate


def _case_data_root(base: str | Path | None, case_id: str, run_index: int) -> Path:
    if base is None:
        root = Path(tempfile.gettempdir()) / "BG6022-v3-benchmark-data"
    else:
        root = Path(base)
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    return root.resolve() / f"bench_{run_index}_{digest}"


def _default_data_root(benchmark_dir: str | Path) -> Path:
    project_root = Path(benchmark_dir).resolve().parents[1]
    return project_root.parent / f"{project_root.name}-benchmark-data"


def _isolated_config(config: AppConfig, data_root: Path) -> AppConfig:
    data_root.mkdir(parents=True, exist_ok=True)
    runtime = config.runtime.model_copy(update={"data_root": str(data_root)})
    return config.model_copy(
        update={"runtime": runtime, "data_root_path": str(data_root)},
        deep=True,
    )


def _validate_compute_budget(config: AppConfig) -> None:
    runtime = config.runtime
    expected = {"cores": 4, "memory_mb": 1024, "maxcore_mb": 192, "max_concurrent_jobs": 1}
    actual = {name: getattr(runtime, name) for name in expected}
    if actual != expected:
        raise BenchmarkConfigurationError(
            f"live ORCA benchmark budget must match the active project "
            f"budget {expected}; got {actual}"
        )


def _required_config(config: AppConfig | None) -> AppConfig:
    if config is None:
        raise BenchmarkModeDisabled("live benchmark cases require an AppConfig")
    return config


__all__ = [
    "BenchmarkModeDisabled",
    "run_case",
]
