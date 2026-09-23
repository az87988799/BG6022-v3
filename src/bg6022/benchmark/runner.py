"""Execute offline, replay, and explicitly enabled live benchmark cases."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bg6022.agent import INPUT_GEOMETRY_PLACEHOLDER, Agent
from bg6022.config import AppConfig
from bg6022.llm import LlmClient, LlmError
from bg6022.models import InputReference, Plan, Request, Result, Run
from bg6022.planner import (
    IntakeOutput,
    PlanProposal,
    intake_message,
    plan_message,
    proposal_to_plan,
    request_from_intake,
    validate_request_plan,
)
from bg6022.session import (
    artifact_path,
    create_run,
    publish_step_result,
    register_file_artifact,
    run_directory,
    save_run,
    sha256_file,
)
from bg6022.tools.registry import ToolRegistry, build_registry

from .loader import BenchmarkConfigurationError, load_fixture
from .models import BenchmarkCase, CaseObservation
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
) -> list[CaseObservation]:
    """Run the requested repetitions; live work is impossible without explicit flags."""

    _check_live_permissions(
        case,
        allow_live_llm=allow_live_llm,
        allow_live_orca=allow_live_orca,
        allow_live_pubchem=allow_live_pubchem,
        config=config,
    )
    if config is not None and case.requires_orca:
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
            observation = _run_live_llm(case, run_index=run_index, config=_required_config(config))
        elif case.mode == "live_orca":
            observation = _run_live_orca(
                case,
                run_index=run_index,
                benchmark_dir=benchmark_dir,
                config=_required_config(config),
                data_root=data_root,
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


def _run_live_llm(case: BenchmarkCase, *, run_index: int, config: AppConfig) -> CaseObservation:
    started = time.monotonic()
    registry = build_registry(config)
    client = LlmClient(config)
    stage = "intake"
    intake = None
    request = None
    plan = None
    try:
        intake = intake_message(
            client,
            case.prompt,
            result_catalog=registry.result_capabilities(),
            geometry_catalog=[],
            capability_catalog=registry.result_capabilities(),
            registry=registry,
        )
        request = request_from_intake(
            case.prompt,
            intake,
            request_id=f"benchmark_{case.id}_{run_index}",
            registry=registry,
        )
        if request.requirements and not request.missing_fields:
            stage = "planner"
            proposal = plan_message(client, request, registry=registry)
            stage = "plan_validation"
            plan = proposal_to_plan(
                request,
                proposal,
                registry,
                plan_id=f"plan_{case.id}_{run_index}",
                artifact_aliases={
                    "initial_geometry": INPUT_GEOMETRY_PLACEHOLDER,
                    "provided_geometry": INPUT_GEOMETRY_PLACEHOLDER,
                    "input_geometry": INPUT_GEOMETRY_PLACEHOLDER,
                },
            )
        status = "blocked" if plan is None or request.missing_fields else "completed"
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status=status,
            stage="complete" if status == "completed" else "answer",
            intake=intake,
            request=request,
            plan=plan,
            response_text=intake.answer,
            llm_calls=client.calls,
            elapsed_seconds=time.monotonic() - started,
        )
    except LlmError as error:
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage=stage,
            intake=intake,
            request=request,
            plan=plan,
            llm_calls=client.calls,
            elapsed_seconds=time.monotonic() - started,
            error_category=error.category,
            error_message=str(error),
        )
    except (TypeError, ValueError, OSError) as error:
        return observation_from_runtime(
            case_id=case.id,
            run_index=run_index,
            status="failed",
            stage=stage,
            intake=intake,
            request=request,
            plan=plan,
            llm_calls=client.calls,
            elapsed_seconds=time.monotonic() - started,
            error_category="validation_error",
            error_message=str(error),
        )


def _run_live_orca(
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
    return root.resolve() / f"bench_{case_id}_{run_index}"


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
