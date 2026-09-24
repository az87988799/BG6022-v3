"""Run Benchmark 1.2 items through the public Agent conversation surface."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from bg6022.agent import Agent, AgentResponse
from bg6022.config import AppConfig
from bg6022.llm import LlmClient
from bg6022.models import Result, Run
from bg6022.session import load_run, run_directory
from bg6022.tools.registry import ToolRegistry, build_registry

from .dataset12 import Benchmark12Item, ScenarioTurn
from .models import Benchmark12Observation, Benchmark12TurnObservation
from .observers import count_orca_attempts, orca_attempt_outcomes


class Benchmark12DriverError(RuntimeError):
    """The public conversation harness exceeded its bounded interaction contract."""


class PublicConversationDriver:
    """Drive only ``Agent.handle_message`` and record the public responses."""

    def __init__(
        self,
        agent: Agent,
        llm: Any,
        registry: ToolRegistry,
        *,
        max_turns: int = 8,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.agent = agent
        self.llm = llm
        self.registry = registry
        self.max_turns = max_turns
        self.turns: list[Benchmark12TurnObservation] = []
        self.responses: list[AgentResponse] = []
        self.latest_run: Run | None = None

    def run(self, script: Sequence[ScenarioTurn]) -> list[Benchmark12TurnObservation]:
        for script_turn in script:
            if len(self.turns) >= self.max_turns:
                raise Benchmark12DriverError(
                    f"conversation exceeded the {self.max_turns}-turn bound"
                )
            previous_run = self.latest_run
            response = self._send(script_turn.message_template)
            if (
                previous_run is not None
                and previous_run.status == "waiting"
                and previous_run.waiting_for == "confirmation"
                and response.run is not None
                and response.run.id == previous_run.id
                and response.run.waiting_for == "confirmation"
                and _plan_signature(response.run) == _plan_signature(previous_run)
                and script_turn.message_template.strip().casefold()
                not in {"确认", "/confirm", "confirm"}
            ):
                # A clarification or rejected patch must not be turned into an
                # implicit approval merely because the previous revision waited.
                break
            if script_turn.confirm_after and response.run is not None:
                if response.run.status == "waiting" and response.run.waiting_for == "confirmation":
                    if len(self.turns) >= self.max_turns:
                        raise Benchmark12DriverError(
                            "turn bound leaves no room for the public confirmation message"
                        )
                    response = self._send("确认")
            if script_turn.stop_after:
                break
        return list(self.turns)

    def _send(self, message: str) -> AgentResponse:
        call_start = len(getattr(self.llm, "calls", []))
        response = self.agent.handle_message(message)
        calls = list(getattr(self.llm, "calls", []))[call_start:]
        run = response.run
        if run is not None:
            self.latest_run = run
        observed_run = self.latest_run
        self.responses.append(response)
        self.turns.append(
            Benchmark12TurnObservation(
                index=len(self.turns) + 1,
                user_message=message,
                response_text=response.text[:20000],
                run_id=observed_run.id if observed_run is not None else None,
                run_status=observed_run.status if observed_run is not None else None,
                waiting_for=observed_run.waiting_for if observed_run is not None else None,
                llm_purposes=[str(_dump(call).get("purpose", "unknown")) for call in calls],
                orca_attempts_after_turn=count_orca_attempts(observed_run, self.registry),
                delivery_status=(
                    str(response.delivery.get("status"))
                    if response.delivery.get("status") is not None
                    else None
                ),
                delivery_properties=_delivery_properties(response.delivery),
            )
        )
        return response


def run_item(
    item: Benchmark12Item,
    *,
    config: AppConfig,
    data_root: str | Path,
    run_index: int,
    max_turns: int = 8,
) -> Benchmark12Observation:
    """Execute one item with an isolated Agent session and production registry."""

    started = time.monotonic()
    isolated_root = case_data_root(data_root, item.id, run_index)
    isolated_root.mkdir(parents=True, exist_ok=False)
    runtime = config.runtime.model_copy(update={"data_root": str(isolated_root)})
    isolated_config = config.model_copy(
        update={"runtime": runtime, "data_root_path": str(isolated_root)},
        deep=True,
    )
    registry = build_registry(isolated_config)
    llm = LlmClient(isolated_config)
    session_id = f"bench12_{hashlib.sha256(item.id.encode('utf-8')).hexdigest()[:16]}_{run_index}"
    agent = Agent(isolated_config, registry, llm=llm, session_id=session_id)
    driver = PublicConversationDriver(agent, llm, registry, max_turns=max_turns)
    script = item.script or (
        ScenarioTurn(
            message_template=item.prompt,
            confirm_after=item.ground_truth.require_confirmation
            and item.ground_truth.route in {"compute", "compute_then_query"},
        ),
    )
    driver.run(script)
    return observe_full_pipeline(
        item,
        driver,
        llm,
        registry,
        data_root=isolated_root,
        run_index=run_index,
        elapsed_seconds=time.monotonic() - started,
    )


def observe_full_pipeline(
    item: Benchmark12Item,
    driver: PublicConversationDriver,
    llm: Any,
    registry: ToolRegistry,
    *,
    data_root: str | Path,
    run_index: int,
    elapsed_seconds: float,
) -> Benchmark12Observation:
    """Snapshot the evidence produced by the real Agent/Tool/Run pipeline."""

    run = driver.latest_run
    if run is not None:
        try:
            run = load_run(data_root, run.id)
        except ValueError:
            # In-memory state remains useful for failures before the first checkpoint.
            pass
    results = _published_results(data_root, run) if run is not None else []
    last_response = driver.responses[-1] if driver.responses else None
    delivery: dict[str, Any] = {}
    if last_response is not None:
        delivery = dict(last_response.delivery)
    if not delivery and isinstance(getattr(driver.agent, "_session", None), dict):
        session_delivery = driver.agent._session.get("last_delivery", [])
        if isinstance(session_delivery, list) and session_delivery:
            delivery = {"status": "complete", "outputs": session_delivery}
    calls = [_dump(call) for call in getattr(llm, "calls", [])]
    purposes = [str(call.get("purpose", "unknown")) for call in calls]
    legacy_used = any(purpose in {"intake", "planner"} for purpose in purposes)
    attempt_count, orca_successes, orca_failures = orca_attempt_outcomes(run, registry)
    confirmation_turn_index = next(
        (
            turn.index
            for turn in driver.turns
            if turn.user_message.strip().casefold() in {"确认", "/confirm", "confirm"}
        ),
        None,
    )
    pre_confirmation = attempt_count
    if confirmation_turn_index is not None:
        previous = [
            turn.orca_attempts_after_turn
            for turn in driver.turns
            if turn.index < confirmation_turn_index
        ]
        pre_confirmation = previous[-1] if previous else 0
    properties = _delivery_properties(delivery)
    failed_stage, error_category = _failure_facts(item, run, calls, last_response)
    external_calls = _external_call_counts(run, results, registry)
    return Benchmark12Observation(
        item_id=item.id,
        run_index=run_index,
        turns=driver.turns,
        request=run.request.model_dump(mode="json") if run is not None else None,
        plan=run.plan.model_dump(mode="json") if run is not None else None,
        results=results,
        artifacts=[artifact.model_dump(mode="json") for artifact in run.artifact_index]
        if run is not None
        else [],
        final_response_text=last_response.text[:20000] if last_response is not None else None,
        llm_calls=calls,
        orca_attempts=attempt_count,
        orca_successes=orca_successes,
        orca_failures=orca_failures,
        repair_attempts=len(run.repair_records) if run is not None else 0,
        final_status=run.status if run is not None else _no_run_status(item, last_response),
        failed_stage=failed_stage,
        error_category=error_category,
        pre_confirmation_orca_attempts=pre_confirmation,
        post_confirmation_orca_attempts=max(0, attempt_count - pre_confirmation)
        if confirmation_turn_index is not None
        else 0,
        confirmation_turn_index=confirmation_turn_index,
        confirmation_required=any(turn.waiting_for == "confirmation" for turn in driver.turns),
        public_delivery_properties=properties,
        delivery_status=str(delivery.get("status")) if delivery.get("status") else None,
        legacy_control_plane_used=legacy_used,
        external_calls=external_calls,
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
    )


def failed_observation(
    item: Benchmark12Item,
    run_index: int,
    *,
    error: BaseException,
    failed_stage: str = "harness",
) -> Benchmark12Observation:
    """Create bounded evidence when an item fails before an Agent is available."""

    return Benchmark12Observation(
        item_id=item.id,
        run_index=run_index,
        turns=[],
        final_status="exception",
        failed_stage=failed_stage,
        error_category="benchmark_harness_failure",
        final_response_text=str(error)[:1000],
    )


def case_data_root(data_root: str | Path, item_id: str, run_index: int) -> Path:
    if run_index < 1:
        raise ValueError("run_index must be positive")
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
    return Path(data_root).resolve() / f"bench12_{digest}_{run_index:02d}"


def _published_results(data_root: str | Path, run: Run) -> list[dict[str, Any]]:
    root = run_directory(data_root, run.id).resolve()
    snapshots: list[dict[str, Any]] = []
    for relative in run.result_index:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise Benchmark12DriverError("Run result index contains an absolute path")
        candidate = root / relative_path
        resolved = candidate.resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise Benchmark12DriverError("Run result index escapes its Run directory")
        current = candidate
        while current != root:
            if current.is_symlink():
                raise Benchmark12DriverError("Run result index traverses a symlink")
            current = current.parent
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            result = Result.model_validate(payload, strict=True)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise Benchmark12DriverError(f"cannot observe published Result: {error}") from error
        if result.run_id != run.id:
            raise Benchmark12DriverError("published Result belongs to another Run")
        snapshots.append(result.model_dump(mode="json"))
    return snapshots


def _delivery_properties(delivery: dict[str, Any]) -> list[str]:
    outputs = delivery.get("outputs", [])
    if not isinstance(outputs, list):
        return []
    properties = [
        str(item.get("property"))
        for item in outputs
        if isinstance(item, dict) and isinstance(item.get("property"), str)
    ]
    return sorted(set(properties))


def _external_call_counts(
    run: Run | None,
    results: list[dict[str, Any]],
    registry: ToolRegistry,
) -> dict[str, int]:
    counts = {"pubchem_requests": 0, "rdkit_geometry_generations": 0, "orca": 0}
    if run is None:
        return counts
    steps = {step.id: step for step in run.plan.steps}
    for attempt in run.attempts:
        step = steps.get(str(attempt.get("step_id", "")))
        if step is None:
            continue
        if step.tool == "resolve_molecule":
            counts["pubchem_requests"] += 1
        elif step.tool == "generate_geometry":
            counts["rdkit_geometry_generations"] += 1
        elif registry.get(step.tool).execution_budget == "electronic_structure":
            counts["orca"] += 1
    diagnostic_requests = 0
    has_request_diagnostics = False
    for result in results:
        diagnostics = result.get("diagnostics", {})
        attempts = diagnostics.get("lookup_attempts") if isinstance(diagnostics, dict) else None
        if type(attempts) is int and attempts > 0:
            diagnostic_requests += attempts
            has_request_diagnostics = True
    if has_request_diagnostics:
        counts["pubchem_requests"] = diagnostic_requests
    return counts


def _failure_facts(
    item: Benchmark12Item,
    run: Run | None,
    calls: list[dict[str, Any]],
    last_response: AgentResponse | None,
) -> tuple[str | None, str | None]:
    if run is None:
        categories = [str(call.get("category", "")) for call in calls]
        external = {
            "missing_api_key",
            "auth",
            "network_error",
            "rate_limited",
            "temporary_failure",
            "timeout",
            "http_error",
        }
        if any(category in external for category in categories):
            return "semantic", "external_dependency_failure"
        text = last_response.text.casefold() if last_response is not None else ""
        if item.ground_truth.route == "unsupported" and text:
            return None, None
        if item.ground_truth.route == "clarify" and text:
            return None, None
        return "semantic", "product_failure"
    if run.status not in {"failed", "cancelled", "interrupted"}:
        return None, None
    category = str(run.pending_data.get("category") or "")
    llm_external = {
        "missing_api_key",
        "auth",
        "network_error",
        "rate_limited",
        "temporary_failure",
        "timeout",
    }
    if any(str(call.get("category", "")) in llm_external for call in calls):
        return _stage_for_run(run), "external_dependency_failure"
    if category in {
        "executable_missing",
        "environment_error",
        "resource_limit",
        "orca_environment",
    }:
        return "execution", "execution_environment_failure"
    if category in {"scientific_check", "goal_check_not_met", "unverified_result"}:
        return "scientific_check", "product_failure"
    if category in {"pubchem_network_error", "pubchem_timeout", "network_error", "timeout"}:
        return "identity_resolution", "external_dependency_failure"
    return _stage_for_run(run), "product_failure"


def _stage_for_run(run: Run) -> str:
    tool_by_step = {step.id: step.tool for step in run.plan.steps}
    failed = next(
        (
            str(attempt.get("step_id"))
            for attempt in reversed(run.attempts)
            if attempt.get("status") in {"failed", "cancelled", "interrupted"}
        ),
        None,
    )
    tool = tool_by_step.get(failed or "")
    if tool == "resolve_molecule":
        return "identity_resolution"
    if tool == "generate_geometry":
        return "geometry_generation"
    if tool is not None:
        return "execution"
    return "result_publish"


def _no_run_status(item: Benchmark12Item, response: AgentResponse | None) -> str:
    if response is None:
        return "not_started"
    text = response.text.casefold()
    if item.ground_truth.route == "unsupported" and any(
        marker in text for marker in ("不支持", "unsupported", "未启动计算", "没有启动计算")
    ):
        return "unsupported"
    if item.ground_truth.route == "clarify" and any(
        marker in text for marker in ("请明确", "请说明", "请直接说明", "需要明确")
    ):
        return "clarify"
    return "no_run"


def _plan_signature(run: Run) -> str:
    return json.dumps(
        run.plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


__all__ = [
    "Benchmark12DriverError",
    "PublicConversationDriver",
    "case_data_root",
    "failed_observation",
    "observe_full_pipeline",
    "run_item",
]
