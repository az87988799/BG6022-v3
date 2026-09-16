"""The single Plan -> Tool -> Result loop used by CLI and chat."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Event
from typing import Any

from .answer import (
    facts_from_result,
    render_already_finished,
    render_clarification,
    render_confirmation,
    render_result,
    render_run,
    render_selected_facts,
    select_facts_for_question,
)
from .config import AppConfig, validate_execution_environment
from .llm import LlmClient, LlmError
from .models import InputReference, Plan, Request, Result, Run, Step, Tool
from .orca.profiles import get_profile, resolve_parameters
from .orca.repair_rules import applicable_repairs, applicable_scf_repair
from .planner import (
    QuerySelection,
    electronic_state_clarification,
    intake_blocking_requirements,
    intake_message,
    normalize_user_explicit_parameters,
    plan_message,
    proposal_to_plan,
    request_from_intake,
    validate_request_plan,
)
from .repair import apply_repair_proposal, propose_repair
from .session import (
    MAX_RECENT_RUNS,
    artifact_path,
    create_run,
    execution_fingerprint,
    find_artifact,
    load_run,
    load_session,
    new_id,
    register_bytes_artifact,
    register_file_artifact,
    run_directory,
    save_result,
    save_run,
    save_session,
    utc_now,
)
from .tools.molecule import parse_xyz_bytes, validate_electronic_state
from .tools.registry import ToolRegistry, merge_explicit_step_parameters

INPUT_GEOMETRY_PLACEHOLDER = "__input_geometry__"
MAX_QUERY_CATALOG_ITEMS = 24
MAX_HISTORY_GEOMETRIES = 8


@dataclass(frozen=True)
class AgentResponse:
    text: str
    run: Run | None = None
    result: Result | None = None


class Agent:
    """Advance one durable Run; model output never executes outside Tools."""

    def __init__(
        self,
        config: AppConfig,
        registry: ToolRegistry,
        *,
        llm: LlmClient | Any | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.llm = llm if llm is not None else LlmClient(config)
        self.session_id = session_id or new_id("session")
        self._cancel_events: dict[str, Event] = {}
        self._query_bindings: dict[tuple[str, str], dict[str, Any]] = {}
        self._request_sequence = 0
        self._active_request: tuple[int, Event] | None = None
        try:
            self._session = load_session(config.data_root_path, self.session_id)
        except ValueError:
            # A corrupt session is not overwritten.  The caller can use /new.
            self._session = {
                "session_id": self.session_id,
                "recent_messages": [],
                "recent_results": [],
                "active_run_id": None,
                "pending_prompt": "session record is invalid; use /new or /exit",
            }

    def execute_plan(
        self,
        request: Request,
        plan: Plan,
        *,
        xyz_path: str | Path,
        cancel: Event | None = None,
        execute: bool = False,
    ) -> tuple[Run, Result]:
        """Compatibility wrapper for explicit CLI executions."""

        if not request.id or plan.request_id != request.id:
            raise ValueError("Plan.request_id must match Request.id")
        if not plan.steps:
            raise ValueError("Plan must contain at least one Step")
        if not any(step.inputs.get("geometry") is not None for step in plan.steps):
            raise ValueError("explicit ORCA execution requires a geometry input")
        if not execute:
            raise PermissionError("execution permission was not explicitly granted")
        plan = validate_request_plan(request, plan, self.registry)
        _validate_request_parameter_scope(request, plan, self.registry)
        # The explicit command has always checked the machine before creating a
        # Run. Chat intentionally defers this check until an ORCA Tool starts.
        if any(self.registry.get(step.tool).execution_budget == "orca" for step in plan.steps):
            validate_execution_environment(self.config)
        geometry_path = Path(xyz_path).resolve()
        parse_xyz_bytes(geometry_path.read_bytes())
        run = Run(
            id=new_id("run"),
            request=request,
            plan=plan,
            resources=self.config.resources,
            execution_permission=True,
            status="planned",
            budget=self._default_budget(),
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        create_run(self.config.data_root_path, run)
        initial_artifact = register_file_artifact(
            self.config.data_root_path,
            run,
            geometry_path,
            artifact_type="molecular_geometry",
            role="input_geometry",
            source=str(geometry_path),
            metadata={"imported_as_raw_bytes": True},
        )
        bound_plan = _bind_input_geometry(plan, initial_artifact.id)
        run.plan = _normalize_explicit_plan(self.registry, bound_plan)
        run.origin_step_map = {step.id: step.origin_step_id or step.id for step in run.plan.steps}
        run.accepted_snapshot = self._acceptance_snapshot(run)
        run.accepted_execution_sha256 = execution_fingerprint(
            run.plan,
            run.resources,
            run.artifact_index,
            snapshot=run.accepted_snapshot,
        )
        save_run(self.config.data_root_path, run)
        result = self.advance(run, cancel=cancel)
        if result is None:
            raise ValueError("explicit CLI execution unexpectedly paused")
        return run, result

    def handle_message(self, message: str) -> AgentResponse:
        """Process one chat message synchronously; the CLI may call this worker-side."""

        text = message
        command = message.strip()
        if not command:
            return AgentResponse("请输入请求。")
        if command.casefold() in {"/confirm", "confirm", "确认"}:
            return self.confirm()
        if command.casefold() in {"/cancel", "cancel", "取消"}:
            return self.cancel()
        if command.casefold() in {"/status", "status", "状态"}:
            return self.status()
        if command.casefold() in {"/new", "new", "新任务"}:
            return self.new_session()

        request_token, request_cancel = self._begin_request()
        self._append_message("user", text)
        llm_call_cursor = self._llm_call_count()
        llm_stage = "intake"
        try:
            self._ensure_request_active(request_token, request_cancel)
            result_catalog = self._build_query_catalog()
            geometry_catalog, geometry_bindings = self._build_geometry_catalog()
            capability_catalog = self.registry.result_capabilities()
            intake = intake_message(
                self.llm,
                text,
                context={
                    "recent_messages": self._session.get("recent_messages", []),
                    "recent_results": self._session.get("recent_results", []),
                },
                result_catalog=result_catalog,
                geometry_catalog=geometry_catalog,
                capability_catalog=capability_catalog,
                registry=self.registry,
                cancel=request_cancel,
            )
            self._persist_llm_diagnostics(llm_call_cursor, stage="intake")
            llm_call_cursor = self._llm_call_count()
            self._ensure_request_active(request_token, request_cancel)

            blocking = intake_blocking_requirements(intake, self.registry)
            if blocking:
                response = AgentResponse(
                    "本次请求还有尚未支持或尚未明确的要求："
                    + "；".join(blocking)
                    + "。请明确这些要求，或重新指定只计算已支持的部分。"
                )
                self._append_message("assistant", response.text)
                return response

            selected_geometry_alias = intake.history_geometry_alias
            history_geometry_requested = _requests_history_geometry(text)
            if selected_geometry_alias is not None and not history_geometry_requested:
                raise ValueError(
                    "a historical geometry can be selected only when the user explicitly "
                    "requests reuse of a previous structure"
                )
            if (
                intake.intent == "chemistry_compute"
                and history_geometry_requested
                and len(geometry_catalog) > 1
            ):
                # The intake model can help interpret a request, but it must not
                # choose among multiple historical structures on the user's behalf.
                explicit_alias = _explicit_history_geometry_alias(text, geometry_catalog)
                if explicit_alias is None:
                    options = "\n".join(
                        f"- {item['alias']}: {item['description']} "
                        f"({_query_system_label(item.get('system', {}))})"
                        for item in geometry_catalog
                    )
                    response = AgentResponse(
                        "当前会话有多个可复用的成功优化结构，请明确选择一个结构别名后再继续：\n"
                        f"{options}\n回复如“复用 geometry_1”。"
                    )
                    self._append_message("assistant", response.text)
                    return response
                selected_geometry_alias = explicit_alias
                intake = intake.model_copy(
                    update={"history_geometry_alias": selected_geometry_alias}
                )
            if (
                selected_geometry_alias is None
                and intake.intent == "chemistry_compute"
                and history_geometry_requested
            ):
                if not geometry_catalog:
                    response = AgentResponse(
                        "当前会话没有找到可安全复用的成功优化结构；我没有改用新结构或启动计算。"
                    )
                    self._append_message("assistant", response.text)
                    return response
                if len(geometry_catalog) > 1:
                    response = AgentResponse(
                        "当前会话有多个可复用的成功优化结构，请说明要使用哪一个分子或任务。"
                    )
                    self._append_message("assistant", response.text)
                    return response
                selected_geometry_alias = str(geometry_catalog[0]["alias"])
                intake = intake.model_copy(
                    update={"history_geometry_alias": selected_geometry_alias}
                )
            if selected_geometry_alias is not None:
                if intake.intent != "chemistry_compute":
                    raise ValueError(
                        "a historical geometry alias can only be used for a calculation"
                    )
                if selected_geometry_alias not in geometry_bindings:
                    raise ValueError(
                        "intake selected a history geometry outside the verified catalog"
                    )
                if intake.structure_input.get("xyz_text") or intake.structure_input.get("xyz"):
                    raise ValueError(
                        "a request cannot combine a historical geometry alias with inline XYZ"
                    )

            current = self._coerce_run(None)
            normalized_parameters = normalize_user_explicit_parameters(
                text,
                intake.explicit_parameters,
                intake.electronic_state_candidates,
            )
            explicit_parameters = normalized_parameters.explicit_parameters
            if normalized_parameters.clarification_fields and (
                intake.intent == "chemistry_compute"
                or (
                    current is not None
                    and current.status == "waiting"
                    and current.waiting_for in {"clarification", "confirmation"}
                )
            ):
                response = AgentResponse(
                    electronic_state_clarification(normalized_parameters), run=current
                )
                self._append_message("assistant", response.text)
                return response
            if (
                current is not None
                and current.status == "waiting"
                and intake.molecule_query
                and current.waiting_for == "clarification"
                and current.pending_data.get("category") == "ambiguous_molecule"
                and not _looks_like_molecule_change(text)
            ):
                response = self._apply_molecule_clarification(
                    current,
                    intake.molecule_query,
                    intake.molecule_input_kind,
                    cancel=request_cancel,
                )
                self._append_message("assistant", response.text)
                return response
            if (
                current is not None
                and current.status == "waiting"
                and _is_parameter_continuation(current, intake, text, explicit_parameters)
            ):
                response = self._apply_parameter_update(
                    current, explicit_parameters, cancel=request_cancel
                )
                self._append_message("assistant", response.text)
                return response
            if (
                current is not None
                and current.status == "waiting"
                and intake.molecule_query
                and _looks_like_molecule_change(text)
            ):
                response = self._apply_molecule_update(
                    current,
                    intake.molecule_query,
                    intake.molecule_input_kind,
                    cancel=request_cancel,
                )
                self._append_message("assistant", response.text)
                return response

            if intake.intent in {"chemistry_qa", "daily_qa"}:
                return self._answer_question(text, cancel=request_cancel)
            if intake.intent == "context_query":
                return self._answer_context(
                    text,
                    selection=intake.query_selection,
                    catalog=result_catalog,
                    cancel=request_cancel,
                )

            request = request_from_intake(
                text,
                intake,
                request_id=new_id("request"),
                registry=self.registry,
                normalized_parameters=normalized_parameters,
            )
            self._ensure_request_active(request_token, request_cancel)
            plan = None
            validation_feedback = None
            max_revisions = int(self.config.repair.max_plan_revisions)
            for revision in range(max_revisions + 1):
                llm_stage = "planner"
                proposal = plan_message(
                    self.llm,
                    request,
                    registry=self.registry,
                    context={
                        "recent_messages": self._session.get("recent_messages", []),
                        "recent_results": self._session.get("recent_results", []),
                        "geometry_catalog": (
                            [
                                item
                                for item in geometry_catalog
                                if item.get("alias") == selected_geometry_alias
                            ]
                            if selected_geometry_alias is not None
                            else []
                        ),
                    },
                    validation_feedback=validation_feedback,
                    cancel=request_cancel,
                )
                self._persist_llm_diagnostics(llm_call_cursor, stage="planner")
                llm_call_cursor = self._llm_call_count()
                self._ensure_request_active(request_token, request_cancel)
                try:
                    plan = proposal_to_plan(
                        request,
                        proposal,
                        self.registry,
                        plan_id=new_id("plan"),
                        artifact_aliases=_selected_artifact_aliases(
                            request, selected_geometry_alias
                        ),
                    )
                    _validate_selected_geometry_binding(plan, request, selected_geometry_alias)
                    break
                except ValueError as error:
                    if revision >= max_revisions:
                        raise ValueError(
                            f"Plan failed local validation after {revision} correction(s): {error}"
                        ) from error
                    validation_feedback = (
                        "The previous candidate Plan was rejected by local validation. "
                        f"Correct these issues without changing the Request: {error}"
                    )
            if plan is None:
                raise ValueError("Planner did not produce a locally valid Plan")
            self._ensure_request_active(request_token, request_cancel)
            run = self._create_chat_run(
                request,
                plan,
                history_geometry_binding=(
                    geometry_bindings.get(selected_geometry_alias)
                    if selected_geometry_alias is not None
                    else None
                ),
            )
            self._session["active_run_id"] = run.id
            self._save_session()
            result = self.advance(run, cancel=request_cancel)
            response = self._response_for_run(run, result)
            self._append_message("assistant", response.text)
            return response
        except LlmError as error:
            stage = error.purpose or llm_stage
            self._persist_llm_diagnostics(
                llm_call_cursor,
                stage=stage,
                failure_category=error.category,
            )
            if error.category == "cancelled" or request_cancel.is_set():
                response = AgentResponse("当前请求已取消，尚未执行。")
            else:
                stage_label = {
                    "intake": "请求解析阶段",
                    "planner": "计划生成阶段",
                }.get(stage)
                if stage == "intake" and error.category == "ambiguous_result":
                    response = AgentResponse(
                        "Opt 和 SP 同时请求时，通用‘能量’目标不唯一。请明确选择 "
                        "sp_electronic_energy（单点电子能）或 "
                        "opt_final_electronic_energy（优化末态电子能）；未启动计算。"
                    )
                elif stage_label is not None:
                    response = AgentResponse(
                        f"{stage_label}未获得有效模型响应（{error.category}）；"
                        "我已停止，没有生成或启动新的计算。"
                    )
                else:
                    response = AgentResponse(
                        f"模型调用未获得有效响应（{error.category}）；任务已停止。"
                    )
            self._append_message("assistant", response.text)
            return response
        except (ValueError, OSError) as error:
            response = AgentResponse(f"请求无法规划：{error}")
            self._append_message("assistant", response.text)
            return response
        finally:
            self._finish_request(request_token, request_cancel)

    def advance(self, run: Run, *, cancel: Event | None = None) -> Result | None:
        """Continue the current Run until a result, wait point, or terminal state."""

        cancel_event = cancel or self._cancel_events.setdefault(run.id, Event())
        if run.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return self._latest_result(run)
        if run.waiting_for is not None:
            return self._latest_result(run)
        run.status = "running"
        run.start_active_interval()
        save_run(self.config.data_root_path, run)
        last_result: Result | None = self._latest_result(run)
        try:
            while True:
                if cancel_event.is_set():
                    run.status = "cancelled"
                    run.waiting_for = None
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                if self._remaining_active_seconds(run) <= 0:
                    run.status = "failed"
                    run.pending_data = {
                        "category": "timeout",
                        "reason": "Run active time budget exhausted",
                    }
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                if _requested_results_satisfied(self.config.data_root_path, run, self.registry):
                    run.status = "succeeded"
                    run.waiting_for = None
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                step = _next_ready_step(run, self.config.data_root_path)
                if step is None:
                    blocked_checks = _unmet_goal_checks(
                        self.config.data_root_path, run, self.registry
                    )
                    run.status = "failed"
                    if blocked_checks:
                        run.pending_data = {
                            "category": "goal_not_met",
                            "reason": (
                                "A required scientific check did not reach its required status"
                            ),
                            "blocked_goal_checks": blocked_checks,
                        }
                    else:
                        run.pending_data = {
                            "category": "plan_incomplete",
                            "reason": (
                                "No executable Step remains before requested results were satisfied"
                            ),
                        }
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                tool = self.registry.get(step.tool)
                if tool.parameter_preparation == "orca_electronic_state":
                    preparing_step_id = step.id
                    try:
                        if tool.requires_compute_permission and not run.execution_permission:
                            # Resolve every ORCA Step before showing one confirmation
                            # for the whole Plan. Later operations often consume a
                            # future geometry port, so structure facts are traced
                            # back through that port to the already prepared input.
                            for planned_step in run.plan.steps:
                                planned_tool = self.registry.get(planned_step.tool)
                                if planned_tool.parameter_preparation != "orca_electronic_state":
                                    continue
                                preparing_step_id = planned_step.id
                                prepared_step = self._prepare_orca_step(run, planned_step)
                                if prepared_step is None:
                                    run.finish_active_interval()
                                    save_run(self.config.data_root_path, run)
                                    return last_result
                            step = next(item for item in run.plan.steps if item.id == step.id)
                        else:
                            prepared_step = self._prepare_orca_step(run, step)
                            if prepared_step is None:
                                run.finish_active_interval()
                                save_run(self.config.data_root_path, run)
                                return last_result
                            # Parameter resolution may replace a deferred Step. The
                            # exact replacement must be used for preview, fingerprint,
                            # budget reservation, and execution in this same turn.
                            step = prepared_step
                    except (TypeError, ValueError, OSError) as error:
                        run.status = "failed"
                        run.waiting_for = None
                        run.pending_data = {
                            "category": "parameter_preparation",
                            "reason": str(error),
                            "step_id": preparing_step_id,
                        }
                        run.finish_active_interval()
                        save_run(self.config.data_root_path, run)
                        return last_result
                    tool = self.registry.get(step.tool)
                if tool.requires_compute_permission and not run.execution_permission:
                    self._prepare_confirmation(run, step)
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                if tool.requires_compute_permission and not run.accepted_snapshot:
                    run.pending_data["permission_source"] = "config.confirm_before_compute=false"
                    run.accepted_snapshot = self._acceptance_snapshot(run)
                    run.accepted_execution_sha256 = execution_fingerprint(
                        run.plan,
                        run.resources,
                        run.artifact_index,
                        snapshot=run.accepted_snapshot,
                    )
                if not self._reserve_attempt(run, step, tool):
                    run.status = "failed"
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                run.step_status[step.id] = "running"
                save_run(self.config.data_root_path, run)
                try:
                    result = tool.execute(step, run, cancel=cancel_event)
                except (PermissionError, ValueError, OSError) as error:
                    run.status = "failed"
                    run.pending_data = {
                        "category": "execution_boundary",
                        "reason": str(error),
                        "step_id": step.id,
                    }
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    raise
                last_result = result
                result.step_fingerprint = _step_fingerprint(step)
                step_parameter_sources = run.parameter_sources_by_step.get(step.id)
                if step_parameter_sources:
                    result.parameter_sources = dict(step_parameter_sources)
                elif run.pending_data.get("parameter_sources"):
                    result.parameter_sources = dict(run.pending_data["parameter_sources"])
                result.input_bindings = {
                    name: artifact.id
                    for name, reference in step.inputs.items()
                    if (artifact := self._artifact_from_reference(run, reference)) is not None
                }
                result.input_artifact_ids = list(result.input_bindings.values())
                save_result(self.config.data_root_path, run, result)
                result_path = result.attempt_relative_path + "/result.json"
                if result_path not in run.result_index:
                    run.result_index.append(result_path)
                run.step_status[step.id] = result.status
                if result.status == "succeeded":
                    run.current_results[step.id] = result_path
                    run.pending_data = {}
                    self._record_result_summary(run, result)
                    save_run(self.config.data_root_path, run)
                    continue
                if result.status == "needs_input":
                    run.status = "waiting"
                    run.waiting_for = "clarification"
                    run.pending_data = {
                        **result.diagnostics,
                        **result.clarification,
                        "step_id": step.id,
                        "result_path": result.attempt_relative_path + "/result.json",
                    }
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return result
                save_run(self.config.data_root_path, run)
                if self._try_repair(run, step, result, cancel_event):
                    run.start_active_interval()
                    continue
                run.status = result.status
                run.finish_active_interval()
                save_run(self.config.data_root_path, run)
                return result
        except KeyboardInterrupt:
            cancel_event.set()
            run.status = "cancelled"
            run.finish_active_interval()
            save_run(self.config.data_root_path, run)
            raise
        finally:
            if run.status in {"succeeded", "failed", "cancelled", "interrupted"}:
                run.finish_active_interval()
                save_run(self.config.data_root_path, run)

    def confirm(self, run: Run | str | None = None) -> AgentResponse:
        current = self._coerce_run(run)
        if current is None:
            return AgentResponse("当前没有等待确认的计算。")
        if current.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            result = self._latest_result(current)
            return AgentResponse(
                render_already_finished(
                    current,
                    result,
                    self.registry,
                    structure=self._result_structure(current, result),
                    **self._failure_render_context(current),
                ),
                run=current,
                result=result,
            )
        if current.waiting_for != "confirmation":
            return AgentResponse(self._waiting_text(current), run=current)
        current.execution_permission = True
        current.waiting_for = None
        current.accepted_snapshot = self._acceptance_snapshot(current)
        current.accepted_execution_sha256 = execution_fingerprint(
            current.plan,
            current.resources,
            current.artifact_index,
            snapshot=current.accepted_snapshot,
        )
        save_run(self.config.data_root_path, current)
        try:
            result = self.advance(
                current, cancel=self._cancel_events.setdefault(current.id, Event())
            )
        except (PermissionError, ValueError, OSError) as error:
            current.status = "failed"
            current.pending_data = {"category": "execution_boundary", "reason": str(error)}
            save_run(self.config.data_root_path, current)
            return self._response_for_run(current, self._latest_result(current))
        response = self._response_for_run(current, result)
        self._append_message("assistant", response.text)
        return response

    def cancel(self, run: Run | str | None = None) -> AgentResponse:
        active_request = self._active_request
        if active_request is not None:
            active_request[1].set()
        current = self._coerce_run(run)
        if current is None:
            if active_request is not None:
                return AgentResponse("已请求取消当前请求。")
            return AgentResponse("当前没有活动中的计算。")
        event = self._cancel_events.setdefault(current.id, Event())
        event.set()
        if current.status == "waiting" and current.status not in {
            "succeeded",
            "failed",
            "cancelled",
            "interrupted",
        }:
            current.status = "cancelled"
            current.waiting_for = None
            save_run(self.config.data_root_path, current)
        if active_request is not None:
            return AgentResponse("已请求取消当前请求。", run=current)
        return AgentResponse("已请求取消当前计算。", run=current)

    def request_cancel(self) -> AgentResponse:
        """Signal cancellation without mutating a Run owned by the worker.

        The input thread can call this while intake or planning is still in
        flight, before a Run exists.  In that case the request Event is the
        cancellation boundary and no durable Run is fabricated.
        """

        active_request = self._active_request
        if active_request is not None:
            active_request[1].set()
        run_id = self._session.get("active_run_id")
        if run_id and active_request is None:
            event = self._cancel_events.setdefault(str(run_id), Event())
            event.set()
            return AgentResponse("已请求取消当前计算。")
        if active_request is not None:
            return AgentResponse("已请求取消当前请求。")
        return AgentResponse("当前没有活动中的计算。")

    def status(self) -> AgentResponse:
        current = self._coerce_run(None)
        if current is None:
            return AgentResponse("当前没有活动任务。")
        result = self._latest_result(current)
        return AgentResponse(
            render_run(
                current,
                result,
                self.registry,
                structure=self._result_structure(current, result),
                **self._failure_render_context(current),
            ),
            run=current,
            result=result,
        )

    def new_session(self) -> AgentResponse:
        current = self._coerce_run(None)
        if current is not None and current.status in {"running", "waiting"}:
            self.cancel(current)
        if self._active_request is not None:
            self._active_request[1].set()
        self._request_sequence += 1
        self._active_request = None
        self.session_id = new_id("session")
        self._session = {
            "session_id": self.session_id,
            "recent_messages": [],
            "recent_results": [],
            "active_run_id": None,
            "pending_prompt": None,
        }
        self._save_session()
        return AgentResponse("已开始新会话。")

    def _create_chat_run(
        self,
        request: Request,
        plan: Plan,
        *,
        history_geometry_binding: Mapping[str, Any] | None = None,
    ) -> Run:
        _validate_request_parameter_scope(request, plan, self.registry)
        history_alias = request.structure_input.get("history_geometry_alias")
        verified_history = (
            self._verify_history_geometry_binding(history_geometry_binding)
            if history_alias is not None and history_geometry_binding is not None
            else None
        )
        if history_alias is not None and verified_history is None:
            raise ValueError("selected history geometry has no verified source binding")
        run = Run(
            id=new_id("run"),
            request=request,
            plan=plan,
            resources=self.config.resources,
            execution_permission=not self.config.runtime.confirm_before_compute,
            status="planned",
            budget=self._default_budget(),
            origin_step_map={step.id: step.origin_step_id or step.id for step in plan.steps},
            session_id=self.session_id,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        create_run(self.config.data_root_path, run)
        alias_replacements: dict[str, str] = {}
        if verified_history is not None:
            source_run, source_step, source_result, source_artifact, source_path = verified_history
            copied = register_file_artifact(
                self.config.data_root_path,
                run,
                source_path,
                artifact_type="molecular_geometry",
                role="input_geometry",
                source="history:verified_optimized_geometry",
                extension=".xyz",
                metadata={
                    "history_source_role": "optimized_geometry",
                    "history_source_run_id": source_run.id,
                    "history_source_step_id": source_step.id,
                    "history_source_attempt": source_result.attempt,
                    "history_source_artifact_id": source_artifact.id,
                    "history_source_sha256": source_artifact.sha256,
                },
            )
            if copied.sha256 != source_artifact.sha256:
                raise ValueError("copied history geometry hash differs from its verified source")
            assert isinstance(history_alias, str)
            alias_replacements[history_alias] = copied.id
        input_artifact = self._seed_structure_input(run)
        if input_artifact is not None:
            alias_replacements["request_geometry"] = input_artifact.id
            alias_replacements[INPUT_GEOMETRY_PLACEHOLDER] = input_artifact.id
        updated_steps = []
        for step in run.plan.steps:
            inputs = {
                name: (
                    InputReference(artifact_id=alias_replacements[reference.artifact_id])
                    if reference.artifact_id in alias_replacements
                    else reference
                )
                for name, reference in step.inputs.items()
            }
            if inputs != step.inputs:
                updated_steps.append(step.model_copy(update={"inputs": inputs}))
            else:
                updated_steps.append(step)
        run.plan = Plan.model_validate(
            {**run.plan.model_dump(mode="python"), "steps": updated_steps}, strict=True
        )
        save_run(self.config.data_root_path, run)
        return run

    def _seed_structure_input(self, run: Run) -> Any | None:
        value = run.request.structure_input
        xyz_text = value.get("xyz_text") or value.get("xyz") if isinstance(value, dict) else None
        if xyz_text is None:
            return None
        if not isinstance(xyz_text, str):
            raise ValueError("chat structure_input.xyz_text must be text")
        geometry_bytes = xyz_text.encode("utf-8")
        parse_xyz_bytes(geometry_bytes)
        artifact = register_bytes_artifact(
            self.config.data_root_path,
            run,
            geometry_bytes,
            artifact_type="molecular_geometry",
            role="input_geometry",
            source="chat:inline_xyz",
            extension=".xyz",
            metadata={"imported_as_raw_bytes": True, "source": "chat"},
        )
        return artifact

    def _prepare_orca_step(self, run: Run, step: Step) -> Step | None:
        tool = self.registry.get(step.tool)
        original_parameters = dict(step.parameters)
        checked_parameters = tool.validate_parameters(original_parameters, allow_deferred=True)
        supplied_parameters = {
            name: checked_parameters[name]
            for name in original_parameters
            if name in checked_parameters
        }
        parameter_fields = _tool_parameter_fields(tool)
        if (
            run.accepted_snapshot
            and "charge" in step.parameters
            and "multiplicity" in step.parameters
        ):
            validated = tool.validate_parameters(step.parameters)
            _validate_orca_profile(validated)
            return step.model_copy(update={"parameters": validated})
        if (
            run.pending_data.get("step_id") == step.id
            and run.pending_data.get("parameters") == step.parameters
            and run.pending_data.get("parameter_sources")
        ):
            validated = self.registry.get(step.tool).validate_parameters(step.parameters)
            _validate_orca_profile(validated)
            return step.model_copy(update={"parameters": validated})
        facts = self._known_structure_facts(run, step)
        resolution = resolve_parameters(
            run.request.explicit_parameters,
            facts,
            supplied_parameters,
            self.config.defaults,
            user_modifications=run.request.user_modifications,
            parameter_fields=parameter_fields,
        )
        if resolution.missing_fields:
            run.status = "waiting"
            run.waiting_for = "clarification"
            run.pending_data = {
                "question": "Please provide the missing electronic state parameters.",
                "step_id": step.id,
                "missing_fields": list(resolution.missing_fields),
                "parameter_sources": resolution.parameter_sources,
                "parameters": dict(step.parameters),
            }
            return None
        validated = tool.validate_parameters(resolution.effective_parameters)
        _validate_orca_profile(validated)
        replacement = Step.model_validate(
            {**step.model_dump(mode="python"), "parameters": validated}, strict=True
        )
        run.plan = _replace_step(run.plan, replacement)
        run.pending_data = {
            "step_id": step.id,
            "parameters": dict(validated),
            "parameter_sources": resolution.parameter_sources,
            "effective_parameters": validated,
        }
        run.parameter_sources_by_step[step.id] = dict(resolution.parameter_sources)
        return replacement

    def _apply_parameter_update(
        self, run: Run, parameters: dict[str, Any], *, cancel: Event | None = None
    ) -> AgentResponse:
        if cancel is not None and cancel.is_set():
            return AgentResponse("当前请求已取消。", run=run)
        editable_steps = [
            item for item in run.plan.steps if self.registry.get(item.tool).request_parameters
        ]
        if not editable_steps:
            return AgentResponse("等待中的任务没有可编辑的计算步骤。", run=run)
        if not parameters:
            return AgentResponse("没有识别到可应用的计算参数修改。", run=run)

        candidate_request = run.request.model_copy(
            update={
                "explicit_parameters": {
                    **run.request.explicit_parameters,
                    **parameters,
                },
                "user_modifications": {
                    **run.request.user_modifications,
                    **parameters,
                },
            }
        )
        try:
            _validate_request_parameter_scope(candidate_request, run.plan, self.registry)
            candidate_steps: list[Step] = []
            candidate_sources: dict[str, dict[str, str]] = {}
            for step in run.plan.steps:
                tool = self.registry.get(step.tool)
                if not tool.request_parameters:
                    candidate_steps.append(step)
                    continue

                if tool.parameter_preparation != "orca_electronic_state":
                    replacement = merge_explicit_step_parameters(tool, step, candidate_request)
                    candidate_steps.append(replacement)
                    candidate_sources[step.id] = {
                        name: (
                            "user_modification"
                            if name in candidate_request.user_modifications
                            else "request_explicit"
                        )
                        for name in tool.request_parameters
                        if name in replacement.parameters
                        and (
                            name in candidate_request.user_modifications
                            or name in candidate_request.explicit_parameters
                        )
                    }
                    continue

                parameter_fields = _tool_parameter_fields(tool)
                step_patch = {
                    name: value for name, value in parameters.items() if name in parameter_fields
                }
                merged = dict(step.parameters)
                merged.update(step_patch)
                checked = tool.validate_parameters(merged, allow_deferred=True)
                supplied = {name: checked[name] for name in merged if name in checked}
                resolution = resolve_parameters(
                    candidate_request.explicit_parameters,
                    self._known_structure_facts(run, step),
                    supplied,
                    self.config.defaults,
                    user_modifications=candidate_request.user_modifications,
                    parameter_fields=parameter_fields,
                )
                effective = tool.validate_parameters(
                    resolution.effective_parameters,
                    allow_deferred=bool(resolution.missing_fields),
                )
                if tool.execution_budget == "orca":
                    _validate_orca_profile(effective)
                    reference = step.inputs.get("geometry")
                    if (
                        "charge" in effective
                        and "multiplicity" in effective
                        and reference is not None
                        and self._artifact_from_reference(run, reference) is not None
                    ):
                        # Validate against a geometry already available at this
                        # boundary. A downstream Step may reference an Opt port
                        # that is intentionally not produced until after confirmation.
                        self._validate_candidate_electronic_state(run, step, effective)
                replacement = Step.model_validate(
                    {**step.model_dump(mode="python"), "parameters": effective}, strict=True
                )
                candidate_steps.append(replacement)
                candidate_sources[step.id] = dict(resolution.parameter_sources)

            candidate_plan = Plan.model_validate(
                {
                    **run.plan.model_dump(mode="python"),
                    "steps": candidate_steps,
                    "revision": run.plan.revision + 1,
                },
                strict=True,
            )
            candidate_plan = self.registry.validate_plan(candidate_plan)
            _validate_request_parameter_scope(candidate_request, candidate_plan, self.registry)
        except (TypeError, ValueError) as error:
            return AgentResponse(f"参数修改已拒绝（rejected）：{error}", run=run)

        changed_steps = [
            step.id
            for old, step in zip(run.plan.steps, candidate_plan.steps, strict=True)
            if old.parameters != step.parameters
        ]
        run.request = candidate_request
        run.plan = candidate_plan
        run.execution_permission = not self.config.runtime.confirm_before_compute
        run.accepted_snapshot = {}
        run.accepted_execution_sha256 = None
        for changed_step_id in changed_steps:
            _invalidate_current_results(run, changed_step_id)
        run.parameter_sources_by_step.update(candidate_sources)
        run.waiting_for = None
        run.status = "running"
        run.pending_data = {}
        save_run(self.config.data_root_path, run)
        try:
            result = self.advance(
                run,
                cancel=cancel or self._cancel_events.setdefault(run.id, Event()),
            )
        except (PermissionError, ValueError, OSError) as error:
            run.status = "failed"
            run.pending_data = {"category": "execution_boundary", "reason": str(error)}
            save_run(self.config.data_root_path, run)
            return AgentResponse(f"计算无法继续：{error}", run=run)
        return self._response_for_run(run, result)

    def _apply_molecule_clarification(
        self,
        run: Run,
        query: str,
        input_kind: str | None,
        *,
        cancel: Event | None = None,
    ) -> AgentResponse:
        return self._apply_molecule_update(run, query, input_kind, cancel=cancel)

    def _apply_molecule_update(
        self,
        run: Run,
        query: str,
        input_kind: str | None,
        *,
        cancel: Event | None = None,
    ) -> AgentResponse:
        if cancel is not None and cancel.is_set():
            return AgentResponse("当前请求已取消。", run=run)
        step_id = str(run.pending_data.get("step_id", ""))
        step = next((item for item in run.plan.steps if item.id == step_id), None)
        if step is None or step.tool != "resolve_molecule":
            resolve_steps = [item for item in run.plan.steps if item.tool == "resolve_molecule"]
            step = resolve_steps[0] if len(resolve_steps) == 1 else None
        if step is None:
            return AgentResponse("等待中的任务没有分子解析步骤。", run=run)
        kind = input_kind or ("cid" if query.isdecimal() else "name")
        try:
            replacement = Step.model_validate(
                {
                    **step.model_dump(mode="python"),
                    "parameters": {"query": query, "input_kind": kind},
                },
                strict=True,
            )
            self.registry.get(step.tool).validate_parameters(replacement.parameters)
            candidate_plan = _replace_step(run.plan, replacement)
            candidate_plan = Plan.model_validate(
                {
                    **candidate_plan.model_dump(mode="python"),
                    "revision": candidate_plan.revision + 1,
                },
                strict=True,
            )
            candidate_plan = self.registry.validate_plan(candidate_plan)
        except ValueError as error:
            return AgentResponse(f"该分子选择已拒绝（rejected）：{error}", run=run)
        run.plan = candidate_plan
        _invalidate_current_results(run, step.id)
        run.accepted_snapshot = {}
        run.accepted_execution_sha256 = None
        run.status = "running"
        run.waiting_for = None
        run.pending_data = {}
        save_run(self.config.data_root_path, run)
        result = self.advance(
            run,
            cancel=cancel or self._cancel_events.setdefault(run.id, Event()),
        )
        return self._response_for_run(run, result)

    def _try_repair(self, run: Run, step: Step, result: Result, cancel: Event) -> bool:
        if not self.config.repair.enabled:
            return False
        options = applicable_repairs(run, step, result) + applicable_scf_repair(run, step, result)
        if not options or self.llm is None:
            return False
        if run.plan_revisions >= int(run.budget.get("max_plan_revisions", 2)):
            run.pending_data = {
                "budget_exhausted": "max_plan_revisions",
                "step_id": step.id,
            }
            save_run(self.config.data_root_path, run)
            return False
        if self._remaining_active_seconds(run) <= 0:
            run.pending_data = {
                "category": "timeout",
                "reason": "Run active time budget exhausted before repair proposal",
            }
            save_run(self.config.data_root_path, run)
            return False
        max_extra = int(
            run.budget.get(
                "max_extra_orca_executions",
                self.config.repair.max_extra_orca_executions,
            )
        )
        if run.extra_orca_executions >= max_extra:
            return False
        try:
            proposal = propose_repair(
                self.llm,
                run=run,
                step=step,
                result=result,
                options=options,
                cancel=cancel,
                remaining_timeout_seconds=self._remaining_active_seconds(run),
            )
        except LlmError as error:
            run.pending_data = {"repair_unavailable": error.category, "reason": str(error)}
            save_run(self.config.data_root_path, run)
            return False
        if proposal is None or proposal.action == "none":
            return False
        option = next((item for item in options if item.action == proposal.action), None)
        if option is None:
            return False
        try:
            replacement, record = apply_repair_proposal(
                proposal, option=option, run=run, step=step, result=result
            )
        except ValueError as error:
            run.pending_data = {"repair_rejected": str(error)}
            save_run(self.config.data_root_path, run)
            return False
        run.plan = _replace_step(run.plan, replacement)
        run.plan = self.registry.validate_plan(run.plan)
        run.plan = Plan.model_validate(
            {**run.plan.model_dump(mode="python"), "revision": run.plan.revision + 1},
            strict=True,
        )
        run.plan_revisions += 1
        record["derived_plan_sha256"] = _plan_fingerprint(run.plan)
        run.repair_records.append(record)
        _invalidate_current_results(run, step.id)
        save_run(self.config.data_root_path, run)
        return True

    def _reserve_attempt(self, run: Run, step: Step, tool: Tool) -> bool:
        if tool.execution_budget != "orca":
            return True
        origin = run.origin_step_map.get(step.id, step.origin_step_id or step.id)
        known_origin = step.id in run.origin_step_map or origin in run.origin_step_map.values()
        new_science_step = bool(run.accepted_snapshot) and not known_origin
        if step.id not in run.origin_step_map:
            run.origin_step_map[step.id] = origin
        count = int(run.attempt_counts.get(origin, 0))
        if count >= int(run.budget.get("max_attempts_per_science_step", 3)):
            run.pending_data = {
                "budget_exhausted": "max_attempts_per_science_step",
                "step_id": step.id,
            }
            return False
        if (count > 0 or new_science_step) and run.extra_orca_executions >= int(
            run.budget.get("max_extra_orca_executions", 3)
        ):
            run.pending_data = {
                "budget_exhausted": "max_extra_orca_executions",
                "step_id": step.id,
            }
            return False
        run.attempt_counts[origin] = count + 1
        if count > 0 or new_science_step:
            run.extra_orca_executions += 1
        return True

    def _prepare_confirmation(self, run: Run, step: Step) -> None:
        run.status = "waiting"
        run.waiting_for = "confirmation"
        run.pending_data = self._preview(run, step)

    def _preview(self, run: Run, step: Step) -> dict[str, Any]:
        artifacts = self._input_artifacts(run, step)
        structure = dict(self._known_structure_facts(run, step))
        geometry_facts = self._preview_geometry_facts(run, step)
        structure.update(geometry_facts)
        if not geometry_facts.get("geometry_available"):
            # Molecule facts may contain the implicit-hydrogen graph count.  It
            # is not a substitute for the atom count in the actual XYZ input.
            structure.pop("atom_count", None)
        return {
            "request": {
                "description": run.request.description,
                "operations": list(run.request.operations),
                "requested_results": [
                    target.model_dump(mode="json") for target in run.request.requested_results
                ],
            },
            "operation": (self.registry.get(step.tool).operations or [None])[0]
            or {"optimize_geometry": "Opt", "single_point": "SP"}.get(step.tool, "Tool"),
            "step_id": step.id,
            "tool": step.tool,
            "plan_steps": self._preview_plan_steps(run),
            "parameters": dict(step.parameters),
            "result_targets": self._preview_result_targets(run, step),
            "parameter_sources": run.parameter_sources_by_step.get(step.id, {}),
            "structure": structure,
            "artifact_bindings": [
                {
                    "id": artifact.id,
                    "sha256": artifact.sha256,
                    "role": artifact.role,
                    "artifact_type": artifact.artifact_type,
                    "source": artifact.source,
                    "structure_source": artifact.metadata.get("structure_source"),
                }
                for artifact in artifacts
            ],
            "resources": dict(run.resources),
            "allowed_repairs": self._allowed_repairs(run, step),
            "repair_scope": self._repair_scope(run),
            "budget": dict(run.budget),
        }

    def _preview_plan_steps(self, run: Run) -> list[dict[str, Any]]:
        step_numbers = {step.id: index for index, step in enumerate(run.plan.steps, start=1)}
        summaries: list[dict[str, Any]] = []
        for index, step in enumerate(run.plan.steps, start=1):
            tool = self.registry.get(step.tool)
            inputs: list[dict[str, Any]] = []
            for name, reference in step.inputs.items():
                if reference.step_id is not None:
                    source_index = step_numbers.get(reference.step_id)
                    inputs.append(
                        {
                            "name": name,
                            "source_step": f"步骤 {source_index}" if source_index else "未知步骤",
                            "port": reference.port,
                        }
                    )
                    continue
                artifact = self._artifact_from_reference(run, reference)
                inputs.append(
                    {
                        "name": name,
                        "artifact_role": artifact.role if artifact is not None else "待绑定结构",
                        "history_geometry": bool(
                            artifact is not None
                            and artifact.source == "history:verified_optimized_geometry"
                        ),
                    }
                )
            goals = [
                {
                    "source_step": (
                        f"步骤 {step_numbers[goal.source_step_id]}"
                        if goal.source_step_id in step_numbers
                        else "未知步骤"
                    ),
                    "check": goal.check,
                    "required_status": goal.required_status,
                }
                for goal in step.goal_checks
            ]
            summaries.append(
                {
                    "index": index,
                    "tool": tool.name,
                    "operations": list(tool.operations),
                    "parameters": dict(step.parameters),
                    "inputs": inputs,
                    "goal_checks": goals,
                    "requested_results": self._preview_result_targets(run, step),
                }
            )
        return summaries

    def _preview_result_targets(self, run: Run, step: Step) -> list[dict[str, str]]:
        try:
            tool = self.registry.get(step.tool)
        except ValueError:
            return []
        targets: list[dict[str, str]] = []
        for target in run.plan.requested_results:
            if target.step_id != step.id:
                continue
            name = target.check or target.port or target.field
            if name is None:
                continue
            metadata = dict(tool.result_metadata.get(name, {}))
            metadata.setdefault("label", name.replace("_", " "))
            metadata.setdefault("description", tool.description)
            item = {
                "name": name,
                "kind": (
                    "check"
                    if target.check is not None
                    else "port"
                    if target.port is not None
                    else "field"
                ),
                "label": metadata["label"],
                "description": metadata["description"],
            }
            if item not in targets:
                targets.append(item)
        return targets

    def _preview_geometry_facts(self, run: Run, step: Step) -> dict[str, Any]:
        reference = step.inputs.get("geometry")
        if reference is None:
            return {"geometry_available": False, "geometry_error": "geometry input is unavailable"}
        artifact = self._artifact_from_reference(run, reference)
        if artifact is None or artifact.artifact_type != "molecular_geometry":
            return {"geometry_available": False, "geometry_error": "geometry input is unavailable"}
        try:
            geometry = parse_xyz_bytes(
                artifact_path(self.config.data_root_path, run, artifact).read_bytes()
            )
        except (OSError, ValueError) as error:
            return {"geometry_available": False, "geometry_error": str(error)}
        return {
            "geometry_available": True,
            "atom_count": geometry.atom_count,
            "atom_symbols": list(geometry.symbols),
        }

    def _acceptance_snapshot(self, run: Run) -> dict[str, Any]:
        bound: list[dict[str, Any]] = []
        for step in run.plan.steps:
            for artifact in self._input_artifacts(run, step):
                item = {"step_id": step.id, "id": artifact.id, "sha256": artifact.sha256}
                if item not in bound:
                    bound.append(item)
        repair_scope = self._repair_scope(run)
        allowed_repairs = sorted(
            {
                action
                for step_scope in repair_scope.get("steps", {}).values()
                if isinstance(step_scope, dict)
                for action in step_scope.get("actions", {})
            }
        )
        return {
            "request": run.request.model_dump(mode="json"),
            "plan": run.plan.model_dump(mode="json"),
            "resources": dict(run.resources),
            "bound_inputs": bound,
            "allowed_repairs": allowed_repairs,
            "repair_scope": repair_scope,
            "budgets": dict(run.budget),
            "origin_step_map": dict(run.origin_step_map),
            "repair_record_start": len(run.repair_records),
        }

    def _known_structure_facts(self, run: Run, step: Step | None = None) -> dict[str, Any]:
        """Read facts only from the molecule that produced this Step's geometry."""

        return self._known_structure_facts_from_step(run, step, seen=set())

    def _known_structure_facts_from_step(
        self, run: Run, step: Step | None, *, seen: set[str]
    ) -> dict[str, Any]:
        if step is None:
            return {}
        if step.id in seen:
            return {}
        seen.add(step.id)
        geometry_reference = step.inputs.get("geometry")
        if geometry_reference is None:
            return {}
        geometry_artifact = self._artifact_from_reference(run, geometry_reference)
        if geometry_artifact is None:
            if geometry_reference.step_id is None:
                return {}
            source_step = next(
                (item for item in run.plan.steps if item.id == geometry_reference.step_id), None
            )
            return self._known_structure_facts_from_step(run, source_step, seen=seen)
        molecule_artifact = self._molecule_artifact_for_geometry(run, geometry_artifact, seen=set())
        if molecule_artifact is None:
            return {}
        try:
            payload = json.loads(
                artifact_path(self.config.data_root_path, run, molecule_artifact).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        facts = payload.get("facts")
        if isinstance(facts, dict):
            enriched = dict(facts)
            enriched.setdefault("source", payload.get("source"))
            enriched.setdefault("query", payload.get("query"))
            enriched.setdefault("input_kind", payload.get("input_kind"))
            return enriched
        return {}

    def _validate_candidate_electronic_state(
        self, run: Run, step: Step, parameters: dict[str, Any]
    ) -> None:
        reference = step.inputs.get("geometry")
        if reference is None:
            return
        geometry_artifact = self._artifact_from_reference(run, reference)
        if geometry_artifact is None:
            raise ValueError("candidate geometry is not a current successful artifact")
        try:
            geometry = parse_xyz_bytes(
                artifact_path(self.config.data_root_path, run, geometry_artifact).read_bytes()
            )
        except (OSError, ValueError) as error:
            raise ValueError("candidate geometry cannot be validated") from error
        validate_electronic_state(
            geometry,
            charge=parameters["charge"],
            multiplicity=parameters["multiplicity"],
        )

    def _molecule_artifact_for_geometry(
        self, run: Run, artifact: Any, *, seen: set[str]
    ) -> Any | None:
        if artifact.id in seen:
            return None
        seen.add(artifact.id)
        if artifact.artifact_type == "molecule" and artifact.role == "resolved_molecule":
            return artifact
        molecule_id = artifact.metadata.get("molecule_artifact_id")
        if isinstance(molecule_id, str):
            try:
                molecule = find_artifact(run, molecule_id)
            except ValueError:
                molecule = None
            if (
                molecule is not None
                and molecule.artifact_type == "molecule"
                and molecule.role == "resolved_molecule"
            ):
                return molecule
        for attempt in reversed(run.attempts):
            if (
                attempt.get("step_id") != artifact.step_id
                or attempt.get("attempt") != artifact.attempt
                or artifact.id not in attempt.get("artifact_ids", [])
            ):
                continue
            input_id = attempt.get("input_geometry_artifact_id")
            if not isinstance(input_id, str):
                continue
            try:
                input_artifact = find_artifact(run, input_id)
            except ValueError:
                input_artifact = None
            if input_artifact is not None:
                return self._molecule_artifact_for_geometry(run, input_artifact, seen=seen)
        if artifact.step_id is None:
            return None
        producer = next((item for item in run.plan.steps if item.id == artifact.step_id), None)
        if producer is None:
            return None
        upstream_name = "molecule" if producer.tool == "generate_geometry" else "geometry"
        upstream_reference = producer.inputs.get(upstream_name)
        if upstream_reference is None:
            return None
        upstream = self._artifact_from_reference(run, upstream_reference)
        if upstream is None:
            return None
        return self._molecule_artifact_for_geometry(run, upstream, seen=seen)

    def _repair_scope(self, run: Run) -> dict[str, Any]:
        iteration_increase_allowed = _iteration_increase_allowed(run.request.description)
        scopes: dict[str, Any] = {}
        for step in run.plan.steps:
            tool = self.registry.get(step.tool)
            try:
                capabilities = set(tool.applicable_repair_capabilities(step.parameters))
            except (TypeError, ValueError):
                capabilities = set()
            mutable_parameters: list[str] = []
            immutable_parameters = ["method_profile", "environment", "charge", "multiplicity"]
            actions: dict[str, Any] = {}
            if "restart_optimization" in capabilities:
                mutable_parameters.append("geom_maxiter")
                immutable_parameters.append("scf_maxiter")
                if iteration_increase_allowed:
                    actions["restart_optimization"] = {
                        "fields": ["geom_maxiter"],
                        "maximum": 1000,
                        "maximum_is_program_cap": True,
                    }
            if "increase_scf_maxiter" in capabilities:
                mutable_parameters.append("scf_maxiter")
                immutable_parameters.append("geom_maxiter")
                if iteration_increase_allowed:
                    actions["increase_scf_maxiter"] = {
                        "fields": ["scf_maxiter"],
                        "maximum": 1000,
                        "maximum_is_program_cap": True,
                    }
            if not capabilities.intersection({"restart_optimization", "increase_scf_maxiter"}):
                continue
            scopes[step.id] = {
                "origin_step_id": run.origin_step_map.get(step.id, step.origin_step_id or step.id),
                "mutable_parameters": mutable_parameters,
                "immutable_parameters": immutable_parameters,
                "geometry_rule": (
                    "only a program-validated restart_candidate from this Step"
                    if "restart_optimization" in capabilities
                    else "retain the accepted geometry reference"
                ),
                "actions": actions,
            }
        return {
            "version": 1,
            "iteration_increase_allowed": iteration_increase_allowed,
            "steps": scopes,
            "resources_immutable": dict(run.resources),
            "max_attempts_per_science_step": run.budget.get("max_attempts_per_science_step"),
            "max_extra_orca_executions": run.budget.get("max_extra_orca_executions"),
        }

    def _allowed_repairs(self, run: Run, step: Step) -> list[str]:
        scope = self._repair_scope(run).get("steps", {})
        step_scope = scope.get(step.id, {}) if isinstance(scope, dict) else {}
        actions = step_scope.get("actions", {}) if isinstance(step_scope, dict) else {}
        if not isinstance(actions, dict):
            return []
        return sorted(str(action) for action in actions)

    def _input_artifacts(self, run: Run, step: Step) -> list[Any]:
        artifacts: list[Any] = []
        for reference in step.inputs.values():
            artifact = self._artifact_from_reference(run, reference)
            if artifact is not None and artifact not in artifacts:
                artifacts.append(artifact)
        return artifacts

    def _artifact_from_reference(self, run: Run, reference: InputReference) -> Any | None:
        if reference.artifact_id is not None:
            try:
                return find_artifact(run, reference.artifact_id)
            except ValueError:
                return None
        if reference.step_id is None:
            return None
        relative = run.current_results.get(reference.step_id)
        if not relative:
            return None
        result = _load_bound_result(self.config.data_root_path, run, relative)
        if result is None or result.status != "succeeded":
            return None
        upstream = next((item for item in run.plan.steps if item.id == reference.step_id), None)
        if upstream is None or result.step_fingerprint != _step_fingerprint(upstream):
            return None
        try:
            artifact_id = result.output_ports.get(reference.port)
            if not artifact_id or artifact_id not in result.artifact_ids:
                return None
            artifact = find_artifact(run, artifact_id)
            if artifact.step_id != reference.step_id or artifact.attempt != result.attempt:
                return None
            return artifact
        except ValueError:
            return None

    def _latest_result(self, run: Run) -> Result | None:
        if not run.result_index:
            return None
        for relative in reversed(run.result_index):
            result = _load_bound_result(self.config.data_root_path, run, relative)
            if result is not None:
                return result
        return None

    def _result_structure(self, run: Run, result: Result | None) -> dict[str, Any] | None:
        if result is None:
            return None
        step = next((item for item in run.plan.steps if item.id == result.step_id), None)
        if step is None:
            return None
        structure = self._query_structure(run, step, result)
        return structure or None

    def _coerce_run(self, run: Run | str | None) -> Run | None:
        if isinstance(run, Run):
            return run
        run_id = run or self._session.get("active_run_id")
        if not run_id:
            return None
        try:
            return load_run(self.config.data_root_path, str(run_id))
        except ValueError:
            return None

    def _default_budget(self) -> dict[str, Any]:
        return {
            "max_attempts_per_science_step": self.config.repair.max_attempts_per_science_step,
            "max_extra_orca_executions": self.config.repair.max_extra_orca_executions,
            "max_plan_revisions": self.config.repair.max_plan_revisions,
        }

    def _remaining_active_seconds(self, run: Run) -> float:
        limit = float(
            run.resources.get(
                "run_active_timeout_seconds", self.config.runtime.run_active_timeout_seconds
            )
        )
        return limit - run.current_active_seconds()

    def _response_for_run(self, run: Run, result: Result | None) -> AgentResponse:
        # A successful preparation step is not the user's requested scientific
        # result.  Waiting state always wins over the last intermediate Result.
        if run.status == "waiting":
            return AgentResponse(self._waiting_text(run), run=run, result=result)
        structure = self._result_structure(run, result)
        if run.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return AgentResponse(
                render_run(
                    run,
                    result,
                    self.registry,
                    structure=structure,
                    **self._failure_render_context(run),
                ),
                run=run,
                result=result,
            )
        if result is not None:
            return AgentResponse(
                render_result(run, result, self.registry, structure=structure),
                run=run,
                result=result,
            )
        return AgentResponse(render_run(run, registry=self.registry), run=run)

    def _waiting_text(self, run: Run) -> str:
        if run.waiting_for == "confirmation":
            return "计算等待确认（waiting for confirmation）。\n" + render_confirmation(
                run.pending_data
            )
        return "任务正在等待补充信息。\n" + render_clarification(run.pending_data)

    def _begin_request(self) -> tuple[int, Event]:
        self._request_sequence += 1
        token = self._request_sequence
        event = Event()
        self._active_request = (token, event)
        return token, event

    def _finish_request(self, token: int, event: Event) -> None:
        if self._active_request == (token, event):
            self._active_request = None

    def _ensure_request_active(self, token: int, event: Event) -> None:
        if event.is_set() or self._active_request != (token, event):
            raise LlmError("request cancelled", category="cancelled")

    def _append_message(self, role: str, content: str) -> None:
        messages = self._session.setdefault("recent_messages", [])
        messages.append({"role": role, "content": content})
        self._session["recent_messages"] = messages[-12:]
        self._save_session()

    def _record_result_summary(self, run: Run, result: Result) -> None:
        summaries = self._session.get("recent_results", [])
        recent_by_run: list[dict[str, Any]] = []
        if isinstance(summaries, list):
            for item in summaries:
                if not isinstance(item, Mapping):
                    continue
                run_id = item.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    continue
                recent_by_run = [entry for entry in recent_by_run if entry["run_id"] != run_id]
                recent_by_run.append(dict(item))
        recent_by_run = [entry for entry in recent_by_run if entry["run_id"] != run.id]
        recent_by_run.append(
            {
                "run_id": run.id,
                "step_id": result.step_id,
                "status": result.status,
                "result_path": f"{result.attempt_relative_path}/result.json",
                "attempt": result.attempt,
            }
        )
        self._session["recent_results"] = recent_by_run[-MAX_RECENT_RUNS:]
        self._session["active_run_id"] = run.id
        self._save_session()

    def _save_session(self) -> None:
        try:
            save_session(self.config.data_root_path, self.session_id, self._session)
        except OSError:
            pass

    def _persist_llm_diagnostics(
        self,
        start_index: int,
        *,
        stage: str,
        failure_category: str | None = None,
    ) -> None:
        """Keep bounded, credential-free model-call facts with session context."""

        diagnostics = self._session.get("llm_diagnostics", [])
        if not isinstance(diagnostics, list):
            diagnostics = []
        calls = getattr(self.llm, "calls", [])
        if isinstance(calls, list):
            for call in calls[start_index:]:
                if not is_dataclass(call):
                    continue
                item = asdict(call)
                item["stage"] = stage
                diagnostics.append(item)
        if failure_category is not None:
            diagnostics.append({"stage": stage, "category": failure_category})
        self._session["llm_diagnostics"] = diagnostics[-32:]
        self._save_session()

    def _llm_call_count(self) -> int:
        calls = getattr(self.llm, "calls", [])
        return len(calls) if isinstance(calls, list) else 0

    def _build_query_catalog(self) -> list[dict[str, Any]]:
        """Build short, public references for valid facts in this session.

        The catalog deliberately contains no Run IDs, paths, hashes, or
        numeric values. Facts are grouped under program-generated subject
        references, while their declared properties remain machine-readable.
        """

        self._query_bindings = {}
        subject_refs: dict[str, str] = {}

        def _subject_ref(run: Run, step: Step) -> str:
            task_key = f"{run.id}:{step.id}"
            if task_key in subject_refs:
                return subject_refs[task_key]
            ref = f"t{len(subject_refs) + 1}"
            subject_refs[task_key] = ref
            return ref

        def _ordered_steps(run: Run) -> list[Step]:
            by_id = {step.id: step for step in run.plan.steps}
            prioritized: list[Step] = []
            seen: set[str] = set()
            for target in run.plan.requested_results:
                candidate = by_id.get(target.step_id) if target.step_id is not None else None
                if candidate is None:
                    name = target.port or target.field
                    matching: list[Step] = []
                    if name is not None:
                        for step in run.plan.steps:
                            try:
                                tool = self.registry.get(step.tool)
                            except ValueError:
                                continue
                            declared = (
                                tool.output_ports if target.port is not None else tool.results
                            )
                            if name in declared:
                                matching.append(step)
                    if len(matching) == 1:
                        candidate = matching[0]
                if candidate is not None and candidate.id not in seen:
                    prioritized.append(candidate)
                    seen.add(candidate.id)
            return prioritized + [step for step in run.plan.steps if step.id not in seen]

        active_run_id = self._session.get("active_run_id")
        indexed_ids: list[str] = []
        if isinstance(active_run_id, str) and active_run_id:
            indexed_ids.append(active_run_id)
        for summary in reversed(self._session.get("recent_results", [])):
            if not isinstance(summary, Mapping):
                continue
            run_id = summary.get("run_id")
            if isinstance(run_id, str) and run_id and run_id not in indexed_ids:
                indexed_ids.append(run_id)

        catalog: list[dict[str, Any]] = []
        for run_id in indexed_ids:
            try:
                run = load_run(self.config.data_root_path, run_id)
            except ValueError:
                continue
            # A missing session_id is tolerated only because this Run was
            # explicitly indexed by this session.  A present foreign session
            # is never imported into the catalog.
            if run.session_id not in {None, self.session_id}:
                continue
            for step in _ordered_steps(run):
                if len(catalog) >= MAX_QUERY_CATALOG_ITEMS:
                    return catalog
                relative = run.current_results.get(step.id)
                if not isinstance(relative, str):
                    continue
                result = _load_bound_result(self.config.data_root_path, run, relative)
                if result is None or not self._query_result_is_valid(run, step, result, relative):
                    continue
                try:
                    tool = self.registry.get(step.tool)
                except ValueError:
                    continue
                structure = self._query_structure(run, step, result)
                for name, expected_type in tool.results.items():
                    if name in tool.output_ports or name not in result.values:
                        continue
                    property_name = tool.result_properties.get(name)
                    if property_name is None:
                        continue
                    value = result.values[name]
                    if not _query_value_is_compatible(value, expected_type):
                        continue
                    subject_ref = _subject_ref(run, step)
                    binding = {
                        "session_id": self.session_id,
                        "run_id": run.id,
                        "step_id": step.id,
                        "result_path": relative,
                        "attempt": result.attempt,
                        "step_fingerprint": _step_fingerprint(step),
                        "kind": "field",
                        "name": name,
                        "property": property_name,
                    }
                    self._query_bindings[(subject_ref, property_name)] = binding
                    catalog.append(
                        self._public_query_entry(
                            subject_ref,
                            run,
                            step,
                            tool,
                            name=name,
                            kind="field",
                            expected_type=expected_type,
                            property_name=property_name,
                            structure=structure,
                        )
                    )
                    if len(catalog) >= MAX_QUERY_CATALOG_ITEMS:
                        return catalog
                for name, expected_type in tool.output_ports.items():
                    property_name = tool.result_properties.get(name)
                    if property_name is None:
                        continue
                    artifact = self._query_port_artifact(run, step, result, name, expected_type)
                    if artifact is None:
                        continue
                    subject_ref = _subject_ref(run, step)
                    binding = {
                        "session_id": self.session_id,
                        "run_id": run.id,
                        "step_id": step.id,
                        "result_path": relative,
                        "attempt": result.attempt,
                        "step_fingerprint": _step_fingerprint(step),
                        "kind": "port",
                        "name": name,
                        "artifact_id": artifact.id,
                        "property": property_name,
                    }
                    self._query_bindings[(subject_ref, property_name)] = binding
                    catalog.append(
                        self._public_query_entry(
                            subject_ref,
                            run,
                            step,
                            tool,
                            name=name,
                            kind="port",
                            expected_type=expected_type,
                            property_name=property_name,
                            structure=structure,
                        )
                    )
                    if len(catalog) >= MAX_QUERY_CATALOG_ITEMS:
                        return catalog
        return catalog

    def _build_geometry_catalog(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Expose bounded aliases for current, verified optimized geometries."""

        indexed_ids: list[str] = []
        active_run_id = self._session.get("active_run_id")
        if isinstance(active_run_id, str) and active_run_id:
            indexed_ids.append(active_run_id)
        for summary in reversed(self._session.get("recent_results", [])):
            if not isinstance(summary, Mapping):
                continue
            run_id = summary.get("run_id")
            if isinstance(run_id, str) and run_id and run_id not in indexed_ids:
                indexed_ids.append(run_id)

        catalog: list[dict[str, Any]] = []
        bindings: dict[str, dict[str, Any]] = {}
        for run_id in indexed_ids:
            try:
                run = load_run(self.config.data_root_path, run_id)
            except ValueError:
                continue
            if run.session_id not in {None, self.session_id}:
                continue
            for step in reversed(run.plan.steps):
                if len(catalog) >= MAX_HISTORY_GEOMETRIES:
                    return catalog, bindings
                try:
                    tool = self.registry.get(step.tool)
                except ValueError:
                    continue
                if tool.name != "optimize_geometry":
                    continue
                relative = run.current_results.get(step.id)
                if not isinstance(relative, str):
                    continue
                result = _load_bound_result(self.config.data_root_path, run, relative)
                if result is None or not self._query_result_is_valid(run, step, result, relative):
                    continue
                artifact = self._query_port_artifact(
                    run, step, result, "optimized_geometry", "molecular_geometry"
                )
                if artifact is None or artifact.role != "optimized_geometry":
                    continue
                try:
                    geometry_path = artifact_path(self.config.data_root_path, run, artifact)
                    geometry = parse_xyz_bytes(geometry_path.read_bytes())
                except (OSError, ValueError):
                    continue
                alias = f"geometry_{len(catalog) + 1}"
                binding = {
                    "session_id": self.session_id,
                    "run_id": run.id,
                    "step_id": step.id,
                    "result_path": relative,
                    "attempt": result.attempt,
                    "step_fingerprint": _step_fingerprint(step),
                    "artifact_id": artifact.id,
                    "sha256": artifact.sha256,
                }
                bindings[alias] = binding
                system = self._query_structure(run, step, result)
                catalog.append(
                    {
                        "alias": alias,
                        "description": run.request.description[:160],
                        "created_at": run.created_at,
                        "run_status": run.status,
                        "system": system,
                        "geometry": {
                            "role": "verified optimized_geometry",
                            "atom_count": geometry.atom_count,
                        },
                        "calculation": {
                            "method_profile": step.parameters.get("method_profile"),
                            "environment": step.parameters.get("environment"),
                            "charge": step.parameters.get("charge"),
                            "multiplicity": step.parameters.get("multiplicity"),
                        },
                    }
                )
        return catalog, bindings

    def _verify_history_geometry_binding(
        self, binding: Mapping[str, Any]
    ) -> tuple[Run, Step, Result, Any, Path]:
        """Revalidate source result and bytes immediately before copying them."""

        if binding.get("session_id") != self.session_id:
            raise ValueError("history geometry belongs to another session")
        run_id = binding.get("run_id")
        step_id = binding.get("step_id")
        relative = binding.get("result_path")
        if not all(isinstance(item, str) and item for item in (run_id, step_id, relative)):
            raise ValueError("history geometry binding is malformed")
        source_run = load_run(self.config.data_root_path, run_id)
        if source_run.session_id not in {None, self.session_id}:
            raise ValueError("history geometry source is outside this session")
        step = next((item for item in source_run.plan.steps if item.id == step_id), None)
        result = _load_bound_result(self.config.data_root_path, source_run, relative)
        if (
            step is None
            or step.tool != "optimize_geometry"
            or result is None
            or source_run.current_results.get(step.id) != relative
            or result.status != "succeeded"
            or result.run_id != source_run.id
            or result.step_id != step.id
            or result.attempt != binding.get("attempt")
            or result.step_fingerprint != _step_fingerprint(step)
            or result.step_fingerprint != binding.get("step_fingerprint")
        ):
            raise ValueError("history geometry no longer belongs to a current successful result")
        artifact = self._query_port_artifact(
            source_run, step, result, "optimized_geometry", "molecular_geometry"
        )
        if (
            artifact is None
            or artifact.id != binding.get("artifact_id")
            or artifact.role != "optimized_geometry"
            or artifact.step_id != step.id
            or artifact.attempt != result.attempt
            or artifact.sha256 != binding.get("sha256")
        ):
            raise ValueError("history geometry artifact binding is invalid")
        path = artifact_path(self.config.data_root_path, source_run, artifact)
        parse_xyz_bytes(path.read_bytes())
        return source_run, step, result, artifact, path

    def _public_query_entry(
        self,
        subject_ref: str,
        run: Run,
        step: Step,
        tool: Any,
        *,
        name: str,
        kind: str,
        expected_type: str,
        property_name: str,
        structure: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(tool.result_metadata.get(name, {}))
        metadata.setdefault("label", name.replace("_", " "))
        metadata.setdefault("description", tool.description)
        item = {
            "subject_ref": subject_ref,
            "active_task": run.id == self._session.get("active_run_id"),
            "task": {
                "description": run.request.description,
                "status": run.status,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            },
            "system": structure,
            "step": {
                "purpose": _step_purpose(step),
                "tool": tool.name,
                "method_profile": step.parameters.get("method_profile"),
                "environment": step.parameters.get("environment"),
                "charge": step.parameters.get("charge"),
                "multiplicity": step.parameters.get("multiplicity"),
            },
            "result": {
                "name": name,
                "kind": kind,
                "property": property_name,
                "label": metadata["label"],
                "description": metadata["description"],
                "unit": expected_type if kind == "field" else None,
                "artifact_type": expected_type if kind == "port" else None,
                "caveat": metadata.get("caveat"),
                "validity": "verified",
            },
        }
        return item

    def _load_query_fact(self, subject_ref: str, property_name: str) -> dict[str, Any] | None:
        """Reload one private catalog binding and verify it has not changed."""

        resolved = self._query_bindings.get((subject_ref, property_name))
        if resolved is None:
            return None
        if (
            resolved.get("session_id") != self.session_id
            or resolved.get("property") != property_name
        ):
            return None
        run_id = resolved.get("run_id")
        if not isinstance(run_id, str):
            return None
        try:
            run = load_run(self.config.data_root_path, run_id)
        except ValueError:
            return None
        if run.session_id not in {None, self.session_id}:
            return None
        relative = resolved.get("result_path")
        step_id = resolved.get("step_id")
        if not isinstance(relative, str) or not isinstance(step_id, str):
            return None
        step = next((item for item in run.plan.steps if item.id == step_id), None)
        result = _load_bound_result(self.config.data_root_path, run, relative)
        if step is None or result is None:
            return None
        if not self._query_result_is_valid(run, step, result, relative):
            return None
        if result.attempt != resolved.get("attempt") or result.step_fingerprint != resolved.get(
            "step_fingerprint"
        ):
            return None
        try:
            tool = self.registry.get(step.tool)
        except ValueError:
            return None
        name = resolved.get("name")
        kind = resolved.get("kind")
        if not isinstance(name, str) or kind not in {"field", "port"}:
            return None
        expected_type = tool.results.get(name) if kind == "field" else tool.output_ports.get(name)
        if not isinstance(expected_type, str) or tool.result_properties.get(name) != property_name:
            return None
        value: Any
        if kind == "field":
            if name not in result.values or not _query_value_is_compatible(
                result.values[name], expected_type
            ):
                return None
            value = result.values[name]
        else:
            artifact = self._query_port_artifact(run, step, result, name, expected_type)
            if artifact is None or artifact.id != resolved.get("artifact_id"):
                return None
            value = {"artifact_id": artifact.id}
        structure = self._query_structure(run, step, result)
        metadata = dict(tool.result_metadata.get(name, {}))
        metadata.setdefault("label", name.replace("_", " "))
        metadata.setdefault("description", tool.description)
        return {
            "task_key": f"{run.id}:{step.id}",
            "subject_ref": subject_ref,
            "task_description": run.request.description,
            "task_created_at": run.created_at,
            "system": _query_system_label(structure),
            "_run": run,
            "_result": result,
            "run_status": run.status,
            "result_status": result.status,
            "step_id": step.id,
            "step_tool": step.tool,
            "method_profile": step.parameters.get("method_profile"),
            "environment": step.parameters.get("environment"),
            "name": name,
            "kind": kind,
            "value": value,
            "expected_type": expected_type,
            "result_property": property_name,
            "metadata": metadata,
        }

    def _failure_render_context(self, run: Run) -> dict[str, Any]:
        if run.status != "failed":
            return {}
        facts = self._current_run_facts(run)
        return {
            "partial_facts": facts,
            "repairs": run.repair_records,
            "incomplete_targets": self._incomplete_target_labels(run, facts),
        }

    def _current_run_facts(self, run: Run) -> list[dict[str, Any]]:
        """Return only outputs still bound to current, verified successful attempts."""

        facts: list[dict[str, Any]] = []
        for step in run.plan.steps:
            relative = run.current_results.get(step.id)
            if not isinstance(relative, str):
                continue
            result = _load_bound_result(self.config.data_root_path, run, relative)
            if result is None or not self._query_result_is_valid(run, step, result, relative):
                continue
            try:
                tool = self.registry.get(step.tool)
            except ValueError:
                continue
            structure = self._query_structure(run, step, result)
            values = {
                name: value
                for name, expected_type in tool.results.items()
                if name not in tool.output_ports
                and (value := result.values.get(name)) is not None
                and _query_value_is_compatible(value, expected_type)
            }
            ports = {
                name: artifact.id
                for name, expected_type in tool.output_ports.items()
                if (artifact := self._query_port_artifact(run, step, result, name, expected_type))
                is not None
            }
            safe_result = result.model_copy(update={"values": values, "output_ports": ports})
            facts.extend(facts_from_result(run, safe_result, self.registry, structure=structure))
        return facts

    def _incomplete_target_labels(
        self, run: Run, facts: list[dict[str, Any]] | None = None
    ) -> list[str]:
        facts = self._current_run_facts(run) if facts is None else facts
        verified = {(fact.get("step_id"), fact.get("kind"), fact.get("name")) for fact in facts}
        labels: list[str] = []
        for target in run.plan.requested_results:
            if target.check is not None:
                candidates = [
                    step
                    for step in run.plan.steps
                    if target.check in self.registry.get(step.tool).scientific_checks
                    and (target.step_id is None or target.step_id == step.id)
                ]
                if len(candidates) == 1:
                    step = candidates[0]
                    relative = run.current_results.get(step.id)
                    result = (
                        _load_bound_result(self.config.data_root_path, run, relative)
                        if relative is not None
                        else None
                    )
                    check = (
                        result.scientific_checks.get(target.check)
                        if result is not None and result.step_fingerprint == _step_fingerprint(step)
                        else None
                    )
                    if (
                        check is not None
                        and check.status == "passed"
                        and _result_check_input_is_bound(
                            self.config.data_root_path, run, result, check
                        )
                    ):
                        continue
                check_labels = {
                    "frequency_complete": "完整频率检查",
                    "local_minimum_supported": "局部极小值检查",
                }
                label = check_labels.get(target.check, target.check.replace("_", " "))
                if label not in labels:
                    labels.append(label)
                continue

            kind = "port" if target.port is not None else "field"
            name = target.port or target.field
            if name is None:
                continue
            requested_operations = set(run.request.operations)
            if (
                name == "energy"
                and "SP" in requested_operations
                and "Opt" not in requested_operations
            ):
                name = "sp_electronic_energy"
            elif (
                name == "energy"
                and "Opt" in requested_operations
                and "SP" not in requested_operations
            ):
                name = "opt_final_electronic_energy"
            elif name in {"geometry", "molecular_geometry"} and "Opt" in requested_operations:
                kind, name = "port", "optimized_geometry"
            else:
                aliases = {
                    "sp_energy": ("field", "sp_electronic_energy"),
                    "opt_energy": ("field", "opt_final_electronic_energy"),
                    "optimized_geometry": ("port", "optimized_geometry"),
                    "frequency": ("field", "vibrational_frequencies"),
                    "frequencies": ("field", "vibrational_frequencies"),
                }
                kind, name = aliases.get(name, (kind, name))

            producers = [
                step
                for step in run.plan.steps
                if name in _declared_target_names(self.registry, step, kind)
            ]
            candidates = [
                step for step in producers if target.step_id is None or target.step_id == step.id
            ]
            complete = any((step.id, kind, name) in verified for step in candidates)
            if complete:
                continue
            label = {
                "sp_electronic_energy": "单点电子能",
                "opt_final_electronic_energy": "优化后的电子能",
                "optimized_geometry": "优化后的几何",
                "geometry": "初始几何",
                "interatomic_distance": "原子间距离",
            }.get(name, name.replace("_", " "))
            label_step = candidates[0] if candidates else producers[0] if producers else None
            if label_step is not None:
                label = (
                    self.registry.get(label_step.tool)
                    .result_metadata.get(name, {})
                    .get("label", label)
                )
            if label not in labels:
                labels.append(label)

        for operation in run.request.operations:
            expected_steps = [
                step
                for step in run.plan.steps
                if operation in self.registry.get(step.tool).operations
            ]
            operation_complete = bool(expected_steps) and all(
                (relative := run.current_results.get(step.id)) is not None
                and (result := _load_bound_result(self.config.data_root_path, run, relative))
                is not None
                and self._query_result_is_valid(run, step, result, relative)
                for step in expected_steps
            )
            if not operation_complete:
                operation_label = {
                    "SP": "单点计算",
                    "Opt": "几何优化",
                    "Freq": "频率计算",
                }[operation]
                labels.insert(0, operation_label)
        return list(dict.fromkeys(labels))

    def _query_result_is_valid(self, run: Run, step: Step, result: Result, relative: str) -> bool:
        if (
            result.run_id != run.id
            or result.step_id != step.id
            or result.status != "succeeded"
            or run.current_results.get(step.id) != relative
            or result.attempt < 1
            or result.step_fingerprint != _step_fingerprint(step)
        ):
            return False
        expected_relative = f"{result.attempt_relative_path.rstrip('/')}/result.json"
        if not result.attempt_relative_path or expected_relative != relative:
            return False
        attempt_records = [item for item in run.attempts if item.get("step_id") == step.id]
        if attempt_records:
            matching_attempts = [
                item
                for item in attempt_records
                if item.get("attempt") == result.attempt
                and item.get("phase") == "finished"
                and item.get("status") == "succeeded"
            ]
            if len(matching_attempts) != 1:
                return False
            recorded_artifacts = matching_attempts[0].get("artifact_ids")
            if isinstance(recorded_artifacts, list) and any(
                artifact_id not in recorded_artifacts for artifact_id in result.artifact_ids
            ):
                return False
        for input_name, reference in step.inputs.items():
            artifact = self._artifact_from_reference(run, reference)
            if artifact is None or artifact.run_id != run.id:
                return False
            try:
                artifact_path(self.config.data_root_path, run, artifact)
            except (OSError, ValueError):
                return False
            if artifact.id not in result.input_artifact_ids:
                return False
            if result.input_bindings.get(input_name) != artifact.id:
                return False
        if step.tool == "frequency":
            required_checks = (
                "runner_succeeded",
                "exit_code_zero",
                "process_tree_empty",
                "normal_termination",
                "stdout_valid_utf8",
                "stdout_within_size_limit",
                "stderr_within_size_limit",
                "scf_converged",
                "input_hashes_match",
                "frequency_section_complete",
                "frequency_values_finite",
                "frequency_mode_indices_match_hessian",
                "hessian_present",
                "hessian_valid",
            )
            if any(result.checks.get(name) is not True for name in required_checks):
                return False
            if not _query_value_is_compatible(
                result.values.get("vibrational_frequencies"), "frequency"
            ):
                return False
            artifact = self._query_port_artifact(run, step, result, "hessian", "orca_hessian")
            if artifact is None or artifact.role != "verified_hessian":
                return False
        return True

    def _query_port_artifact(
        self,
        run: Run,
        step: Step,
        result: Result,
        name: str,
        expected_type: str,
    ) -> Any | None:
        artifact_id = result.output_ports.get(name)
        if (
            not isinstance(artifact_id, str)
            or name not in result.output_ports
            or artifact_id not in result.artifact_ids
        ):
            return None
        try:
            artifact = find_artifact(run, artifact_id)
            if (
                artifact.run_id != run.id
                or artifact.step_id != step.id
                or artifact.attempt != result.attempt
                or artifact.artifact_type != expected_type
                or artifact.role == "restart_candidate"
            ):
                return None
            path = artifact_path(self.config.data_root_path, run, artifact)
            if expected_type == "molecular_geometry":
                parse_xyz_bytes(path.read_bytes())
            return artifact
        except (OSError, ValueError):
            return None

    def _query_structure(self, run: Run, step: Step, result: Result) -> dict[str, Any]:
        artifacts: list[Any] = []
        for reference in step.inputs.values():
            artifact = self._artifact_from_reference(run, reference)
            if artifact is not None:
                artifacts.append(artifact)
        for artifact_id in result.output_ports.values():
            try:
                artifact = find_artifact(run, artifact_id)
            except ValueError:
                continue
            if artifact not in artifacts:
                artifacts.append(artifact)
        for artifact in artifacts:
            molecule = (
                artifact
                if artifact.artifact_type == "molecule" and artifact.role == "resolved_molecule"
                else self._molecule_artifact_for_geometry(run, artifact, seen=set())
            )
            if molecule is None:
                continue
            try:
                payload = json.loads(
                    artifact_path(self.config.data_root_path, run, molecule).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            facts = payload.get("facts")
            if not isinstance(facts, dict):
                continue
            return {
                key: facts[key]
                for key in ("formula", "title", "query", "cid", "atom_count")
                if key in facts and isinstance(facts[key], (str, int, float))
            }
        return {}

    def _answer_question(self, question: str, *, cancel: Event | None = None) -> AgentResponse:
        try:
            text = self.llm.complete_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer the question without claiming an unperformed computation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": question,
                                "recent_messages": self._session.get("recent_messages", [])[-12:],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                purpose="answer",
                cancel=cancel,
            )
        except LlmError as error:
            text = (
                "当前请求已取消。"
                if error.category == "cancelled"
                else f"当前配置的模型无法回答：{error}"
            )
        response = AgentResponse(text)
        self._append_message("assistant", text)
        return response

    def _answer_context(
        self,
        question: str,
        *,
        selection: QuerySelection | None,
        catalog: list[Mapping[str, Any]],
        cancel: Event | None = None,
    ) -> AgentResponse:
        if cancel is not None and cancel.is_set():
            return AgentResponse("当前请求已取消。")
        if selection is None:
            text = render_clarification({"status": "unavailable", "missing_description": question})
            response = AgentResponse(text)
            self._append_message("assistant", text)
            return response
        if selection.status != "selected":
            text = render_clarification(selection.model_dump(mode="python"))
            response = AgentResponse(text)
            self._append_message("assistant", text)
            return response

        catalog_refs = {
            str(item.get("subject_ref"))
            for item in catalog
            if isinstance(item, Mapping) and isinstance(item.get("subject_ref"), str)
        }
        if any(target.subject_ref not in catalog_refs for target in selection.targets):
            text = render_clarification({"binding_invalid": True})
            response = AgentResponse(text)
            self._append_message("assistant", text)
            return response
        selected_facts: list[dict[str, Any]] = []
        for target in selection.targets:
            fact = self._load_query_fact(target.subject_ref, target.property)
            if fact is None:
                text = "所选任务尚未得到所问性质；已保存的其他性质不能替代它。"
                response = AgentResponse(text)
                self._append_message("assistant", text)
                return response
            selected_facts.append(fact)
        targets = [target.model_dump(mode="python") for target in selection.targets]
        facts, covered = select_facts_for_question(targets, selected_facts)
        if not covered:
            text = "本次任务尚未得到所问性质；已保存的其他性质不能替代它。"
            response = AgentResponse(text)
            self._append_message("assistant", text)
            return response
        text = render_selected_facts(facts)
        first = facts[0] if facts else {}
        run = first.get("_run")
        result = first.get("_result")
        response = AgentResponse(text, run=run, result=result)
        self._append_message("assistant", text)
        return response


def _bind_input_geometry(plan: Plan, artifact_id: str) -> Plan:
    steps: list[Step] = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        reference = inputs.get("geometry")
        if reference is not None and reference.artifact_id == INPUT_GEOMETRY_PLACEHOLDER:
            inputs["geometry"] = InputReference(artifact_id=artifact_id)
        steps.append(step.model_copy(update={"inputs": inputs}))
    return plan.model_copy(update={"steps": steps})


def _normalize_explicit_plan(registry: ToolRegistry, plan: Plan) -> Plan:
    """Resolve defaults before an explicit Run receives its acceptance snapshot."""

    steps: list[Step] = []
    for step in plan.steps:
        tool = registry.get(step.tool)
        if tool.requires_compute_permission:
            parameters = tool.validate_parameters(step.parameters)
            if tool.execution_budget == "orca":
                _validate_orca_profile(parameters)
            step = step.model_copy(update={"parameters": parameters})
        steps.append(step)
    return plan.model_copy(update={"steps": steps})


def _is_parameter_continuation(
    run: Run,
    intake: Any,
    message: str,
    explicit_parameters: dict[str, Any],
) -> bool:
    if (
        intake.intent != "chemistry_compute"
        or not explicit_parameters
        or intake.molecule_query
        or intake.structure_input
        or intake.history_geometry_alias
        or intake.requested_results
    ):
        return False
    if intake.operations and run.request.operations:
        if intake.operations != run.request.operations:
            return False
    if _looks_like_molecule_change(message) and not _looks_like_parameter_only_change(message):
        return False
    return run.waiting_for in {"clarification", "confirmation"}


def _selected_artifact_aliases(
    request: Request, selected_history_alias: str | None
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if selected_history_alias is not None:
        aliases[selected_history_alias] = selected_history_alias
    structure_input = request.structure_input
    if isinstance(structure_input, Mapping) and (
        structure_input.get("xyz_text") is not None or structure_input.get("xyz") is not None
    ):
        aliases["request_geometry"] = "request_geometry"
        aliases[INPUT_GEOMETRY_PLACEHOLDER] = INPUT_GEOMETRY_PLACEHOLDER
    return aliases


def _requests_history_geometry(message: str) -> bool:
    history = r"刚才|之前|此前|上次|上一个|最近|历史|previous|prior|last|recent"
    geometry = r"结构|几何|优化结果|优化后的|geometry|structure|optimized"
    explicit_alias = re.search(r"(?<![A-Za-z0-9_-])geometry_\d+(?![A-Za-z0-9_-])", message, re.I)
    alias_action = re.search(
        r"use|using|reuse|select|choose|take|apply|使用|复用|沿用|采用|选择|基于|选用",
        message,
        re.I,
    )
    if explicit_alias is not None and alias_action is not None:
        denied_alias_action = re.search(
            r"(?:do\s+not|don't|never|without|avoid|must\s+not|should\s+not|"
            r"不要(?:再)?|不(?:要)?|别|禁止)"
            r".{0,16}(?:use|using|reuse|select|choose|take|apply|使用|复用|沿用|采用|选择|基于|选用)"
            r".{0,24}geometry_\d+",
            message,
            re.I,
        )
        return denied_alias_action is None
    relation = re.search(
        rf"(?:{history}).{{0,32}}(?:{geometry})|"
        rf"(?:{geometry}).{{0,32}}(?:{history})",
        message,
        re.I,
    )
    if relation is None:
        return False
    if re.search(
        r"(?:do\s+not|don't|never|without|avoid|exclude|must\s+not|should\s+not|"
        r"不要(?:再)?|不(?:要)?|别|禁止)"
        r".{0,12}(?:use|using|reuse|select|choose|take|apply|使用|复用|沿用|采用|选择|用)"
        r".{0,32}(?:刚才|之前|此前|上次|上一个|最近|历史|previous|prior|last|recent)"
        r".{0,24}(?:结构|几何|优化结果|优化后的|geometry|structure|optimized)|"
        r"(?:刚才|之前|此前|上次|上一个|最近|历史|previous|prior|last|recent)"
        r".{0,32}(?:结构|几何|优化结果|优化后的|geometry|structure|optimized)"
        r".{0,12}(?:不要(?:再)?|不(?:要)?|别|禁止).{0,8}"
        r"(?:use|using|reuse|select|choose|take|apply|使用|复用|沿用|采用|选择|用)",
        message,
        re.I,
    ):
        return False
    action = (
        r"use|using|reuse|select|choose|take|apply|run|perform|calculate|compute|conduct|"
        r"single[ -]?point|\bSP\b|\bFreq\b|\bOpt\b|"
        r"使用|复用|沿用|采用|选择|基于|继续|运行|执行|计算|单点|频率|优化"
    )
    return bool(
        re.search(rf"(?:{action}).{{0,64}}(?:{history}|{geometry})", message, re.I)
        or re.search(rf"(?:{history}|{geometry}).{{0,64}}(?:{action})", message, re.I)
    )


def _explicit_history_geometry_alias(
    message: str, geometry_catalog: list[dict[str, Any]]
) -> str | None:
    """Return an alias only when the user named exactly one offered alias."""

    available = {
        str(item["alias"]).casefold(): str(item["alias"])
        for item in geometry_catalog
        if isinstance(item.get("alias"), str)
    }
    mentioned = {
        match.casefold()
        for match in re.findall(r"(?<![A-Za-z0-9_-])geometry_\d+(?![A-Za-z0-9_-])", message, re.I)
    }
    selected = mentioned & available.keys()
    if len(selected) != 1:
        return None
    return available[next(iter(selected))]


def _validate_selected_geometry_binding(
    plan: Plan, request: Request, selected_history_alias: str | None
) -> None:
    selected_alias = selected_history_alias
    if (
        selected_alias is None
        and isinstance(request.structure_input, Mapping)
        and (
            request.structure_input.get("xyz_text") is not None
            or request.structure_input.get("xyz") is not None
        )
    ):
        selected_alias = "request_geometry"
    if selected_alias is None:
        return
    if any(step.tool in {"resolve_molecule", "generate_geometry"} for step in plan.steps):
        raise ValueError(
            "a supplied geometry must be used directly without new structure preparation"
        )
    if not any(
        (reference := step.inputs.get("geometry")) is not None
        and reference.artifact_id == selected_alias
        for step in plan.steps
    ):
        raise ValueError("Plan does not use the selected supplied geometry as a calculation input")


def _looks_like_molecule_change(message: str) -> bool:
    return bool(
        re.search(
            r"(?:改成|改为|改用|换成|换为|换用|换一个|使用.+代替|换.+instead|"
            r"change.+to|switch.+to|use.+instead|instead\s+of)",
            message,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_parameter_only_change(message: str) -> bool:
    """Avoid treating a scoped numerical parameter edit as a molecule change."""

    parameter_change = re.compile(
        r"(?:几何优化|优化几何|geometry optimization|geom_maxiter|SCF|scf_maxiter)"
        r".{0,16}?(?:改成|改为|设为|设置为|change(?:d)?\s+to|set\s+to|=)\s*[-+]?\d+",
        re.IGNORECASE,
    )
    matches = list(parameter_change.finditer(message))
    if not matches:
        return False
    remainder = parameter_change.sub("", message)
    return not _looks_like_molecule_change(remainder)


def _iteration_increase_allowed(message: str) -> bool:
    """Honor an explicit request not to raise an iteration limit."""

    return not bool(
        re.search(
            r"(?:不得|不要|禁止|不允许|不能|无需|不需要)\s*"
            r"(?:再?\s*)?(?:增加|提高|放宽|上调)\s*(?:迭代|步数|次数|上限|maxiter)?|"
            r"(?:do\s*not|don't|without)\s+(?:increase|raise|relax|change)"
            r"\s+(?:the\s+)?(?:iteration|iterations|maxiter|iteration\s+limit)",
            message,
            flags=re.IGNORECASE,
        )
    )


def _replace_step(plan: Plan, replacement: Step) -> Plan:
    return Plan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "steps": [replacement if step.id == replacement.id else step for step in plan.steps],
        },
        strict=True,
    )


def _tool_parameter_fields(tool: Tool) -> frozenset[str]:
    if tool.parameter_type is None:
        return frozenset()
    return frozenset(tool.parameter_type.model_fields)


def _validate_request_parameter_scope(request: Request, plan: Plan, registry: ToolRegistry) -> None:
    """Require each explicit request-level parameter to have a consumer in this Plan."""

    explicit_names = {
        name
        for parameters in (request.explicit_parameters, request.user_modifications)
        for name, value in parameters.items()
        if value is not None
    }
    if not explicit_names:
        return
    supported_names: set[str] = set()
    for step in plan.steps:
        tool = registry.get(step.tool)
        supported_names.update(tool.request_parameters)
    unscoped = sorted(explicit_names - supported_names)
    if unscoped:
        raise ValueError(
            "request parameter(s) have no compatible calculation step in this Plan: "
            + ", ".join(unscoped)
        )


def _next_ready_step(run: Run, data_root: str | None = None) -> Step | None:
    for step in run.plan.steps:
        if step.id in run.current_results:
            continue
        if all(
            reference.step_id is None or reference.step_id in run.current_results
            for reference in step.inputs.values()
        ) and all(
            data_root is not None
            and _goal_check_requirement_met(data_root, run, requirement, dependent_step=step)
            for requirement in step.goal_checks
        ):
            return step
    return None


def _goal_check_requirement_met(
    data_root: str,
    run: Run,
    requirement: Any,
    *,
    dependent_step: Step | None = None,
) -> bool:
    source_step = next(
        (item for item in run.plan.steps if item.id == requirement.source_step_id), None
    )
    relative = run.current_results.get(requirement.source_step_id)
    if source_step is None or relative is None:
        return False
    result = _load_bound_result(data_root, run, relative)
    if (
        result is None
        or result.status != "succeeded"
        or result.run_id != run.id
        or result.step_id != source_step.id
        or result.step_fingerprint != _step_fingerprint(source_step)
    ):
        return False
    check = result.scientific_checks.get(requirement.check)
    return bool(
        check is not None
        and check.status == requirement.required_status
        and _result_check_input_is_bound(data_root, run, result, check)
        and (
            dependent_step is None
            or _goal_check_geometry_matches(requirement, source_step, dependent_step)
        )
    )


def _goal_check_geometry_matches(requirement: Any, source_step: Step, dependent_step: Step) -> bool:
    """Keep a local-minimum gate attached to the exact geometry it verified."""

    if requirement.check != "local_minimum_supported":
        return True
    source_geometry = source_step.inputs.get("geometry")
    dependent_geometry = dependent_step.inputs.get("geometry")
    return (
        source_step.tool == "frequency"
        and source_geometry is not None
        and source_geometry == dependent_geometry
    )


def _result_check_input_is_bound(data_root: str, run: Run, result: Result, check: Any) -> bool:
    """Bind a scientific check to the exact, hash-verified geometry input."""

    artifact_id = result.input_bindings.get("geometry")
    if (
        not isinstance(artifact_id, str)
        or artifact_id not in result.input_artifact_ids
        or not isinstance(check.input_geometry_sha256, str)
    ):
        return False
    try:
        artifact = find_artifact(run, artifact_id)
        if (
            artifact.run_id != run.id
            or artifact.artifact_type != "molecular_geometry"
            or artifact.sha256 != check.input_geometry_sha256
        ):
            return False
        artifact_path(data_root, run, artifact)
    except (OSError, ValueError):
        return False
    return True


def _unmet_goal_checks(data_root: str, run: Run, registry: ToolRegistry) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for step in run.plan.steps:
        if step.id in run.current_results:
            continue
        for requirement in step.goal_checks:
            source_step = next(
                (item for item in run.plan.steps if item.id == requirement.source_step_id), None
            )
            relative = run.current_results.get(requirement.source_step_id)
            source_result = (
                _load_bound_result(data_root, run, relative) if relative is not None else None
            )
            if (
                source_step is None
                or source_result is None
                or source_result.status != "succeeded"
                or source_result.step_fingerprint != _step_fingerprint(source_step)
            ):
                continue
            check = source_result.scientific_checks.get(requirement.check)
            actual_status = check.status if check is not None else "unverified"
            input_bound = (
                check is not None
                and _result_check_input_is_bound(data_root, run, source_result, check)
                and _goal_check_geometry_matches(requirement, source_step, step)
            )
            if actual_status != requirement.required_status or not input_bound:
                blocked.append(
                    {
                        "dependent_step_id": step.id,
                        "source_step_id": requirement.source_step_id,
                        "check": requirement.check,
                        "required_status": requirement.required_status,
                        "actual_status": actual_status if input_bound else "unverified",
                        "reason": (
                            check.reason
                            if check is not None and input_bound
                            else "scientific check is not bound to the dependent Step geometry"
                            if check is not None
                            else "check result is missing"
                        ),
                    }
                )
    for target in run.plan.requested_results:
        if target.check is None:
            continue
        candidates = [
            step
            for step in run.plan.steps
            if target.check in _declared_target_names(registry, step, "check")
        ]
        if target.step_id is not None:
            candidates = [step for step in candidates if step.id == target.step_id]
        if len(candidates) != 1:
            continue
        source_step = candidates[0]
        relative = run.current_results.get(source_step.id)
        source_result = (
            _load_bound_result(data_root, run, relative) if relative is not None else None
        )
        if (
            source_result is None
            or source_result.status != "succeeded"
            or source_result.step_fingerprint != _step_fingerprint(source_step)
        ):
            continue
        check = source_result.scientific_checks.get(target.check)
        actual_status = check.status if check is not None else "unverified"
        input_bound = check is not None and _result_check_input_is_bound(
            data_root, run, source_result, check
        )
        if actual_status != "passed" or not input_bound:
            blocked.append(
                {
                    "dependent_step_id": None,
                    "source_step_id": source_step.id,
                    "check": target.check,
                    "required_status": "passed",
                    "actual_status": actual_status if input_bound else "unverified",
                    "reason": check.reason if check is not None else "check result is missing",
                }
            )
    return blocked


def _declared_target_names(registry: ToolRegistry, step: Step, kind: str) -> set[str]:
    tool = registry.get(step.tool)
    if kind == "port":
        return set(tool.output_ports)
    if kind == "check":
        return set(tool.scientific_checks)
    return set(tool.results) - set(tool.output_ports)


def _invalidate_current_results(run: Run, changed_step_id: str) -> None:
    """Drop current outputs for a changed step and every downstream consumer."""

    invalidated = {changed_step_id}
    changed = True
    while changed:
        changed = False
        for step in run.plan.steps:
            if step.id in invalidated:
                continue
            if any(reference.step_id in invalidated for reference in step.inputs.values()) or any(
                requirement.source_step_id in invalidated for requirement in step.goal_checks
            ):
                invalidated.add(step.id)
                changed = True
    for step_id in invalidated:
        run.current_results.pop(step_id, None)
        run.parameter_sources_by_step.pop(step_id, None)
        run.step_status[step_id] = "planned"


def _requested_results_satisfied(data_root: str, run: Run, registry: ToolRegistry) -> bool:
    for operation in run.request.operations:
        required_steps = [
            step for step in run.plan.steps if operation in registry.get(step.tool).operations
        ]
        if not required_steps:
            return False
        for step in required_steps:
            relative = run.current_results.get(step.id)
            result = _load_bound_result(data_root, run, relative) if relative else None
            if (
                result is None
                or result.run_id != run.id
                or result.step_id != step.id
                or result.attempt < 1
                or result.status != "succeeded"
                or result.step_fingerprint != _step_fingerprint(step)
            ):
                return False
    if run.plan.requested_results:
        for target in run.plan.requested_results:
            step_id = target.step_id
            if step_id is None:
                # Old M0 targets were unqualified; the registry ensures this
                # is unique, so the first producer is the only legal binding.
                kind = (
                    "check"
                    if target.check is not None
                    else "port"
                    if target.port is not None
                    else "field"
                )
                name = target.check or target.port or target.field
                matches = [
                    step.id
                    for step in run.plan.steps
                    if name is not None and name in _declared_target_names(registry, step, kind)
                ]
                if len(matches) != 1:
                    return False
                step_id = matches[0]
            relative = run.current_results.get(step_id)
            if relative is None:
                return False
            result = _load_bound_result(data_root, run, relative)
            if result is None or result.status != "succeeded" or result.step_id != step_id:
                return False
            step = next((item for item in run.plan.steps if item.id == step_id), None)
            if step is None or result.step_fingerprint != _step_fingerprint(step):
                return False
            if target.field is not None and target.field not in result.values:
                return False
            if target.field is not None:
                tool = registry.get(step.tool)
                expected_type = tool.results.get(target.field)
                if (
                    expected_type is None
                    or target.field in tool.output_ports
                    or not _query_value_is_compatible(result.values[target.field], expected_type)
                ):
                    return False
            if target.check is not None:
                check = result.scientific_checks.get(target.check)
                if check is None or check.status != "passed":
                    return False
                if not _result_check_input_is_bound(data_root, run, result, check):
                    return False
            if target.port is not None:
                tool = registry.get(step.tool)
                artifact_id = result.output_ports.get(target.port)
                if (
                    target.port not in tool.output_ports
                    or artifact_id is None
                    or artifact_id not in result.artifact_ids
                ):
                    return False
                try:
                    artifact = find_artifact(run, artifact_id)
                    artifact_path(data_root, run, artifact)
                except (OSError, ValueError):
                    return False
                if (
                    artifact.step_id != step.id
                    or artifact.attempt != result.attempt
                    or artifact.artifact_type != tool.output_ports[target.port]
                    or artifact.role == "restart_candidate"
                ):
                    return False
        return True
    for step in run.plan.steps:
        relative = run.current_results.get(step.id)
        result = _load_bound_result(data_root, run, relative) if relative else None
        if (
            result is None
            or result.status != "succeeded"
            or result.step_id != step.id
            or result.step_fingerprint != _step_fingerprint(step)
        ):
            return False
    return True


def _load_bound_result(data_root: str, run: Run, relative: str) -> Result | None:
    root = run_directory(data_root, run.id).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or path.name != "result.json":
        return None
    try:
        return Result.model_validate(json.loads(path.read_text(encoding="utf-8")), strict=True)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _step_fingerprint(step: Step) -> str:
    import hashlib

    payload = json.dumps(
        step.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _query_value_is_compatible(value: Any, declared_type: str) -> bool:
    """Accept only finite values whose persisted unit/type matches the Tool."""

    if declared_type == "frequency":
        if not isinstance(value, Mapping) or value.get("complete") is not True:
            return False
        if value.get("unit") != "cm^-1" or type(value.get("scaling_applied")) is not bool:
            return False
        modes = value.get("modes")
        if not isinstance(modes, list) or not modes:
            return False
        indices: list[int] = []
        for mode in modes:
            if not isinstance(mode, Mapping):
                return False
            index, number = mode.get("index"), mode.get("value")
            if type(index) is not int or type(number) not in {int, float}:
                return False
            if mode.get("unit") != "cm^-1" or not math.isfinite(float(number)):
                return False
            indices.append(index)
        if indices != list(range(len(indices))):
            return False
        factor = value.get("scaling_factor")
        return factor is None or (
            type(factor) in {int, float} and math.isfinite(float(factor)) and factor > 0
        )
    if declared_type == "Eh":
        if not isinstance(value, Mapping):
            return False
        raw = value.get("value")
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            return False
        if value.get("unit") != "Eh":
            return False
        token = value.get("token")
        if token is not None:
            if not isinstance(token, str):
                return False
            try:
                if not math.isfinite(float(token)):
                    return False
            except ValueError:
                return False
        return True
    if declared_type == "angstrom":
        if not isinstance(value, Mapping):
            return False
        raw = value.get("value")
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            return False
        if value.get("unit") != "angstrom":
            return False
        atom_indices = value.get("atom_indices")
        if atom_indices is not None:
            if (
                not isinstance(atom_indices, list)
                or len(atom_indices) != 2
                or any(type(item) is not int or item < 1 for item in atom_indices)
                or atom_indices[0] == atom_indices[1]
            ):
                return False
        atom_symbols = value.get("atom_symbols")
        if atom_symbols is not None and (
            not isinstance(atom_symbols, list)
            or len(atom_symbols) != 2
            or any(not isinstance(item, str) or not item for item in atom_symbols)
        ):
            return False
        return True
    if declared_type == "integer":
        if isinstance(value, Mapping):
            value = value.get("value")
        return type(value) is int
    if declared_type == "text":
        if isinstance(value, Mapping):
            value = value.get("value")
        return type(value) is str
    if isinstance(value, Mapping):
        raw = value.get("value")
        if type(raw) in {int, float} and not math.isfinite(float(raw)):
            return False
        unit = value.get("unit")
        if unit is not None and unit != declared_type:
            return False
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _step_purpose(step: Step) -> str:
    return {
        "resolve_molecule": "解析分子身份",
        "generate_geometry": "生成初始几何",
        "single_point": "计算单点电子能",
        "optimize_geometry": "进行几何优化并检查收敛",
        "frequency": "计算振动频率并检查 Hessian",
        "geometry_distance": "测量指定原子间距离",
    }.get(step.tool, step.tool)


def _query_system_label(structure: Mapping[str, Any]) -> str:
    formula = structure.get("formula")
    if formula == "H2O":
        return "水分子（H₂O）"
    title = structure.get("title")
    if isinstance(title, str) and title and title.casefold() not in {"o", "water"}:
        return title
    if isinstance(formula, str) and formula:
        return formula.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    query = structure.get("query")
    if isinstance(query, str) and query:
        return query
    return "该体系"


def _validate_orca_profile(parameters: dict[str, Any]) -> None:
    profile = get_profile(parameters["method_profile"])
    if parameters["environment"] not in profile.supported_environments:
        raise ValueError(
            f"environment {parameters['environment']!r} is not implemented for {profile.name!r}"
        )


def _plan_fingerprint(plan: Plan) -> str:
    import hashlib

    payload = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["Agent", "AgentResponse", "INPUT_GEOMETRY_PLACEHOLDER"]
