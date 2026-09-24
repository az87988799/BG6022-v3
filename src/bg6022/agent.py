"""The single Plan -> Tool -> Result loop used by CLI and chat."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from threading import Event
from typing import Any

from .answer import (
    AnswerOutput,
    AnswerSection,
    compose_answer,
    facts_from_result,
    render_answer_output,
    render_clarification,
    render_confirmation,
    render_result,
    render_run,
    select_facts_for_question,
    validate_result_answer,
)
from .canonicalize import canonicalize_modification, semantic_to_intake
from .config import AppConfig
from .llm import LlmClient, LlmError
from .models import InputReference, Plan, Request, Result, Run, Step, Subject, Tool
from .molecule_identity import (
    build_identity_constraint,
    canonical_structure,
    extract_explicit_smiles,
    formula_token_from_text,
    identity_matches_facts,
    normalize_formula_token,
)
from .output_contracts import is_compatible_value, public_type_info
from .plan_builder import PlanBuildError, build_plan
from .planner import (
    IntakeOutput,
    IntakeSubjectProposal,
    QuerySelection,
    _request_output_preferences,
    apply_plan_change,
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
from .semantic import semantic_message
from .session import (
    MAX_RECENT_RUNS,
    artifact_path,
    create_run,
    execution_fingerprint,
    find_artifact,
    load_run,
    load_session,
    new_id,
    publish_step_result,
    register_bytes_artifact,
    register_file_artifact,
    run_directory,
    save_run,
    save_session,
    sha256_file,
    utc_now,
)
from .tools.molecule import (
    parse_xyz_bytes,
    resolve_artifact_reference,
)
from .tools.pubchem import _facts_from_smiles
from .tools.registry import ToolRegistry, merge_explicit_step_parameters

INPUT_GEOMETRY_PLACEHOLDER = "__input_geometry__"
MAX_QUERY_CATALOG_ITEMS = 24
MAX_HISTORY_GEOMETRIES = 8
MAX_FILE_PREVIEW_BYTES = 16 * 1024
MAX_FILE_PREVIEW_LINES = 256
MAX_TOTAL_FILE_PREVIEW_BYTES = 64 * 1024
_PENDING_MOLECULE_CATEGORIES = frozenset(
    {
        "ambiguous_molecule",
        "molecule_identity_not_found",
        "molecule_name_not_found",
        "molecule_search_incomplete",
        "molecule_source_unverified",
        "identity_mismatch",
    }
)


def _is_waiting_for_identity(run: Run | None) -> bool:
    return bool(
        run is not None
        and run.status == "waiting"
        and run.waiting_for == "clarification"
        and (
            run.pending_data.get("input_requirement") == "molecule_identity"
            or run.pending_data.get("category") in _PENDING_MOLECULE_CATEGORIES
        )
    )


def _pending_intake_context(run: Run | None) -> dict[str, Any]:
    if run is None or run.status != "waiting":
        return {}
    waiting_step = next(
        (item for item in run.plan.steps if item.id == run.pending_data.get("step_id")),
        None,
    )
    subject_id = waiting_step.subject_id if waiting_step is not None else None
    subject = run.request.subjects.get(subject_id or "", {})
    subject_input = subject.get("structure_input", {}) if isinstance(subject, Mapping) else {}
    identity = (
        subject_input.get("molecule_identity") if isinstance(subject_input, Mapping) else None
    )
    if not isinstance(identity, Mapping):
        identity = run.request.structure_input.get("molecule_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    candidates = run.pending_data.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    public_candidates = [
        {
            key: item[key][:256] if isinstance(item[key], str) else item[key]
            for key in ("choice_id", "cid", "title", "formula")
            if key in item
        }
        for item in candidates[:5]
        if isinstance(item, Mapping)
    ]
    return {
        "waiting_for": run.waiting_for,
        "identity_required": _is_waiting_for_identity(run),
        "can_replace_identity": sum(step.tool == "resolve_molecule" for step in run.plan.steps)
        == 1,
        "category": run.pending_data.get("category"),
        "pending_subject_id": subject_id,
        "raw_query": str(identity.get("raw_query") or "")[:128],
        "lookup_query": str(identity.get("lookup_query") or "")[:128],
        "operations": list(run.request.operations),
        "requirements": [
            {
                "requirement_id": item.id,
                "capability": item.capability,
                "subject_id": item.subject_id,
                "parameters": dict(item.parameters),
            }
            for item in run.request.requirements
        ],
        "requested_results": [
            target.port or target.field or target.check
            for target in run.request.requested_results[:24]
        ],
        "candidates": public_candidates,
    }


@dataclass(frozen=True)
class AgentResponse:
    text: str
    run: Run | None = None
    result: Result | None = None
    files: tuple[dict[str, Any], ...] = ()
    delivery: dict[str, Any] = dataclass_field(default_factory=dict)


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
                "last_delivery": [],
                "active_run_id": None,
                "pending_prompt": "session record is invalid; use /new or /exit",
            }
        self._session.setdefault("last_delivery", [])

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
        # Each Tool owns its environment preflight; the core only invokes the
        # declared hook for capabilities that require one.
        for step in plan.steps:
            self.registry.get(step.tool).preflight()
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
        run.plan = _normalize_explicit_plan(
            self.registry, bound_plan, defaults=self.config.defaults
        )
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
        route_review_used = False
        answer_draft: AnswerOutput | None = None
        answer_error: str | None = None
        try:
            self._ensure_request_active(request_token, request_cancel)
            current = self._coerce_run(None)
            if _is_waiting_for_identity(current):
                selection = _pending_molecule_selection(text, current.pending_data)
                if selection is not None and selection.get("invalid"):
                    response = AgentResponse(str(selection["invalid"]), run=current)
                    self._record_response(response, cancel=request_cancel)
                    return response
                if selection is not None and (
                    selection.get("candidate") is not None or selection.get("explicit_new")
                ):
                    response = self._apply_molecule_clarification(
                        current,
                        str(selection["query"]),
                        str(selection["input_kind"]),
                        candidate=selection.get("candidate"),
                        preserve_identity=bool(selection.get("preserve_identity")),
                        name_evidence=selection.get("name_evidence"),
                        message=text,
                        cancel=request_cancel,
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
            result_catalog = self._build_query_catalog()
            geometry_catalog, geometry_bindings = self._build_geometry_catalog()
            capability_catalog = self.registry.result_capabilities()
            semantic_enabled = bool(self.config.runtime.semantic_planner_v1) and not (
                _is_waiting_for_identity(current)
            )
            pending_tasks, pending_ref_to_requirement = _semantic_pending_tasks(
                current, self.registry
            )
            if (
                semantic_enabled
                and pending_tasks
                and _ambiguous_iteration_parameter_change(text)
            ):
                response = AgentResponse(
                    "“迭代上限”可能指几何优化迭代或 SCF 电子迭代。"
                    "请明确要修改哪一种；当前任务未更改。",
                    run=current,
                )
                self._record_response(response, cancel=request_cancel)
                return response
            if semantic_enabled:
                semantic = semantic_message(
                    self.llm,
                    text,
                    registry=self.registry,
                    recent_context={
                        "recent_messages": self._session.get("recent_messages", []),
                        "recent_results": self._session.get("recent_results", []),
                        "last_delivery": self._session.get("last_delivery", []),
                    },
                    pending_tasks=pending_tasks,
                    result_catalog=result_catalog,
                    cancel=request_cancel,
                )
                self._persist_llm_diagnostics(llm_call_cursor, stage="semantic")
                if semantic.mode == "modify":
                    if current is None:
                        response = AgentResponse("当前没有可修改的等待任务。")
                    else:
                        try:
                            requirement_id, patch, constraint_patch = canonicalize_modification(
                                semantic.modification,
                                pending_ref_to_requirement=pending_ref_to_requirement,
                                run=current,
                                registry=self.registry,
                            )
                            response = self._apply_parameter_update(
                                current,
                                patch,
                                requirement_id=requirement_id,
                                requirement_constraint_patch=constraint_patch,
                                cancel=request_cancel,
                            )
                        except (TypeError, ValueError) as error:
                            response = AgentResponse(
                                f"参数修改已拒绝（rejected）：{error}", run=current
                            )
                    self._record_response(response, cancel=request_cancel)
                    return response
                if semantic.mode == "clarify":
                    response = AgentResponse(str(semantic.clarification), run=current)
                    self._record_response(response, cancel=request_cancel)
                    return response
                if semantic.mode == "unsupported":
                    response = AgentResponse(
                        "当前工具目录不支持这些计算要求："
                        + "；".join(semantic.unsupported_requirements)
                        + "。本次没有启动计算。",
                        run=current,
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                intake = semantic_to_intake(text, semantic, registry=self.registry)
            else:
                intake = intake_message(
                    self.llm,
                    text,
                    context={
                        "recent_messages": self._session.get("recent_messages", []),
                        "recent_results": self._session.get("recent_results", []),
                        "last_delivery": self._session.get("last_delivery", []),
                    },
                    result_catalog=result_catalog,
                    geometry_catalog=geometry_catalog,
                    capability_catalog=capability_catalog,
                    registry=self.registry,
                    pending_context=_pending_intake_context(current),
                    cancel=request_cancel,
                )
                self._persist_llm_diagnostics(llm_call_cursor, stage="intake")
            llm_call_cursor = self._llm_call_count()
            self._ensure_request_active(request_token, request_cancel)

            answer_stage_eligible = intake.intent in {"chemistry_qa", "daily_qa"} or (
                intake.intent == "context_query"
                and (intake.query_selection is None or intake.query_selection.status != "selected")
            )
            if answer_stage_eligible:
                llm_stage = "answer"
                answer_draft, answer_error = self._compose_answer_draft(
                    text,
                    mode="knowledge",
                    capability_catalog=capability_catalog,
                    available_outputs=result_catalog,
                    context={"recent_messages": self._session.get("recent_messages", [])[-12:]},
                    cancel=request_cancel,
                )
                llm_call_cursor = self._llm_call_count()
                if (
                    answer_draft is not None
                    and answer_draft.action == "needs_tools"
                    and not route_review_used
                ):
                    route_targets = self._validated_tool_targets(answer_draft, capability_catalog)
                    if route_targets:
                        route_review_used = True
                        llm_stage = "intake"
                        validation_feedback = (
                            "The previous route classified this as ordinary question "
                            "answering, but the public answer stage identified a concrete "
                            f"registered Tool target: {route_targets}. Re-evaluate the original "
                            "user message and preserve the molecule, source, and requested "
                            "result exactly; do not add an operation that the user did not request."
                        )
                        if semantic_enabled:
                            semantic = semantic_message(
                                self.llm,
                                text,
                                registry=self.registry,
                                recent_context={
                                    "recent_messages": self._session.get("recent_messages", []),
                                    "recent_results": self._session.get("recent_results", []),
                                    "last_delivery": self._session.get("last_delivery", []),
                                },
                                pending_tasks=pending_tasks,
                                result_catalog=result_catalog,
                                validation_feedback=validation_feedback,
                                cancel=request_cancel,
                            )
                            self._persist_llm_diagnostics(llm_call_cursor, stage="semantic")
                            if semantic.mode in {"compute", "qa", "context_query"}:
                                intake = semantic_to_intake(
                                    text, semantic, registry=self.registry
                                )
                            elif semantic.mode == "unsupported":
                                response = AgentResponse(
                                    "当前工具目录不支持这些计算要求："
                                    + "；".join(semantic.unsupported_requirements),
                                    run=current,
                                )
                                self._record_response(response, cancel=request_cancel)
                                return response
                            elif semantic.mode == "clarify":
                                response = AgentResponse(
                                    str(semantic.clarification), run=current
                                )
                                self._record_response(response, cancel=request_cancel)
                                return response
                            elif semantic.mode == "modify":
                                response = AgentResponse(
                                    "当前没有可修改的等待任务。", run=current
                                )
                                self._record_response(response, cancel=request_cancel)
                                return response
                            else:
                                intake = IntakeOutput(intent="chemistry_qa")
                        else:
                            intake = intake_message(
                                self.llm,
                                text,
                                context={
                                    "recent_messages": self._session.get("recent_messages", []),
                                    "recent_results": self._session.get("recent_results", []),
                                    "last_delivery": self._session.get("last_delivery", []),
                                },
                                result_catalog=result_catalog,
                                geometry_catalog=geometry_catalog,
                                capability_catalog=capability_catalog,
                                registry=self.registry,
                                pending_context=_pending_intake_context(current),
                                validation_feedback=validation_feedback,
                                cancel=request_cancel,
                            )
                            self._persist_llm_diagnostics(llm_call_cursor, stage="intake")
                        llm_call_cursor = self._llm_call_count()
                        self._ensure_request_active(request_token, request_cancel)
                        answer_draft = None
                        answer_error = None
                        if intake.intent in {"chemistry_qa", "daily_qa"}:
                            answer_draft = AnswerOutput(
                                action="clarify",
                                clarification=(
                                    "我识别到这可能需要一个已注册工具产生具体结果，"
                                    "但还无法形成合法的工具请求；请明确要获取的结构或文件。"
                                ),
                            )

            if intake.intent in {"chemistry_qa", "daily_qa"}:
                return self._answer_question(
                    text,
                    draft=answer_draft,
                    error=answer_error,
                    cancel=request_cancel,
                )
            if intake.intent == "context_query":
                return self._answer_context(
                    text,
                    selection=intake.query_selection,
                    catalog=result_catalog,
                    preferences=_request_output_preferences(text, intake.output_preferences),
                    cancel=request_cancel,
                )

            if intake.pending_action == "clarify":
                response = AgentResponse(
                    "你是在补充当前任务的分子身份，还是要发起一个新的计算？"
                    "请直接说明；当前任务仍保留。",
                    run=current,
                )
                self._record_response(response, cancel=request_cancel)
                return response

            if intake.pending_action in {"supplement_identity", "replace_identity"}:
                if current is None:
                    raise ValueError("waiting task disappeared before identity update")
                query, kind = _validated_identity_reply(intake, message=text)
                self._ensure_request_active(request_token, request_cancel)
                response = self._apply_molecule_clarification(
                    current,
                    query,
                    kind,
                    preserve_identity=intake.pending_action == "supplement_identity",
                    name_evidence=intake.molecule_name_evidence,
                    message=text,
                    cancel=request_cancel,
                )
                self._record_response(response, cancel=request_cancel)
                return response

            if _is_waiting_for_identity(current) and intake.pending_action == "none":
                stripped = text.strip()
                if (
                    intake.molecule_input_kind == "smiles"
                    and re.fullmatch(r"[-A-Za-z0-9@+_=#\\/%().:\[\]]+", stripped)
                    and intake.molecule_query != stripped
                ):
                    response = AgentResponse(
                        "当前身份补充未匹配到候选、CID 或可验证的 SMILES；原任务保持等待。",
                        run=current,
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response

            parameter_continuation = (
                current is not None
                and current.status == "waiting"
                and _is_parameter_continuation(
                    current,
                    intake,
                    text,
                    dict(intake.explicit_parameters),
                )
            )
            index_parameter_names = (
                self.registry.request_index_parameter_fields_for_plan(current.plan)
                if parameter_continuation
                else self.registry.request_index_parameter_fields(
                    intake.operations, intake.requested_results
                )
            )
            if not parameter_continuation and intake.requirements:
                index_parameter_names.update(
                    self.registry.request_index_parameter_fields_for_requirements(
                        intake.requirements
                    )
                )
            normalized_parameters = normalize_user_explicit_parameters(
                text,
                intake.explicit_parameters,
                intake.electronic_state_candidates,
                parameter_names=tuple(index_parameter_names),
            )
            explicit_parameters = normalized_parameters.explicit_parameters
            blocking = intake_blocking_requirements(intake, self.registry)
            if normalized_parameters.parameter_issues:
                response = AgentResponse(
                    _parameter_issue_clarification(normalized_parameters.parameter_issues),
                    run=current,
                )
                self._record_response(response, cancel=request_cancel)
                return response
            if blocking:
                if (
                    current is not None
                    and current.status == "waiting"
                    and _is_parameter_continuation(current, intake, text, explicit_parameters)
                    and not intake.unresolved_results
                    and _pending_missing_fields_are_scoped(current, intake, self.registry)
                ):
                    response = self._apply_parameter_update(
                        current,
                        explicit_parameters,
                        requirement_id=intake.parameter_target_requirement_id,
                        cancel=request_cancel,
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                response = AgentResponse(
                    "本次请求还有尚未支持或尚未明确的要求："
                    + "；".join(blocking)
                    + "。请明确这些要求，或重新指定只计算已支持的部分。"
                )
                self._record_response(response, cancel=request_cancel)
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
                        "当前会话有多个可复用的成功结构，请明确选择一个结构别名后再继续：\n"
                        f"{options}\n回复如“复用 geometry_1”。"
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                selected_geometry_alias = explicit_alias
                intake = _intake_with_history_geometry(intake, selected_geometry_alias)
            if (
                selected_geometry_alias is None
                and intake.intent == "chemistry_compute"
                and history_geometry_requested
            ):
                if not geometry_catalog:
                    response = AgentResponse(
                        "当前会话没有找到可安全复用的成功结构；我没有改用新结构或启动计算。"
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                if len(geometry_catalog) > 1:
                    response = AgentResponse(
                        "当前会话有多个可复用的成功结构，请说明要使用哪一个分子或任务。"
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                selected_geometry_alias = str(geometry_catalog[0]["alias"])
                intake = _intake_with_history_geometry(intake, selected_geometry_alias)
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
                self._record_response(response, cancel=request_cancel)
                return response
            if (
                current is not None
                and current.status == "waiting"
                and _is_parameter_continuation(current, intake, text, explicit_parameters)
            ):
                response = self._apply_parameter_update(
                    current,
                    explicit_parameters,
                    requirement_id=intake.parameter_target_requirement_id,
                    cancel=request_cancel,
                )
                self._record_response(response, cancel=request_cancel)
                return response
            request = request_from_intake(
                text,
                intake,
                request_id=new_id("request"),
                registry=self.registry,
                normalized_parameters=normalized_parameters,
            )
            self._ensure_request_active(request_token, request_cancel)
            if semantic_enabled:
                llm_stage = "planning"
                try:
                    plan = build_plan(
                        request,
                        registry=self.registry,
                        plan_id=new_id("plan"),
                    )
                except PlanBuildError as error:
                    response = AgentResponse(
                        "当前请求存在无法唯一确定的执行依赖："
                        f"{error}。请明确后再继续；本次没有启动计算。"
                    )
                    self._record_response(response, cancel=request_cancel)
                    return response
                self._ensure_request_active(request_token, request_cancel)
            else:
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
                            "last_delivery": self._session.get("last_delivery", []),
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
                                "Plan failed local validation after "
                                f"{revision} correction(s): {error}"
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
            response = self._response_for_run(run, result, cancel=request_cancel)
            self._record_response(response, cancel=request_cancel)
            return response
        except LlmError as error:
            stage = error.purpose or llm_stage
            if stage == "answer":
                llm_call_cursor = self._llm_call_count()
            self._persist_llm_diagnostics(
                llm_call_cursor,
                stage=stage,
                failure_category=error.category,
            )
            if error.category == "cancelled" or request_cancel.is_set():
                completed_run = locals().get("run")
                if isinstance(completed_run, Run) and completed_run.status in {
                    "succeeded",
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    response = self._response_for_run(
                        completed_run,
                        self._latest_result(completed_run),
                        cancel=request_cancel,
                    )
                else:
                    response = AgentResponse("当前请求已取消，尚未执行。")
            else:
                stage_label = {
                    "intake": "请求解析阶段",
                    "semantic": "语义解析阶段",
                    "planner": "计划生成阶段",
                }.get(stage)
                if stage == "intake" and error.category == "ambiguous_result":
                    response = AgentResponse(
                        "Opt 和 SP 同时请求时，通用‘能量’目标不唯一。请明确选择 "
                        "sp_electronic_energy（单点电子能）或 "
                        "opt_final_electronic_energy（优化末态电子能）；未启动计算。"
                    )
                elif stage == "intake" and error.category == "schema_error":
                    diagnostic_text = "；".join(
                        f"{item.get('path', '$')}: {item.get('message', '')}"
                        for item in error.diagnostics
                    )
                    if "electronic_state_parameter_invalid" in diagnostic_text:
                        response = AgentResponse(
                            "电子态参数取值无效（不是受支持的整数值）：电荷和自旋多重度"
                            "必须是严格整数。"
                            "请更正参数；本次没有生成或启动计算。"
                        )
                    elif "outside the Tool catalog" in diagnostic_text:
                        response = AgentResponse(
                            f"请求中有未注册的 Tool 参数字段（{diagnostic_text[:300]}）。"
                            "请使用参数目录中的字段；本次没有生成或启动计算。"
                        )
                    else:
                        response = AgentResponse(
                            f"请求字段未通过校验（{diagnostic_text[:300]}）。"
                            "请按提示修正；本次没有生成或启动计算。"
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
            self._record_response(response, cancel=request_cancel)
            return response
        except (ValueError, OSError) as error:
            response = AgentResponse(f"请求无法规划：{error}")
            self._record_response(response, cancel=request_cancel)
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
                step = _next_ready_step(run, self.config.data_root_path, registry=self.registry)
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
                if tool.requires_compute_permission and not run.execution_permission:
                    try:
                        self._validate_known_plan_parameters(run)
                    except (TypeError, ValueError, OSError) as error:
                        run.status = "failed"
                        run.waiting_for = None
                        run.pending_data = {
                            "category": "parameter_validation",
                            "reason": str(error),
                        }
                        run.finish_active_interval()
                        save_run(self.config.data_root_path, run)
                        return last_result
                if tool.preparation_function is not None:
                    preparing_step_id = step.id
                    try:
                        if tool.requires_compute_permission and not run.execution_permission:
                            # Resolve every prepared Tool Step before one confirmation
                            # for the whole Plan. Later operations often consume a
                            # future output port, so structure facts are traced
                            # back through that port to the already prepared input.
                            for planned_step in run.plan.steps:
                                planned_tool = self.registry.get(planned_step.tool)
                                if planned_tool.preparation_function is None:
                                    continue
                                preparing_step_id = planned_step.id
                                prepared_step = self._prepare_tool_step(
                                    run, planned_step, planned_tool
                                )
                                if prepared_step is None:
                                    run.finish_active_interval()
                                    save_run(self.config.data_root_path, run)
                                    return last_result
                            step = next(item for item in run.plan.steps if item.id == step.id)
                        else:
                            prepared_step = self._prepare_tool_step(run, step, tool)
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
                try:
                    invoked = self._invoke_step(run, step, tool, cancel_event)
                    if invoked is None:
                        run.status = "failed"
                        run.finish_active_interval()
                        save_run(self.config.data_root_path, run)
                        return last_result
                    result, result_path = invoked
                except Exception as error:
                    run.status = "failed"
                    run.step_status[step.id] = "failed"
                    run.pending_data = {
                        "category": "execution_boundary",
                        "reason": str(error),
                        "exception_type": type(error).__name__,
                        "step_id": step.id,
                    }
                    run.finish_active_interval()
                    try:
                        save_run(self.config.data_root_path, run)
                    except Exception:
                        pass
                    return last_result
                last_result = result
                if result.status == "succeeded":
                    run.pending_data = {}
                    self._record_result_summary(run, result)
                    try:
                        save_run(self.config.data_root_path, run)
                    except OSError as error:
                        run.status = "failed"
                        run.pending_data = {
                            "category": "result_publish_failed",
                            "reason": str(error),
                            "step_id": step.id,
                            "result_path": result_path,
                        }
                        run.finish_active_interval()
                        try:
                            save_run(self.config.data_root_path, run)
                        except Exception:
                            pass
                        return result
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
            try:
                save_run(self.config.data_root_path, run)
            except Exception:
                pass
            raise
        finally:
            if run.status in {"succeeded", "failed", "cancelled", "interrupted"}:
                run.finish_active_interval()
                try:
                    save_run(self.config.data_root_path, run)
                except Exception:
                    pass

    def _invoke_step(
        self, run: Run, step: Step, tool: Tool, cancel: Event
    ) -> tuple[Result, str] | None:
        """Own one Tool call's budget, attempt record, frozen inputs, and publish edge."""

        if tool.requires_compute_permission and not run.execution_permission:
            raise PermissionError("Run does not have permission to invoke this compute Tool")
        if not self._reserve_attempt(run, step, tool):
            return None
        expected_bindings, expected_hashes = self._frozen_step_inputs(run, step)
        frozen_inputs = {
            name: find_artifact(run, artifact_id) for name, artifact_id in expected_bindings.items()
        }
        attempt = (
            max(
                (
                    int(item.get("attempt", 0))
                    for item in run.attempts
                    if item.get("step_id") == step.id
                ),
                default=0,
            )
            + 1
        )
        relative = f"{step.id}/attempt-{attempt:02d}"
        workdir = run_directory(self.config.data_root_path, run.id) / relative
        workdir.mkdir(parents=True, exist_ok=False)
        attempt_record = {
            "step_id": step.id,
            "attempt": attempt,
            "phase": "prepared",
            "status": "running",
            "relative_path": workdir.relative_to(self.config.data_root_path).as_posix(),
            "result_relative_path": relative,
            "artifact_ids": [],
            "output_ports": {},
            "input_artifact_ids": list(expected_bindings.values()),
        }
        run.attempts.append(attempt_record)
        run.step_status[step.id] = "running"
        save_run(self.config.data_root_path, run)
        from .tools.runtime import ToolCallContext

        context = ToolCallContext(
            data_root=self.config.data_root_path,
            run=run,
            step=step,
            attempt=attempt,
            cancel=cancel,
            workdir=workdir,
            relative_attempt_path=relative,
            frozen_inputs=frozen_inputs,
            attempt_record=attempt_record,
        )
        try:
            result = tool.execute(step, context)
            if (
                result.run_id != run.id
                or result.step_id != step.id
                or result.attempt != attempt
                or result.attempt_relative_path != relative
            ):
                raise ValueError("Tool Result does not match its active ToolCallContext")
            attempt_record.update(
                {
                    "phase": "finished",
                    "status": result.status,
                    "result_category": result.diagnostics.get("category"),
                    "artifact_ids": list(result.artifact_ids),
                    "output_ports": dict(result.output_ports),
                    "input_artifact_ids": list(result.input_artifact_ids),
                }
            )
            step_parameter_sources = run.parameter_sources_by_step.get(step.id)
            if not step_parameter_sources and run.pending_data.get("parameter_sources"):
                step_parameter_sources = dict(run.pending_data["parameter_sources"])
            result_path = publish_step_result(
                self.config.data_root_path,
                run,
                step,
                tool,
                result,
                expected_input_bindings=expected_bindings,
                expected_input_hashes=expected_hashes,
                parameter_sources=step_parameter_sources,
            )
            return result, result_path
        except Exception:
            attempt_record.update(
                {"phase": "finished", "status": "failed", "result_category": "execution_boundary"}
            )
            run.step_status[step.id] = "failed"
            try:
                save_run(self.config.data_root_path, run)
            except Exception:
                pass
            raise

    def confirm(self, run: Run | str | None = None) -> AgentResponse:
        request_token, request_cancel = self._begin_request()
        origin_session = self.session_id
        try:
            current = self._coerce_run(run)
            if current is None:
                response = AgentResponse("当前没有等待确认的计算。")
                self._record_response(
                    response,
                    cancel=request_cancel,
                    request_token=request_token,
                    session_id=origin_session,
                )
                return response
            if current.status in {"succeeded", "failed", "cancelled", "interrupted"}:
                result = self._latest_result(current)
                response = self._response_for_run(
                    current,
                    result,
                    cancel=request_cancel,
                )
                self._record_response(
                    response,
                    cancel=request_cancel,
                    request_token=request_token,
                    session_id=origin_session,
                )
                return response
            if current.waiting_for != "confirmation":
                response = AgentResponse(self._waiting_text(current), run=current)
                self._record_response(
                    response,
                    cancel=request_cancel,
                    request_token=request_token,
                    session_id=origin_session,
                )
                return response
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
            self._cancel_events[current.id] = request_cancel
            try:
                result = self.advance(current, cancel=request_cancel)
            except (PermissionError, ValueError, OSError) as error:
                current.status = "failed"
                current.pending_data = {"category": "execution_boundary", "reason": str(error)}
                save_run(self.config.data_root_path, current)
                result = self._latest_result(current)
            response = self._response_for_run(current, result, cancel=request_cancel)
            self._record_response(
                response,
                cancel=request_cancel,
                request_token=request_token,
                session_id=origin_session,
            )
            return response
        finally:
            self._finish_request(request_token, request_cancel)

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
        self._query_bindings = {}
        self._session = {
            "session_id": self.session_id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
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
        plan = validate_request_plan(request, plan, self.registry)
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
            execution_permission=(
                not self.config.runtime.confirm_before_compute
                and not any(
                    isinstance(item.constraints.get("method_resolution"), Mapping)
                    and item.constraints["method_resolution"].get("status") == "proposed"
                    for item in request.requirements
                )
            ),
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
            (
                source_run,
                source_step,
                source_result,
                source_artifact,
                source_path,
                source_port,
            ) = verified_history
            copied = register_file_artifact(
                self.config.data_root_path,
                run,
                source_path,
                artifact_type="molecular_geometry",
                role="input_geometry",
                source=f"history:verified_{source_artifact.role}",
                extension=".xyz",
                metadata={
                    "history_source_role": source_artifact.role,
                    "history_source_port": source_port,
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
        for alias, input_artifact in self._seed_structure_inputs(run).items():
            alias_replacements[alias] = input_artifact.id
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

    def _seed_structure_inputs(self, run: Run) -> dict[str, Any]:
        """Register caller-supplied XYZ once per subject and expose safe aliases."""

        replacements: dict[str, Any] = {}
        value = run.request.structure_input
        xyz_text = value.get("xyz_text") or value.get("xyz") if isinstance(value, dict) else None
        if xyz_text is not None:
            artifact = self._register_input_geometry(
                run,
                xyz_text,
                value.get("molecule_identity") if isinstance(value, Mapping) else None,
                subject_key=None,
            )
            replacements["request_geometry"] = artifact
            replacements[INPUT_GEOMETRY_PLACEHOLDER] = artifact
        for subject_id, subject in run.request.subjects.items():
            if not isinstance(subject, Mapping):
                continue
            subject_input = subject.get("structure_input", {})
            if not isinstance(subject_input, Mapping):
                continue
            subject_xyz = subject_input.get("xyz_text") or subject_input.get("xyz")
            if subject_xyz is None:
                continue
            subject_key = str(subject.get("key") or subject_id)
            if xyz_text is not None and len(run.request.subjects) == 1 and subject_xyz == xyz_text:
                artifact = replacements["request_geometry"]
            else:
                artifact = self._register_input_geometry(
                    run,
                    subject_xyz,
                    subject_input.get("molecule_identity"),
                    subject_key=subject_key,
                )
            replacements[f"request_geometry_{subject_key}"] = artifact
        return replacements

    def _register_input_geometry(
        self,
        run: Run,
        xyz_text: Any,
        identity: Any,
        *,
        subject_key: str | None,
    ) -> Any:
        if not isinstance(xyz_text, str):
            raise ValueError("chat subject XYZ must be text")
        geometry_bytes = xyz_text.encode("utf-8")
        geometry = parse_xyz_bytes(geometry_bytes)
        if isinstance(identity, Mapping) and identity.get("element_counts") is not None:
            matches, reason = identity_matches_facts(
                identity,
                {
                    "element_counts": dict(Counter(geometry.symbols)),
                    "component_count": 1,
                    "isotopic": False,
                    "formal_charge": 0,
                },
            )
            if not matches:
                raise ValueError(
                    f"inline XYZ does not satisfy subject {subject_key or 'subject_1'}'s "
                    f"molecule formula: {reason}"
                )
        artifact = register_bytes_artifact(
            self.config.data_root_path,
            run,
            geometry_bytes,
            artifact_type="molecular_geometry",
            role="input_geometry",
            source="chat:inline_xyz" if subject_key is None else f"chat:inline_xyz:{subject_key}",
            extension=".xyz",
            metadata={
                "imported_as_raw_bytes": True,
                "source": "chat",
                **({"subject_key": subject_key} if subject_key is not None else {}),
            },
        )
        return artifact

    def _prepare_tool_step(self, run: Run, step: Step, tool: Tool) -> Step | None:
        """Apply the selected Tool's optional domain parameter preparation hook."""

        prepared = tool.prepare(
            step,
            {
                "request": run.request,
                "structure_facts": self._known_structure_facts(run, step),
                "defaults": self.config.defaults,
                "parameters_locked": bool(run.accepted_snapshot),
                "pending_parameters": dict(run.pending_data),
            },
        )
        if prepared.missing_fields or prepared.step is None:
            run.status = "waiting"
            run.waiting_for = "clarification"
            run.pending_data = {
                "category": "parameter_preparation",
                "question": prepared.question
                or "Please provide the missing parameters for this Tool.",
                "step_id": step.id,
                "missing_fields": list(prepared.missing_fields),
                "parameter_sources": dict(prepared.parameter_sources),
                "parameters": dict(step.parameters),
            }
            return None
        replacement = prepared.step
        assert replacement is not None
        if replacement.id != step.id or replacement.tool != step.tool:
            raise ValueError("Tool preparation cannot change the Step identity or capability")
        if replacement != step:
            run.plan = _replace_step(run.plan, replacement)
        run.pending_data = {
            "step_id": step.id,
            "parameters": dict(replacement.parameters),
            "parameter_sources": dict(prepared.parameter_sources),
            "effective_parameters": dict(replacement.parameters),
        }
        run.parameter_sources_by_step[step.id] = dict(prepared.parameter_sources)
        return replacement

    def _apply_parameter_update(
        self,
        run: Run,
        parameters: dict[str, Any],
        *,
        requirement_id: str | None = None,
        requirement_constraint_patch: Mapping[str, Any] | None = None,
        cancel: Event | None = None,
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

        editable_steps = [
            step
            for step in editable_steps
            if set(parameters) & set(self.registry.get(step.tool).request_parameters)
        ]
        scoped_requirement_id = requirement_id
        if scoped_requirement_id is not None:
            selected_requirement = next(
                (item for item in run.request.requirements if item.id == scoped_requirement_id),
                None,
            )
            if selected_requirement is None:
                return AgentResponse("参数修改已拒绝：目标要求项不属于当前任务。", run=run)
            editable_steps = [
                step for step in editable_steps if step.requirement_id == scoped_requirement_id
            ]
        elif len(editable_steps) == 1:
            # A single unlabelled target remains a request-wide update for
            # backwards compatibility; only an explicit target uses a scoped map.
            scoped_requirement_id = None
        elif len({step.requirement_id or f"step:{step.id}" for step in editable_steps}) > 1:
            choices = "\n".join(
                [
                    *(
                        f"- {item.id}: {item.capability} / {item.subject_id}"
                        for item in run.request.requirements
                        if any(step.requirement_id == item.id for step in editable_steps)
                    ),
                    *(
                        f"- {step.id}: {step.tool} / {step.subject_id or 'unknown subject'}"
                        for step in editable_steps
                        if step.requirement_id is None
                    ),
                ]
            )
            return AgentResponse(
                "当前有多个参数作用域，请明确只修改其中一个要求项：\n" + choices,
                run=run,
            )
        if not editable_steps:
            return AgentResponse("参数修改与所选要求项没有兼容的参数字段。", run=run)

        updated_requirements = []
        for requirement in run.request.requirements:
            if scoped_requirement_id is not None and requirement.id != scoped_requirement_id:
                updated_requirements.append(requirement)
                continue
            allowed = set(self.registry.get(requirement.capability).request_parameters)
            patch = {name: value for name, value in parameters.items() if name in allowed}
            if not patch:
                updated_requirements.append(requirement)
                continue
            constraints = dict(requirement.constraints)
            sources = dict(constraints.get("parameter_sources", {}))
            sources.update({name: "user_modification" for name in patch})
            constraints["parameter_sources"] = sources
            if (
                requirement.id == scoped_requirement_id
                and requirement_constraint_patch
            ):
                constraints.update(dict(requirement_constraint_patch))
            updated_requirements.append(
                requirement.model_copy(
                    update={
                        "parameters": {**requirement.parameters, **patch},
                        "constraints": constraints,
                    }
                )
            )
        candidate_request = run.request.model_copy(update={"requirements": updated_requirements})
        try:
            _validate_request_parameter_scope(candidate_request, run.plan, self.registry)
            candidate_steps: list[Step] = []
            candidate_sources: dict[str, dict[str, str]] = {}
            for step in run.plan.steps:
                tool = self.registry.get(step.tool)
                if not tool.request_parameters or step not in editable_steps:
                    candidate_steps.append(step)
                    continue

                if tool.preparation_function is None:
                    replacement = merge_explicit_step_parameters(tool, step, candidate_request)
                    candidate_steps.append(replacement)
                    scoped_explicit: dict[str, Any] = {}
                    scoped_modifications: dict[str, Any] = {}
                    if step.requirement_id is not None:
                        requirement = next(
                            item
                            for item in candidate_request.requirements
                            if item.id == step.requirement_id
                        )
                        scoped_explicit = requirement.parameters
                        scoped_modifications = (
                            candidate_request.user_modifications_by_requirement.get(
                                step.requirement_id, {}
                            )
                        )
                    candidate_sources[step.id] = {
                        name: (
                            "user_modification"
                            if name in scoped_modifications
                            or name in candidate_request.user_modifications
                            else "requirement_explicit"
                            if name in scoped_explicit
                            else "request_explicit"
                        )
                        for name in tool.request_parameters
                        if name in replacement.parameters
                        and (
                            name in scoped_modifications
                            or name in scoped_explicit
                            or name in candidate_request.user_modifications
                            or name in candidate_request.explicit_parameters
                        )
                    }
                    continue

                candidate_step = merge_explicit_step_parameters(tool, step, candidate_request)
                checked = tool.validate_parameters(candidate_step.parameters, allow_deferred=True)
                candidate_step = candidate_step.model_copy(update={"parameters": checked})
                prepared = tool.prepare(
                    candidate_step,
                    {
                        "request": candidate_request,
                        "structure_facts": self._known_structure_facts(run, step),
                        "defaults": self.config.defaults,
                        "parameters_locked": False,
                        "pending_parameters": {},
                    },
                )
                replacement = prepared.step or candidate_step
                candidate_steps.append(replacement)
                candidate_sources[step.id] = dict(prepared.parameter_sources)

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
            self._validate_known_plan_parameters(run, candidate_plan)
        except (TypeError, ValueError) as error:
            return AgentResponse(f"参数修改已拒绝（rejected）：{error}", run=run)

        changed_steps = [
            step.id
            for old, step in zip(run.plan.steps, candidate_plan.steps, strict=True)
            if old.parameters != step.parameters
        ]
        apply_plan_change(
            run,
            candidate_plan,
            self.registry,
            candidate_request=candidate_request,
            changed_step_ids=set(changed_steps),
        )
        has_proposed_method = any(
            isinstance(item.constraints.get("method_resolution"), Mapping)
            and item.constraints["method_resolution"].get("status") == "proposed"
            for item in candidate_request.requirements
        )
        run.execution_permission = (
            not self.config.runtime.confirm_before_compute and not has_proposed_method
        )
        run.accepted_snapshot = {}
        run.accepted_execution_sha256 = None
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
        return self._response_for_run(run, result, cancel=cancel)

    def _apply_molecule_clarification(
        self,
        run: Run,
        query: str,
        input_kind: str | None,
        *,
        candidate: Mapping[str, Any] | None = None,
        preserve_identity: bool = False,
        name_evidence: str | None = None,
        message: str | None = None,
        cancel: Event | None = None,
    ) -> AgentResponse:
        return self._apply_molecule_update(
            run,
            query,
            input_kind,
            candidate=candidate,
            preserve_identity=preserve_identity,
            name_evidence=name_evidence,
            message=message,
            cancel=cancel,
        )

    def _apply_molecule_update(
        self,
        run: Run,
        query: str,
        input_kind: str | None,
        *,
        candidate: Mapping[str, Any] | None = None,
        preserve_identity: bool = False,
        name_evidence: str | None = None,
        message: str | None = None,
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
        selected_candidate_cid = _candidate_cid(candidate)
        if candidate is not None and selected_candidate_cid is None:
            return AgentResponse("当前候选缺少可验证的 CID，无法安全选择。", run=run)
        if selected_candidate_cid is not None:
            query = str(selected_candidate_cid)
            kind = "cid"
        try:
            subject_id = step.subject_id
            if subject_id is None and len(run.request.subjects) == 1:
                subject_id = next(iter(run.request.subjects))
            subject_record = run.request.subjects.get(subject_id or "")
            subject_value = (
                subject_record.model_dump(mode="python")
                if isinstance(subject_record, Subject)
                else dict(subject_record)
                if isinstance(subject_record, Mapping)
                else None
            )
            subject_input_value = (
                subject_record.get("structure_input", {})
                if isinstance(subject_record, Mapping)
                else run.request.structure_input
            )
            structure_input = dict(subject_input_value)
            if candidate is not None:
                identity = structure_input.get("molecule_identity")
                if not isinstance(identity, Mapping):
                    return AgentResponse(
                        "当前候选没有可继承的分子身份约束，无法安全选择。",
                        run=run,
                    )
                selected_identity = dict(identity)
                selected_identity["selected_cid"] = selected_candidate_cid
                selected_identity.pop("selected_smiles", None)
                choice_id = candidate.get("choice_id")
                if isinstance(choice_id, str):
                    selected_identity["selected_choice_id"] = choice_id
                selected_smiles = candidate.get("isomeric_smiles") or candidate.get(
                    "canonical_smiles"
                )
                if isinstance(selected_smiles, str):
                    selected_identity["selected_smiles"] = selected_smiles
                structure_input["molecule_identity"] = selected_identity
            elif preserve_identity:
                identity = structure_input.get("molecule_identity")
                if not isinstance(identity, Mapping):
                    return AgentResponse(
                        "当前输入没有可继承的分子身份约束，无法安全选择。",
                        run=run,
                    )
                selected_identity = dict(identity)
                if kind == "cid":
                    selected_identity["selected_cid"] = int(query)
                    selected_identity.pop("selected_smiles", None)
                    selected_identity.pop("selected_choice_id", None)
                elif kind == "smiles":
                    selected_identity["selected_smiles"] = query
                    selected_identity.pop("selected_cid", None)
                    selected_identity.pop("selected_choice_id", None)
                elif kind == "formula":
                    corrected = build_identity_constraint(
                        message=message or query,
                        query=query,
                        input_kind="formula",
                    )
                    if corrected is None:
                        return AgentResponse("该分子式无法建立身份约束。", run=run)
                    selected_identity = corrected
                elif kind == "name":
                    if selected_identity.get("input_kind") != "name":
                        return AgentResponse(
                            "当前任务不是名称身份，无法只替换名称检索词。", run=run
                        )
                    selected_identity["lookup_query"] = query
                    selected_identity.pop("selected_cid", None)
                    selected_identity.pop("selected_choice_id", None)
                    selected_identity.pop("selected_smiles", None)
                else:
                    return AgentResponse("当前身份补充不是可识别的候选、CID 或 SMILES。", run=run)
                structure_input["molecule_identity"] = selected_identity
            else:
                identity = build_identity_constraint(
                    message=message or query,
                    query=query,
                    input_kind=kind,
                    name_evidence=name_evidence,
                )
                if identity is not None:
                    structure_input["molecule_identity"] = identity
            selected_identity = structure_input.get("molecule_identity")
            local_facts: dict[str, Any] | None = None
            if candidate is not None:
                candidate_smiles = candidate.get("isomeric_smiles") or candidate.get(
                    "canonical_smiles"
                )
                if isinstance(candidate_smiles, str) and selected_candidate_cid is not None:
                    local_facts = _facts_from_smiles(candidate_smiles)
                    local_facts["cid"] = selected_candidate_cid
            elif preserve_identity and kind == "smiles":
                local_facts = _facts_from_smiles(query)
            if isinstance(selected_identity, Mapping) and local_facts is not None:
                matches, reason = identity_matches_facts(selected_identity, local_facts)
                if not matches:
                    return AgentResponse(
                        "该分子选择已拒绝（rejected）：" + (reason or "结构身份与原分子式不一致"),
                        run=run,
                    )
            if subject_record is not None and subject_id is not None:
                subject_value["structure_input"] = structure_input
                subjects = dict(run.request.subjects)
                subjects[subject_id] = (
                    subject_record.model_copy(update={"structure_input": structure_input})
                    if isinstance(subject_record, Subject)
                    else subject_value
                )
                candidate_request = run.request.model_copy(update={"subjects": subjects})
            else:
                candidate_request = run.request.model_copy(
                    update={"structure_input": structure_input}
                )
        except (TypeError, ValueError) as error:
            return AgentResponse(f"该分子选择已拒绝（rejected）：{error}", run=run)
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
            candidate_plan = validate_request_plan(
                candidate_request,
                candidate_plan,
                self.registry,
            )
        except ValueError as error:
            return AgentResponse(f"该分子选择已拒绝（rejected）：{error}", run=run)
        apply_plan_change(
            run,
            candidate_plan,
            self.registry,
            candidate_request=candidate_request,
            changed_step_ids={step.id},
        )
        run.execution_permission = not self.config.runtime.confirm_before_compute
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
        return self._response_for_run(run, result, cancel=cancel)

    def _try_repair(self, run: Run, step: Step, result: Result, cancel: Event) -> bool:
        if not self.config.repair.enabled:
            return False
        tool = self.registry.get(step.tool)
        options = tool.repair_options(run, step, result)
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
        category = tool.execution_budget
        max_extra = self._extra_execution_limit(run, category)
        used_extra = run.extra_executions_by_category.get(category, 0)
        if category != "none" and used_extra >= max_extra:
            return False
        try:
            proposal = propose_repair(
                self.llm,
                run=run,
                step=step,
                result=result,
                options=options,
                budget_category=tool.execution_budget,
                cancel=cancel,
                remaining_timeout_seconds=self._remaining_active_seconds(run),
            )
        except LlmError as error:
            run.pending_data = {"repair_unavailable": error.category, "reason": str(error)}
            save_run(self.config.data_root_path, run)
            return False
        if proposal is None or proposal.selected_option_id == "none":
            return False
        option = next(
            (item for item in options if item.option_id == proposal.selected_option_id), None
        )
        if option is None:
            return False
        try:
            candidate_plan, record = apply_repair_proposal(
                proposal,
                option=option,
                run=run,
                step=step,
                result=result,
                tool=tool,
            )
        except ValueError as error:
            run.pending_data = {"repair_rejected": str(error)}
            save_run(self.config.data_root_path, run)
            return False
        patch = record.get("parameter_patch", {})
        old_parameters = record.get("old_parameters", {})
        new_parameters = record.get("new_parameters", {})
        verified_diff = {
            step.id: {
                str(name): (old_parameters.get(name), new_parameters.get(name)) for name in patch
            }
        }
        apply_plan_change(
            run,
            candidate_plan,
            self.registry,
            changed_step_ids={step.id},
            allow_verified_repair_diff=verified_diff,
        )
        run.plan_revisions += 1
        record["derived_plan_sha256"] = _plan_fingerprint(run.plan)
        run.repair_records.append(record)
        save_run(self.config.data_root_path, run)
        return True

    def _reserve_attempt(self, run: Run, step: Step, tool: Tool) -> bool:
        category = tool.execution_budget
        if category == "none":
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
        used_extra = run.extra_executions_by_category.get(category, 0)
        if (count > 0 or new_science_step) and used_extra >= self._extra_execution_limit(
            run, category
        ):
            run.pending_data = {
                "budget_exhausted": "max_extra_executions",
                "budget_category": category,
                "step_id": step.id,
            }
            return False
        run.attempt_counts[origin] = count + 1
        if count > 0 or new_science_step:
            run.extra_executions_by_category[category] = used_extra + 1
        return True

    def _extra_execution_limit(self, run: Run, category: str) -> int:
        configured = run.budget.get("max_extra_executions_by_category", {})
        if isinstance(configured, dict) and category in configured:
            return int(configured[category])
        return int(self.config.repair.execution_budgets_by_category.get(category, 0))

    def _prepare_confirmation(self, run: Run, step: Step) -> None:
        run.status = "waiting"
        run.waiting_for = "confirmation"
        run.pending_data = self._preview(run, step)

    def _validate_known_plan_parameters(self, run: Run, plan: Plan | None = None) -> None:
        """Validate parameters against any trusted geometry already available."""

        candidate_plan = plan or run.plan
        for step in candidate_plan.steps:
            tool = self.registry.get(step.tool)
            if tool.parameter_validation_function is None:
                continue
            context = self._parameter_validation_context(run, candidate_plan, step)
            try:
                tool.validate_parameters(
                    step.parameters,
                    allow_deferred=True,
                    context=context,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"step {step.id} has invalid parameters: {error}") from error

    def _parameter_validation_context(self, run: Run, plan: Plan, step: Step) -> dict[str, Any]:
        """Build context from validated geometry, never from model-provided counts."""

        tool = self.registry.get(step.tool)
        for input_name, input_type in tool.input_ports.items():
            if input_type != "molecular_geometry":
                continue
            reference = step.inputs.get(input_name)
            if reference is None:
                continue
            atom_count = self._geometry_atom_count_for_reference(run, plan, reference, seen=set())
            if atom_count is not None:
                return {"geometry_atom_count": atom_count}
        return {}

    def _geometry_atom_count_for_reference(
        self,
        run: Run,
        plan: Plan,
        reference: InputReference,
        *,
        seen: set[tuple[str | None, str | None, str | None]],
    ) -> int | None:
        """Resolve a count from an artifact or a declared geometry-preserving edge."""

        key = (reference.artifact_id, reference.step_id, reference.port)
        if key in seen:
            return None
        seen.add(key)

        if reference.artifact_id is not None:
            try:
                artifact = resolve_artifact_reference(
                    self.config, run, reference, expected_type="molecular_geometry"
                )
                geometry = parse_xyz_bytes(
                    artifact_path(self.config.data_root_path, run, artifact).read_bytes()
                )
            except (OSError, TypeError, ValueError):
                return None
            return geometry.atom_count

        if reference.step_id is None:
            return None
        try:
            artifact = self._artifact_from_reference(run, reference)
        except (OSError, TypeError, ValueError):
            artifact = None
        if artifact is not None:
            try:
                geometry = parse_xyz_bytes(
                    artifact_path(self.config.data_root_path, run, artifact).read_bytes()
                )
            except (OSError, TypeError, ValueError):
                return None
            return geometry.atom_count

        steps_by_id = {item.id: item for item in plan.steps}
        producer = steps_by_id.get(reference.step_id)
        if producer is None:
            return None
        producer_tool = self.registry.get(producer.tool)
        input_name = producer_tool.geometry_output_input_ports.get(reference.port or "")
        if input_name is None:
            return None
        producer_reference = producer.inputs.get(input_name)
        if producer_reference is None:
            return None
        return self._geometry_atom_count_for_reference(run, plan, producer_reference, seen=seen)

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
            "operation": (self.registry.get(step.tool).operations or ["Tool"])[0],
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
            "method_resolution_proposals": [
                {
                    "requirement_id": requirement.id,
                    **dict(requirement.constraints["method_resolution"]),
                }
                for requirement in run.request.requirements
                if isinstance(requirement.constraints.get("method_resolution"), Mapping)
                and requirement.constraints["method_resolution"].get("status") == "proposed"
            ],
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
                            artifact is not None and artifact.source.startswith("history:verified_")
                        ),
                    }
                )
            goals = []
            for goal in step.goal_checks:
                source_step = next(
                    (item for item in run.plan.steps if item.id == goal.source_step_id), None
                )
                source_tool = (
                    self.registry.get(source_step.tool) if source_step is not None else None
                )
                goals.append(
                    {
                        "source_step": (
                            f"步骤 {step_numbers[goal.source_step_id]}"
                            if goal.source_step_id in step_numbers
                            else "未知步骤"
                        ),
                        "check": goal.check,
                        "label": (
                            source_tool.result_metadata.get(goal.check, {}).get(
                                "label", goal.check.replace("_", " ")
                            )
                            if source_tool is not None
                            else goal.check.replace("_", " ")
                        ),
                        "required_status": goal.required_status,
                    }
                )
            summaries.append(
                {
                    "index": index,
                    "tool": tool.name,
                    "tool_label": tool.display_name or tool.name,
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
        same_capability_requirements = [
            item for item in run.request.requirements if item.capability == step.tool
        ]
        for target in run.plan.requested_results:
            if target.step_id is not None and target.step_id != step.id:
                continue
            if target.requirement_id is not None and target.requirement_id != step.requirement_id:
                continue
            if (
                target.requirement_id is None
                and target.step_id is None
                and len(same_capability_requirements) > 1
            ):
                continue
            if not any(
                descriptor["name"] == (target.check or target.port or target.field)
                for descriptor in tool.public_outputs()
            ):
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
            mutable_parameters: set[str] = set()
            declared_parameters = (
                set(tool.parameter_type.model_fields) if tool.parameter_type is not None else set()
            )
            actions: dict[str, Any] = {}
            for action in sorted(capabilities):
                fields = tool.repair_parameter_fields.get(action, [])
                mutable_parameters.update(fields)
                if iteration_increase_allowed:
                    limits = tool.repair_parameter_limits.get(action, {})
                    maximum = min(limits.values()) if limits else None
                    actions[action] = {
                        "fields": list(fields),
                        "limits": dict(limits),
                        "maximum": maximum,
                        "input_aliases": list(tool.repair_input_aliases.get(action, [])),
                    }
            if not capabilities:
                continue
            scopes[step.id] = {
                "origin_step_id": run.origin_step_map.get(step.id, step.origin_step_id or step.id),
                "mutable_parameters": sorted(mutable_parameters),
                "immutable_parameters": sorted(declared_parameters - mutable_parameters),
                "actions": actions,
            }
        return {
            "version": 1,
            "iteration_increase_allowed": iteration_increase_allowed,
            "steps": scopes,
            "resources_immutable": dict(run.resources),
            "max_attempts_per_science_step": run.budget.get("max_attempts_per_science_step"),
            "max_extra_executions_by_category": run.budget.get(
                "max_extra_executions_by_category", {}
            ),
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

    def _frozen_step_inputs(self, run: Run, step: Step) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve and hash-check every bound input immediately before invocation."""

        bindings: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for input_name, reference in step.inputs.items():
            artifact = self._artifact_from_reference(run, reference)
            if artifact is None:
                raise ValueError(f"Step input {input_name!r} is not a current verified Artifact")
            path = artifact_path(self.config.data_root_path, run, artifact)
            actual_hash = sha256_file(path)
            if actual_hash != artifact.sha256:
                raise ValueError(f"Step input {input_name!r} changed before Tool invocation")
            bindings[input_name] = artifact.id
            hashes[artifact.id] = actual_hash
        return bindings, hashes

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
        category_limits = self.config.repair.execution_budgets_by_category
        return {
            "max_attempts_per_science_step": self.config.repair.max_attempts_per_science_step,
            "max_extra_executions_by_category": category_limits,
            "max_plan_revisions": self.config.repair.max_plan_revisions,
        }

    def _remaining_active_seconds(self, run: Run) -> float:
        limit = float(
            run.resources.get(
                "run_active_timeout_seconds", self.config.runtime.run_active_timeout_seconds
            )
        )
        return limit - run.current_active_seconds()

    def _response_for_run(
        self, run: Run, result: Result | None, *, cancel: Event | None = None
    ) -> AgentResponse:
        # A successful preparation step is not the user's requested scientific
        # result.  Waiting state always wins over the last intermediate Result.
        if run.status == "waiting":
            return AgentResponse(self._waiting_text(run), run=run, result=result)
        structure = self._result_structure(run, result)
        if run.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            facts, files, unavailable = self._collect_requested_outputs(run, result, cancel=cancel)
            if run.status != "succeeded" and run.plan.requested_results:
                facts, files = self._include_intermediate_facts(run, facts, files, cancel=cancel)
            default_text = render_run(
                run,
                result,
                self.registry,
                structure=structure,
                **self._failure_render_context(run),
            )
            delivery: dict[str, Any] = {}
            status_override = "cancelled" if run.status == "cancelled" else None
            if facts or unavailable or status_override is not None:
                rendered_facts, delivery = self._render_verified_delivery(
                    run,
                    facts,
                    files,
                    unavailable=unavailable,
                    cancel=cancel,
                    status_override=status_override,
                )
                default_text = (
                    rendered_facts
                    if run.status == "succeeded"
                    else f"{default_text}\n{rendered_facts}"
                )
            return AgentResponse(
                default_text,
                run=run,
                result=result,
                files=tuple(files),
                delivery=delivery,
            )
        if result is not None:
            return AgentResponse(
                render_result(run, result, self.registry, structure=structure),
                run=run,
                result=result,
            )
        return AgentResponse(render_run(run, registry=self.registry), run=run)

    def _include_intermediate_facts(
        self,
        run: Run,
        facts: list[dict[str, Any]],
        files: list[dict[str, Any]],
        *,
        cancel: Event | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Display verified intermediate facts without treating them as targets."""

        current = self._current_run_facts(run)
        if not current:
            return facts, files
        combined = list(facts)
        existing = {(fact.get("step_id"), fact.get("kind"), fact.get("name")) for fact in combined}
        used_refs = {
            str(fact.get("output_ref"))
            for fact in combined
            if isinstance(fact.get("output_ref"), str)
        }
        next_index = 1
        for fact in current:
            identity = (fact.get("step_id"), fact.get("kind"), fact.get("name"))
            if identity in existing:
                continue
            while f"out_{next_index}" in used_refs:
                next_index += 1
            optional = dict(fact)
            optional["output_ref"] = f"out_{next_index}"
            combined.append(optional)
            existing.add(identity)
            used_refs.add(optional["output_ref"])
            next_index += 1
        if len(combined) == len(facts):
            return facts, files
        # Rebuild descriptors so optional port facts and target facts share the
        # same stable output references.  Target coverage remains represented
        # by the separate ``unavailable`` list passed to the renderer.
        return combined, self._files_for_facts(combined, cancel=cancel)

    def _collect_requested_outputs(
        self, run: Run, result: Result | None = None, *, cancel: Event | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Select every current, verified producer named by the Plan targets.

        The third return value is deliberately separate from the collected
        facts.  A missing Artifact must not silently shrink the user's target
        set and then be reported as a complete result.
        """

        facts: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        unavailable: list[str] = []
        targets = list(run.plan.requested_results)
        if not targets and result is not None:
            return facts_from_result(run, result, self.registry), [], []
        for target_index, target in enumerate(targets, start=1):
            kind, name = _normalized_run_target(run, target, self.registry)
            if kind is None or name is None:
                unavailable.append(_target_identity(target, kind, name))
                continue
            step = self._target_producer(run, target, kind, name)
            if step is None:
                unavailable.append(_target_identity(target, kind, name))
                continue
            relative = run.current_results.get(step.id)
            if not isinstance(relative, str):
                unavailable.append(_target_identity(target, kind, name))
                continue
            bound = _load_bound_result(self.config.data_root_path, run, relative)
            if bound is None or not self._query_result_is_valid(run, step, bound, relative):
                unavailable.append(_target_identity(target, kind, name))
                continue
            tool = self.registry.get(step.tool)
            structure = self._query_structure(run, step, bound)
            selected = bound
            if kind == "field":
                if name not in bound.values or not _query_value_is_compatible(
                    bound.values[name], tool.results.get(name, "")
                ):
                    unavailable.append(_target_identity(target, kind, name))
                    continue
                selected = bound.model_copy(
                    update={
                        "values": {name: bound.values[name]},
                        "output_ports": {},
                        "scientific_checks": {},
                    }
                )
            elif kind == "port":
                artifact = self._query_port_artifact(
                    run, step, bound, name, tool.output_ports.get(name, "")
                )
                if artifact is None:
                    unavailable.append(_target_identity(target, kind, name))
                    continue
                selected = bound.model_copy(
                    update={
                        "values": {},
                        "output_ports": {name: artifact.id},
                        "scientific_checks": {},
                    }
                )
            else:
                check = bound.scientific_checks.get(name)
                if check is None or check.status != "passed":
                    unavailable.append(_target_identity(target, kind, name))
                    continue
                selected = bound.model_copy(
                    update={"values": {}, "output_ports": {}, "scientific_checks": {name: check}}
                )
            selected_facts = facts_from_result(run, selected, self.registry, structure=structure)
            for fact in selected_facts:
                output_ref = f"out_{target_index}"
                fact["output_ref"] = output_ref
                fact["_run"] = run
                fact["_result"] = bound
                facts.append(fact)
                if fact.get("kind") == "port":
                    artifact_id = fact.get("value", {}).get("artifact_id")
                    if isinstance(artifact_id, str):
                        try:
                            artifact = find_artifact(run, artifact_id)
                        except ValueError:
                            continue
                        file_info = self._artifact_file_descriptor(
                            run,
                            step,
                            bound,
                            artifact,
                            ref=output_ref,
                            expected_type=str(fact.get("expected_type") or artifact.artifact_type),
                            display_name=str(
                                _mapping(fact.get("metadata")).get("label") or path_label(artifact)
                            ),
                            cancel=cancel,
                        )
                        if file_info is not None:
                            files.append(file_info)
                        else:
                            unavailable.append(_target_identity(target, kind, name))
        self._limit_total_previews(files)
        return facts, files, unavailable

    def _target_producer(self, run: Run, target: Any, kind: str, name: str) -> Step | None:
        candidates = [
            step
            for step in run.plan.steps
            if name in _declared_target_names(self.registry, step, kind)
        ]
        if target.step_id is not None:
            candidates = [step for step in candidates if step.id == target.step_id]
        return candidates[0] if len(candidates) == 1 else None

    def _artifact_file_descriptor(
        self,
        run: Run,
        step: Step,
        result: Result,
        artifact: Any,
        *,
        ref: str,
        expected_type: str | None = None,
        display_name: str | None = None,
        cancel: Event | None = None,
    ) -> dict[str, Any] | None:
        try:
            path = artifact_path(self.config.data_root_path, run, artifact)
            expected_size = path.stat().st_size
        except (OSError, ValueError):
            return None
        digest = hashlib.sha256()
        prefix = bytearray()
        try:
            with path.open("rb") as handle:
                while True:
                    if cancel is not None and cancel.is_set():
                        return None
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    if len(prefix) < MAX_FILE_PREVIEW_BYTES:
                        prefix.extend(chunk[: MAX_FILE_PREVIEW_BYTES - len(prefix)])
        except OSError:
            return None
        if digest.hexdigest() != artifact.sha256 or expected_size != artifact.size_bytes:
            return None
        preview_text: str | None = None
        preview_complete = False
        preview_reason: str | None = None
        previewable = expected_type in {
            "molecular_geometry",
            "energy_data",
            "molecule",
            "text_file",
            "file",
        } or str(getattr(artifact, "metadata", {}).get("mime_type", "")).startswith("text/")
        if previewable:
            decoded, boundary_truncated = _decode_utf8_preview(bytes(prefix))
            if decoded is None:
                preview_reason = "文件不是可严格解码的 UTF-8 文本"
            else:
                lines = decoded.splitlines(keepends=True)
                complete_prefix = len(prefix) >= expected_size
                if boundary_truncated:
                    complete_prefix = False
                if len(lines) > MAX_FILE_PREVIEW_LINES:
                    lines = lines[:MAX_FILE_PREVIEW_LINES]
                    complete_prefix = False
                if len(prefix) >= MAX_FILE_PREVIEW_BYTES:
                    complete_prefix = False
                preview_text = "".join(lines)
                preview_complete = complete_prefix
                if not preview_complete:
                    preview_reason = "文件正文超过单次预览上限"
        else:
            preview_reason = "该 Artifact 类型不支持正文预览"
        mime_type = _artifact_mime_type(expected_type, path.suffix)
        return {
            "ref": ref,
            "output_ref": ref,
            "artifact_id": artifact.id,
            "filename": path.name,
            "display_name": display_name or path.name,
            "path": str(path),
            "sha256": artifact.sha256,
            "size_bytes": expected_size,
            "role": artifact.role,
            "source": artifact.source,
            "step_id": step.id,
            "attempt": result.attempt,
            "mime_type": mime_type,
            "preview_text": preview_text,
            "preview_complete": preview_complete,
            "preview_reason": preview_reason,
        }

    @staticmethod
    def _public_answer_outputs(
        facts: list[Mapping[str, Any]], files: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        files_by_ref = {
            str(item.get("output_ref") or item.get("ref")): item
            for item in files
            if item.get("output_ref") or item.get("ref")
        }
        outputs: list[dict[str, Any]] = []
        for fact in facts:
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), Mapping) else {}
            expected_type = fact.get("expected_type")
            kind = str(fact.get("kind") or "field")
            contract = public_type_info(str(expected_type or ""), kind=kind)
            file_info = files_by_ref.get(str(fact.get("output_ref")))
            outputs.append(
                {
                    "ref": fact.get("output_ref"),
                    "kind": kind,
                    "name": fact.get("name"),
                    "property": fact.get("result_property"),
                    "type": expected_type,
                    "unit": contract.get("unit"),
                    "mime_type": contract.get("mime_type"),
                    "shape": contract.get("shape"),
                    "supported_views": list(contract.get("supported_views", [])),
                    "label": metadata.get("label"),
                    "description": metadata.get("description"),
                    "caveat": metadata.get("caveat"),
                    "verified_value": (fact.get("value") if kind in {"field", "check"} else None),
                    "task_context": {
                        key: fact.get(key)
                        for key in (
                            "system",
                            "step_tool",
                            "method_profile",
                            "environment",
                            "result_status",
                        )
                        if fact.get(key) is not None
                    },
                    "file": (
                        {
                            "available": True,
                            "supports_view": ["auto", "code", "link"],
                            "preview_complete": bool(file_info.get("preview_complete")),
                        }
                        if file_info is not None
                        else None
                    ),
                }
            )
        return outputs

    @staticmethod
    def _answer_draft_covers_outputs(
        draft: AnswerOutput | None,
        facts: list[Mapping[str, Any]],
        *,
        files: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        preferences: Mapping[str, Any] | None = None,
    ) -> bool:
        outputs = Agent._answer_output_map(facts, files)
        required = [str(fact.get("output_ref")) for fact in facts if fact.get("output_ref")]
        try:
            validate_result_answer(draft, outputs, required, preferences=preferences)
        except ValueError:
            return False
        return True

    @staticmethod
    def _answer_output_map(
        facts: list[Mapping[str, Any]], files: list[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        files_by_ref = {
            str(item.get("output_ref") or item.get("ref")): item
            for item in files
            if item.get("output_ref") or item.get("ref")
        }
        task_keys = {
            str(fact.get("task_key")) for fact in facts if fact.get("task_key") is not None
        }
        outputs: dict[str, dict[str, Any]] = {}
        for fact in facts:
            ref = fact.get("output_ref")
            if not isinstance(ref, str) or not ref:
                continue
            if isinstance(fact, dict):
                fact["_include_task_identity"] = len(task_keys) > 1
            outputs[ref] = {
                "ref": ref,
                "kind": fact.get("kind"),
                "type": fact.get("expected_type"),
                "fact": fact,
                "file": files_by_ref.get(ref),
                "supported_views": list(
                    public_type_info(
                        str(fact.get("expected_type") or ""),
                        kind=str(fact.get("kind") or "field"),
                    ).get("supported_views", [])
                ),
            }
        return outputs

    @staticmethod
    def _answer_goal_context(
        run: Run, facts: list[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Bind presentation goals to current verified outputs, without planning work."""

        requirements = {item.id: item for item in run.request.requirements}
        goals: list[dict[str, Any]] = []
        notes: list[str] = []
        for goal in run.request.answer_goals:
            calculations: list[dict[str, Any]] = []
            for requirement_id in goal.requirement_ids:
                requirement = requirements.get(requirement_id)
                fact = next(
                    (
                        item
                        for item in facts
                        if item.get("requirement_id") == requirement_id
                        and goal.output
                        in {str(item.get("name") or ""), str(item.get("result_property") or "")}
                    ),
                    None,
                )
                if requirement is None or fact is None:
                    calculations = []
                    break
                calculations.append(
                    {
                        "capability": requirement.capability,
                        "method_profile": fact.get("method_profile"),
                        "output_ref": fact.get("output_ref"),
                    }
                )
            if len(calculations) != len(goal.requirement_ids):
                continue
            goals.append(
                {
                    "kind": goal.kind,
                    "mode": goal.mode,
                    "output": goal.output,
                    "calculations": calculations,
                }
            )
            if goal.mode == "side_by_side" and all(
                item["capability"] == "optimize_geometry" for item in calculations
            ):
                notes.append(
                    "以下能量分别来自各自优化后的几何；这些几何可能不同，绝对能量差"
                    "不能单独用于判断哪种方法更准确。"
                )
        return goals, ("\n".join(dict.fromkeys(notes)) or None)

    def _render_verified_delivery(
        self,
        run: Run,
        facts: list[dict[str, Any]],
        files: list[dict[str, Any]],
        *,
        unavailable: list[str],
        cancel: Event | None,
        question: str | None = None,
        preferences: Mapping[str, Any] | None = None,
        status_override: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        for index, fact in enumerate(facts, start=1):
            fact.setdefault("output_ref", f"out_{index}")
        file_refs = {
            str(item.get("output_ref") or item.get("ref"))
            for item in files
            if item.get("output_ref") or item.get("ref")
        }
        renderable: list[dict[str, Any]] = []
        unavailable_targets = list(unavailable)
        for fact in facts:
            ref = str(fact["output_ref"])
            if fact.get("kind") == "port" and ref not in file_refs:
                label = str(_mapping(fact.get("metadata")).get("label") or fact.get("name") or ref)
                if label not in unavailable_targets:
                    unavailable_targets.append(label)
                continue
            renderable.append(fact)
        delivery_preferences = dict(
            preferences if preferences is not None else run.request.output_preferences
        )
        cancelled = status_override == "cancelled" or (cancel is not None and cancel.is_set())
        required = [str(fact["output_ref"]) for fact in renderable]
        if not required:
            if cancelled:
                missing = (
                    f"\n尚未交付：{'；'.join(unavailable_targets)}；未重新计算。"
                    if unavailable_targets
                    else ""
                )
                return (
                    "本次结果呈现已取消。已完成的科学结果仍保留；本次未将其标记为完整交付。"
                    + missing,
                    {
                        "status": "cancelled",
                        "rendered_refs": [],
                        "unavailable_targets": unavailable_targets,
                        "outputs": [],
                    },
                )
            missing = (
                f"\n尚未交付：{'；'.join(unavailable_targets)}；未重新计算。"
                if unavailable_targets
                else ""
            )
            return (
                "本次目标暂无可交付的当前文件或结果；未重新计算。" + missing,
                {
                    "status": "unavailable",
                    "rendered_refs": [],
                    "unavailable_targets": unavailable_targets,
                    "outputs": [],
                },
            )
        outputs = self._answer_output_map(renderable, files)
        answer_goal_context, answer_goal_note = self._answer_goal_context(run, renderable)
        draft: AnswerOutput | None = None
        if not cancelled:
            try:
                draft, _error = self._compose_answer_draft(
                    question or run.request.description,
                    mode="result",
                    available_outputs=self._public_answer_outputs(renderable, files),
                    required_outputs=required,
                    context={
                        "run_status": run.status,
                        "output_preferences": delivery_preferences,
                        "answer_goals": answer_goal_context,
                    },
                    cancel=cancel,
                )
            except LlmError as error:
                # The provider can observe cancellation after the initial
                # check above.  Re-read the event so the delivery cannot be
                # reported as complete after a cancelled answer call.
                if error.category == "cancelled":
                    cancelled = True
                draft = None
            cancelled = cancelled or (cancel is not None and cancel.is_set())
        fallback = AnswerOutput(
            action="respond",
            sections=[
                AnswerSection(
                    format="auto",
                    heading="results",
                    output_refs=required,
                    detail=str(delivery_preferences.get("detail", "normal")),
                    text=None,
                )
            ],
        )
        if cancelled:
            text = render_answer_output(
                fallback,
                outputs_by_ref=outputs,
                required_refs=required,
                preferences=delivery_preferences,
            )
            text = f"{text}\n本次结果呈现已取消。已完成的科学结果仍保留；本次未将其标记为完整交付。"
        elif self._answer_draft_covers_outputs(
            draft,
            renderable,
            files=files,
            preferences=delivery_preferences,
        ):
            try:
                text = render_answer_output(
                    draft,
                    outputs_by_ref=outputs,
                    required_refs=required,
                    preferences=delivery_preferences,
                )
            except ValueError:
                draft = None
                text = render_answer_output(
                    fallback,
                    outputs_by_ref=outputs,
                    required_refs=required,
                    preferences=delivery_preferences,
                )
        else:
            text = render_answer_output(
                fallback,
                outputs_by_ref=outputs,
                required_refs=required,
                preferences=delivery_preferences,
            )
        if answer_goal_note and not cancelled:
            text = f"{answer_goal_note}\n{text}"
        if unavailable_targets:
            text = f"{text}\n尚未交付：{'；'.join(unavailable_targets)}；未重新计算。"
        locators: list[dict[str, Any]] = []
        for fact in renderable:
            ref = str(fact["output_ref"])
            file_info = next(
                (item for item in files if str(item.get("output_ref") or item.get("ref")) == ref),
                None,
            )
            result = fact.get("_result")
            fact_run = fact.get("_run")
            locator = {
                "output_ref": ref,
                "run_id": fact_run.id if isinstance(fact_run, Run) else run.id,
                "step_id": fact.get("step_id"),
                "subject_ref": fact.get("subject_ref"),
                "kind": fact.get("kind"),
                "name": fact.get("name"),
                "property": fact.get("result_property"),
                "attempt": result.attempt if isinstance(result, Result) else None,
                "step_fingerprint": (
                    result.step_fingerprint if isinstance(result, Result) else None
                ),
            }
            if file_info is not None:
                locator.update(
                    {
                        "artifact_id": file_info.get("artifact_id"),
                        "path": file_info.get("path"),
                        "sha256": file_info.get("sha256"),
                        "filename": file_info.get("filename"),
                        "preview_complete": file_info.get("preview_complete"),
                    }
                )
            locators.append(locator)
        if cancelled:
            status = "cancelled"
        elif unavailable_targets:
            status = "partial"
        else:
            status = "complete"
        return text, {
            "status": status,
            "rendered_refs": [str(fact["output_ref"]) for fact in renderable],
            "unavailable_targets": unavailable_targets,
            "outputs": locators,
        }

    def _files_for_facts(
        self, facts: list[Mapping[str, Any]], *, cancel: Event | None = None
    ) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for index, fact in enumerate(facts, start=1):
            run = fact.get("_run")
            result = fact.get("_result")
            if not isinstance(run, Run) or not isinstance(result, Result):
                continue
            step = next((item for item in run.plan.steps if item.id == fact.get("step_id")), None)
            if step is None or fact.get("kind") != "port":
                continue
            value = fact.get("value")
            artifact_id = value.get("artifact_id") if isinstance(value, Mapping) else None
            if not isinstance(artifact_id, str):
                continue
            try:
                artifact = find_artifact(run, artifact_id)
            except ValueError:
                continue
            output_ref = str(fact.get("output_ref") or f"out_{index}")
            descriptor = self._artifact_file_descriptor(
                run,
                step,
                result,
                artifact,
                ref=output_ref,
                expected_type=str(fact.get("expected_type") or artifact.artifact_type),
                display_name=str(
                    _mapping(fact.get("metadata")).get("label") or path_label(artifact)
                ),
                cancel=cancel,
            )
            if descriptor is not None:
                files.append(descriptor)
        self._limit_total_previews(files)
        return files

    @staticmethod
    def _limit_total_previews(files: list[dict[str, Any]]) -> None:
        remaining = MAX_TOTAL_FILE_PREVIEW_BYTES
        for item in files:
            preview = item.get("preview_text")
            if not isinstance(preview, str):
                continue
            encoded = preview.encode("utf-8")
            if len(encoded) <= remaining:
                remaining -= len(encoded)
                continue
            prefix = encoded[:remaining].decode("utf-8", errors="ignore")
            lines = prefix.splitlines(keepends=True)
            complete_lines = lines if not lines or lines[-1].endswith(("\n", "\r")) else lines[:-1]
            item["preview_text"] = "".join(complete_lines)
            item["preview_complete"] = False
            item["preview_reason"] = "总文件预览上限为 64 KiB"
            remaining = 0

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

    def _append_message(self, role: str, content: str, *, save: bool = True) -> None:
        messages = self._session.setdefault("recent_messages", [])
        bounded_content = content if len(content) <= 4000 else content[:3997] + "..."
        messages.append({"role": role, "content": bounded_content})
        self._session["recent_messages"] = messages[-12:]
        if save:
            self._save_session()

    def _record_response(
        self,
        response: AgentResponse,
        *,
        cancel: Event | None = None,
        request_token: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """Persist a bounded conversational summary and delivery locators."""

        if cancel is not None and cancel.is_set():
            return
        if request_token is not None:
            if self._active_request is None or self._active_request[0] != request_token:
                return
        if session_id is not None and self.session_id != session_id:
            return
        summary = re.sub(r"```.*?```", "[已交付文件正文省略]", response.text, flags=re.DOTALL)
        self._append_message("assistant", summary, save=False)
        delivery = response.delivery
        if isinstance(delivery, Mapping) and delivery.get("outputs"):
            outputs = [
                dict(item) for item in delivery.get("outputs", []) if isinstance(item, Mapping)
            ]
            if delivery.get("status") == "complete":
                self._session["last_delivery"] = outputs[-8:]
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
        last_delivery = [
            item for item in self._session.get("last_delivery", []) if isinstance(item, Mapping)
        ]

        def _subject_ref(run: Run, step: Step) -> str:
            task_key = f"{run.id}:{step.id}"
            if task_key in subject_refs:
                return subject_refs[task_key]
            if step.tool == "resolve_molecule":
                molecule_count = sum(ref.startswith("m") for ref in subject_refs.values())
                ref = f"m{molecule_count + 1}"
            else:
                task_count = sum(ref.startswith("t") for ref in subject_refs.values())
                ref = f"t{task_count + 1}"
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
                for output in tool.public_outputs():
                    kind = str(output["kind"])
                    name = str(output["name"])
                    expected_type = str(output["type"])
                    property_name = str(output["property"])
                    artifact = None
                    if kind == "field":
                        if name not in result.values:
                            continue
                        if not _query_value_is_compatible(result.values[name], expected_type):
                            continue
                    elif kind == "port":
                        artifact = self._query_port_artifact(run, step, result, name, expected_type)
                        if artifact is None:
                            continue
                    elif kind == "check":
                        check = result.scientific_checks.get(name)
                        if check is None or not is_compatible_value(
                            check.model_dump(mode="python"), "scientific_check"
                        ):
                            continue
                        if not _result_check_input_is_bound(
                            self.config.data_root_path,
                            run,
                            step,
                            result,
                            name,
                            check,
                            self.registry,
                        ):
                            continue
                    else:
                        continue
                    subject_ref = _subject_ref(run, step)
                    binding = {
                        "session_id": self.session_id,
                        "run_id": run.id,
                        "step_id": step.id,
                        "result_path": relative,
                        "attempt": result.attempt,
                        "step_fingerprint": _step_fingerprint(step),
                        "kind": kind,
                        "name": name,
                        "property": property_name,
                    }
                    if artifact is not None:
                        binding.update(
                            {
                                "artifact_id": artifact.id,
                                "artifact_sha256": artifact.sha256,
                                "artifact_size": artifact.size_bytes,
                            }
                        )
                    recently_delivered = self._delivery_locator_matches(
                        binding,
                        artifact=artifact,
                        last_delivery=last_delivery,
                    )
                    key = (subject_ref, property_name)
                    if key in self._query_bindings:
                        raise ValueError(
                            "ambiguous query binding: "
                            f"subject {subject_ref!r} exposes property {property_name!r} twice"
                        )
                    self._query_bindings[key] = binding
                    catalog.append(
                        self._public_query_entry(
                            subject_ref,
                            run,
                            step,
                            tool,
                            name=name,
                            kind=kind,
                            expected_type=expected_type,
                            property_name=property_name,
                            structure=structure,
                            check_status=(
                                result.scientific_checks[name].status if kind == "check" else None
                            ),
                            recently_delivered=recently_delivered,
                        )
                    )
                    if len(catalog) >= MAX_QUERY_CATALOG_ITEMS:
                        return catalog
        return catalog

    @staticmethod
    def _delivery_locator_matches(
        binding: Mapping[str, Any],
        *,
        artifact: Any | None,
        last_delivery: list[Mapping[str, Any]],
    ) -> bool:
        """Match a saved delivery only after its current result is revalidated."""

        for item in last_delivery:
            if any(
                item.get(key) != binding.get(key)
                for key in ("run_id", "step_id", "property", "kind")
            ):
                continue
            if item.get("attempt") != binding.get("attempt"):
                continue
            if item.get("step_fingerprint") != binding.get("step_fingerprint"):
                continue
            if artifact is not None:
                if item.get("artifact_id") != artifact.id:
                    continue
                if item.get("sha256") != artifact.sha256:
                    continue
            return True
        return False

    def _build_geometry_catalog(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Expose bounded aliases for verified molecular-geometry outputs."""

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
                relative = run.current_results.get(step.id)
                if not isinstance(relative, str):
                    continue
                result = _load_bound_result(self.config.data_root_path, run, relative)
                if result is None or not self._query_result_is_valid(run, step, result, relative):
                    continue
                for port, expected_type in tool.output_ports.items():
                    if len(catalog) >= MAX_HISTORY_GEOMETRIES:
                        return catalog, bindings
                    if expected_type != "molecular_geometry":
                        continue
                    artifact = self._query_port_artifact(run, step, result, port, expected_type)
                    if artifact is None or artifact.role not in {
                        "initial_geometry",
                        "optimized_geometry",
                    }:
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
                        "port": port,
                        "role": artifact.role,
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
                                "role": f"verified {artifact.role}",
                                "port": port,
                                "atom_count": geometry.atom_count,
                            },
                            "calculation": {
                                "tool": tool.name,
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
    ) -> tuple[Run, Step, Result, Any, Path, str]:
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
        try:
            source_tool = self.registry.get(step.tool)
        except ValueError as error:
            raise ValueError("history geometry source Tool is unavailable") from error
        port = binding.get("port")
        if not isinstance(port, str) or source_tool.output_ports.get(port) != "molecular_geometry":
            raise ValueError("history geometry binding does not name a geometry output port")
        artifact = self._query_port_artifact(source_run, step, result, port, "molecular_geometry")
        if (
            artifact is None
            or artifact.id != binding.get("artifact_id")
            or artifact.role not in {"initial_geometry", "optimized_geometry"}
            or (binding.get("role") is not None and artifact.role != binding.get("role"))
            or artifact.step_id != step.id
            or artifact.attempt != result.attempt
            or artifact.sha256 != binding.get("sha256")
        ):
            raise ValueError("history geometry artifact binding is invalid")
        path = artifact_path(self.config.data_root_path, source_run, artifact)
        parse_xyz_bytes(path.read_bytes())
        return source_run, step, result, artifact, path, port

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
        check_status: str | None = None,
        recently_delivered: bool = False,
    ) -> dict[str, Any]:
        metadata = dict(tool.result_metadata.get(name, {}))
        metadata.setdefault("label", name.replace("_", " "))
        metadata.setdefault("description", tool.description)
        contract = public_type_info(expected_type, kind=kind)
        item = {
            "subject_ref": subject_ref,
            "active_task": run.id == self._session.get("active_run_id"),
            "recently_delivered": recently_delivered,
            "task": {
                "description": run.request.description,
                "status": run.status,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            },
            "system": structure,
            "step": {
                "purpose": tool.display_name or tool.description,
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
                "type": contract["type"],
                "unit": contract["unit"],
                "mime_type": contract["mime_type"],
                "supported_views": list(contract.get("supported_views", [])),
                "artifact_type": expected_type if kind == "port" else None,
                "caveat": metadata.get("caveat"),
                "validity": "verified",
                **({"check_status": check_status} if kind == "check" else {}),
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
        if not isinstance(name, str) or kind not in {"field", "port", "check"}:
            return None
        descriptor = next(
            (
                item
                for item in tool.public_outputs()
                if item["kind"] == kind and item["name"] == name
            ),
            None,
        )
        if descriptor is None or descriptor["property"] != property_name:
            return None
        expected_type = str(descriptor["type"])
        value: Any
        if kind == "field":
            if name not in result.values or not _query_value_is_compatible(
                result.values[name], expected_type
            ):
                return None
            value = result.values[name]
        elif kind == "port":
            artifact = self._query_port_artifact(run, step, result, name, expected_type)
            if (
                artifact is None
                or artifact.id != resolved.get("artifact_id")
                or (
                    resolved.get("artifact_sha256") is not None
                    and artifact.sha256 != resolved.get("artifact_sha256")
                )
            ):
                return None
            value = {"artifact_id": artifact.id}
        else:
            check = result.scientific_checks.get(name)
            if check is None or not is_compatible_value(
                check.model_dump(mode="python"), "scientific_check"
            ):
                return None
            if not _result_check_input_is_bound(
                self.config.data_root_path, run, step, result, name, check, self.registry
            ):
                return None
            value = check.model_dump(mode="python")
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
            current_facts = facts_from_result(run, safe_result, self.registry, structure=structure)
            for fact in current_facts:
                fact["_run"] = run
                fact["_result"] = result
            facts.extend(current_facts)
        return facts

    def _incomplete_target_labels(
        self, run: Run, facts: list[dict[str, Any]] | None = None
    ) -> list[str]:
        facts = self._current_run_facts(run) if facts is None else facts
        verified = {(fact.get("step_id"), fact.get("kind"), fact.get("name")) for fact in facts}
        labels: list[str] = []
        for target in run.plan.requested_results:
            if target.check is not None:
                step = None
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
                            self.config.data_root_path,
                            run,
                            step,
                            result,
                            target.check,
                            check,
                            self.registry,
                        )
                    ):
                        continue
                label = (
                    self.registry.get(step.tool)
                    .result_metadata.get(target.check, {})
                    .get("label", target.check.replace("_", " "))
                    if step is not None
                    else target.check.replace("_", " ")
                )
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
        tool = self.registry.get(step.tool)
        if tool.scientific_checks and not tool.validate_result(run, step, result):
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
            if not _artifact_integrity_matches(path, artifact):
                return None
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

    def _compose_answer_draft(
        self,
        question: str,
        *,
        mode: str,
        capability_catalog: list[Mapping[str, Any]] | None = None,
        available_outputs: list[Mapping[str, Any]] | None = None,
        required_outputs: list[str] | tuple[str, ...] = (),
        context: Mapping[str, Any] | None = None,
        cancel: Event | None = None,
    ) -> tuple[AnswerOutput | None, str | None]:
        start = self._llm_call_count()
        failure_category: str | None = None
        if not callable(getattr(self.llm, "complete_json", None)):
            return None, "answer protocol is unavailable on the configured model client"
        try:
            draft = compose_answer(
                self.llm,
                question=question,
                mode=mode,  # type: ignore[arg-type]
                capability_catalog=capability_catalog or self.registry.result_capabilities(),
                available_outputs=available_outputs or (),
                required_outputs=required_outputs,
                context=context,
                cancel=cancel,
            )
            if not isinstance(draft, AnswerOutput):
                draft = AnswerOutput.model_validate(draft, strict=True)
        except LlmError as error:
            failure_category = error.category
            if error.category == "cancelled":
                raise
            return None, str(error)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            failure_category = "schema_error"
            return None, str(error)
        finally:
            self._persist_llm_diagnostics(
                start,
                stage="answer",
                failure_category=failure_category,
            )
        return draft, None

    @staticmethod
    def _validated_tool_targets(
        draft: AnswerOutput, capability_catalog: list[Mapping[str, Any]]
    ) -> tuple[str, ...]:
        names = {str(item.get("name")) for item in capability_catalog if item.get("name")}
        requested = tuple(dict.fromkeys(str(item) for item in draft.requested_results))
        if not requested or any(item not in names for item in requested):
            return ()
        return requested

    def _answer_question(
        self,
        question: str,
        *,
        draft: AnswerOutput | None = None,
        error: str | None = None,
        cancel: Event | None = None,
    ) -> AgentResponse:
        if cancel is not None and cancel.is_set():
            return AgentResponse(
                "当前请求已取消。",
                delivery={
                    "status": "cancelled",
                    "rendered_refs": [],
                    "unavailable_targets": [],
                    "outputs": [],
                },
            )
        if draft is not None and draft.action == "respond":
            text = render_answer_output(draft)
            if not text:
                text = "当前配置的模型没有生成可展示的回答。"
        elif draft is not None and draft.action == "clarify":
            text = render_answer_output(draft) or "请进一步明确要查询的对象或性质。"
        elif error:
            text = f"当前配置的模型无法回答：{error}"
        else:
            text = "当前配置的模型没有生成可展示的回答。"
        response = AgentResponse(text)
        self._record_response(response, cancel=cancel)
        return response

    def _answer_context(
        self,
        question: str,
        *,
        selection: QuerySelection | None,
        catalog: list[Mapping[str, Any]],
        preferences: Mapping[str, Any] | None = None,
        cancel: Event | None = None,
    ) -> AgentResponse:
        if cancel is not None and cancel.is_set():
            return AgentResponse(
                "当前请求已取消。",
                delivery={
                    "status": "cancelled",
                    "rendered_refs": [],
                    "unavailable_targets": [],
                    "outputs": [],
                },
            )
        if selection is None:
            text = render_clarification({"status": "unavailable", "missing_description": question})
            response = AgentResponse(text)
            self._record_response(response, cancel=cancel)
            return response
        if selection.status != "selected":
            text = render_clarification(selection.model_dump(mode="python"))
            response = AgentResponse(text)
            self._record_response(response, cancel=cancel)
            return response

        catalog_refs = {
            str(item.get("subject_ref"))
            for item in catalog
            if isinstance(item, Mapping) and isinstance(item.get("subject_ref"), str)
        }
        if any(target.subject_ref not in catalog_refs for target in selection.targets):
            text = render_clarification({"binding_invalid": True})
            response = AgentResponse(text)
            self._record_response(response, cancel=cancel)
            return response
        selected_facts: list[dict[str, Any]] = []
        unavailable: list[str] = []
        for target in selection.targets:
            fact = self._load_query_fact(target.subject_ref, target.property)
            if fact is None:
                unavailable.append(_query_target_label(target, catalog))
                continue
            selected_facts.append(fact)
        targets = [target.model_dump(mode="python") for target in selection.targets]
        facts, _covered = select_facts_for_question(targets, selected_facts)
        if not facts:
            status = "cancelled" if cancel is not None and cancel.is_set() else "unavailable"
            text = "本次所选结果当前不可交付；未重新计算。"
            if unavailable:
                text += f"\n尚未交付：{'；'.join(unavailable)}。"
            response = AgentResponse(
                text,
                delivery={
                    "status": status,
                    "rendered_refs": [],
                    "unavailable_targets": unavailable,
                    "outputs": [],
                },
            )
            self._record_response(response, cancel=cancel)
            return response
        for index, fact in enumerate(facts, start=1):
            fact["output_ref"] = f"out_{index}"
        files = self._files_for_facts(facts, cancel=cancel)
        first = facts[0] if facts else {}
        run = first.get("_run")
        result = first.get("_result")
        if not isinstance(run, Run):
            response = AgentResponse(
                "所选结果的来源任务已无法重新验证；未重新计算。",
                files=tuple(files),
                delivery={
                    "status": "unavailable",
                    "rendered_refs": [],
                    "unavailable_targets": [question],
                    "outputs": [],
                },
            )
            self._record_response(response, cancel=cancel)
            return response
        text, delivery = self._render_verified_delivery(
            run,
            facts,
            files,
            unavailable=unavailable,
            cancel=cancel,
            question=question,
            preferences=preferences,
        )
        response = AgentResponse(
            text,
            run=run,
            result=result,
            files=tuple(files),
            delivery=delivery,
        )
        self._record_response(response, cancel=cancel)
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _target_identity(target: Any, kind: str | None, name: str | None) -> str:
    step_id = getattr(target, "step_id", None) or "unbound"
    return f"{step_id}:{kind or 'unknown'}:{name or 'unknown'}"


def _query_target_label(target: Any, catalog: list[Mapping[str, Any]]) -> str:
    subject_ref = getattr(target, "subject_ref", None)
    property_name = getattr(target, "property", None)
    for item in catalog:
        if not isinstance(item, Mapping) or item.get("subject_ref") != subject_ref:
            continue
        result = item.get("result")
        if not isinstance(result, Mapping) or result.get("property") != property_name:
            continue
        return str(result.get("label") or property_name)
    return str(property_name or "所选结果")


def path_label(artifact: Any) -> str:
    role = getattr(artifact, "role", None)
    return {
        "initial_geometry": "初始 XYZ 结构文件",
        "optimized_geometry": "优化后的 XYZ 结构文件",
        "verified_hessian": "已验证 Hessian 文件",
    }.get(str(role), str(role or "已验证文件"))


def _public_unit(expected_type: Any) -> str | None:
    return {
        "Eh": "Eh",
        "angstrom": "Å",
        "degree": "°",
        "frequency": "cm⁻¹",
    }.get(str(expected_type))


def _artifact_mime_type(expected_type: Any, suffix: str | None) -> str:
    if expected_type == "molecular_geometry":
        return "chemical/x-xyz"
    if expected_type == "molecule":
        return "application/json"
    if expected_type == "energy_data":
        return "application/json"
    if expected_type == "text_file":
        return "text/plain"
    if expected_type == "orca_hessian":
        return "text/plain"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def _artifact_integrity_matches(path: Path, artifact: Any) -> bool:
    try:
        if path.stat().st_size != artifact.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == artifact.sha256
    except OSError:
        return False


def _decode_utf8_preview(data: bytes) -> tuple[str | None, bool]:
    """Decode a bounded UTF-8 prefix without treating a split code point as invalid."""

    for trim in range(0, min(4, len(data) + 1)):
        candidate = data[: len(data) - trim] if trim else data
        try:
            return candidate.decode("utf-8"), trim > 0
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data":
                return None, False
    return None, False


def _normalize_explicit_plan(
    registry: ToolRegistry, plan: Plan, *, defaults: Any | None = None
) -> Plan:
    """Resolve defaults before an explicit Run receives its acceptance snapshot."""

    steps: list[Step] = []
    for step in plan.steps:
        tool = registry.get(step.tool)
        if tool.requires_compute_permission:
            original_parameters = dict(step.parameters)
            checked_parameters = tool.validate_parameters(original_parameters, allow_deferred=True)
            supplied_parameters = {
                name: checked_parameters[name]
                for name in original_parameters
                if name in checked_parameters
            }
            if defaults is not None and tool.preparation_function is not None:
                supplied_parameters.setdefault("method_profile", defaults.method_profile)
                supplied_parameters.setdefault("environment", defaults.environment)
            parameters = tool.validate_parameters(supplied_parameters)
            step = step.model_copy(update={"parameters": parameters})
        steps.append(step)
    return plan.model_copy(update={"steps": steps})


def _is_parameter_continuation(
    run: Run,
    intake: Any,
    message: str,
    explicit_parameters: dict[str, Any],
) -> bool:
    if intake.intent != "chemistry_compute" or not explicit_parameters:
        return False
    compatibility_updates = getattr(intake, "__dict__", {})
    if (
        intake.molecule_query
        or intake.subjects
        or intake.requirements
        or intake.structure_input
        or intake.history_geometry_alias
        or intake.requested_results
        or compatibility_updates.get("molecule_query")
        or compatibility_updates.get("subjects")
        or compatibility_updates.get("requirements")
        or compatibility_updates.get("structure_input")
        or compatibility_updates.get("history_geometry_alias")
        or compatibility_updates.get("requested_results")
    ):
        return False
    if intake.operations and run.request.operations:
        if intake.operations != run.request.operations:
            return False
    if (
        intake.parameter_target_requirement_id is not None
        and intake.parameter_target_requirement_id
        not in {item.id for item in run.request.requirements}
    ):
        return False
    if _looks_like_molecule_change(message) and not _looks_like_parameter_only_change(message):
        return False
    return run.waiting_for in {"clarification", "confirmation"}


def _pending_missing_fields_are_scoped(run: Run, intake: Any, registry: ToolRegistry) -> bool:
    """Allow a waiting update only for fields owned by the current plan."""

    known = set(run.request.missing_fields)
    pending = run.pending_data.get("missing_fields")
    if isinstance(pending, list):
        known.update(item for item in pending if isinstance(item, str))
    known.update(
        field for step in run.plan.steps for field in registry.get(step.tool).request_parameters
    )
    return set(intake.missing_fields) <= known


def _semantic_pending_tasks(
    run: Run | None,
    registry: ToolRegistry,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Give Semantic LLM short task refs without persistent Requirement IDs."""

    if run is None or run.status != "waiting":
        return [], {}
    tasks: list[dict[str, Any]] = []
    ref_map: dict[str, str] = {}
    for index, requirement in enumerate(run.request.requirements, start=1):
        ref = f"t{index}"
        parameters = dict(requirement.parameters)
        profile = parameters.pop("method_profile", None)
        resolution = requirement.constraints.get("method_resolution")
        method_request = (
            resolution.get("request")
            if isinstance(resolution, Mapping) and isinstance(resolution.get("request"), str)
            else None
        )
        if method_request is None and isinstance(profile, str):
            method_request = next(
                (
                    str(item["display_name"])
                    for item in registry.method_capability_catalog()
                    if item["name"] == profile
                ),
                None,
            )
        task = {
            "task_ref": ref,
            "capability": requirement.capability,
            "parameters": parameters,
        }
        if method_request is not None:
            task["method_request"] = method_request
        tasks.append(task)
        ref_map[ref] = requirement.id
    return tasks, ref_map


def _intake_with_history_geometry(intake: IntakeOutput, alias: str) -> IntakeOutput:
    if intake.intent != "chemistry_compute":
        raise ValueError("historical geometry can only bind a calculation Request")
    subjects = dict(intake.subjects)
    if not subjects:
        subjects["subject_1"] = IntakeSubjectProposal(key="subject_1")
    if len(subjects) != 1:
        raise ValueError("historical geometry must identify exactly one calculation subject")
    subject_key, subject = next(iter(subjects.items()))
    if subject.inline_xyz is not None:
        raise ValueError("a request cannot combine historical geometry and inline XYZ")
    subjects[subject_key] = subject.model_copy(update={"history_geometry_alias": alias})
    return intake.model_copy(update={"subjects": subjects})


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
    for subject_id, subject in request.subjects.items():
        if not isinstance(subject, Mapping):
            continue
        subject_input = subject.get("structure_input", {})
        if not isinstance(subject_input, Mapping) or not (
            subject_input.get("xyz_text") is not None or subject_input.get("xyz") is not None
        ):
            continue
        key = str(subject.get("key") or subject_id)
        alias = f"request_geometry_{key}"
        aliases[alias] = alias
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


def _pending_molecule_selection(
    message: str, pending_data: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Parse only bounded molecule choices from the current pending snapshot."""

    candidates = pending_data.get("candidates")
    candidate_list = (
        [item for item in candidates if isinstance(item, Mapping)]
        if isinstance(candidates, list)
        else []
    )
    stripped = message.strip()
    candidate_label = re.fullmatch(r"(?i)(?:候选(?:编号)?|candidate\s*[_-]?\s*)(\d+)", stripped)
    if candidate_label is not None:
        number = int(candidate_label.group(1))
        for index, candidate in enumerate(candidate_list, start=1):
            choice_id = str(candidate.get("choice_id") or f"candidate_{index}")
            if choice_id.casefold() == f"candidate_{number}" or (
                "choice_id" not in candidate and index == number
            ):
                return _candidate_selection(candidate)
        return {
            "invalid": (
                f"候选编号 {number} 不在当前候选快照中；请回复列出的编号、CID 或明确 SMILES。"
            )
        }

    numeric = re.fullmatch(r"\d+", stripped)
    if numeric is not None:
        number = int(numeric.group(0))
        if 1 <= number <= len(candidate_list):
            return _candidate_selection(candidate_list[number - 1])
        if candidate_list:
            return {
                "invalid": (
                    f"纯数字 {number} 不在当前候选快照的编号范围内；如需按 CID 选择，"
                    "请明确写成“CID "
                    f"{number}”。"
                )
            }
        return {"invalid": "当前没有可按序号选择的候选；如需按 CID 选择，请明确写成“CID 数字”。"}

    cid_match = re.fullmatch(r"CID\s*[:#]?\s*(\d+)", stripped, re.IGNORECASE)
    if cid_match is not None:
        cid = int(cid_match.group(1))
        for candidate in candidate_list:
            if _candidate_cid(candidate) == cid:
                return _candidate_selection(candidate)
        return {
            "query": str(cid),
            "input_kind": "cid",
            "explicit_new": True,
            "preserve_identity": True,
        }

    for candidate in candidate_list:
        title = candidate.get("title") or candidate.get("Title")
        if isinstance(title, str) and title.strip() and title.casefold() == stripped.casefold():
            return _candidate_selection(candidate)

    explicit_smiles = ()
    try:
        explicit_smiles = extract_explicit_smiles(stripped)
    except ValueError:
        return {"invalid": "SMILES 标签后没有可验证的完整结构。"}
    if (
        len(explicit_smiles) == 1
        and explicit_smiles[0]["start"] == 0
        and explicit_smiles[0]["end"] == len(stripped)
    ):
        smiles_query = str(explicit_smiles[0]["raw_query"])
        matched = _candidate_with_same_structure(candidate_list, smiles_query)
        if matched is not None:
            return _candidate_selection(matched)
        return {
            "query": smiles_query,
            "input_kind": "smiles",
            "explicit_new": True,
            "preserve_identity": True,
        }

    if not re.fullmatch(r"[A-Za-z0-9₀-₉]+", stripped):
        return None
    try:
        formula = formula_token_from_text(stripped)
    except ValueError:
        formula = None
    if formula == stripped:
        try:
            normalized = normalize_formula_token(formula)
        except ValueError:
            return {"invalid": "该分子式无法安全解析，请提供明确身份。"}
        return {
            "query": normalized,
            "input_kind": "formula",
            "explicit_new": True,
            "preserve_identity": True,
        }

    return None


def _validated_identity_reply(intake: Any, *, message: str) -> tuple[str, str]:
    """Validate the identity payload before applying it to a waiting Run."""

    query = intake.molecule_query
    kind = intake.molecule_input_kind
    if not isinstance(query, str) or not query.strip():
        raise ValueError("identity reply has no query")
    query = query.strip()
    stripped = message.strip()

    if kind == "name":
        from .planner import _validate_lookup_name

        _validate_lookup_name(intake, message=message)
        return query, kind

    if kind == "smiles":
        labelled = extract_explicit_smiles(message)
        if labelled:
            if len(labelled) != 1 or labelled[0]["raw_query"] != query:
                raise ValueError("SMILES reply must preserve the explicit structure")
        elif stripped != query:
            raise ValueError("bare SMILES must exactly match this message")
        canonical_structure(query)
        return query, kind

    if kind == "cid":
        matches = list(
            re.finditer(
                r"(?i)(?<![A-Za-z0-9_])CID\s*[:：#=]?\s*"
                r"([^\s，,；;。!?！？]+)",
                message,
            )
        )
        if len(matches) != 1:
            raise ValueError("CID reply requires exactly one explicit CID")
        raw_cid = matches[0].group(1)
        if (
            re.fullmatch(r"[0-9]{1,16}", raw_cid) is None
            or re.fullmatch(r"[0-9]{1,16}", query) is None
            or int(query) <= 0
            or int(query) != int(raw_cid)
        ):
            raise ValueError("CID reply must preserve a complete positive integer")
        return str(int(query)), kind

    if kind == "formula":
        try:
            raw = formula_token_from_text(message)
        except ValueError as error:
            raise ValueError("formula reply must preserve the complete user formula") from error
        if raw is None or normalize_formula_token(raw) != query:
            raise ValueError("formula reply must preserve the complete user formula")
        return query, kind

    raise ValueError("use a supported name, explicit CID, formula, or SMILES")


def _candidate_selection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    cid = _candidate_cid(candidate)
    if cid is None:
        return {"invalid": "当前候选缺少可验证的 CID，无法安全选择。"}
    return {"query": str(cid), "input_kind": "cid", "candidate": candidate}


def _candidate_with_same_structure(
    candidates: list[Mapping[str, Any]], smiles: str
) -> Mapping[str, Any] | None:
    try:
        expected = canonical_structure(smiles)
    except ValueError:
        return None
    for candidate in candidates:
        candidate_smiles = candidate.get("isomeric_smiles") or candidate.get("canonical_smiles")
        if not isinstance(candidate_smiles, str):
            continue
        try:
            if canonical_structure(candidate_smiles) == expected:
                return candidate
        except ValueError:
            continue
    return None


def _candidate_cid(candidate: Mapping[str, Any] | None) -> int | None:
    if not isinstance(candidate, Mapping):
        return None
    value = candidate.get("cid") or candidate.get("CID")
    try:
        cid = int(value)
    except (TypeError, ValueError):
        return None
    return cid if cid > 0 else None


def _looks_like_parameter_only_change(message: str) -> bool:
    """Avoid treating a scoped numerical parameter edit as a molecule change."""

    parameter_change = re.compile(
        r"(?:几何优化|优化几何|geometry optimization|geom_maxiter|SCF|scf_maxiter|"
        r"atom_[ijk])"
        r".{0,16}?(?:改成|改为|设为|设置为|change(?:d)?\s+to|set\s+to|=)\s*[-+]?\d+",
        re.IGNORECASE,
    )
    matches = list(parameter_change.finditer(message))
    if not matches:
        return False
    remainder = parameter_change.sub("", message)
    return not _looks_like_molecule_change(remainder)


def _parameter_issue_clarification(issues: Mapping[str, str]) -> str:
    labels = {
        "atom_i": "第一个原子索引",
        "atom_j": "第二个原子索引",
        "atom_k": "第三个原子索引",
    }
    details = "；".join(
        f"{labels.get(name, name)}：{reason}" for name, reason in sorted(issues.items())
    )
    return f"本轮参数没有更新，也没有启动计算。请提供完整的 1-based 整数索引（{details}）。"


def _has_explicit_atom_index_update(message: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])atom_[ijk](?![A-Za-z0-9_]).{0,16}?"
            r"(?:=|:|改成|改为|设为|设置为)\s*[-+]?\d+(?!\d)",
            message,
            re.IGNORECASE,
        )
    )


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


def _ambiguous_iteration_parameter_change(message: str) -> bool:
    generic_iteration = re.search(
        r"(?:迭代(?:上限|次数|步数|限)?|(?:iteration|iterations|maxiter)\s*"
        r"(?:limit|cap|upper\s*bound)?)",
        message,
        flags=re.IGNORECASE,
    )
    numeric_assignment = re.search(
        r"(?:设(?:置)?为|改为|调整到|设置成|上限为|to|=|:)\s*[-+]?\d+",
        message,
        flags=re.IGNORECASE,
    )
    parameter_is_specific = re.search(
        r"(?:几何|结构优化|几何优化|优化步数|优化迭代|geometry|geom_maxiter|"
        r"\bscf\b|自洽|电子迭代|scf_maxiter)",
        message,
        flags=re.IGNORECASE,
    )
    return (
        generic_iteration is not None
        and numeric_assignment is not None
        and parameter_is_specific is None
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
    for requirement in request.requirements:
        tool = registry.get(requirement.capability)
        for source_name, mapping in (
            ("requirement parameters", requirement.parameters),
            (
                "requirement user modifications",
                request.user_modifications_by_requirement.get(requirement.id, {}),
            ),
        ):
            unknown = sorted(set(mapping) - set(tool.request_parameters))
            if unknown:
                raise ValueError(
                    f"{source_name} for {requirement.id!r} include fields outside "
                    f"{tool.name}: {unknown}"
                )
            if mapping:
                tool.validate_parameter_patch(mapping)


def _next_ready_step(
    run: Run, data_root: str | None = None, registry: ToolRegistry | None = None
) -> Step | None:
    for step in run.plan.steps:
        if step.id in run.current_results:
            continue
        if all(
            reference.step_id is None or reference.step_id in run.current_results
            for reference in step.inputs.values()
        ) and all(
            data_root is not None
            and registry is not None
            and _goal_check_requirement_met(
                data_root,
                run,
                requirement,
                registry,
                dependent_step=step,
            )
            for requirement in step.goal_checks
        ):
            return step
    return None


def _goal_check_requirement_met(
    data_root: str,
    run: Run,
    requirement: Any,
    registry: ToolRegistry,
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
        and _result_check_input_is_bound(
            data_root,
            run,
            source_step,
            result,
            requirement.check,
            check,
            registry,
        )
        and (
            dependent_step is None
            or _goal_check_inputs_match(requirement, source_step, dependent_step, registry)
        )
    )


def _goal_check_inputs_match(
    requirement: Any,
    source_step: Step,
    dependent_step: Step,
    registry: ToolRegistry,
) -> bool:
    """Keep a check's declared source input attached to an identical consumer input."""

    source_tool = registry.get(source_step.tool)
    input_name = source_tool.scientific_check_input_ports.get(requirement.check)
    if input_name is None:
        return True
    source_reference = source_step.inputs.get(input_name)
    source_type = source_tool.input_ports.get(input_name)
    dependent_tool = registry.get(dependent_step.tool)
    return source_reference is not None and any(
        declared_type == source_type and dependent_step.inputs.get(name) == source_reference
        for name, declared_type in dependent_tool.input_ports.items()
    )


def _result_check_input_is_bound(
    data_root: str,
    run: Run,
    step: Step,
    result: Result,
    check_name: str,
    check: Any,
    registry: ToolRegistry,
) -> bool:
    """Bind a scientific check to the exact, hash-verified input it declares."""

    tool = registry.get(step.tool)
    input_name = tool.scientific_check_input_ports.get(check_name)
    if input_name is None:
        return check.input_geometry_sha256 is None and not check.input_artifact_sha256_by_port
    input_type = tool.input_ports.get(input_name)
    artifact_id = result.input_bindings.get(input_name)
    if not isinstance(artifact_id, str) or artifact_id not in result.input_artifact_ids:
        return False
    expected_hash = check.input_artifact_sha256_by_port.get(input_name)
    if expected_hash is None and input_type == "molecular_geometry":
        expected_hash = check.input_geometry_sha256
    if not isinstance(expected_hash, str):
        return False
    try:
        artifact = find_artifact(run, artifact_id)
        if (
            artifact.run_id != run.id
            or artifact.artifact_type != input_type
            or artifact.sha256 != expected_hash
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
                and _result_check_input_is_bound(
                    data_root,
                    run,
                    source_step,
                    source_result,
                    requirement.check,
                    check,
                    registry,
                )
                and _goal_check_inputs_match(requirement, source_step, step, registry)
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
            data_root,
            run,
            source_step,
            source_result,
            target.check,
            check,
            registry,
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


def _normalized_run_target(
    run: Run, target: Any, registry: ToolRegistry
) -> tuple[str | None, str | None]:
    """Normalize legacy target spellings against the current Tool directory."""

    if target.check is not None:
        return "check", target.check
    kind = "port" if target.port is not None else "field"
    name = target.port or target.field
    if name is None:
        return None, None
    operations = set(run.request.operations)
    if name == "energy":
        if operations == {"SP"}:
            return "field", "sp_electronic_energy"
        if operations == {"Opt"}:
            return "field", "opt_final_electronic_energy"
    if name in {"geometry", "molecular_geometry"}:
        if "Opt" in operations:
            return "port", "optimized_geometry"
        return "port", "geometry"
    aliases = {
        "sp_energy": ("field", "sp_electronic_energy"),
        "opt_energy": ("field", "opt_final_electronic_energy"),
        "frequency": ("field", "vibrational_frequencies"),
        "frequencies": ("field", "vibrational_frequencies"),
        "optimized_geometry": ("port", "optimized_geometry"),
    }
    if name in aliases:
        return aliases[name]
    if kind == "field":
        # Old Plans sometimes encoded a port in field.  Resolve only if the
        # producer directory proves that this name is a port.
        port_matches = [
            step
            for step in run.plan.steps
            if name in _declared_target_names(registry, step, "port")
        ]
        if port_matches:
            return "port", name
    return kind, name


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
            target_kind, target_name = _normalized_run_target(run, target, registry)
            if target_kind is None or target_name is None:
                return False
            step_id = target.step_id
            if step_id is None and target.requirement_id is not None:
                matches = [
                    step.id
                    for step in run.plan.steps
                    if step.requirement_id == target.requirement_id
                ]
                if len(matches) != 1:
                    return False
                step_id = matches[0]
            elif step_id is None:
                # Old M0 targets were unqualified; the registry ensures this
                # is unique, so the first producer is the only legal binding.
                matches = [
                    step.id
                    for step in run.plan.steps
                    if target_name in _declared_target_names(registry, step, target_kind)
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
            if target_kind == "field":
                tool = registry.get(step.tool)
                expected_type = tool.results.get(target_name)
                if (
                    expected_type is None
                    or target_name in tool.output_ports
                    or target_name not in result.values
                    or not _query_value_is_compatible(result.values[target_name], expected_type)
                ):
                    return False
            if target_kind == "check":
                check = result.scientific_checks.get(target_name)
                if check is None or check.status != "passed":
                    return False
                if not _result_check_input_is_bound(
                    data_root, run, step, result, target_name, check, registry
                ):
                    return False
            if target_kind == "port":
                tool = registry.get(step.tool)
                artifact_id = result.output_ports.get(target_name)
                if (
                    target_name not in tool.output_ports
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
                    or artifact.artifact_type != tool.output_ports[target_name]
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
    """Accept only values covered by a registered public output contract."""

    return is_compatible_value(value, declared_type)


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
