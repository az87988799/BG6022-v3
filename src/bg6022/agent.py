"""The single Plan -> Tool -> Result loop used by CLI and chat."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from .answer import context_answer, render_result, render_run
from .config import AppConfig, validate_execution_environment
from .llm import LlmClient, LlmError
from .models import InputReference, Plan, Request, Result, Run, Step
from .orca.profiles import get_profile, resolve_parameters
from .orca.repair_rules import applicable_repairs, applicable_scf_repair
from .planner import intake_message, plan_message, proposal_to_plan, request_from_intake
from .repair import apply_repair_proposal, propose_repair
from .session import (
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
from .tools.molecule import parse_xyz_bytes
from .tools.registry import ToolRegistry

INPUT_GEOMETRY_PLACEHOLDER = "__input_geometry__"


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
        plan = self.registry.validate_plan(plan)
        # The explicit command has always checked the machine before creating a
        # Run. Chat intentionally defers this check until an ORCA Tool starts.
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
        run.plan = _bind_input_geometry(plan, initial_artifact.id)
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

        text = message.strip()
        if not text:
            return AgentResponse("Please enter a request.")
        if text.casefold() in {"/confirm", "confirm", "确认"}:
            return self.confirm()
        if text.casefold() in {"/cancel", "cancel", "取消"}:
            return self.cancel()
        if text.casefold() in {"/status", "status", "状态"}:
            return self.status()
        if text.casefold() in {"/new", "new", "新任务"}:
            return self.new_session()

        self._append_message("user", text)
        try:
            intake = intake_message(
                self.llm,
                text,
                context={
                    "recent_messages": self._session.get("recent_messages", []),
                    "recent_results": self._session.get("recent_results", []),
                },
            )
        except (LlmError, ValueError) as error:
            response = AgentResponse(f"I could not interpret this request: {error}")
            self._append_message("assistant", response.text)
            return response

        current = self._coerce_run(None)
        if (
            current is not None
            and current.status == "waiting"
            and current.waiting_for in {"clarification", "confirmation"}
            and intake.explicit_parameters
        ):
            response = self._apply_parameter_update(current, intake.explicit_parameters)
            self._append_message("assistant", response.text)
            return response
        if (
            current is not None
            and current.status == "waiting"
            and current.waiting_for == "clarification"
            and current.pending_data.get("category") == "ambiguous_molecule"
            and intake.molecule_query
        ):
            response = self._apply_molecule_clarification(
                current, intake.molecule_query, intake.molecule_input_kind
            )
            self._append_message("assistant", response.text)
            return response

        if intake.intent in {"chemistry_qa", "daily_qa"}:
            return self._answer_question(text)
        if intake.intent == "context_query":
            return self._answer_context(text)
        try:
            request = request_from_intake(text, intake, request_id=new_id("request"))
            proposal = plan_message(
                self.llm,
                request,
                registry=self.registry,
                context={
                    "recent_messages": self._session.get("recent_messages", []),
                    "recent_results": self._session.get("recent_results", []),
                },
            )
            plan = proposal_to_plan(request, proposal, self.registry, plan_id=new_id("plan"))
            request = request.model_copy(update={"requested_results": plan.requested_results})
            run = self._create_chat_run(request, plan)
            self._session["active_run_id"] = run.id
            self._save_session()
            result = self.advance(run)
            if result is not None:
                response = AgentResponse(render_result(run, result), run=run, result=result)
            elif run.status == "waiting":
                response = AgentResponse(self._waiting_text(run), run=run)
            else:
                response = AgentResponse(render_run(run), run=run)
            self._append_message("assistant", response.text)
            return response
        except (LlmError, ValueError, OSError) as error:
            response = AgentResponse(f"The request could not be planned: {error}")
            self._append_message("assistant", response.text)
            return response

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
                step = _next_ready_step(run)
                if step is None:
                    run.status = "failed"
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                tool = self.registry.get(step.tool)
                if tool.requires_compute_permission and not self._prepare_orca_step(run, step):
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
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
                if not self._reserve_attempt(run, step):
                    run.status = "failed"
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    return last_result
                run.step_status[step.id] = "running"
                save_run(self.config.data_root_path, run)
                try:
                    result = tool.execute(step, run, cancel=cancel_event)
                except (PermissionError, ValueError, OSError):
                    run.status = "failed"
                    run.finish_active_interval()
                    save_run(self.config.data_root_path, run)
                    raise
                last_result = result
                result.step_fingerprint = _step_fingerprint(step)
                if run.pending_data.get("parameter_sources"):
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
            return AgentResponse("There is no prepared calculation to confirm.")
        if current.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            result = self._latest_result(current)
            return AgentResponse(render_run(current, result), run=current, result=result)
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
            return AgentResponse(
                f"The calculation was refused at the ORCA execution boundary: {error}",
                run=current,
            )
        if result is not None:
            response = AgentResponse(render_result(current, result), run=current, result=result)
        elif current.status == "waiting":
            response = AgentResponse(self._waiting_text(current), run=current)
        else:
            response = AgentResponse(render_run(current), run=current)
        self._append_message("assistant", response.text)
        return response

    def cancel(self, run: Run | str | None = None) -> AgentResponse:
        current = self._coerce_run(run)
        if current is None:
            return AgentResponse("There is no active calculation.")
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
        return AgentResponse(f"Cancellation requested for {current.id}.", run=current)

    def request_cancel(self) -> AgentResponse:
        """Signal cancellation without loading/saving a Run owned by a worker."""

        run_id = self._session.get("active_run_id")
        if not run_id:
            return AgentResponse("There is no active calculation.")
        event = self._cancel_events.setdefault(str(run_id), Event())
        event.set()
        return AgentResponse(f"Cancellation requested for {run_id}.")

    def status(self) -> AgentResponse:
        current = self._coerce_run(None)
        if current is None:
            return AgentResponse("No active Run.")
        result = self._latest_result(current)
        return AgentResponse(render_run(current, result), run=current, result=result)

    def new_session(self) -> AgentResponse:
        current = self._coerce_run(None)
        if current is not None and current.status in {"running", "waiting"}:
            self.cancel(current)
        self.session_id = new_id("session")
        self._session = {
            "session_id": self.session_id,
            "recent_messages": [],
            "recent_results": [],
            "active_run_id": None,
            "pending_prompt": None,
        }
        self._save_session()
        return AgentResponse("Started a new session.")

    def _create_chat_run(self, request: Request, plan: Plan) -> Run:
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
        self._seed_structure_input(run)
        save_run(self.config.data_root_path, run)
        return run

    def _seed_structure_input(self, run: Run) -> None:
        value = run.request.structure_input
        xyz_text = value.get("xyz_text") or value.get("xyz") if isinstance(value, dict) else None
        if xyz_text is None:
            return
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
        steps = []
        for step in run.plan.steps:
            inputs = {}
            for name, reference in step.inputs.items():
                if reference.artifact_id in {"request_geometry", INPUT_GEOMETRY_PLACEHOLDER}:
                    inputs[name] = InputReference(artifact_id=artifact.id)
                else:
                    inputs[name] = reference
            steps.append(step.model_copy(update={"inputs": inputs}))
        run.plan = Plan.model_validate(
            {**run.plan.model_dump(mode="python"), "steps": steps}, strict=True
        )

    def _prepare_orca_step(self, run: Run, step: Step) -> bool:
        if (
            run.accepted_snapshot
            and "charge" in step.parameters
            and "multiplicity" in step.parameters
        ):
            validated = self.registry.get(step.tool).validate_parameters(step.parameters)
            _validate_orca_profile(validated)
            return True
        if (
            run.pending_data.get("step_id") == step.id
            and run.pending_data.get("parameters") == step.parameters
            and run.pending_data.get("parameter_sources")
        ):
            validated = self.registry.get(step.tool).validate_parameters(step.parameters)
            _validate_orca_profile(validated)
            return True
        facts = self._known_structure_facts(run)
        resolution = resolve_parameters(
            run.request.explicit_parameters,
            facts,
            step.parameters,
            self.config.defaults,
            user_modifications=run.request.user_modifications,
        )
        if resolution.missing_fields:
            run.status = "waiting"
            run.waiting_for = "clarification"
            run.pending_data = {
                "question": "Please provide the missing electronic state parameters.",
                "step_id": step.id,
                "missing_fields": list(resolution.missing_fields),
                "parameter_sources": resolution.parameter_sources,
            }
            return False
        validated = self.registry.get(step.tool).validate_parameters(
            resolution.effective_parameters
        )
        _validate_orca_profile(validated)
        replacement = Step.model_validate(
            {**step.model_dump(mode="python"), "parameters": validated}, strict=True
        )
        run.plan = _replace_step(run.plan, replacement)
        run.pending_data = {
            "step_id": step.id,
            "parameter_sources": resolution.parameter_sources,
            "effective_parameters": validated,
        }
        return True

    def _apply_parameter_update(self, run: Run, parameters: dict[str, Any]) -> AgentResponse:
        step_id = str(run.pending_data.get("step_id", ""))
        step = next((item for item in run.plan.steps if item.id == step_id), None)
        if step is None:
            step = next(
                (
                    item
                    for item in run.plan.steps
                    if item.tool in {"single_point", "optimize_geometry"}
                ),
                None,
            )
        if step is None:
            return AgentResponse("The pending Run has no editable calculation step.", run=run)
        merged = dict(step.parameters)
        merged.update(parameters)
        try:
            self.registry.get(step.tool).validate_parameters(merged)
            replacement = Step.model_validate(
                {**step.model_dump(mode="python"), "parameters": merged}, strict=True
            )
            run.request = run.request.model_copy(
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
            run.plan = _replace_step(run.plan, replacement)
            run.plan = Plan.model_validate(
                {**run.plan.model_dump(mode="python"), "revision": run.plan.revision + 1},
                strict=True,
            )
            run.plan = self.registry.validate_plan(run.plan)
            run.execution_permission = not self.config.runtime.confirm_before_compute
            run.accepted_snapshot = {}
            run.accepted_execution_sha256 = None
            _invalidate_current_results(run, step.id)
            run.waiting_for = None
            run.status = "running"
            run.pending_data = {}
            save_run(self.config.data_root_path, run)
            result = self.advance(run)
            if result is not None:
                return AgentResponse(render_result(run, result), run=run, result=result)
            return AgentResponse(self._waiting_text(run), run=run)
        except ValueError as error:
            return AgentResponse(f"That parameter change was rejected: {error}", run=run)

    def _apply_molecule_clarification(
        self,
        run: Run,
        query: str,
        input_kind: str | None,
    ) -> AgentResponse:
        step_id = str(run.pending_data.get("step_id", ""))
        step = next((item for item in run.plan.steps if item.id == step_id), None)
        if step is None or step.tool != "resolve_molecule":
            return AgentResponse("The pending Run has no molecule-resolution step.", run=run)
        kind = input_kind or ("cid" if query.isdecimal() else "name")
        replacement = Step.model_validate(
            {
                **step.model_dump(mode="python"),
                "parameters": {"query": query, "input_kind": kind},
            },
            strict=True,
        )
        try:
            self.registry.get(step.tool).validate_parameters(replacement.parameters)
            run.plan = _replace_step(run.plan, replacement)
            run.plan = Plan.model_validate(
                {**run.plan.model_dump(mode="python"), "revision": run.plan.revision + 1},
                strict=True,
            )
            run.plan = self.registry.validate_plan(run.plan)
        except ValueError as error:
            return AgentResponse(f"That molecule choice was rejected: {error}", run=run)
        run.status = "running"
        run.waiting_for = None
        run.pending_data = {}
        save_run(self.config.data_root_path, run)
        result = self.advance(run)
        if result is not None:
            return AgentResponse(render_result(run, result), run=run, result=result)
        return AgentResponse(self._waiting_text(run), run=run)

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
        if len(run.repair_records) >= self.config.repair.max_extra_orca_executions:
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

    def _reserve_attempt(self, run: Run, step: Step) -> bool:
        if step.tool not in {"single_point", "optimize_geometry"}:
            return True
        origin = run.origin_step_map.get(step.id, step.origin_step_id or step.id)
        count = int(run.attempt_counts.get(origin, 0))
        if count >= int(run.budget.get("max_attempts_per_science_step", 3)):
            run.pending_data = {
                "budget_exhausted": "max_attempts_per_science_step",
                "step_id": step.id,
            }
            return False
        if count > 0 and run.extra_orca_executions >= int(
            run.budget.get("max_extra_orca_executions", 3)
        ):
            run.pending_data = {
                "budget_exhausted": "max_extra_orca_executions",
                "step_id": step.id,
            }
            return False
        run.attempt_counts[origin] = count + 1
        if count > 0:
            run.extra_orca_executions += 1
        return True

    def _prepare_confirmation(self, run: Run, step: Step) -> None:
        run.status = "waiting"
        run.waiting_for = "confirmation"
        run.pending_data = self._preview(run, step)

    def _preview(self, run: Run, step: Step) -> dict[str, Any]:
        artifacts = self._input_artifacts(run, step)
        structure = self._known_structure_facts(run)
        return {
            "operation": "Opt" if step.tool == "optimize_geometry" else "SP",
            "step_id": step.id,
            "tool": step.tool,
            "parameters": dict(step.parameters),
            "parameter_sources": run.pending_data.get("parameter_sources", {}),
            "structure": structure,
            "artifact_bindings": [
                {"id": artifact.id, "sha256": artifact.sha256, "role": artifact.role}
                for artifact in artifacts
            ],
            "resources": dict(run.resources),
            "allowed_repairs": [
                "restart_optimization",
                "increase_scf_maxiter (only if separately verified)",
            ],
            "budget": dict(run.budget),
        }

    def _acceptance_snapshot(self, run: Run) -> dict[str, Any]:
        bound: list[dict[str, Any]] = []
        for step in run.plan.steps:
            for artifact in self._input_artifacts(run, step):
                item = {"step_id": step.id, "id": artifact.id, "sha256": artifact.sha256}
                if item not in bound:
                    bound.append(item)
        return {
            "request": run.request.model_dump(mode="json"),
            "plan": run.plan.model_dump(mode="json"),
            "resources": dict(run.resources),
            "bound_inputs": bound,
            "allowed_repairs": ["restart_optimization", "increase_scf_maxiter"],
            "budgets": dict(run.budget),
        }

    def _known_structure_facts(self, run: Run) -> dict[str, Any]:
        for artifact in reversed(run.artifact_index):
            if artifact.artifact_type != "molecule" or artifact.role != "resolved_molecule":
                continue
            try:
                payload = json.loads(
                    artifact_path(self.config.data_root_path, run, artifact).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            facts = payload.get("facts")
            if isinstance(facts, dict):
                return facts
        return {}

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
        try:
            artifact_id = result.output_ports.get(reference.port)
            return find_artifact(run, artifact_id) if artifact_id else None
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

    def _waiting_text(self, run: Run) -> str:
        if run.waiting_for == "confirmation":
            preview = run.pending_data
            return (
                "Prepared a calculation and need your confirmation: "
                f"{preview.get('operation', 'calculation')} "
                f"with {preview.get('parameters', {})}. Type /confirm to run it."
            )
        return "More information is required: " + json.dumps(run.pending_data, ensure_ascii=False)

    def _append_message(self, role: str, content: str) -> None:
        messages = self._session.setdefault("recent_messages", [])
        messages.append({"role": role, "content": content})
        self._session["recent_messages"] = messages[-12:]
        self._save_session()

    def _record_result_summary(self, run: Run, result: Result) -> None:
        summaries = self._session.setdefault("recent_results", [])
        summaries.append(
            {
                "run_id": run.id,
                "step_id": result.step_id,
                "status": result.status,
                "values": result.values,
            }
        )
        self._session["recent_results"] = summaries[-3:]
        self._session["active_run_id"] = run.id
        self._save_session()

    def _save_session(self) -> None:
        try:
            save_session(self.config.data_root_path, self.session_id, self._session)
        except OSError:
            pass

    def _answer_question(self, question: str) -> AgentResponse:
        try:
            text = self.llm.complete_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer the question without claiming an unperformed computation."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                purpose="answer",
            )
        except LlmError as error:
            text = f"I cannot answer through the configured model: {error}"
        response = AgentResponse(text)
        self._append_message("assistant", text)
        return response

    def _answer_context(self, question: str) -> AgentResponse:
        run = self._coerce_run(None)
        result = self._context_result(run) if run is not None else None
        text = (
            context_answer(run, result, question)
            if run is not None
            else "No saved calculation result is available."
        )
        response = AgentResponse(text, run=run, result=result)
        self._append_message("assistant", text)
        return response

    def _context_result(self, run: Run) -> Result | None:
        for step in reversed(run.plan.steps):
            relative = run.current_results.get(step.id)
            if not relative:
                continue
            result = _load_bound_result(self.config.data_root_path, run, relative)
            if result is not None and result.status == "succeeded" and result.values:
                return result
        return self._latest_result(run)


def _bind_input_geometry(plan: Plan, artifact_id: str) -> Plan:
    steps: list[Step] = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        reference = inputs.get("geometry")
        if reference is not None and reference.artifact_id == INPUT_GEOMETRY_PLACEHOLDER:
            inputs["geometry"] = InputReference(artifact_id=artifact_id)
        steps.append(step.model_copy(update={"inputs": inputs}))
    return plan.model_copy(update={"steps": steps})


def _replace_step(plan: Plan, replacement: Step) -> Plan:
    return Plan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "steps": [replacement if step.id == replacement.id else step for step in plan.steps],
        },
        strict=True,
    )


def _next_ready_step(run: Run) -> Step | None:
    for step in run.plan.steps:
        if step.id in run.current_results:
            continue
        if all(
            reference.step_id is None or reference.step_id in run.current_results
            for reference in step.inputs.values()
        ):
            return step
    return None


def _invalidate_current_results(run: Run, changed_step_id: str) -> None:
    """Drop current outputs for a changed step and every downstream consumer."""

    invalidated = {changed_step_id}
    changed = True
    while changed:
        changed = False
        for step in run.plan.steps:
            if step.id in invalidated:
                continue
            if any(reference.step_id in invalidated for reference in step.inputs.values()):
                invalidated.add(step.id)
                changed = True
    for step_id in invalidated:
        run.current_results.pop(step_id, None)
        run.step_status[step_id] = "planned"


def _requested_results_satisfied(data_root: str, run: Run, registry: ToolRegistry) -> bool:
    if run.plan.requested_results:
        for target in run.plan.requested_results:
            step_id = target.step_id
            if step_id is None:
                # Old M0 targets were unqualified; the registry ensures this
                # is unique, so the first producer is the only legal binding.
                kind = "port" if target.port is not None else "field"
                name = target.port or target.field
                matches = [
                    step.id
                    for step in run.plan.steps
                    if name is not None
                    and (
                        name in registry.get(step.tool).output_ports
                        if kind == "port"
                        else name in registry.get(step.tool).results
                        and name not in registry.get(step.tool).output_ports
                    )
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
