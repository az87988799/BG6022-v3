"""Shared command-line entry point for package scripts and ``python -m``."""

from __future__ import annotations

import argparse
import json
import platform
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .agent import INPUT_GEOMETRY_PLACEHOLDER, Agent, AgentResponse
from .config import AppConfig, load_config, runtime_summary, validate_execution_environment
from .diagnostics import configure_rdkit_logging
from .models import InputReference, Plan, Request, Result, ResultTarget, Step
from .orca.input import OrcaInputSpec, render_input
from .orca.runner import probe_orca_version
from .session import ChatSessionLock, load_run, new_id, read_execution_guard, run_directory
from .tools.molecule import parse_xyz_file, validate_electronic_state
from .tools.registry import build_registry, describe_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bg6022", description="BG6022-v3 ORCA agent")
    parser.add_argument("--config", help="path to the TOML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check local configuration and platform")
    doctor.add_argument("--probe-orca", action="store_true", help="probe the ORCA banner")
    commands.add_parser("tools", help="list registered executable tools")

    commands.add_parser("chat", help="start the local natural-language agent")

    run_tool = commands.add_parser("run-tool", help="preview or execute one registered tool")
    run_tool.add_argument(
        "tool",
        choices=tuple(
            item["name"]
            for item in describe_tools()
            if item.get("requires_compute_permission") is True
        ),
    )
    run_tool.add_argument("--xyz", required=True, help="input XYZ file")
    run_tool.add_argument("--charge", required=True, type=_strict_int)
    run_tool.add_argument("--multiplicity", required=True, type=_strict_int)
    run_tool.add_argument("--method-profile", default=None)
    run_tool.add_argument("--environment", default=None)
    run_tool.add_argument("--scf-maxiter", type=_strict_int, default=None)
    run_tool.add_argument("--geom-maxiter", type=_strict_int, default=None)
    run_tool.add_argument("--execute", action="store_true", help="grant execution permission")

    show_run = commands.add_parser(
        "show-run", help="read one durable Run without starting a process"
    )
    show_run.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "tools":
            print(json.dumps(describe_tools(), ensure_ascii=True, indent=2))
            return 0
        if args.command == "chat":
            return _chat(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "show-run":
            return _show_run(args)
        if args.command == "run-tool":
            return _run_tool(args)
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("cancelled by user", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, PermissionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        if (
            args.command == "run-tool"
            and args.execute
            and isinstance(error, OSError)
            and not isinstance(error, FileNotFoundError)
        ):
            return 1
        return 2
    return 2


def _doctor(args: argparse.Namespace) -> int:
    config = _require_config(args)
    payload = runtime_summary(config)
    payload["python_311_supported"] = sys.version_info[:2] == (3, 11)
    payload["platform"] = platform.platform()
    try:
        validate_execution_environment(config)
        payload["execution_environment"] = {"ok": True}
    except (ValueError, OSError) as error:
        payload["execution_environment"] = {"ok": False, "reason": str(error)}
    try:
        guard = read_execution_guard(config.data_root_path)
        payload["execution_guard"] = guard
    except RuntimeError as error:
        payload["execution_guard"] = {"ok": False, "reason": str(error)}
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if args.probe_orca:
        result = probe_orca_version(
            config.executable_path,
            data_root=config.data_root_path,
        )
        print(json.dumps({"orca_probe": result}, ensure_ascii=True, indent=2))
        if not result.get("ok"):
            return 1
    return 0 if payload["execution_environment"]["ok"] else 1


def _show_run(args: argparse.Namespace) -> int:
    config = _require_config(args)
    run = load_run(config.data_root_path, args.run_id)
    payload = run.model_dump(mode="json")
    results: list[dict[str, object]] = []
    had_errors = False
    root = run_directory(config.data_root_path, run.id).resolve()
    for relative in run.result_index:
        candidate = (root / relative).resolve()
        if root not in candidate.parents or candidate.name != "result.json":
            results.append({"path": relative, "error": "result path escapes the Run directory"})
            had_errors = True
            continue
        try:
            result = Result.model_validate(
                json.loads(candidate.read_text(encoding="utf-8")), strict=True
            )
            results.append({"path": relative, "result": result.model_dump(mode="json")})
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as error:
            results.append({"path": relative, "error": str(error)})
            had_errors = True
    payload["results"] = results
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 1 if had_errors else 0


def _run_tool(args: argparse.Namespace) -> int:
    config = _require_config(args)
    configure_rdkit_logging(config.data_root_path)
    geometry = parse_xyz_file(args.xyz)
    method = args.method_profile or config.defaults.method_profile
    environment = args.environment or config.defaults.environment
    validate_electronic_state(geometry, charge=args.charge, multiplicity=args.multiplicity)
    parameters: dict[str, object] = {
        "method_profile": method,
        "environment": environment,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
    }
    if args.scf_maxiter is not None:
        parameters["scf_maxiter"] = args.scf_maxiter
    if args.tool == "optimize_geometry" and args.geom_maxiter is not None:
        parameters["geom_maxiter"] = args.geom_maxiter
    if args.tool == "single_point" and args.geom_maxiter is not None:
        raise ValueError("--geom-maxiter is only valid for optimize_geometry")

    registry = build_registry(config)
    tool = registry.get(args.tool)
    validated = tool.validate_parameters(parameters)
    spec = OrcaInputSpec(
        operation="SP" if args.tool == "single_point" else "Opt",
        method_profile=validated["method_profile"],
        environment=validated["environment"],
        charge=validated["charge"],
        multiplicity=validated["multiplicity"],
        cores=config.runtime.cores,
        maxcore_mb=config.runtime.maxcore_mb,
        scf_maxiter=validated.get("scf_maxiter"),
        geom_maxiter=validated.get("geom_maxiter"),
    )
    rendered = render_input(spec).decode("ascii")
    if not args.execute:
        print("preview: no calculation executed")
        print(
            json.dumps(
                {
                    "tool": args.tool,
                    "geometry_atoms": geometry.atom_count,
                    **validated,
                    **config.resources,
                },
                indent=2,
            )
        )
        print("input.inp:")
        print(rendered, end="")
        return 0

    request = Request(
        id=new_id("request"),
        description=f"explicit CLI execution of {args.tool}",
        operations=list(tool.operations),
        requested_results=_result_targets(tool),
        explicit_parameters=validated,
    )
    plan = Plan(
        id=new_id("plan"),
        request_id=request.id,
        steps=[
            Step(
                id="compute",
                tool=args.tool,
                parameters=validated,
                inputs={"geometry": InputReference(artifact_id=INPUT_GEOMETRY_PLACEHOLDER)},
            )
        ],
        requested_results=_result_targets(tool),
    )
    agent = Agent(config, registry)
    run, result = agent.execute_plan(
        request,
        plan,
        xyz_path=args.xyz,
        execute=True,
    )
    in_memory_status = run.status
    checkpoint: dict[str, object] = {"status": "verified"}
    try:
        authoritative = load_run(config.data_root_path, run.id)
        if authoritative.status != in_memory_status:
            checkpoint = {
                "status": "mismatch",
                "authoritative_status": authoritative.status,
                "in_memory_status": in_memory_status,
            }
    except (OSError, ValueError) as error:
        checkpoint = {
            "status": "not_readable",
            "exception_type": type(error).__name__,
            "reason": str(error),
        }
    payload = {
        "run_id": run.id,
        "status": in_memory_status,
        "result_status": result.status,
        "values": result.values,
        "checks": result.checks,
        "diagnostics": {
            "category": result.diagnostics.get("category"),
            "reason": result.diagnostics.get("reason"),
            "raw_paths": result.diagnostics.get("raw_paths", {}),
        },
        "result_path": str(
            Path(config.data_root_path)
            / "runs"
            / run.id
            / result.attempt_relative_path
            / "result.json"
        ),
        "run_path": str(Path(config.data_root_path) / "runs" / run.id / "run.json"),
        "artifact_ids": result.artifact_ids,
        "run_checkpoint": checkpoint,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return _result_exit_code(result, run_status=in_memory_status)


def _result_exit_code(result: Result, *, run_status: str | None = None) -> int:
    category = result.diagnostics.get("category")
    if result.status == "cancelled" or category == "cancelled":
        return 130
    if category == "timeout":
        return 124
    if run_status in {"failed", "interrupted", "cancelled"}:
        return 130 if run_status == "cancelled" else 1
    return 0 if result.status == "succeeded" and run_status in {None, "succeeded"} else 1


def _result_targets(tool) -> list[ResultTarget]:
    return [
        ResultTarget(port=name) if name in tool.output_ports else ResultTarget(field=name)
        for name in tool.results
    ]


def _chat(args: argparse.Namespace) -> int:
    config = _require_config(args)
    from .tools.registry import build_registry

    try:
        with ChatSessionLock(config.data_root_path):
            configure_rdkit_logging(config.data_root_path)
            agent = Agent(config, build_registry(config))
            return _chat_loop(agent)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


@dataclass(frozen=True)
class _QueuedMessage:
    text: str
    confirmation_run_id: str | None = None
    confirmation_fingerprint: str | None = None
    confirmation_queued: bool = False


class _DisplayedConfirmation:
    """Snapshot only the confirmation preview that the display loop showed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: tuple[str, str] | None = None

    def observe(self, agent: Agent, response: AgentResponse) -> None:
        run = response.run
        token = (
            agent.confirmation_token(run)
            if run is not None and run.waiting_for == "confirmation"
            else None
        )
        with self._lock:
            self._token = token

    def snapshot(self) -> tuple[str, str] | None:
        with self._lock:
            return self._token


def _chat_loop(agent: Agent) -> int:
    """Keep input, display, and the one Agent worker separate."""

    incoming: queue.Queue[_QueuedMessage | None] = queue.Queue()
    outgoing: queue.Queue[AgentResponse] = queue.Queue()
    stop_input = threading.Event()
    displayed_confirmation = _DisplayedConfirmation()

    def read_input() -> None:
        while not stop_input.is_set():
            try:
                message = input("> ")
                normalized = message.casefold()
                # Signal the Event from the input side immediately.  The
                # worker still owns durable Run mutation and will process the
                # queued command after its current call returns.
                if normalized in {
                    "/cancel",
                    "cancel",
                    "取消",
                    "/exit",
                    "exit",
                    "退出",
                    "/new",
                    "new",
                    "新任务",
                }:
                    agent.request_cancel()
                if normalized in {"/confirm", "confirm", "确认"}:
                    token = displayed_confirmation.snapshot()
                    incoming.put(
                        _QueuedMessage(
                            message,
                            confirmation_run_id=token[0] if token else None,
                            confirmation_fingerprint=token[1] if token else None,
                            confirmation_queued=True,
                        )
                    )
                else:
                    incoming.put(_QueuedMessage(message))
            except EOFError:
                agent.request_cancel()
                incoming.put(None)
                return
            except KeyboardInterrupt:
                agent.request_cancel()
                incoming.put(_QueuedMessage("/exit"))
                return

    def work() -> None:
        while True:
            item = incoming.get()
            if item is None:
                agent.request_cancel()
                return
            message = item.text
            try:
                if message.casefold() in {"/exit", "exit", "退出"}:
                    agent.request_cancel()
                    outgoing.put(AgentResponse("Exiting after active work is cleaned up."))
                    return
                if message.casefold() in {"/cancel", "cancel", "取消"}:
                    outgoing.put(agent.cancel())
                    continue
                outgoing.put(
                    agent.handle_message(
                        message,
                        confirmation_run_id=item.confirmation_run_id,
                        confirmation_fingerprint=item.confirmation_fingerprint,
                        confirmation_queued=item.confirmation_queued,
                    )
                )
            except Exception as error:  # noqa: BLE001 - keep the chat worker alive per request
                outgoing.put(AgentResponse(f"This request failed safely: {error}"))

    reader = threading.Thread(target=read_input, name="bg6022-chat-input", daemon=True)
    worker = threading.Thread(target=work, name="bg6022-agent-worker", daemon=True)
    reader.start()
    worker.start()
    print("BG6022-v3 chat. Use /confirm, /status, /cancel, /new, or /exit.")
    exiting = False
    try:
        while worker.is_alive() or reader.is_alive() or not outgoing.empty():
            try:
                response = outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            print(response.text)
            displayed_confirmation.observe(agent, response)
            if response.text.startswith("Exiting after"):
                exiting = True
                stop_input.set()
                break
    finally:
        if exiting or not worker.is_alive():
            stop_input.set()
        if worker.is_alive():
            agent.cancel()
            worker.join(timeout=5)
    return 0


def _require_config(args: argparse.Namespace) -> AppConfig:
    if not args.config:
        raise ValueError("--config is required for this command")
    return load_config(args.config)


def _strict_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if str(result) != value and not (value.startswith("+") and str(result) == value[1:]):
        raise argparse.ArgumentTypeError("must be a base-10 integer")
    return result


__all__ = ["build_parser", "main"]
