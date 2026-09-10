"""Shared command-line entry point for package scripts and ``python -m``."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .agent import INPUT_GEOMETRY_PLACEHOLDER, Agent
from .config import load_config, runtime_summary
from .models import InputReference, Plan, Request, Step
from .orca.input import OrcaInputSpec, render_input
from .orca.runner import probe_orca_version
from .session import load_run, new_id
from .tools.molecule import parse_xyz_file, validate_electronic_state
from .tools.registry import build_registry, describe_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bg6022", description="BG6022-v3 ORCA tool foundation")
    parser.add_argument("--config", help="path to the TOML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check local configuration and platform")
    doctor.add_argument("--probe-orca", action="store_true", help="probe the ORCA banner")
    commands.add_parser("tools", help="list registered executable tools")

    run_tool = commands.add_parser("run-tool", help="preview or execute one registered tool")
    run_tool.add_argument("tool", choices=("single_point", "optimize_geometry"))
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
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _doctor(args: argparse.Namespace) -> int:
    config = _require_config(args)
    print(json.dumps(runtime_summary(config), ensure_ascii=True, indent=2))
    print(f"python_311_supported: {sys.version_info[:2] == (3, 11)}")
    print(f"platform: {platform.platform()}")
    if args.probe_orca:
        result = probe_orca_version(config.executable_path)
        print(json.dumps({"orca_probe": result}, ensure_ascii=True, indent=2))
        if not result.get("ok"):
            return 1
    return 0


def _show_run(args: argparse.Namespace) -> int:
    config = _require_config(args)
    run = load_run(config.data_root_path, args.run_id)
    print(json.dumps(run.model_dump(mode="json"), ensure_ascii=True, indent=2))
    return 0


def _run_tool(args: argparse.Namespace) -> int:
    config = _require_config(args)
    geometry = parse_xyz_file(args.xyz)
    method = args.method_profile or config.defaults.method_profile
    environment = args.environment or config.defaults.environment
    validate_electronic_state(geometry, charge=args.charge, multiplicity=args.multiplicity)
    parameters = {
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
    validated = _validate_parameters(tool.parameter_model, parameters)
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
        requested_results=list(tool.results),
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
        requested_results=list(tool.results),
    )
    agent = Agent(config, registry)
    run, result = agent.execute_plan(request, plan, xyz_path=args.xyz)
    payload = {
        "run_id": run.id,
        "status": result.status,
        "values": result.values,
        "checks": result.checks,
        "result_path": str(
            Path(config.data_root_path)
            / "runs"
            / run.id
            / result.attempt_relative_path
            / "result.json"
        ),
        "run_path": str(Path(config.data_root_path) / "runs" / run.id / "run.json"),
        "artifact_ids": result.artifact_ids,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if result.status == "cancelled":
        return 130
    return 0 if result.status == "succeeded" else 1


def _validate_parameters(model_name: str, parameters: dict[str, object]) -> dict[str, object]:
    from .tools.orca import OptimizeParameters, SinglePointParameters

    models = {
        OptimizeParameters.__name__: OptimizeParameters,
        SinglePointParameters.__name__: SinglePointParameters,
    }
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"unknown parameter model: {model_name}")
    return model.model_validate(parameters, strict=True).model_dump(
        mode="python", exclude_none=True
    )


def _require_config(args: argparse.Namespace):
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
