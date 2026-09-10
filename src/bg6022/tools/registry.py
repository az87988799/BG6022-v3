"""Central registry for the tools that are actually implemented."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Plan, Tool
from bg6022.orca.profiles import get_profile
from bg6022.tools.orca import make_optimize_tool, make_single_point_tool

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        tool_list = list(tools)
        self._tools = {tool.name: tool for tool in tool_list}
        if len(self._tools) != len(tool_list):
            raise ValueError("tool names must be unique")

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"tool is not registered: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def describe(self) -> list[dict[str, Any]]:
        return [self._tools[name].description_json() for name in self.names()]

    def validate_plan(self, plan: Plan) -> None:
        if not plan.steps:
            raise ValueError("plan must contain at least one Step")
        step_ids: set[str] = set()
        produced_ports: dict[str, dict[str, str]] = {}
        for step in plan.steps:
            if not _ID_RE.fullmatch(step.id) or step.id in {".", ".."}:
                raise ValueError(f"step id is not a simple domain identifier: {step.id!r}")
            if step.id in step_ids:
                raise ValueError(f"plan step id is duplicated: {step.id}")
            tool = self.get(step.tool)
            try:
                parameters = tool.validate_parameters(step.parameters)
                profile = get_profile(parameters["method_profile"])
                if parameters["environment"] not in profile.supported_environments:
                    raise ValueError(
                        f"environment {parameters['environment']!r} is not implemented "
                        f"for {profile.name!r}"
                    )
            except ValueError as error:
                raise ValueError(f"step {step.id} has invalid parameters: {error}") from error
            unknown_inputs = set(step.inputs) - set(tool.input_ports)
            if unknown_inputs:
                raise ValueError(f"step {step.id} has unknown inputs: {sorted(unknown_inputs)}")
            missing_inputs = set(tool.input_ports) - set(step.inputs)
            if missing_inputs:
                raise ValueError(f"step {step.id} is missing inputs: {sorted(missing_inputs)}")
            for input_name, reference in step.inputs.items():
                if reference.step_id is not None:
                    if reference.step_id not in step_ids:
                        raise ValueError(
                            f"step {step.id} must reference an earlier step, "
                            f"got {reference.step_id}"
                        )
                    upstream_ports = produced_ports[reference.step_id]
                    if reference.port not in upstream_ports:
                        raise ValueError(
                            f"step {step.id} references undeclared output port "
                            f"{reference.step_id}.{reference.port}"
                        )
                    expected_type = tool.input_ports[input_name]
                    if upstream_ports[reference.port] != expected_type:
                        raise ValueError(
                            f"step {step.id} input type does not match "
                            f"{reference.step_id}.{reference.port}"
                        )
            produced_ports[step.id] = dict(tool.output_ports)
            step_ids.add(step.id)


def build_registry(config: AppConfig | None = None) -> ToolRegistry:
    return ToolRegistry([make_single_point_tool(config), make_optimize_tool(config)])


def describe_tools() -> list[dict[str, Any]]:
    """Return descriptions without loading a config or creating any executable closure."""
    return build_registry().describe()


__all__ = ["ToolRegistry", "build_registry", "describe_tools"]
