"""Central registry for the tools that are actually implemented."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Plan, ResultTarget, Tool
from bg6022.tools.molecule import make_generate_geometry_tool
from bg6022.tools.orca import make_optimize_tool, make_single_point_tool
from bg6022.tools.pubchem import make_resolve_molecule_tool

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

    def validate_plan(self, plan: Plan) -> Plan:
        """Validate generic Tool contracts and return a dependency-ordered Plan.

        ORCA method/profile checks stay inside the ORCA adapter.  This method
        only knows the declared ports, parameter models, and result metadata of
        each registered Tool.
        """

        if not plan.steps:
            raise ValueError("plan must contain at least one Step")
        step_by_id: dict[str, Any] = {}
        for step in plan.steps:
            if not _ID_RE.fullmatch(step.id) or step.id in {".", ".."}:
                raise ValueError(f"step id is not a simple domain identifier: {step.id!r}")
            if step.id in step_by_id:
                raise ValueError(f"plan step id is duplicated: {step.id}")
            step_by_id[step.id] = step

        dependencies: dict[str, set[str]] = {step.id: set() for step in plan.steps}
        for step in plan.steps:
            tool = self.get(step.tool)
            if not tool.available:
                raise ValueError(f"tool {step.tool!r} is not available in this environment")
            try:
                tool.validate_parameters(step.parameters, allow_deferred=True)
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
                    if reference.step_id == step.id:
                        raise ValueError(
                            f"step {step.id} must reference an earlier step, "
                            f"got {reference.step_id}"
                        )
                    upstream = step_by_id.get(reference.step_id)
                    if upstream is None:
                        raise ValueError(
                            f"step {step.id} references an unknown step {reference.step_id}"
                        )
                    upstream_tool = self.get(upstream.tool)
                    if reference.port not in upstream_tool.output_ports:
                        raise ValueError(
                            f"step {step.id} references undeclared output port "
                            f"{reference.step_id}.{reference.port}"
                        )
                    expected_type = tool.input_ports[input_name]
                    if upstream_tool.output_ports[reference.port] != expected_type:
                        raise ValueError(
                            f"step {step.id} input type does not match "
                            f"{reference.step_id}.{reference.port}"
                        )
                    dependencies[step.id].add(reference.step_id)

        ordered = _topological_order(plan.steps, dependencies)
        plan = _normalize_legacy_targets(plan, ordered, self)
        _validate_requested_results(plan.requested_results, ordered, self)
        if [step.id for step in ordered] == [step.id for step in plan.steps]:
            return plan
        return plan.model_copy(update={"steps": ordered})


def build_registry(config: AppConfig | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            make_resolve_molecule_tool(config),
            make_generate_geometry_tool(config),
            make_single_point_tool(config),
            make_optimize_tool(config),
        ]
    )


def describe_tools() -> list[dict[str, Any]]:
    """Return descriptions without loading a config or creating any executable closure."""
    return build_registry().describe()


__all__ = ["ToolRegistry", "build_registry", "describe_tools"]


def _topological_order(steps: list[Any], dependencies: dict[str, set[str]]) -> list[Any]:
    remaining = {step.id: step for step in steps}
    ordered: list[Any] = []
    while remaining:
        ready = [
            step
            for step in steps
            if step.id in remaining and not (dependencies[step.id] & remaining.keys())
        ]
        if not ready:
            raise ValueError("plan dependency graph contains a cycle")
        for step in ready:
            ordered.append(step)
            remaining.pop(step.id)
    return ordered


def _validate_requested_results(
    targets: list[ResultTarget], steps: list[Any], registry: ToolRegistry
) -> None:
    producers: dict[tuple[str, str], list[str]] = {}
    for step in steps:
        tool = registry.get(step.tool)
        for field in tool.results:
            if field not in tool.output_ports:
                producers.setdefault(("field", field), []).append(step.id)
        for port in tool.output_ports:
            producers.setdefault(("port", port), []).append(step.id)
    for target in targets:
        kind = "port" if target.port is not None else "field"
        name = target.port or target.field
        assert name is not None
        matches = producers.get((kind, name), [])
        if target.step_id is not None:
            if target.step_id not in {step.id for step in steps}:
                raise ValueError(f"requested result references unknown step {target.step_id}")
            matches = [step_id for step_id in matches if step_id == target.step_id]
        if not matches:
            raise ValueError(
                f"requested result has no producer: {target.step_id or '*'}:{kind}:{name}"
            )
        if target.step_id is None and len(matches) > 1:
            raise ValueError(f"requested result is ambiguous without step_id: {kind}:{name}")


def _normalize_legacy_targets(plan: Plan, ordered: list[Any], registry: ToolRegistry) -> Plan:
    """Convert old string-style output-port targets using Tool metadata only."""

    output_port_names = {port for step in ordered for port in registry.get(step.tool).output_ports}
    targets = []
    changed = False
    for target in plan.requested_results:
        if target.field in output_port_names:
            targets.append(target.model_copy(update={"field": None, "port": target.field}))
            changed = True
        else:
            targets.append(target)
    return plan.model_copy(update={"requested_results": targets}) if changed else plan
