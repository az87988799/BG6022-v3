"""Central registry for the tools that are actually implemented."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Plan, Tool
from bg6022.tools.orca import make_optimize_tool, make_single_point_tool


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
        step_ids: set[str] = set()
        produced_ports: dict[str, set[str]] = {}
        for step in plan.steps:
            if step.id in step_ids:
                raise ValueError(f"plan step id is duplicated: {step.id}")
            step_ids.add(step.id)
            tool = self.get(step.tool)
            unknown_inputs = set(step.inputs) - set(tool.input_ports)
            if unknown_inputs:
                raise ValueError(f"step {step.id} has unknown inputs: {sorted(unknown_inputs)}")
            for reference in step.inputs.values():
                if reference.step_id is not None:
                    if reference.step_id not in step_ids:
                        raise ValueError(
                            f"step {step.id} must reference an earlier step, "
                            f"got {reference.step_id}"
                        )
                    if reference.port not in produced_ports[reference.step_id]:
                        raise ValueError(
                            f"step {step.id} references undeclared output port "
                            f"{reference.step_id}.{reference.port}"
                        )
            produced_ports[step.id] = set(tool.output_ports)
        if not plan.steps:
            raise ValueError("plan must contain at least one Step")


def build_registry(config: AppConfig) -> ToolRegistry:
    return ToolRegistry([make_single_point_tool(config), make_optimize_tool(config)])


def describe_tools() -> list[dict[str, Any]]:
    """Return descriptions without loading a config or creating any executable closure."""

    from bg6022.models import Tool
    from bg6022.tools.orca import OptimizeParameters, SinglePointParameters

    descriptions = [
        Tool(
            name="single_point",
            description=(
                "Run an independent ORCA single-point calculation on a registered geometry."
            ),
            parameter_model=SinglePointParameters.__name__,
            parameter_schema=SinglePointParameters.model_json_schema(),
            input_ports={"geometry": "molecular_geometry"},
            results={"sp_electronic_energy": "Eh"},
            success_conditions=["normal ORCA termination", "SCF convergence", "finite energy"],
            repair_capabilities=["failed facts retained for M1"],
        ),
        Tool(
            name="optimize_geometry",
            description=(
                "Optimize a registered geometry and return its converged final geometry and energy."
            ),
            parameter_model=OptimizeParameters.__name__,
            parameter_schema=OptimizeParameters.model_json_schema(),
            input_ports={"geometry": "molecular_geometry"},
            output_ports={"optimized_geometry": "molecular_geometry"},
            results={
                "opt_final_electronic_energy": "Eh",
                "optimized_geometry": "molecular_geometry",
            },
            success_conditions=[
                "normal ORCA termination",
                "SCF convergence",
                "Opt convergence",
                "geometry binding",
            ],
            repair_capabilities=["failed Opt may publish restart_candidate for M1"],
        ),
    ]
    return [item.description_json() for item in descriptions]


__all__ = ["ToolRegistry", "build_registry", "describe_tools"]
