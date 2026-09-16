"""Central registry for the tools that are actually implemented."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from bg6022.config import AppConfig
from bg6022.models import Plan, Request, ResultTarget, Step, Tool
from bg6022.orca.profiles import method_capability_catalog
from bg6022.output_contracts import public_type_info
from bg6022.tools.geometry_distance import make_geometry_distance_tool
from bg6022.tools.molecule import make_generate_geometry_tool
from bg6022.tools.orca import make_frequency_tool, make_optimize_tool, make_single_point_tool
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

    def result_capabilities(self) -> list[dict[str, Any]]:
        """Describe user-requestable result targets from registered Tool contracts.

        The capability list is derived from the same declarations used by Plan
        validation. If a Tool declares a value as both a result and an output
        port, the port is canonical because it carries the geometry type.
        """

        capabilities: list[dict[str, Any]] = []
        for tool_name in self.names():
            tool = self._tools[tool_name]
            if not tool.available:
                continue
            declared = [("port", name) for name in tool.output_ports]
            declared.extend(
                ("field", name) for name in tool.results if name not in tool.output_ports
            )
            declared.extend(("check", name) for name in tool.scientific_checks)
            for kind, name in declared:
                metadata = tool.result_metadata.get(name, {})
                capabilities.append(
                    {
                        "name": name,
                        "kind": kind,
                        "tool": tool.name,
                        "operations": list(tool.operations),
                        "property": tool.result_properties.get(name),
                        **public_type_info(
                            (
                                tool.output_ports[name]
                                if kind == "port"
                                else tool.results[name]
                                if kind == "field"
                                else "scientific_check"
                            ),
                            kind=kind,
                        ),
                        "label": metadata.get("label", name),
                        "description": metadata.get("description")
                        or tool.scientific_checks.get(name),
                        "caveat": metadata.get("caveat"),
                    }
                )
        return capabilities

    def request_parameter_capabilities(self) -> list[dict[str, Any]]:
        """Describe request-level parameters accepted by available compute Tools."""

        return [
            {
                "tool": tool.name,
                "operations": list(tool.operations),
                "schema": tool.parameter_schema,
                "request_parameters": list(tool.request_parameters),
                "deferred_parameters": list(tool.deferred_parameters),
            }
            for tool in (self.get(name) for name in self.names())
            if tool.available and tool.request_parameters and tool.parameter_type is not None
        ]

    def method_capability_catalog(self) -> list[dict[str, Any]]:
        """Return the shared registry-backed method profile directory."""

        return method_capability_catalog()

    def tools_for_request(
        self,
        operations: Iterable[str],
        requested_results: Iterable[str | ResultTarget] = (),
    ) -> list[Tool]:
        """Return the available Tools involved in one request's semantics.

        Operation-bearing Tools are selected by operation.  Operation-free
        result producers are selected from the canonical result capability
        catalog, which keeps tools such as ``geometry_distance`` in scope even
        when the Request intentionally has no ORCA operation.
        """

        requested_operations = set(operations)
        selected: dict[str, Tool] = {
            tool.name: tool
            for tool in (self.get(name) for name in self.names())
            if tool.available
            and requested_operations
            and bool(requested_operations & set(tool.operations))
        }
        for raw_target in requested_results:
            if isinstance(raw_target, ResultTarget):
                target_name = raw_target.check or raw_target.port or raw_target.field
            else:
                target_name = raw_target
            if not target_name:
                continue
            try:
                target = self.resolve_result_target(
                    str(target_name), requested_operations, canonical_only=False
                )
            except ValueError:
                continue
            identity = _target_identity(target)
            for item in self.result_capabilities():
                if (item["kind"], item["name"]) == identity:
                    selected[item["tool"]] = self.get(item["tool"])
        return [selected[name] for name in sorted(selected)]

    def request_parameter_fields(
        self,
        operations: Iterable[str],
        requested_results: Iterable[str | ResultTarget] = (),
    ) -> set[str]:
        """Return only request-level fields declared by involved Tools."""

        return {
            field
            for tool in self.tools_for_request(operations, requested_results)
            for field in tool.request_parameters
        }

    def resolve_result_target(
        self,
        name: str,
        operations: Iterable[str],
        *,
        canonical_only: bool = False,
    ) -> ResultTarget:
        """Normalize one current or legacy result name against Tool declarations."""

        requested_operations = set(operations)
        capabilities = self.result_capabilities()

        def compatible(item: dict[str, Any]) -> bool:
            producers = set(item["operations"])
            return not producers or bool(producers & requested_operations)

        if (
            not canonical_only
            and name in {"geometry", "molecular_geometry"}
            and "Opt" in requested_operations
        ):
            return self.resolve_result_target(
                "optimized_geometry",
                requested_operations,
                canonical_only=True,
            )

        exact = [item for item in capabilities if item["name"] == name]
        if exact:
            matches = [item for item in exact if compatible(item)]
            if not matches:
                raise ValueError(
                    f"requested result {name!r} is not produced by the requested operation(s)"
                )
            return _target_from_unique_capability(name, matches)

        if canonical_only:
            raise ValueError(f"requested result is not in the Tool capability catalog: {name!r}")

        # These aliases are retained only for older Intake output and stored
        # Requests. New model output is constrained to canonical capability names.
        exact_aliases = {
            "sp_energy": "sp_electronic_energy",
            "opt_energy": "opt_final_electronic_energy",
            "frequency": "vibrational_frequencies",
            "frequencies": "vibrational_frequencies",
            "optimized_geometry": "optimized_geometry",
        }
        if name in exact_aliases:
            canonical = exact_aliases[name]
            matches = [
                item for item in capabilities if item["name"] == canonical and compatible(item)
            ]
            if matches:
                return _target_from_unique_capability(name, matches)
            raise ValueError(
                f"legacy result alias {name!r} is not supported by the requested operation(s)"
            )

        property_aliases = {
            "energy": "electronic_energy",
            "electronic_energy": "electronic_energy",
            "geometry": "molecular_geometry",
            "molecular_geometry": "molecular_geometry",
            "distance": "distance",
            "interatomic_distance": "distance",
            "bond_length": "distance",
        }
        property_name = property_aliases.get(name)
        if property_name is None:
            raise ValueError(f"requested result is not in the Tool capability catalog: {name!r}")

        matches = [
            item for item in capabilities if item["property"] == property_name and compatible(item)
        ]
        # A geometry alias after an explicit Opt refers to the optimized output;
        # otherwise it remains the initial geometry produced by the geometry Tool.
        if property_name == "molecular_geometry" and "Opt" in requested_operations:
            optimized = [item for item in matches if item["name"] == "optimized_geometry"]
            if optimized:
                matches = optimized
        if not matches:
            raise ValueError(
                f"requested result {name!r} is not produced by the requested operation(s)"
            )
        identities = {(item["kind"], item["name"]) for item in matches}
        if len(identities) != 1:
            if property_name == "electronic_energy":
                raise ValueError(
                    "the request includes both Opt and SP, so ‘energy’ is ambiguous; "
                    "specify opt_final_electronic_energy or sp_electronic_energy"
                )
            raise ValueError(f"requested result alias {name!r} is ambiguous")
        return _target_from_unique_capability(name, matches)

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
            for requirement in step.goal_checks:
                if requirement.source_step_id == step.id:
                    raise ValueError(f"step {step.id} cannot require its own scientific check")
                source = step_by_id.get(requirement.source_step_id)
                if source is None:
                    raise ValueError(
                        f"step {step.id} goal check references unknown step "
                        f"{requirement.source_step_id}"
                    )
                source_tool = self.get(source.tool)
                if requirement.check not in source_tool.scientific_checks:
                    raise ValueError(
                        f"step {step.id} references undeclared scientific check "
                        f"{requirement.source_step_id}.{requirement.check}"
                    )
                if requirement.check == "local_minimum_supported":
                    source_geometry = source.inputs.get("geometry")
                    dependent_geometry = step.inputs.get("geometry")
                    if (
                        source_tool.name != "frequency"
                        or source_geometry is None
                        or dependent_geometry is None
                        or source_geometry != dependent_geometry
                    ):
                        raise ValueError(
                            f"step {step.id} must use the same geometry checked by "
                            f"{source.id}.local_minimum_supported"
                        )
                dependencies[step.id].add(requirement.source_step_id)

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
            make_frequency_tool(config),
            make_geometry_distance_tool(config),
        ]
    )


def describe_tools() -> list[dict[str, Any]]:
    """Return descriptions without loading a config or creating any executable closure."""
    return build_registry().describe()


def merge_explicit_step_parameters(tool: Tool, step: Step, request: Request) -> Step:
    """Project Request parameters into one Step without inventing defaults.

    The merge order is deliberately explicit: planner candidate values are
    the base, Request explicit values override them, and user modifications
    override both.  Validation uses the Tool's own parameter model while the
    returned mapping preserves which keys were actually supplied.
    """

    allowed = set(tool.request_parameters)
    merged = dict(step.parameters)
    for source in (request.explicit_parameters, request.user_modifications):
        merged.update(
            {key: value for key, value in source.items() if key in allowed and value is not None}
        )
    tool.validate_parameters(merged, allow_deferred=True)
    return step.model_copy(update={"parameters": merged})


__all__ = [
    "ToolRegistry",
    "build_registry",
    "describe_tools",
    "merge_explicit_step_parameters",
]


def _target_from_unique_capability(
    requested_name: str, matches: list[dict[str, Any]]
) -> ResultTarget:
    identities = {(item["kind"], item["name"]) for item in matches}
    if len(identities) != 1:
        raise ValueError(f"requested result alias {requested_name!r} is ambiguous")
    kind, canonical_name = next(iter(identities))
    return ResultTarget(**{kind: canonical_name})


def _target_identity(target: ResultTarget) -> tuple[str, str]:
    if target.check is not None:
        return "check", target.check
    if target.port is not None:
        return "port", target.port
    assert target.field is not None
    return "field", target.field


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
        for check in tool.scientific_checks:
            producers.setdefault(("check", check), []).append(step.id)
    for target in targets:
        kind = (
            "check" if target.check is not None else "port" if target.port is not None else "field"
        )
        name = target.check or target.port or target.field
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
