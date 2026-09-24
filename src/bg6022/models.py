"""The seven durable runtime objects used by the v3 execution foundation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from bg6022.tools.runtime import ToolCallContext

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictStr,
    TypeAdapter,
    create_model,
    field_validator,
    model_validator,
)

RunStatus = Literal[
    "planned",
    "running",
    "waiting",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
ResultStatus = Literal["succeeded", "failed", "cancelled", "interrupted", "needs_input"]
Operation = Literal["SP", "Opt", "Freq"]
ScientificCheckStatus = Literal["passed", "not_met", "unverified"]

_OUTPUT_PREFERENCE_DEFAULTS = {
    "layout": "auto",
    "file_content": "auto",
    "detail": "normal",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def validate_output_preferences(value: Any) -> dict[str, str]:
    """Validate the small non-scientific presentation preference contract."""

    if value is None:
        return dict(_OUTPUT_PREFERENCE_DEFAULTS)
    if not isinstance(value, dict):
        raise TypeError("output_preferences must be an object")
    allowed = {
        "layout": {"auto", "plain", "table"},
        "file_content": {"auto", "show", "link_only"},
        "detail": {"brief", "normal", "full"},
    }
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown output preference(s): {unknown}")
    normalized = dict(_OUTPUT_PREFERENCE_DEFAULTS)
    for name, options in allowed.items():
        if name not in value:
            continue
        item = value[name]
        if type(item) is not str or item not in options:
            raise ValueError(f"invalid output preference {name!r}: {item!r}")
        normalized[name] = item
    return normalized


class ResultTarget(StrictModel):
    """A value or output port requested from one logical step.

    This is a nested value contract, not another persisted runtime object.  The
    optional ``step_id`` keeps old M0 plans readable while M1 plans identify the
    producer explicitly.
    """

    step_id: str | None = None
    requirement_id: str | None = None
    field: str | None = None
    port: str | None = None
    check: str | None = None

    @model_validator(mode="after")
    def _one_target_kind(self) -> ResultTarget:
        if sum(value is not None for value in (self.field, self.port, self.check)) != 1:
            raise ValueError("result target must contain exactly one field, port, or check")
        return self


class RequirementInputBinding(StrictModel):
    """A source for one input port of a Requirement."""

    source_requirement_id: str | None = None
    source_port: str | None = None
    artifact_alias: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> RequirementInputBinding:
        if self.artifact_alias is not None:
            if self.source_requirement_id is not None or self.source_port is not None:
                raise ValueError("artifact_alias cannot be combined with a Requirement source")
            return self
        if (self.source_requirement_id is None) != (self.source_port is None):
            raise ValueError("Requirement input source needs both requirement id and port")
        if self.source_requirement_id is None:
            raise ValueError("Requirement input must reference a Requirement or artifact alias")
        return self


class Requirement(StrictModel):
    """One user-requested capability instance, nested inside Request."""

    id: str
    subject_id: str = "subject_1"
    capability: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    input_bindings: dict[str, RequirementInputBinding] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "subject_id", "capability")
    @classmethod
    def _nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("requirement identity fields must not be blank")
        return value

    @field_validator("outputs")
    @classmethod
    def _unique_outputs(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("requirement outputs must be unique")
        return value


class Subject(StrictModel):
    """A named scientific subject nested inside Request."""

    key: str
    molecule_query: str | None = None
    molecule_input_kind: Literal["name", "cid", "smiles", "formula"] | None = None
    molecule_name_evidence: str | None = None
    structure_input: dict[str, Any] = Field(default_factory=dict)


class AnswerGoal(StrictModel):
    """A presentation goal over outputs from requested Requirements."""

    kind: Literal["compare"]
    requirement_ids: list[str]
    output: str
    mode: Literal["side_by_side", "numeric_difference"] = "side_by_side"

    @field_validator("requirement_ids")
    @classmethod
    def _distinct_requirements(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("a comparison needs at least two Requirements")
        if len(value) != len(set(value)):
            raise ValueError("AnswerGoal requirement ids must be unique")
        return value


class RequiredGeometryBinding(StrictModel):
    """A user-required source for one calculation's geometry input.

    This is a nested Request value contract, not a durable runtime object.
    ``source_operation=None`` with ``source_port='initial_geometry'`` means
    retain the original input geometry; otherwise the named operation and
    output port identify the required upstream source.
    """

    consumer_operation: Operation | None = None
    consumer_tool: str | None = None
    consumer_requirement_id: str | None = None
    consumer_requirement_key: str | None = None
    input_port: str
    source_operation: Operation | None = None
    source_requirement_id: str | None = None
    source_requirement_key: str | None = None
    source_port: str

    @model_validator(mode="after")
    def _one_consumer_selector(self) -> RequiredGeometryBinding:
        if (
            sum(
                value is not None
                for value in (
                    self.consumer_operation,
                    self.consumer_tool,
                    self.consumer_requirement_id,
                    self.consumer_requirement_key,
                )
            )
            != 1
        ):
            raise ValueError(
                "geometry binding must identify one consumer operation, Tool, or requirement"
            )
        if (
            sum(
                value is not None
                for value in (
                    self.source_operation,
                    self.source_requirement_id,
                    self.source_requirement_key,
                )
            )
            > 1
        ):
            raise ValueError(
                "geometry binding must select at most one source operation or requirement"
            )
        if (
            self.source_operation is None
            and self.source_requirement_id is None
            and self.source_requirement_key is None
            and self.source_port != "initial_geometry"
        ):
            raise ValueError("an unselected geometry source must use initial_geometry")
        return self


_REQUIRED_GEOMETRY_BINDINGS = TypeAdapter(list[RequiredGeometryBinding])


class Request(StrictModel):
    id: str
    description: str
    source: Literal["cli", "chat"] = "cli"
    original_text: str | None = None
    subjects: dict[str, Subject] = Field(default_factory=dict)
    requirements: list[Requirement] = Field(default_factory=list)
    answer_goals: list[AnswerGoal] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    output_preferences: dict[str, str] = Field(
        default_factory=lambda: dict(_OUTPUT_PREFERENCE_DEFAULTS)
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Request:
        """Translate legacy operation updates into canonical Requirements."""

        changes = dict(update or {})
        legacy_operation = changes.pop("operation", None)
        legacy_operations = changes.pop("operations", None)
        if legacy_operation is not None:
            if legacy_operations is not None and list(legacy_operations) != [legacy_operation]:
                raise ValueError("legacy operation conflicts with operations")
            legacy_operations = [legacy_operation]
        if legacy_operations is not None:
            from bg6022.tools.registry import build_registry

            registry = build_registry()
            previous = list(changes.get("requirements", self.requirements))
            used: set[str] = set()
            subject_id = next((item.subject_id for item in previous), "subject_1")
            if not changes.get("subjects", self.subjects):
                changes["subjects"] = {subject_id: Subject(key=subject_id)}
            replacements: list[Requirement] = []
            for index, operation in enumerate(legacy_operations, start=1):
                matches = [
                    registry.get(name)
                    for name in registry.names()
                    if operation in registry.get(name).operations
                ]
                if len(matches) != 1:
                    raise ValueError(f"legacy operation {operation!r} has no unique Tool")
                capability = matches[0].name
                existing = next(
                    (
                        item
                        for item in previous
                        if item.capability == capability and item.id not in used
                    ),
                    None,
                )
                if existing is not None:
                    used.add(existing.id)
                    replacements.append(existing)
                else:
                    replacements.append(
                        Requirement(
                            id=f"legacy_req_update_{index}",
                            subject_id=subject_id,
                            capability=capability,
                        )
                    )
            changes["requirements"] = replacements
        return super().model_copy(update=changes, deep=deep)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_keys = {
            "operation",
            "operations",
            "requested_results",
            "explicit_parameters",
            "user_modifications",
            "user_modifications_by_requirement",
            "structure_input",
        }
        if not legacy_keys.intersection(data):
            requirements = data.get("requirements")
            if isinstance(requirements, list):
                raw_requirements = [
                    item.model_dump(mode="python")
                    if isinstance(item, Requirement)
                    else dict(item)
                    if isinstance(item, Mapping)
                    else item
                    for item in requirements
                ]
                if any(
                    isinstance(item, dict) and item.get("capability") == "energy_difference"
                    for item in raw_requirements
                ):
                    from bg6022.tools.registry import build_registry

                    registry = build_registry()
                    data["requirements"] = [
                        {
                            **item,
                            "capability": registry.canonical_tool_name(
                                str(item.get("capability", ""))
                            ),
                        }
                        if isinstance(item, dict)
                        else item
                        for item in raw_requirements
                    ]
            return data

        from bg6022.tools.registry import build_registry

        registry = build_registry()
        legacy_operation = data.pop("operation", None)
        operations = list(data.pop("operations", []))
        if legacy_operation is not None:
            if operations and operations != [legacy_operation]:
                raise ValueError("legacy operation conflicts with operations")
            operations = [legacy_operation]
        raw_targets = list(data.pop("requested_results", []))
        explicit_parameters = dict(data.pop("explicit_parameters", {}))
        user_modifications = dict(data.pop("user_modifications", {}))
        modifications_by_requirement = dict(data.pop("user_modifications_by_requirement", {}))
        legacy_structure = dict(data.pop("structure_input", {}))
        legacy_bindings = list(legacy_structure.pop("required_bindings", []))

        raw_subjects = dict(data.get("subjects", {}))
        subjects: dict[str, dict[str, Any]] = {}
        for subject_id, raw_subject in raw_subjects.items():
            if isinstance(raw_subject, Subject):
                subjects[str(subject_id)] = raw_subject.model_dump(mode="python")
                continue
            item = dict(raw_subject)
            key = str(item.pop("key", subject_id))
            subject_structure = dict(item.pop("structure_input", {}))
            subjects[str(subject_id)] = {
                "key": key,
                **item,
                "structure_input": subject_structure,
            }

        if legacy_structure:
            if not subjects:
                subjects["subject_1"] = {"key": "subject_1", "structure_input": {}}
            if len(subjects) == 1:
                only_subject = next(iter(subjects.values()))
                subject_structure = dict(only_subject.get("structure_input", {}))
                subject_structure.update(legacy_structure)
                only_subject["structure_input"] = subject_structure

        requirements = [
            item.model_dump(mode="python") if isinstance(item, Requirement) else dict(item)
            for item in data.get("requirements", [])
        ]
        requirements = [
            {
                **item,
                "capability": registry.canonical_tool_name(str(item.get("capability", ""))),
            }
            for item in requirements
        ]
        if not requirements:
            for index, operation in enumerate(operations, start=1):
                matches = [
                    tool
                    for name in registry.names()
                    if (tool := registry.get(name)).operations and operation in tool.operations
                ]
                if len(matches) != 1:
                    raise ValueError(f"legacy operation {operation!r} has no unique Tool")
                requirements.append(
                    {
                        "id": f"legacy_req_{index}",
                        "subject_id": next(iter(subjects), "subject_1"),
                        "capability": matches[0].name,
                        "parameters": {},
                        "outputs": [],
                    }
                )

        # Resolve legacy result producers before translating legacy input bindings;
        # an operation-free Tool may be the consumer of such a binding.
        for raw_target in raw_targets:
            target = (
                raw_target.model_dump(mode="python")
                if isinstance(raw_target, ResultTarget)
                else {"field": raw_target}
                if isinstance(raw_target, str)
                else dict(raw_target)
            )
            name = target.get("check") or target.get("port") or target.get("field")
            if not name:
                continue
            try:
                canonical = registry.resolve_result_target(str(name), operations)
            except ValueError:
                continue
            target_name = canonical.check or canonical.port or canonical.field
            if not target_name:
                continue
            target_kind = "check" if canonical.check else "port" if canonical.port else "field"
            producers = {
                str(item["tool"])
                for item in registry.result_capabilities()
                if item["name"] == target_name and item["kind"] == target_kind
            }
            if len(producers) != 1:
                continue
            producer = next(iter(producers))
            if not any(item.get("capability") == producer for item in requirements):
                requirements.append(
                    {
                        "id": str(
                            target.get("requirement_id") or f"legacy_req_{len(requirements) + 1}"
                        ),
                        "subject_id": next(iter(subjects), "subject_1"),
                        "capability": producer,
                        "parameters": {},
                        "outputs": [],
                    }
                )

        for raw_binding in legacy_bindings:
            binding = RequiredGeometryBinding.model_validate(raw_binding, strict=True)
            consumers = [
                item
                for item in requirements
                if (
                    binding.consumer_requirement_id is not None
                    and item.get("id") == binding.consumer_requirement_id
                )
                or (
                    binding.consumer_tool is not None
                    and item.get("capability") == binding.consumer_tool
                )
                or (
                    binding.consumer_operation is not None
                    and binding.consumer_operation
                    in registry.get(str(item.get("capability"))).operations
                )
            ]
            if len(consumers) != 1:
                raise ValueError("legacy geometry binding does not identify one Requirement")
            consumer = consumers[0]
            inputs = consumer.setdefault("input_bindings", {})
            if binding.source_requirement_id is not None:
                inputs[binding.input_port] = {
                    "source_requirement_id": binding.source_requirement_id,
                    "source_port": binding.source_port,
                }
            elif binding.source_operation is not None:
                sources = [
                    item
                    for item in requirements
                    if binding.source_operation
                    in registry.get(str(item.get("capability"))).operations
                ]
                if len(sources) != 1:
                    raise ValueError(
                        "legacy geometry binding does not identify one source Requirement"
                    )
                inputs[binding.input_port] = {
                    "source_requirement_id": sources[0]["id"],
                    "source_port": binding.source_port,
                }
            elif binding.source_port == "initial_geometry":
                inputs[binding.input_port] = {"artifact_alias": "initial_geometry"}

        if raw_targets:
            for raw_target in raw_targets:
                target = (
                    raw_target.model_dump(mode="python")
                    if isinstance(raw_target, ResultTarget)
                    else {"field": raw_target}
                    if isinstance(raw_target, str)
                    else dict(raw_target)
                )
                name = target.get("check") or target.get("port") or target.get("field")
                if not name:
                    continue
                try:
                    canonical = registry.resolve_result_target(str(name), operations)
                except ValueError:
                    continue
                target_name = canonical.check or canonical.port or canonical.field
                if not target_name:
                    continue
                matching_tools = [
                    str(item["tool"])
                    for item in registry.result_capabilities()
                    if item["name"] == target_name
                    and item["kind"]
                    == ("check" if canonical.check else "port" if canonical.port else "field")
                ]
                producer = matching_tools[0] if len(matching_tools) == 1 else None
                matches = [item for item in requirements if item.get("capability") == producer]
                if not matches and producer is not None:
                    requirement_id = str(
                        target.get("requirement_id") or f"legacy_req_{len(requirements) + 1}"
                    )
                    requirement = {
                        "id": requirement_id,
                        "subject_id": next(iter(subjects), "subject_1"),
                        "capability": producer,
                        "parameters": {},
                        "outputs": [],
                    }
                    requirements.append(requirement)
                    matches = [requirement]
                for requirement in matches:
                    output = str(target_name)
                    if output not in requirement.setdefault("outputs", []):
                        requirement["outputs"].append(output)

        modifications_by_id = modifications_by_requirement
        for requirement in requirements:
            capability = str(requirement.get("capability", ""))
            try:
                allowed_parameters = set(registry.get(capability).request_parameters)
            except ValueError:
                allowed_parameters = set()
            parameters = dict(requirement.get("parameters", {}))
            for name, parameter_value in explicit_parameters.items():
                if name in allowed_parameters:
                    parameters.setdefault(name, parameter_value)
            for name, parameter_value in user_modifications.items():
                if name in allowed_parameters:
                    parameters[name] = parameter_value
            for name, parameter_value in dict(
                modifications_by_id.get(str(requirement.get("id", "")), {})
            ).items():
                if name in allowed_parameters:
                    parameters[name] = parameter_value
            requirement["parameters"] = parameters

        if requirements and not subjects:
            subjects["subject_1"] = {"key": "subject_1", "structure_input": {}}

        data["subjects"] = subjects
        data["requirements"] = requirements
        return data

    @field_validator("requirements")
    @classmethod
    def _unique_requirement_ids(cls, value: list[Requirement]) -> list[Requirement]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement ids must be unique")
        return value

    @model_validator(mode="after")
    def _requirement_subjects_are_known(self) -> Request:
        known_subjects = set(self.subjects)
        missing = sorted(
            {
                item.subject_id
                for item in self.requirements
                if known_subjects and item.subject_id not in known_subjects
            }
        )
        if missing:
            raise ValueError(f"requirements refer to unknown subject ids: {missing}")
        requirement_ids = {item.id for item in self.requirements}
        for requirement in self.requirements:
            for input_name, binding in requirement.input_bindings.items():
                if (
                    binding.source_requirement_id is not None
                    and binding.source_requirement_id not in requirement_ids
                ):
                    raise ValueError(
                        f"requirement {requirement.id!r} input {input_name!r} refers to "
                        "an unknown source Requirement"
                    )
        for goal in self.answer_goals:
            unknown = sorted(set(goal.requirement_ids) - requirement_ids)
            if unknown:
                raise ValueError(f"AnswerGoal refers to unknown Requirements: {unknown}")
        return self

    @field_validator("output_preferences", mode="before")
    @classmethod
    def _validate_output_preferences(cls, value: Any) -> dict[str, str]:
        return validate_output_preferences(value)

    @property
    def operations(self) -> list[Operation]:
        """Read-only compatibility derived from canonical Requirements."""

        from bg6022.tools.registry import build_registry

        registry = build_registry()
        return [
            operation
            for requirement in self.requirements
            for operation in registry.get(requirement.capability).operations
        ]

    @property
    def requested_results(self) -> list[ResultTarget]:
        """Read-only compatibility projection derived from Requirement outputs."""

        from bg6022.tools.registry import build_registry

        registry = build_registry()
        counts: dict[str, int] = {}
        for requirement in self.requirements:
            counts[requirement.capability] = counts.get(requirement.capability, 0) + 1
        targets: list[ResultTarget] = []
        for requirement in self.requirements:
            tool = registry.get(requirement.capability)
            for output in requirement.outputs:
                descriptor = next(
                    (item for item in tool.public_outputs() if item["name"] == output), None
                )
                if descriptor is None:
                    continue
                identity: dict[str, str] = {str(descriptor["kind"]): output}
                if counts[requirement.capability] > 1:
                    identity["requirement_id"] = requirement.id
                target = ResultTarget(**identity)
                if target not in targets:
                    targets.append(target)
        return sorted(targets, key=lambda target: target.port != "geometry")

    @property
    def explicit_parameters(self) -> dict[str, Any]:
        """Compatibility view of unambiguous canonical Requirement parameters."""

        values: dict[str, Any] = {}
        conflicts: set[str] = set()
        for requirement in self.requirements:
            for name, value in requirement.parameters.items():
                if name in values and values[name] != value:
                    conflicts.add(name)
                else:
                    values[name] = value
        return {name: value for name, value in values.items() if name not in conflicts}

    @property
    def user_modifications(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for requirement in self.requirements:
            sources = requirement.constraints.get("parameter_sources", {})
            if not isinstance(sources, Mapping):
                continue
            for name, source in sources.items():
                if source == "user_modification" and name in requirement.parameters:
                    values[name] = requirement.parameters[name]
        return values

    @property
    def user_modifications_by_requirement(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for requirement in self.requirements:
            sources = requirement.constraints.get("parameter_sources", {})
            if not isinstance(sources, Mapping):
                continue
            updates = {
                name: requirement.parameters[name]
                for name, source in sources.items()
                if source == "user_modification" and name in requirement.parameters
            }
            if updates:
                result[requirement.id] = updates
        return result

    @property
    def structure_input(self) -> dict[str, Any]:
        """Read-only compatibility view for a single-subject legacy caller."""

        if len(self.subjects) != 1:
            return {}
        value = dict(next(iter(self.subjects.values())).structure_input)
        bindings: list[dict[str, Any]] = []
        from bg6022.tools.registry import build_registry

        registry = build_registry()
        requirement_counts: dict[str, int] = {}
        for requirement in self.requirements:
            requirement_counts[requirement.capability] = (
                requirement_counts.get(requirement.capability, 0) + 1
            )
        requirements_by_id = {item.id: item for item in self.requirements}
        for requirement in self.requirements:
            if not requirement.input_bindings:
                continue
            consumer_tool = registry.get(requirement.capability)
            for input_port, binding in requirement.input_bindings.items():
                if consumer_tool.input_ports.get(input_port) != "molecular_geometry":
                    continue
                if requirement_counts[requirement.capability] > 1:
                    consumer = {"consumer_requirement_id": requirement.id}
                elif len(consumer_tool.operations) == 1:
                    consumer = {"consumer_operation": consumer_tool.operations[0]}
                else:
                    consumer = {"consumer_tool": requirement.capability}
                if binding.artifact_alias is not None:
                    source = {
                        "source_operation": None,
                        "source_port": binding.artifact_alias,
                    }
                else:
                    source_requirement = requirements_by_id[binding.source_requirement_id]
                    source_tool = registry.get(source_requirement.capability)
                    if len(source_tool.operations) == 1:
                        source = {
                            "source_operation": source_tool.operations[0],
                            "source_port": binding.source_port,
                        }
                    else:
                        source = {
                            "source_requirement_id": binding.source_requirement_id,
                            "source_port": binding.source_port,
                        }
                bindings.append(
                    {
                        **consumer,
                        "input_port": input_port,
                        **source,
                    }
                )
        if bindings:
            value["required_bindings"] = bindings
        elif value.get("history_geometry_alias") is not None:
            value["required_bindings"] = []
        return value

    @property
    def operation(self) -> Operation | None:
        """Read-only compatibility for old single-operation callers."""

        operations = self.operations
        return operations[0] if len(operations) == 1 else None

    def with_subject_structure_input(
        self, subject_id: str, structure_input: Mapping[str, Any]
    ) -> Request:
        subject = self.subjects.get(subject_id)
        if subject is None:
            raise ValueError(f"unknown subject id {subject_id!r}")
        subjects = dict(self.subjects)
        subjects[subject_id] = subject.model_copy(update={"structure_input": dict(structure_input)})
        return self.model_copy(update={"subjects": subjects})


class InputReference(StrictModel):
    """A logical artifact or an earlier step port, never an arbitrary path."""

    artifact_id: str | None = None
    step_id: str | None = None
    port: str | None = None

    @model_validator(mode="after")
    def _exactly_one_reference(self) -> InputReference:
        artifact = self.artifact_id is not None
        step = self.step_id is not None or self.port is not None
        if artifact == step:
            raise ValueError("input reference must be an artifact_id or a step_id/port pair")
        if step and (not self.step_id or not self.port):
            raise ValueError("step input references require both step_id and port")
        return self


class GoalCheckRequirement(StrictModel):
    """A source Tool check that must have a declared status before this Step runs."""

    source_step_id: str
    check: str
    required_status: ScientificCheckStatus = "passed"


class ScientificCheckResult(StrictModel):
    """Program-computed evidence for a named scientific goal check."""

    status: ScientificCheckStatus
    input_geometry_sha256: str | None = None
    input_artifact_sha256_by_port: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class Step(StrictModel):
    id: str
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputReference] = Field(default_factory=dict)
    goal_checks: list[GoalCheckRequirement] = Field(default_factory=list)
    requirement_id: str | None = None
    subject_id: str | None = None
    origin_step_id: str | None = None


class Plan(StrictModel):
    id: str
    revision: int = 1
    request_id: str
    steps: list[Step]
    requested_results: list[ResultTarget] = Field(default_factory=list)

    @field_validator("requested_results", mode="before")
    @classmethod
    def _load_legacy_result_targets(cls, value: Any) -> Any:
        if value is None:
            return []
        return [{"field": item} if isinstance(item, str) else item for item in value]

    @field_validator("revision")
    @classmethod
    def _positive_revision(cls, value: int) -> int:
        if value < 1:
            raise ValueError("plan revision must be positive")
        return value


class Artifact(StrictModel):
    id: str
    artifact_type: str
    role: str
    relative_path: str
    size_bytes: int
    sha256: str
    source: str
    run_id: str
    step_id: str | None = None
    attempt: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("size_bytes")
    @classmethod
    def _nonnegative_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("artifact size cannot be negative")
        return value


class Result(StrictModel):
    run_id: str
    step_id: str
    attempt: int
    status: ResultStatus
    values: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    scientific_checks: dict[str, ScientificCheckResult] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    output_ports: dict[str, str] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    attempt_relative_path: str = ""
    clarification: dict[str, Any] = Field(default_factory=dict)
    parameter_sources: dict[str, str] = Field(default_factory=dict)
    step_fingerprint: str | None = None
    input_bindings: dict[str, str] = Field(default_factory=dict)


class Run(StrictModel):
    id: str
    request: Request
    plan: Plan
    resources: dict[str, Any]
    execution_permission: bool = False
    accepted_execution_sha256: str | None = None
    status: RunStatus = "planned"
    step_status: dict[str, str] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_index: list[Artifact] = Field(default_factory=list)
    result_index: list[str] = Field(default_factory=list)
    current_results: dict[str, str] = Field(default_factory=dict)
    waiting_for: Literal["confirmation", "clarification", "repair", None] = None
    pending_data: dict[str, Any] = Field(default_factory=dict)
    parameter_sources_by_step: dict[str, dict[str, str]] = Field(default_factory=dict)
    accepted_snapshot: dict[str, Any] = Field(default_factory=dict)
    repair_records: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    attempt_counts: dict[str, int] = Field(default_factory=dict)
    extra_orca_executions: int = 0
    extra_executions_by_category: dict[str, int] = Field(default_factory=dict)
    plan_revisions: int = 0
    origin_step_map: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None
    active_seconds: float = 0.0
    created_at: str
    updated_at: str

    _active_interval_started_at: float | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_execution_budget(cls, value: Any) -> Any:
        """Read pre-category Run budgets into the generic Tool category contract."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        budget = dict(data.get("budget", {}))
        category_limits = dict(budget.get("max_extra_executions_by_category", {}))
        legacy_limit = budget.get("max_extra_orca_executions")
        legacy_limit = category_limits.pop("orca", legacy_limit)
        if legacy_limit is not None:
            category_limits.setdefault("electronic_structure", legacy_limit)
        if category_limits:
            budget["max_extra_executions_by_category"] = category_limits
        budget.pop("max_extra_orca_executions", None)
        data["budget"] = budget
        execution_counts = dict(data.get("extra_executions_by_category", {}))
        legacy_count = execution_counts.pop("orca", data.get("extra_orca_executions", 0))
        if legacy_count:
            execution_counts.setdefault("electronic_structure", legacy_count)
        data["extra_executions_by_category"] = execution_counts
        return data

    def start_active_interval(self, *, now: float | None = None) -> None:
        """Start the one active execution interval, without changing its total."""

        if self._active_interval_started_at is None:
            self._active_interval_started_at = time.monotonic() if now is None else now

    @property
    def active_interval_open(self) -> bool:
        return self._active_interval_started_at is not None

    def current_active_seconds(self, *, now: float | None = None) -> float:
        """Return persisted active time plus the currently open interval."""

        elapsed = max(0.0, float(self.active_seconds))
        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            elapsed += max(0.0, current - self._active_interval_started_at)
        return elapsed

    def checkpoint_active(self, *, now: float | None = None) -> float:
        """Accumulate the open interval exactly once and leave it open."""

        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            self.active_seconds += max(0.0, current - self._active_interval_started_at)
            self._active_interval_started_at = current
        return self.active_seconds

    def finish_active_interval(self, *, now: float | None = None) -> float:
        """Accumulate and close the current active interval."""

        if self._active_interval_started_at is not None:
            current = time.monotonic() if now is None else now
            self.active_seconds += max(0.0, current - self._active_interval_started_at)
            self._active_interval_started_at = None
        return self.active_seconds


ExecuteFunction = Callable[[Step, Any], Result]
ParameterValidationFunction = Callable[[dict[str, Any], Mapping[str, Any]], None]
ResultValidationFunction = Callable[[Run, Step, Result], bool]
ResultProperty = StrictStr


@dataclass(frozen=True)
class ToolPreparation:
    """Transient result from a Tool's optional parameter preparation hook."""

    step: Step | None
    missing_fields: tuple[str, ...] = ()
    parameter_sources: Mapping[str, str] = dataclass_field(default_factory=dict)
    question: str | None = None


@dataclass(frozen=True)
class RepairOption:
    """A bounded, evidence-backed action offered by one Tool adapter."""

    action: str
    failed_step_id: str
    parameter_patch: dict[str, Any]
    input_aliases: dict[str, str]
    evidence_refs: tuple[str, ...]
    reason: str

    @property
    def option_id(self) -> str:
        return self.action

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.parameter_patch

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "parameters": dict(self.parameter_patch),
            "input_aliases": list(self.input_aliases),
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


class Tool(StrictModel):
    name: str
    description: str
    display_name: str | None = None
    operations: list[Operation] = Field(default_factory=list)
    parameter_model: str = "none"
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    parameter_type: type[BaseModel] | None = Field(default=None, exclude=True, repr=False)
    input_ports: dict[str, str] = Field(default_factory=dict)
    output_ports: dict[str, str] = Field(default_factory=dict)
    results: dict[str, str] = Field(default_factory=dict)
    result_properties: dict[str, ResultProperty] = Field(default_factory=dict)
    result_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)
    success_conditions: list[str] = Field(default_factory=list)
    planning_role: Literal["task", "preparation"] = "task"
    default_outputs: list[str] = Field(default_factory=list)
    scientific_checks: dict[str, str] = Field(default_factory=dict)
    scientific_check_input_ports: dict[str, str] = Field(default_factory=dict)
    result_check_prerequisites: dict[str, list[str]] = Field(default_factory=dict)
    repair_capabilities: list[str] = Field(default_factory=list)
    repair_parameter_fields: dict[str, list[str]] = Field(default_factory=dict)
    repair_parameter_limits: dict[str, dict[str, int]] = Field(default_factory=dict)
    repair_input_aliases: dict[str, list[str]] = Field(default_factory=dict)
    requires_compute_permission: bool = True
    execution_budget: str = "none"
    deferred_parameters: list[str] = Field(default_factory=list)
    request_parameters: list[str] = Field(default_factory=list)
    geometry_output_input_ports: dict[str, str] = Field(default_factory=dict)
    available: bool = True
    execute_function: ExecuteFunction | None = Field(default=None, exclude=True, repr=False)
    preflight_function: Callable[[], None] | None = Field(default=None, exclude=True, repr=False)
    preparation_function: Callable[[Step, Mapping[str, Any]], ToolPreparation] | None = Field(
        default=None, exclude=True, repr=False
    )
    repair_options_function: Callable[[Run, Step, Result], list[RepairOption]] | None = Field(
        default=None, exclude=True, repr=False
    )
    apply_repair_function: (
        Callable[[RepairOption, Run, Step, Result, Mapping[str, Any]], tuple[Plan, dict[str, Any]]]
        | None
    ) = Field(default=None, exclude=True, repr=False)
    parameter_validation_function: ParameterValidationFunction | None = Field(
        default=None, exclude=True, repr=False
    )
    result_validation_function: ResultValidationFunction | None = Field(
        default=None, exclude=True, repr=False
    )
    repair_capabilities_function: Callable[[dict[str, Any]], list[str]] | None = Field(
        default=None, exclude=True, repr=False
    )

    model_config = ConfigDict(
        extra="forbid", strict=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    @model_validator(mode="after")
    def _validate_result_contract(self) -> Tool:
        """Keep presentation and semantic metadata attached to declared outputs.

        Presentation metadata is descriptive only. Result-property identifiers
        are machine-readable; neither mapping can add an undeclared result.
        """

        declared = set(self.results) | set(self.output_ports) | set(self.scientific_checks)
        if self.display_name is not None and not self.display_name.strip():
            raise ValueError("Tool display_name must be nonempty when supplied")
        undeclared_properties = sorted(set(self.result_properties) - declared)
        if undeclared_properties:
            raise ValueError(f"result properties have undeclared keys: {undeclared_properties}")
        unknown = sorted(set(self.result_metadata) - declared)
        if unknown:
            raise ValueError(f"result metadata has undeclared keys: {unknown}")
        from bg6022.output_contracts import validate_declared_output, validate_declared_type

        for kind, outputs in (("field", self.results), ("port", self.output_ports)):
            for name, expected_type in outputs.items():
                validate_declared_type(name, expected_type, kind=kind)
        for name in self.scientific_checks:
            validate_declared_type(name, "scientific_check", kind="check")
        undeclared_prerequisite_outputs = sorted(
            set(self.result_check_prerequisites) - (set(self.results) | set(self.output_ports))
        )
        if undeclared_prerequisite_outputs:
            raise ValueError(
                "result check prerequisites refer to undeclared outputs: "
                f"{undeclared_prerequisite_outputs}"
            )
        for output, checks in self.result_check_prerequisites.items():
            if not checks or len(checks) != len(set(checks)):
                raise ValueError(
                    f"result check prerequisites for {output!r} must be nonempty and unique"
                )

        for name, property_name in self.result_properties.items():
            if name in self.scientific_checks:
                expected_type = "scientific_check"
                kind = "check"
            elif name in self.output_ports:
                expected_type = self.output_ports[name]
                kind = "port"
            else:
                expected_type = self.results.get(name)
                kind = "field"
            if expected_type is None:
                continue
            validate_declared_output(name, expected_type, property_name, kind=kind)
        allowed = {"label", "description", "caveat"}
        for name, metadata in self.result_metadata.items():
            extra = sorted(set(metadata) - allowed)
            if extra:
                raise ValueError(f"result metadata for {name!r} has unsupported keys: {extra}")
        unknown_geometry_outputs = sorted(
            set(self.geometry_output_input_ports) - set(self.output_ports)
        )
        if unknown_geometry_outputs:
            raise ValueError(
                f"geometry output declarations have undeclared ports: {unknown_geometry_outputs}"
            )
        unknown_geometry_inputs = sorted(
            set(self.geometry_output_input_ports.values()) - set(self.input_ports)
        )
        if unknown_geometry_inputs:
            raise ValueError(
                f"geometry output declarations have undeclared inputs: {unknown_geometry_inputs}"
            )
        incompatible_geometry_bindings = sorted(
            output
            for output, input_name in self.geometry_output_input_ports.items()
            if self.output_ports[output] != "molecular_geometry"
            or self.input_ports[input_name] != "molecular_geometry"
        )
        if incompatible_geometry_bindings:
            raise ValueError(
                "geometry output declarations must connect molecular-geometry ports: "
                f"{incompatible_geometry_bindings}"
            )
        unknown_check_bindings = sorted(
            set(self.scientific_check_input_ports) - set(self.scientific_checks)
        )
        if unknown_check_bindings:
            raise ValueError(
                "scientific check input bindings reference undeclared checks: "
                f"{unknown_check_bindings}"
            )
        unknown_check_inputs = sorted(
            set(self.scientific_check_input_ports.values()) - set(self.input_ports)
        )
        if unknown_check_inputs:
            raise ValueError(
                "scientific check input bindings reference undeclared inputs: "
                f"{unknown_check_inputs}"
            )
        if len(self.request_parameters) != len(set(self.request_parameters)):
            raise ValueError("request_parameters must not contain duplicates")
        if self.request_parameters and self.parameter_type is None:
            raise ValueError("request_parameters require a parameter model")
        if self.execution_budget != self.execution_budget.strip():
            raise ValueError("execution budget category must not contain surrounding whitespace")
        if self.parameter_type is not None:
            unknown_request_parameters = sorted(
                set(self.request_parameters) - set(self.parameter_type.model_fields)
            )
            if unknown_request_parameters:
                raise ValueError(
                    "request_parameters are not fields of the parameter model: "
                    f"{unknown_request_parameters}"
                )
        if len(self.repair_capabilities) != len(set(self.repair_capabilities)):
            raise ValueError("repair_capabilities must not contain duplicates")
        if set(self.repair_parameter_fields) - set(self.repair_capabilities):
            raise ValueError("repair parameter fields must belong to a declared capability")
        if set(self.repair_parameter_limits) - set(self.repair_capabilities):
            raise ValueError("repair parameter limits must belong to a declared capability")
        if set(self.repair_input_aliases) - set(self.repair_capabilities):
            raise ValueError("repair input aliases must belong to a declared capability")
        if self.parameter_type is not None:
            declared_fields = set(self.parameter_type.model_fields)
            for action, fields in self.repair_parameter_fields.items():
                if len(fields) != len(set(fields)) or set(fields) - declared_fields:
                    raise ValueError(
                        f"repair capability {action!r} refers to invalid parameter fields"
                    )
                limits = self.repair_parameter_limits.get(action, {})
                invalid_limit = any(
                    type(limit) is not int or limit < 1 for limit in limits.values()
                )
                if set(limits) - set(fields) or invalid_limit:
                    raise ValueError(f"repair capability {action!r} has invalid parameter limits")
        # Resolve the canonical public directory during registration.  This
        # catches property collisions (including a collision with a check)
        # before a Tool can be exposed to Intake or query handling.
        from bg6022.output_contracts import canonical_public_outputs

        public_outputs = canonical_public_outputs(self)
        declared_public_outputs = {str(item["name"]) for item in public_outputs}
        if len(self.default_outputs) != len(set(self.default_outputs)):
            raise ValueError("default_outputs must not contain duplicates")
        unknown_defaults = sorted(set(self.default_outputs) - declared_public_outputs)
        if unknown_defaults:
            raise ValueError(
                f"default_outputs refer to undeclared public outputs: {unknown_defaults}"
            )
        return self

    def execute(self, step: Step, context: ToolCallContext) -> Result:
        if self.execute_function is None:
            raise RuntimeError(f"tool {self.name!r} has no executable implementation")
        if context.step.id != step.id or context.step.tool != self.name:
            raise ValueError("ToolCallContext does not match the invoked Step and Tool")
        return self.execute_function(step, context)

    def preflight(self) -> None:
        if self.preflight_function is not None:
            self.preflight_function()

    def prepare(self, step: Step, context: Mapping[str, Any]) -> ToolPreparation:
        if self.preparation_function is None:
            return ToolPreparation(step=step)
        prepared = self.preparation_function(step, context)
        if not isinstance(prepared, ToolPreparation):
            raise TypeError(f"tool {self.name!r} returned an invalid preparation result")
        return prepared

    def repair_options(self, run: Run, step: Step, result: Result) -> list[RepairOption]:
        if self.repair_options_function is None:
            return []
        return list(self.repair_options_function(run, step, result))

    def apply_repair(
        self,
        option: RepairOption,
        run: Run,
        step: Step,
        result: Result,
        proposal: Mapping[str, Any],
    ) -> tuple[Plan, dict[str, Any]]:
        if self.apply_repair_function is None:
            raise ValueError(f"tool {self.name!r} does not support repairs")
        return self.apply_repair_function(option, run, step, result, proposal)

    def validate_parameters(
        self,
        parameters: dict[str, Any],
        *,
        allow_deferred: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.parameter_type is None:
            if self.parameter_schema:
                raise ValueError(f"tool {self.name!r} has no parameter model")
            if parameters:
                raise ValueError(f"tool {self.name!r} does not accept parameters")
            return {}
        if allow_deferred:
            missing = {
                name
                for name, field in self.parameter_type.model_fields.items()
                if field.is_required() and name not in parameters
            }
            undeclared = missing - set(self.deferred_parameters)
            if undeclared:
                raise ValueError(f"missing required parameters: {sorted(undeclared)}")
            if missing:
                return _validate_partial_model(self.parameter_type, parameters)
        validated = self.parameter_type.model_validate(parameters, strict=True).model_dump(
            mode="python", exclude_none=True
        )
        if self.parameter_validation_function is not None:
            self.parameter_validation_function(validated, context or {})
        return validated

    def validate_parameter_patch(
        self,
        parameters: dict[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate supplied values without requiring the rest of a Tool's schema."""

        if self.parameter_type is None:
            if parameters:
                raise ValueError(f"tool {self.name!r} does not accept parameters")
            return {}
        validated = _validate_partial_model(self.parameter_type, parameters)
        if self.parameter_validation_function is not None and set(self.request_parameters) <= set(
            validated
        ):
            self.parameter_validation_function(validated, context or {})
        return validated

    def validate_result(self, run: Run, step: Step, result: Result) -> bool:
        """Apply a Tool-local verified-result check when the Tool declares one."""

        if self.result_validation_function is None:
            return True
        return self.result_validation_function(run, step, result) is True

    def applicable_repair_capabilities(self, parameters: dict[str, Any]) -> list[str]:
        """Return repair actions permitted for this Tool and its parameters."""

        if self.repair_capabilities_function is None:
            selected = list(self.repair_capabilities)
        else:
            selected = list(self.repair_capabilities_function(dict(parameters)))
        if not set(selected) <= set(self.repair_capabilities):
            raise ValueError("parameter adapter cannot expand the Tool repair capabilities")
        return selected

    def description_json(self) -> dict[str, Any]:
        description = self.model_dump(mode="json", exclude={"execute_function"})
        description["public_outputs"] = self.public_outputs()
        return description

    def public_outputs(self) -> list[dict[str, Any]]:
        """Return the Tool's single canonical public output directory."""

        from bg6022.output_contracts import canonical_public_outputs

        return canonical_public_outputs(self)


__all__ = [
    "Artifact",
    "AnswerGoal",
    "InputReference",
    "Plan",
    "Request",
    "Requirement",
    "RequirementInputBinding",
    "Result",
    "ResultTarget",
    "ResultProperty",
    "Run",
    "RunStatus",
    "Subject",
    "Step",
    "Tool",
]


def _validate_partial_model(
    model_type: type[BaseModel], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Validate supplied fields without inventing missing scientific values."""

    known = set(model_type.model_fields)
    extra = set(parameters) - known
    if extra:
        raise ValueError(f"extra inputs are not permitted: {sorted(extra)}")

    # ``FieldInfo.annotation`` does not include constraints kept in
    # ``FieldInfo.metadata`` (for example ge/le on iteration limits), and a
    # TypeAdapter would also skip model field validators.  Build a temporary
    # all-optional view that preserves both so deferred q/m values remain
    # absent while every supplied value is validated exactly as it would be in
    # the complete model.
    partial_fields: dict[str, tuple[Any, None]] = {}
    for name, field in model_type.model_fields.items():
        field_data = field.asdict()
        attributes = dict(field_data["attributes"])
        attributes.pop("default", None)
        attributes.pop("default_factory", None)
        annotation = Annotated[
            field_data["annotation"],
            *field_data["metadata"],
            Field(**attributes),
        ]
        partial_fields[name] = (annotation, None)
    partial_model = create_model(
        f"{model_type.__name__}PartialValidation",
        __base__=model_type,
        **partial_fields,
    )
    try:
        validated = partial_model.model_validate(parameters, strict=True)
    except Exception as error:
        errors_method = getattr(error, "errors", None)
        if callable(errors_method):
            try:
                errors = errors_method(include_url=False)
            except TypeError:
                errors = errors_method()
            details = "; ".join(str(item.get("msg") or "invalid parameter") for item in errors)
            if details:
                raise ValueError(details) from error
        raise ValueError(str(error)) from error
    return validated.model_dump(mode="python", exclude_none=True)
