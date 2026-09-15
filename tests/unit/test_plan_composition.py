from __future__ import annotations

import pytest

from bg6022.models import (
    Request,
    ResultTarget,
)
from bg6022.planner import (
    GoalCheckProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    proposal_to_plan,
)
from bg6022.tools.registry import ToolRegistry, build_registry


def _compose(request: Request, proposal: PlanProposal, registry: ToolRegistry, *, plan_id: str):
    return proposal_to_plan(
        request,
        proposal,
        registry,
        plan_id=plan_id,
        artifact_aliases={
            "provided_geometry": "provided_geometry",
            "initial_geometry": "initial_geometry",
            "input_geometry": "input_geometry",
        },
    )


def _frequency_registry() -> ToolRegistry:
    """Return the production registry containing its single Freq Tool."""

    return build_registry()


def _compute_request(
    operations: list[str], requested_results: list[ResultTarget] | None = None
) -> Request:
    return Request(
        id="request_composition",
        description="offline plan composition contract",
        operations=operations,
        requested_results=requested_results or [],
    )


def _proposal_step(
    key: str,
    tool: str,
    *,
    inputs: dict | None = None,
    goal_checks: list[GoalCheckProposal] | None = None,
) -> PlanStepProposal:
    parameters = (
        {"charge": 0, "multiplicity": 1}
        if tool
        in {
            "single_point",
            "optimize_geometry",
            "frequency",
        }
        else {}
    )
    return PlanStepProposal(
        key=key,
        tool=tool,
        parameters=parameters,
        inputs=inputs or {},
        goal_checks=goal_checks or [],
    )


def test_intake_and_plan_schemas_accept_composed_operations_and_targets() -> None:
    intake = IntakeOutput.model_validate(
        {
            "intent": "chemistry_compute",
            "operations": ["Opt", "Freq", "SP"],
            "requested_results": ["frequency_complete", "sp_electronic_energy"],
        }
    )
    proposal = PlanProposal.model_validate(
        {
            "steps": [
                {"key": "opt", "tool": "optimize_geometry"},
                {
                    "key": "freq",
                    "tool": "frequency",
                    "goal_checks": [
                        {
                            "source_step_key": "opt",
                            "check": "optimization_converged",
                            "required_status": "passed",
                        }
                    ],
                },
                {
                    "key": "sp",
                    "tool": "single_point",
                    "goal_checks": [
                        {
                            "source_step_key": "freq",
                            "check": "local_minimum_supported",
                        }
                    ],
                },
            ],
            "requested_results": [
                {"step_key": "freq", "check": "frequency_complete"},
                {"step_key": "sp", "field": "sp_electronic_energy"},
            ],
        }
    )

    assert intake.operations == ["Opt", "Freq", "SP"]
    assert proposal.steps[1].goal_checks[0].source_step_key == "opt"
    assert proposal.requested_results[0].check == "frequency_complete"


def test_single_point_plan_covers_only_sp_and_maps_energy_alias() -> None:
    request = _compute_request(["SP"], [ResultTarget(field="energy")])
    registry = build_registry()
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"artifact_alias": "provided_geometry"}},
            )
        ],
        requested_results=[PlanTargetProposal(step_key="sp", field="sp_electronic_energy")],
    )

    plan = _compose(request, proposal, registry, plan_id="plan_sp")

    assert request.operations == ["SP"]
    operations = [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ]
    assert operations == ["SP"]
    assert plan.steps[0].inputs["geometry"].artifact_id == "provided_geometry"
    assert plan.requested_results == [
        ResultTarget(step_id=plan.steps[0].id, field="sp_electronic_energy")
    ]


def test_opt_plan_covers_only_opt_and_maps_energy_alias() -> None:
    request = _compute_request(
        ["Opt"],
        [ResultTarget(field="energy"), ResultTarget(port="optimized_geometry")],
    )
    registry = build_registry()
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "initial_geometry"}},
            )
        ],
        requested_results=[
            PlanTargetProposal(step_key="opt", field="opt_final_electronic_energy"),
            PlanTargetProposal(step_key="opt", port="optimized_geometry"),
        ],
    )

    plan = _compose(request, proposal, registry, plan_id="plan_opt")
    operations = [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ]

    assert operations == ["Opt"]
    assert [target.field or target.port for target in plan.requested_results] == [
        "opt_final_electronic_energy",
        "optimized_geometry",
    ]


def test_frequency_only_plan_uses_supplied_geometry_without_an_opt_protocol() -> None:
    request = Request(
        id="request_frequency_only",
        description="calculate frequencies for the supplied water geometry",
        operations=["Freq"],
        requested_results=[
            ResultTarget(field="frequency"),
            ResultTarget(check="frequency_complete"),
        ],
        structure_input={
            "xyz_text": (
                "3\nwater\n"
                "O 0.000000 0.000000 0.000000\n"
                "H 0.758602 0.000000 0.504284\n"
                "H -0.758602 0.000000 0.504284\n"
            )
        },
        source="chat",
    )
    registry = _frequency_registry()
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "freq",
                "frequency",
                inputs={"geometry": {"artifact_alias": "request_geometry"}},
            )
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", field="vibrational_frequencies"),
            PlanTargetProposal(step_key="freq", check="frequency_complete"),
        ],
    )

    plan = proposal_to_plan(
        request,
        proposal,
        registry,
        plan_id="plan_frequency_only",
        artifact_aliases={"request_geometry": "request_geometry"},
    )

    assert registry.validate_plan(plan) == plan
    assert [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ] == ["Freq"]
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "frequency"
    assert plan.steps[0].inputs["geometry"].artifact_id == "request_geometry"
    assert plan.steps[0].goal_checks == []
    assert all(target.check != "local_minimum_supported" for target in plan.requested_results)
    assert [target.field or target.check for target in plan.requested_results] == [
        "vibrational_frequencies",
        "frequency_complete",
    ]


def test_opt_freq_plan_covers_frequency_goal_without_adding_sp() -> None:
    request = _compute_request(
        ["Opt", "Freq"],
        [ResultTarget(field="frequency"), ResultTarget(check="frequency_complete")],
    )
    registry = _frequency_registry()
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "initial_geometry"}},
            ),
            _proposal_step(
                "freq",
                "frequency",
                inputs={"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
            ),
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", field="vibrational_frequencies"),
            PlanTargetProposal(step_key="freq", check="frequency_complete"),
        ],
    )

    plan = _compose(request, proposal, registry, plan_id="plan_opt_freq")
    operations = [
        operation for step in plan.steps for operation in registry.get(step.tool).operations
    ]

    assert operations == ["Opt", "Freq"]
    assert all(step.tool != "single_point" for step in plan.steps)
    assert plan.steps[1].inputs["geometry"].step_id == plan.steps[0].id
    assert plan.steps[1].inputs["geometry"].port == "optimized_geometry"


def test_opt_freq_independent_sp_plan_uses_opt_geometry_and_check_references() -> None:
    request = _compute_request(
        ["Opt", "Freq", "SP"],
        [
            ResultTarget(field="frequency"),
            ResultTarget(check="local_minimum_supported"),
            ResultTarget(field="sp_energy"),
        ],
    )
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "initial_geometry"}},
            ),
            _proposal_step(
                "freq",
                "frequency",
                inputs={"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
            ),
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
                goal_checks=[
                    GoalCheckProposal(
                        source_step_key="freq",
                        check="local_minimum_supported",
                    )
                ],
            ),
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", field="vibrational_frequencies"),
            PlanTargetProposal(step_key="freq", check="local_minimum_supported"),
            PlanTargetProposal(step_key="sp", field="sp_electronic_energy"),
        ],
    )

    plan = _compose(request, proposal, _frequency_registry(), plan_id="plan_opt_freq_sp")
    opt, freq, sp = plan.steps

    assert ["Opt", "Freq", "SP"] == [
        operation
        for step in plan.steps
        for operation in _frequency_registry().get(step.tool).operations
    ]
    assert freq.inputs["geometry"].step_id == opt.id
    assert freq.inputs["geometry"].port == "optimized_geometry"
    assert sp.inputs["geometry"].step_id == opt.id
    assert sp.inputs["geometry"].port == "optimized_geometry"
    assert sp.goal_checks[0].source_step_id == freq.id
    assert [target.check or target.field for target in plan.requested_results] == [
        "vibrational_frequencies",
        "local_minimum_supported",
        "sp_electronic_energy",
    ]


def test_requested_local_minimum_check_requires_a_gate_on_every_sp_step() -> None:
    request = _compute_request(
        ["Opt", "Freq", "SP"],
        [
            ResultTarget(check="local_minimum_supported"),
            ResultTarget(field="sp_energy"),
        ],
    )
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "initial_geometry"}},
            ),
            _proposal_step(
                "freq",
                "frequency",
                inputs={"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
            ),
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"step_key": "opt", "port": "optimized_geometry"}},
            ),
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", check="local_minimum_supported"),
            PlanTargetProposal(step_key="sp", field="sp_electronic_energy"),
        ],
    )

    with pytest.raises(ValueError, match="must require local_minimum_supported"):
        _compose(request, proposal, _frequency_registry(), plan_id="plan_missing_gate")


@pytest.mark.parametrize(
    ("requested_operations", "planned_tools", "message"),
    [
        (["Opt", "SP"], ["optimize_geometry"], "does not cover the requested SP calculation"),
        (["SP"], ["optimize_geometry", "single_point"], "adds unrequested operation.*Opt"),
    ],
)
def test_plan_rejects_missing_or_extra_operations(
    requested_operations: list[str], planned_tools: list[str], message: str
) -> None:
    request = _compute_request(requested_operations)
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                f"step_{index}",
                tool,
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
            )
            for index, tool in enumerate(planned_tools)
        ]
    )

    with pytest.raises(ValueError, match=message):
        _compose(request, proposal, build_registry(), plan_id="plan_invalid_ops")


def test_plan_rejects_unknown_goal_check_step_reference() -> None:
    request = _compute_request(["SP"])
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
                goal_checks=[
                    GoalCheckProposal(
                        source_step_key="missing",
                        check="frequency_complete",
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValueError, match="goal check references unknown step key"):
        _compose(request, proposal, build_registry(), plan_id="plan_bad_goal_ref")


def test_plan_rejects_bad_input_port_reference() -> None:
    request = _compute_request(["SP"])
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
            ),
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"step_key": "opt", "port": "missing_port"}},
            ),
        ]
    )

    with pytest.raises(ValueError, match="undeclared output port"):
        _compose(request, proposal, build_registry(), plan_id="plan_bad_refs")


def test_plan_rejects_undeclared_goal_check_name() -> None:
    request = _compute_request(["Freq", "SP"])
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "freq",
                "frequency",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
            ),
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
                goal_checks=[GoalCheckProposal(source_step_key="freq", check="unknown_check")],
            ),
        ]
    )

    with pytest.raises(ValueError, match="undeclared scientific check"):
        _compose(
            request,
            proposal,
            _frequency_registry(),
            plan_id="plan_bad_check_name",
        )


def test_plan_rejects_requested_result_target_with_unknown_step() -> None:
    request = _compute_request(["SP"])
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "sp",
                "single_point",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
            )
        ],
        requested_results=[PlanTargetProposal(step_key="missing", field="sp_electronic_energy")],
    )

    with pytest.raises(ValueError, match="result references unknown step key"):
        _compose(request, proposal, build_registry(), plan_id="plan_bad_target")


def test_plan_rejects_missing_user_requested_result() -> None:
    request = _compute_request(["Opt"], [ResultTarget(field="energy")])
    proposal = PlanProposal(
        steps=[
            _proposal_step(
                "opt",
                "optimize_geometry",
                inputs={"geometry": {"artifact_alias": "input_geometry"}},
            )
        ]
    )

    with pytest.raises(ValueError, match="does not cover the requested result: energy"):
        _compose(request, proposal, build_registry(), plan_id="plan_missing_result")
