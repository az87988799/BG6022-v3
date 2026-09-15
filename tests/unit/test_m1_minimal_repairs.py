from __future__ import annotations

from pathlib import Path

import pytest

from bg6022.agent import Agent
from bg6022.answer import render_run
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Run, Step, Tool
from bg6022.orca.parser import inspect_attempt
from bg6022.orca.profiles import resolve_parameters
from bg6022.planner import (
    ElectronicStateCandidate,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    filter_user_explicit_parameters,
    normalize_user_explicit_parameters,
    proposal_to_plan,
    validate_request_plan,
)
from bg6022.session import utc_now
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.registry import ToolRegistry, build_registry

FIXTURE = Path(__file__).parents[1] / "fixtures" / "orca_6_1_1_water_opt"


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_current_default_runtime_budget_is_4_core_1024_mb(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.resources["cores"] == 4
    assert config.resources["memory_mb"] == 1024
    assert config.resources["maxcore_mb"] == 192
    assert config.resources["max_concurrent_jobs"] == 1


def test_generic_compute_tool_does_not_inherit_orca_preparation_or_budget(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    tool = Tool(
        name="external_compute",
        description="A compute Tool unrelated to ORCA electronic-state preparation.",
        requires_compute_permission=True,
        parameter_preparation="none",
        execution_budget="none",
    )
    registry = ToolRegistry([tool])
    agent = Agent(config, registry)
    run = agent._create_chat_run(
        Request(id="request", description="run external computation"),
        Plan(
            id="plan",
            request_id="request",
            steps=[Step(id="compute", tool=tool.name)],
        ),
    )

    agent.advance(run)

    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    assert agent._reserve_attempt(run, run.plan.steps[0], tool)
    assert run.extra_orca_executions == 0


def test_parser_uses_geometry_settings_and_records_the_line() -> None:
    initial = parse_xyz_bytes((FIXTURE / "geometry.xyz").read_bytes())
    facts = inspect_attempt(
        operation="Opt",
        stdout=FIXTURE / "stdout.out",
        stderr=b"",
        exit_code=0,
        runner_status="succeeded",
        input_geometry=initial,
        effective_geom_maxiter=None,
        output_xyz=FIXTURE / "input.xyz",
    )

    assert facts.effective_geom_maxiter == 50
    assert facts.effective_geom_maxiter_source == "orca_output"
    assert facts.source_locations["effective_geom_maxiter"] == 239


def test_parser_rejects_conflicting_input_and_output_iteration_limits() -> None:
    initial = parse_xyz_bytes(b"1\nhydrogen\nH 0 0 0\n")
    facts = inspect_attempt(
        operation="Opt",
        stdout=(b"Geometry optimization settings:\nMax. no of cycles        MaxIter  .... 50\n"),
        stderr=b"",
        exit_code=1,
        runner_status="failed",
        input_geometry=initial,
        effective_geom_maxiter=1,
    )

    assert facts.effective_geom_maxiter is None
    assert facts.effective_geom_maxiter_error is not None
    assert facts.source_locations["effective_geom_maxiter_error"]


def test_model_qm_values_are_not_scientific_authority() -> None:
    assert filter_user_explicit_parameters("optimize water", {"charge": 0, "multiplicity": 1}) == {}
    assert filter_user_explicit_parameters(
        "charge -1, multiplicity +1", {"charge": -1, "multiplicity": 1}
    ) == {"charge": -1, "multiplicity": 1}
    assert filter_user_explicit_parameters("charge +1", {"charge": -1}) == {"charge": 1}

    resolution = resolve_parameters(
        {},
        {"formal_charge": 0, "radical_electrons": 0, "atom_symbols": ["O", "O"]},
        {"charge": 0, "multiplicity": 1},
        {"method_profile": "r2scan3c", "environment": "gas"},
    )
    assert resolution.effective_parameters["charge"] == 0
    assert "multiplicity" in resolution.missing_fields


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("优化三重态氧气", {"multiplicity": 3}),
        ("这个分子是中性的", {"charge": 0}),
        ("把电荷设为0", {"charge": 0}),
        ("多重度为3，电荷为0", {"charge": 0, "multiplicity": 3}),
    ],
)
def test_chinese_electronic_state_statements_are_parsed(
    message: str, expected: dict[str, int]
) -> None:
    assert filter_user_explicit_parameters(message, {"charge": -1, "multiplicity": 1}) == expected


def test_negated_or_conflicting_spin_words_do_not_become_parameters() -> None:
    assert filter_user_explicit_parameters("这个分子不是中性的", {"charge": 0}) == {}
    assert filter_user_explicit_parameters("不是三重态", {"multiplicity": 3}) == {}
    assert filter_user_explicit_parameters("单重态还是三重态？", {"multiplicity": 1}) == {}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("明确设电荷为0", {"charge": 0}),
        ("明确使用三重态", {"multiplicity": 3}),
        ("把电荷从0改为+1", {"charge": 1}),
        ("不要用单重态，使用三重态", {"multiplicity": 3}),
    ],
)
def test_parameter_normalization_preserves_the_current_affirmative_value(
    message: str, expected: dict[str, int]
) -> None:
    result = normalize_user_explicit_parameters(
        message,
        {"charge": -1, "multiplicity": 1},
        [
            ElectronicStateCandidate(field="charge", raw_value="+1", evidence=message),
        ]
        if "电荷从0改为+1" in message
        else [],
    )

    assert result.explicit_parameters == expected
    for field, value in expected.items():
        assert result.states[field].status == "set"
        assert result.states[field].value == value


@pytest.mark.parametrize(
    ("message", "field"),
    [
        ("多重度设为1.5", "multiplicity"),
        ("电荷设为1e2", "charge"),
        ("多重度设为true", "multiplicity"),
        ("多重度设为0", "multiplicity"),
    ],
)
def test_invalid_electronic_state_is_not_coerced_or_dropped_as_absent(
    message: str, field: str
) -> None:
    result = normalize_user_explicit_parameters(message, {field: 1})

    assert result.states[field].status == "invalid"
    assert field not in result.explicit_parameters
    assert field in result.clarification_fields


def test_model_qm_candidates_need_exact_user_evidence_and_omitted_fields_stay_absent() -> None:
    result = normalize_user_explicit_parameters(
        "优化水",
        {"charge": 0, "multiplicity": 1, "method_profile": "r2scan3c"},
        [ElectronicStateCandidate(field="charge", raw_value="0", evidence="优化水")],
    )

    assert result.explicit_parameters == {"method_profile": "r2scan3c"}
    assert result.states["charge"].status == "absent"
    assert result.states["multiplicity"].status == "absent"


def test_one_field_parameter_update_does_not_erase_other_effective_values() -> None:
    result = normalize_user_explicit_parameters(
        "把电荷从0改为+1",
        {"charge": 0, "multiplicity": 2},
    )
    current = {"charge": 0, "multiplicity": 2, "method_profile": "r2scan3c"}
    updated = {**current, **result.explicit_parameters}

    assert result.explicit_parameters == {"charge": 1}
    assert updated == {"charge": 1, "multiplicity": 2, "method_profile": "r2scan3c"}


def test_conflicting_parameter_statements_require_clarification() -> None:
    result = normalize_user_explicit_parameters("多重度为1或3", {"multiplicity": 1})

    assert result.states["multiplicity"].status == "ambiguous"
    assert result.clarification_fields == ("multiplicity",)
    assert "multiplicity" not in result.explicit_parameters


def test_request_target_and_operation_cannot_be_dropped() -> None:
    registry = build_registry()
    request = Request(
        id="request_sp",
        description="single point energy",
        operation="SP",
        requested_results=[ResultTarget(field="energy")],
    )
    preparation_only = Plan(
        id="plan_prepare",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            )
        ],
    )
    with pytest.raises(ValueError, match="requested SP calculation"):
        validate_request_plan(request, preparation_only, registry)


def test_opt_geometry_target_means_optimized_geometry_not_initial_geometry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(
        id="request_opt_geometry",
        description="optimize water and return geometry",
        operation="Opt",
        requested_results=[ResultTarget(field="geometry")],
        explicit_parameters={"charge": 0, "multiplicity": 1},
    )
    plan = Plan(
        id="plan_opt_geometry",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            ),
            Step(
                id="generate",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={"method_profile": "r2scan3c", "environment": "gas"},
                inputs={"geometry": InputReference(step_id="generate", port="geometry")},
            ),
        ],
        requested_results=[ResultTarget(step_id="generate", port="geometry")],
    )

    with pytest.raises(ValueError, match="does not cover the requested result: geometry"):
        validate_request_plan(request, plan, registry)

    # Even if an invalid plan reaches the runtime boundary, a successful
    # geometry-generation Step cannot complete an Opt Request.
    agent = Agent(config, registry)
    run = agent._create_chat_run(request, registry.validate_plan(plan))
    agent.advance(run)
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    assert "generate" in run.current_results
    assert "opt" not in run.current_results


def test_legacy_output_target_is_narrowly_converted_to_a_port() -> None:
    registry = build_registry()
    plan = Plan(
        id="plan_opt",
        request_id="request_opt",
        steps=[
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={"charge": 0, "multiplicity": 1},
                inputs={"geometry": InputReference(artifact_id="artifact_geometry")},
            )
        ],
        requested_results=["optimized_geometry"],
    )

    normalized = registry.validate_plan(plan)
    assert normalized.requested_results[0].field is None
    assert normalized.requested_results[0].port == "optimized_geometry"


def test_unknown_planner_target_is_a_feedback_error() -> None:
    request = Request(id="request", description="prepare")
    proposal = PlanProposal(
        steps=[
            PlanStepProposal(
                key="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            )
        ],
        requested_results=[PlanTargetProposal(step_key="missing", field="molecule_formula")],
    )

    with pytest.raises(ValueError, match="unknown step key"):
        proposal_to_plan(request, proposal, build_registry(), plan_id="plan")


def test_waiting_render_never_promotes_a_preparation_result_to_run_success() -> None:
    request = Request(id="request", description="optimize")
    step = Step(id="geometry", tool="generate_geometry")
    run = Run(
        id="run",
        request=request,
        plan=Plan(id="plan", request_id=request.id, steps=[step]),
        resources={"cores": 4},
        status="waiting",
        waiting_for="confirmation",
        pending_data={"operation": "Opt", "parameters": {"charge": 0, "multiplicity": 1}},
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        values={"geometry_atom_count": 3},
    )

    text = render_run(run, result)
    assert "waiting for confirmation" in text
    assert "succeeded for step" not in text


def test_failed_run_can_render_valid_partial_results_and_repair_history() -> None:
    request = Request(
        id="request",
        description="optimize water and return energy",
        operation="Opt",
        requested_results=[ResultTarget(field="opt_final_electronic_energy")],
    )
    step = Step(id="opt", tool="optimize_geometry")
    run = Run(
        id="run",
        request=request,
        plan=Plan(
            id="plan",
            request_id=request.id,
            steps=[step],
            requested_results=request.requested_results,
        ),
        resources={"cores": 4},
        status="failed",
        pending_data={"category": "scf_not_converged", "reason": "SCF stopped at its limit"},
        repair_records=[
            {
                "action": "increase_scf_maxiter",
                "failed_attempt": 1,
                "parameter_patch": {"scf_maxiter": 240},
            }
        ],
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=2,
        status="failed",
        diagnostics={"category": "scf_not_converged", "reason": "SCF stopped at its limit"},
        attempt_relative_path="opt/attempt-02",
    )
    partial_facts = [
        {
            "task_key": "run:prior",
            "system": "水分子（H₂O）",
            "step_tool": "single_point",
            "name": "sp_electronic_energy",
            "kind": "field",
            "value": {"value": -76.4, "unit": "Eh", "token": "-76.4"},
            "expected_type": "Eh",
            "metadata": {"label": "已验证的单点电子能"},
        }
    ]

    text = render_run(
        run,
        result,
        partial_facts=partial_facts,
        repairs=run.repair_records,
        incomplete_targets=["几何优化", "优化后的电子能"],
    )

    assert "任务未全部完成" in text
    assert "尚未完成：几何优化、优化后的电子能" in text
    assert "已尝试的修复" in text and "SCF 迭代上限调整为 240" in text
    assert "停止原因：SCF stopped at its limit" in text
    assert "已验证的单点电子能为 **-76.4 Eh**" in text
    assert "attempt-02" not in text and "result.json" not in text

    budget_run = run.model_copy(update={"pending_data": {"budget_exhausted": "max_plan_revisions"}})
    budget_text = render_run(budget_run, result)
    assert "budget_exhausted" in budget_text
    assert "已达到最大计划修订次数" in budget_text


def test_rejected_method_update_does_not_mutate_waiting_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(id="request", description="optimize", operation="Opt")
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={"method_profile": "r2scan3c", "environment": "gas"},
        inputs={"geometry": InputReference(artifact_id="artifact_geometry")},
    )
    run = Run(
        id="run",
        request=request,
        plan=Plan(id="plan", request_id=request.id, steps=[step]),
        resources=config.resources,
        status="waiting",
        waiting_for="clarification",
        pending_data={"step_id": step.id, "missing_fields": ["charge"]},
        budget={"max_attempts_per_science_step": 3, "max_extra_orca_executions": 3},
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    original_request = run.request.model_dump(mode="json")
    original_plan = run.plan.model_dump(mode="json")
    original_pending = dict(run.pending_data)

    response = Agent(config, registry)._apply_parameter_update(run, {"method_profile": "b3lyp"})

    assert "rejected" in response.text
    assert run.request.model_dump(mode="json") == original_request
    assert run.plan.model_dump(mode="json") == original_plan
    assert run.pending_data == original_pending
    assert run.status == "waiting"


def test_repair_scope_disables_iteration_increase_when_user_forbids_it(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    request = Request(
        id="request",
        description="optimize water; do not increase the iteration limit",
        operation="Opt",
    )
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={"charge": 0, "multiplicity": 1},
        inputs={"geometry": InputReference(artifact_id="artifact_geometry")},
    )
    run = Run(
        id="run",
        request=request,
        plan=Plan(id="plan", request_id=request.id, steps=[step]),
        resources=config.resources,
        budget={"max_attempts_per_science_step": 3, "max_extra_orca_executions": 3},
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    scope = Agent(config, build_registry(config))._repair_scope(run)
    assert scope["iteration_increase_allowed"] is False
    assert scope["steps"][step.id]["actions"] == {}
