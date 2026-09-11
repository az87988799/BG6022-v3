from __future__ import annotations

from pathlib import Path

import pytest

from bg6022.agent import Agent
from bg6022.answer import render_run
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Run, Step
from bg6022.orca.parser import inspect_attempt
from bg6022.orca.profiles import resolve_parameters
from bg6022.planner import (
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    filter_user_explicit_parameters,
    proposal_to_plan,
    validate_request_plan,
)
from bg6022.session import utc_now
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.registry import build_registry

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
    assert filter_user_explicit_parameters("charge +1", {"charge": -1}) == {}

    resolution = resolve_parameters(
        {},
        {"formal_charge": 0, "radical_electrons": 0, "atom_symbols": ["O", "O"]},
        {"charge": 0, "multiplicity": 1},
        {"method_profile": "r2scan3c", "environment": "gas"},
    )
    assert resolution.effective_parameters["charge"] == 0
    assert "multiplicity" in resolution.missing_fields


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
