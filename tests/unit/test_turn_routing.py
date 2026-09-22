from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, ResultTarget, Run, Step
from bg6022.molecule_identity import build_identity_constraint
from bg6022.planner import (
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
    validate_request_plan,
)
from bg6022.session import create_run, load_run, save_run, save_session, utc_now
from bg6022.tools.pubchem import PubChemLookup
from bg6022.tools.registry import build_registry


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


def _water_lookup() -> PubChemLookup:
    candidate = {
        "CID": 962,
        "Title": "water",
        "ConnectivitySMILES": "O",
        "MolecularFormula": "H2O",
        "Charge": 0,
    }
    return PubChemLookup(
        query="water",
        input_kind="name",
        url="test:water",
        candidates=(candidate,),
        raw_bytes=b"water",
        attempts=1,
        source_responses=({"url": "test:water", "raw_bytes": b"water"},),
        returned_cid_count=1,
    )


def _waiting_name_run(config: Any, session_id: str) -> Run:
    identity = build_identity_constraint(
        message="优化水",
        query="水",
        input_kind="name",
        name_evidence="水",
    )
    assert identity is not None
    request = Request(
        id="request_waiting_water",
        description="优化水",
        original_text="优化水",
        source="chat",
        operations=["Opt"],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
        structure_input={"molecule_identity": identity},
    )
    plan = Plan(
        id="plan_waiting_water",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "水", "input_kind": "name"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={"geometry": InputReference(step_id="geometry", port="geometry")},
            ),
        ],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
    )
    registry = build_registry(config)
    plan = validate_request_plan(request, plan, registry)
    run = Run(
        id="run_waiting_water",
        request=request,
        plan=plan,
        resources=config.resources,
        execution_permission=False,
        status="waiting",
        waiting_for="clarification",
        pending_data={
            "category": "molecule_name_not_found",
            "input_requirement": "molecule_identity",
            "step_id": "molecule",
            "raw_query": "水",
        },
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    save_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
            "pending_prompt": None,
        },
    )
    return run


def _waiting_formula_candidate_run(config: Any, session_id: str) -> Run:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    request = Request(
        id="request_waiting_c6h14",
        description="优化 C6H14",
        original_text="优化 C6H14",
        source="chat",
        operations=["Opt"],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
        explicit_parameters={"charge": 0, "multiplicity": 1},
        structure_input={"molecule_identity": identity},
    )
    plan = Plan(
        id="plan_waiting_c6h14",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "C6H14", "input_kind": "formula"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={"geometry": InputReference(step_id="geometry", port="geometry")},
            ),
        ],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
    )
    registry = build_registry(config)
    plan = validate_request_plan(request, plan, registry)
    run = Run(
        id="run_waiting_c6h14",
        request=request,
        plan=plan,
        resources=config.resources,
        execution_permission=False,
        status="waiting",
        waiting_for="clarification",
        pending_data={
            "category": "ambiguous_molecule",
            "input_requirement": "molecule_identity",
            "step_id": "molecule",
            "raw_query": "C6H14",
            "candidates": [
                {
                    "choice_id": "candidate_1",
                    "cid": 8058,
                    "title": "hexane",
                    "formula": "C6H14",
                    "canonical_smiles": "CCCCCC",
                    "isomeric_smiles": "CCCCCC",
                },
                {
                    "choice_id": "candidate_2",
                    "cid": 7892,
                    "title": "branched hexane",
                    "formula": "C6H14",
                    "canonical_smiles": "CC(C)CCC",
                    "isomeric_smiles": "CC(C)CCC",
                },
            ],
        },
        session_id=session_id,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    save_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
            "pending_prompt": None,
        },
    )
    return run


def _opt_freq_intake() -> IntakeOutput:
    return IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt", "Freq"],
        molecule_query="water",
        molecule_input_kind="name",
        molecule_name_evidence="水",
        explicit_parameters={"charge": 0, "multiplicity": 1},
        requested_results=["vibrational_frequencies", "frequency_complete"],
        structure_input={
            "required_bindings": [
                {
                    "consumer_operation": "Freq",
                    "input_port": "geometry",
                    "source_operation": "Opt",
                    "source_port": "optimized_geometry",
                }
            ]
        },
    )


def _opt_freq_plan(request: Request) -> PlanProposal:
    assert request.operations == ["Opt", "Freq"]
    return PlanProposal(
        steps=[
            PlanStepProposal(
                key="molecule",
                tool="resolve_molecule",
                parameters={"query": "water", "input_kind": "name"},
            ),
            PlanStepProposal(
                key="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputBindingProposal(step_key="molecule", port="molecule")},
            ),
            PlanStepProposal(
                key="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={"geometry": InputBindingProposal(step_key="geometry", port="geometry")},
            ),
            PlanStepProposal(
                key="freq",
                tool="frequency",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={
                    "geometry": InputBindingProposal(step_key="opt", port="optimized_geometry")
                },
            ),
        ],
        requested_results=[
            PlanTargetProposal(step_key="freq", field="vibrational_frequencies"),
            PlanTargetProposal(step_key="freq", check="frequency_complete"),
        ],
    )


def test_new_complete_request_is_not_swallowed_by_waiting_identity_run(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    old_run = _waiting_name_run(config, "session_new_request")
    registry = build_registry(config)
    seen: dict[str, Any] = {}

    def fake_intake(_llm, message: str, **kwargs: Any) -> IntakeOutput:
        assert message == "优化水，并计算频率"
        seen["pending_context"] = kwargs["pending_context"]
        return _opt_freq_intake()

    def fake_plan(_llm, request: Request, **_kwargs: Any) -> PlanProposal:
        return _opt_freq_plan(request)

    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr("bg6022.agent.plan_message", fake_plan)
    monkeypatch.setattr(
        "bg6022.tools.pubchem.fetch_pubchem", lambda *_args, **_kwargs: _water_lookup()
    )

    orca_calls: list[object] = []

    def fail_if_orca(*_args: Any, **_kwargs: Any) -> None:
        orca_calls.append(object())
        raise AssertionError("a new request must stop for confirmation before ORCA")

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fail_if_orca)
    agent = Agent(config, registry, llm=object(), session_id="session_new_request")

    response = agent.handle_message("优化水，并计算频率")

    assert response.run is not None
    assert response.run.id != old_run.id
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert response.run.request.operations == ["Opt", "Freq"]
    assert [step.tool for step in response.run.plan.steps] == [
        "resolve_molecule",
        "generate_geometry",
        "optimize_geometry",
        "frequency",
    ]
    assert response.run.plan.steps[-1].inputs["geometry"].step_id == response.run.plan.steps[-2].id
    assert response.run.plan.steps[-1].inputs["geometry"].port == "optimized_geometry"
    assert orca_calls == []
    assert seen["pending_context"]["identity_required"] is True
    assert seen["pending_context"]["raw_query"] == "水"
    persisted_old = load_run(config.data_root_path, old_run.id)
    assert persisted_old.status == "waiting"
    assert persisted_old.waiting_for == "clarification"
    assert persisted_old.pending_data["category"] == "molecule_name_not_found"


def test_new_complete_request_is_not_swallowed_by_waiting_formula_candidate_run(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    old_run = _waiting_formula_candidate_run(config, "session_new_formula_request")
    old_snapshot = {
        "request": old_run.request.model_dump(mode="json"),
        "plan": old_run.plan.model_dump(mode="json"),
        "requested_results": [
            target.model_dump(mode="json") for target in old_run.request.requested_results
        ],
        "pending_data": deepcopy(old_run.pending_data),
        "execution_permission": old_run.execution_permission,
    }
    registry = build_registry(config)
    seen: dict[str, Any] = {}

    def fake_intake(_llm, message: str, **kwargs: Any) -> IntakeOutput:
        assert message == "优化水，并计算频率"
        seen["pending_context"] = kwargs["pending_context"]
        return _opt_freq_intake()

    def fake_plan(_llm, request: Request, **_kwargs: Any) -> PlanProposal:
        return _opt_freq_plan(request)

    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr("bg6022.agent.plan_message", fake_plan)
    monkeypatch.setattr(
        "bg6022.tools.pubchem.fetch_pubchem", lambda *_args, **_kwargs: _water_lookup()
    )

    orca_calls: list[object] = []

    def fail_if_orca(*_args: Any, **_kwargs: Any) -> None:
        orca_calls.append(object())
        raise AssertionError("a new request must stop for confirmation before ORCA")

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fail_if_orca)
    agent = Agent(config, registry, llm=object(), session_id="session_new_formula_request")

    response = agent.handle_message("优化水，并计算频率")

    assert response.run is not None
    assert response.run.id != old_run.id
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert response.run.request.operations == ["Opt", "Freq"]
    assert [step.tool for step in response.run.plan.steps] == [
        "resolve_molecule",
        "generate_geometry",
        "optimize_geometry",
        "frequency",
    ]
    assert response.run.request.requested_results == [
        ResultTarget(field="vibrational_frequencies"),
        ResultTarget(check="frequency_complete"),
    ]
    assert response.run.plan.steps[-1].inputs["geometry"].step_id == response.run.plan.steps[-2].id
    assert response.run.plan.steps[-1].inputs["geometry"].port == "optimized_geometry"
    assert orca_calls == []
    assert seen["pending_context"]["identity_required"] is True
    assert seen["pending_context"]["category"] == "ambiguous_molecule"
    assert seen["pending_context"]["raw_query"] == "C6H14"
    assert seen["pending_context"]["candidates"] == [
        {"choice_id": "candidate_1", "cid": 8058, "title": "hexane", "formula": "C6H14"},
        {
            "choice_id": "candidate_2",
            "cid": 7892,
            "title": "branched hexane",
            "formula": "C6H14",
        },
    ]

    persisted_old = load_run(config.data_root_path, old_run.id)
    assert persisted_old.request.model_dump(mode="json") == old_snapshot["request"]
    assert persisted_old.plan.model_dump(mode="json") == old_snapshot["plan"]
    assert [
        target.model_dump(mode="json") for target in persisted_old.request.requested_results
    ] == old_snapshot["requested_results"]
    assert persisted_old.pending_data == old_snapshot["pending_data"]
    assert persisted_old.execution_permission == old_snapshot["execution_permission"]
    assert persisted_old.execution_permission is False


def test_identity_supplement_updates_same_waiting_run_with_structured_intake(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    old_run = _waiting_name_run(config, "session_identity_supplement")
    original_identity = deepcopy(old_run.request.structure_input["molecule_identity"])
    original_targets = [
        target.model_dump(mode="json") for target in old_run.request.requested_results
    ]
    original_execution_permission = old_run.execution_permission
    registry = build_registry(config)
    seen: dict[str, Any] = {}

    def fake_intake(_llm, message: str, **kwargs: Any) -> IntakeOutput:
        assert message == "water"
        seen["pending_context"] = kwargs["pending_context"]
        return IntakeOutput(
            intent="chemistry_compute",
            molecule_query="water",
            molecule_input_kind="name",
            molecule_name_evidence="water",
            pending_action="supplement_identity",
            pending_action_evidence="water",
        )

    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr(
        "bg6022.tools.pubchem.fetch_pubchem", lambda *_args, **_kwargs: _water_lookup()
    )

    agent = Agent(config, registry, llm=object(), session_id="session_identity_supplement")

    def fake_advance(run: Run, *, cancel=None):
        run.status = "waiting"
        run.waiting_for = "confirmation"
        run.pending_data = {"operation": "Opt"}
        save_run(config.data_root_path, run)
        return None

    monkeypatch.setattr(agent, "advance", fake_advance)
    response = agent.handle_message("water")

    assert response.run is not None
    assert response.run.id == old_run.id
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    identity = response.run.request.structure_input["molecule_identity"]
    assert identity["raw_query"] == original_identity["raw_query"]
    assert identity["raw_query"] == "水"
    assert identity["lookup_query"] == "water"
    assert response.run.request.operations == old_run.request.operations == ["Opt"]
    assert [
        target.model_dump(mode="json") for target in response.run.request.requested_results
    ] == original_targets
    assert response.run.execution_permission == original_execution_permission
    assert response.run.execution_permission is False
    assert response.run.plan.steps[0].parameters == {"query": "water", "input_kind": "name"}
    assert seen["pending_context"]["identity_required"] is True
    assert seen["pending_context"]["can_replace_identity"] is True


def test_identity_supplement_keeps_real_advance_until_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    old_run = _waiting_name_run(config, "session_identity_real_advance")
    registry = build_registry(config)

    def fake_intake(_llm, message: str, **_kwargs: Any) -> IntakeOutput:
        assert message == "water"
        return IntakeOutput(
            intent="chemistry_compute",
            molecule_query="water",
            molecule_input_kind="name",
            molecule_name_evidence="water",
            pending_action="supplement_identity",
            pending_action_evidence="water",
        )

    monkeypatch.setattr("bg6022.agent.intake_message", fake_intake)
    monkeypatch.setattr(
        "bg6022.tools.pubchem.fetch_pubchem", lambda *_args, **_kwargs: _water_lookup()
    )
    orca_calls: list[object] = []

    def fail_if_orca(*_args: Any, **_kwargs: Any) -> None:
        orca_calls.append(object())
        raise AssertionError("identity supplement must stop before ORCA confirmation")

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fail_if_orca)
    agent = Agent(config, registry, llm=object(), session_id="session_identity_real_advance")

    response = agent.handle_message("water")

    assert response.run is not None
    assert response.run.id == old_run.id
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert response.run.plan.steps[0].parameters == {"query": "water", "input_kind": "name"}
    assert response.run.current_results.keys() >= {"molecule", "geometry"}
    assert response.run.step_status["molecule"] == "succeeded"
    assert response.run.step_status["geometry"] == "succeeded"
    assert orca_calls == []
