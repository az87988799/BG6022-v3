from __future__ import annotations

from types import SimpleNamespace

from bg6022.canonicalize import (
    canonicalize_modification,
    canonicalize_semantic_request,
    semantic_to_intake,
)
from bg6022.semantic import SemanticProposal
from bg6022.tools.registry import build_registry


def _subject():
    return {
        "key": "subject_1",
        "query": "water",
        "input_kind": "name",
        "evidence": "water",
    }


def _proposal(*, tasks, relations=()):
    return SemanticProposal.model_validate(
        {
            "mode": "compute",
            "subjects": [_subject()],
            "tasks": tasks,
            "relations": list(relations),
        },
        strict=True,
    )


def _task(key, capability, *, method=None, properties=()):
    return {
        "key": key,
        "subject_key": "subject_1",
        "capability": capability,
        "method_request": method,
        "parameters": {},
        "requested_properties": list(properties),
    }


def test_pbe0_is_resolved_by_program_and_terminal_output_is_derived():
    registry = build_registry()
    proposal = _proposal(
        tasks=[_task("t1", "optimize_geometry", method="PBE0", properties=["energy"])]
    )

    request = canonicalize_semantic_request(
        "Optimize water with PBE0 and output its energy.",
        proposal,
        request_id="request_pbe0",
        registry=registry,
    )
    requirement = request.requirements[0]

    assert requirement.parameters["method_profile"] == "pbe0_d3bj_def2svp"
    assert requirement.constraints["method_resolution"]["status"] == "proposed"
    assert requirement.outputs == ["opt_final_electronic_energy"]


def test_chinese_name_uses_english_lookup_and_preserves_original_evidence():
    semantic = SemanticProposal.model_validate(
        {
            "mode": "compute",
            "subjects": [
                {
                    "key": "ethanol",
                    "query": "ethanol",
                    "input_kind": "name",
                    "evidence": "乙醇",
                }
            ],
            "tasks": [
                {
                    "key": "opt",
                    "subject_key": "ethanol",
                    "capability": "optimize_geometry",
                    "method_request": "r²SCAN-3c",
                    "parameters": {},
                    "requested_properties": [],
                }
            ],
            "relations": [],
        },
        strict=True,
    )

    intake = semantic_to_intake("优化乙醇", semantic, registry=build_registry())
    subject = intake.subjects["ethanol"]

    assert subject.molecule_query == "ethanol"
    assert subject.molecule_name_evidence == "乙醇"
    assert subject.molecule_input_kind == "name"


def test_dual_opt_compare_derives_shared_output_and_answer_goal():
    registry = build_registry()
    proposal = _proposal(
        tasks=[
            _task("t1", "optimize_geometry", method="r²SCAN-3c"),
            _task("t2", "optimize_geometry", method="PBE0"),
        ],
        relations=[{"type": "compare", "tasks": ["t1", "t2"], "property": "energy"}],
    )

    request = canonicalize_semantic_request(
        "Compare PBE0 and r²SCAN-3c optimized water energies side by side.",
        proposal,
        request_id="request_compare",
        registry=registry,
    )

    assert [item.outputs for item in request.requirements] == [
        ["opt_final_electronic_energy"],
        ["opt_final_electronic_energy"],
    ]
    assert request.answer_goals[0].output == "opt_final_electronic_energy"
    assert request.answer_goals[0].mode == "side_by_side"


def test_opt_to_freq_relation_binds_typed_geometry_output():
    registry = build_registry()
    proposal = _proposal(
        tasks=[
            _task("t1", "optimize_geometry", method="r²SCAN-3c"),
            _task("t2", "frequency", method="r²SCAN-3c", properties=["frequencies"]),
        ],
        relations=[
            {
                "type": "use_output",
                "source_task": "t1",
                "target_task": "t2",
                "property": "geometry",
            }
        ],
    )

    request = canonicalize_semantic_request(
        "Optimize water, then use that geometry for frequencies.",
        proposal,
        request_id="request_opt_freq",
        registry=registry,
    )
    source, target = request.requirements

    assert target.input_bindings["geometry"].source_requirement_id == source.id
    assert target.input_bindings["geometry"].source_port == "optimized_geometry"
    assert source.outputs == []
    assert target.outputs == ["vibrational_frequencies"]


def test_dual_sp_difference_inserts_same_geometry_derived_requirement():
    registry = build_registry()
    proposal = _proposal(
        tasks=[
            _task("t1", "single_point", method="r²SCAN-3c"),
            _task("t2", "single_point", method="PBE0"),
        ],
        relations=[{"type": "difference", "tasks": ["t1", "t2"]}],
    )

    request = canonicalize_semantic_request(
        "Calculate the PBE0 minus r²SCAN-3c single-point energy difference for water.",
        proposal,
        request_id="request_difference",
        registry=registry,
    )
    left, right, difference = request.requirements

    assert left.input_bindings["geometry"].artifact_alias == "initial_geometry"
    assert right.input_bindings["geometry"].artifact_alias == "initial_geometry"
    assert difference.capability == "same_geometry_method_energy_difference"
    assert difference.input_bindings["energy_a"].source_requirement_id == left.id
    assert difference.input_bindings["energy_b"].source_requirement_id == right.id
    assert difference.outputs == ["method_energy_difference"]
    assert left.outputs == right.outputs == []


def test_method_modification_resolves_user_text_and_preserves_constraint_patch():
    registry = build_registry()
    initial = _proposal(tasks=[_task("t1", "optimize_geometry", method="r²SCAN-3c")])
    request = canonicalize_semantic_request(
        "Optimize water with r²SCAN-3c.",
        initial,
        request_id="request_modify",
        registry=registry,
    )
    run = SimpleNamespace(request=request)
    modification = SemanticProposal.model_validate(
        {
            "mode": "modify",
            "modification": {
                "target_task_ref": "t1",
                "method_request": "PBE0",
                "parameters": {"geom_maxiter": 120},
            },
        },
        strict=True,
    ).modification

    requirement_id, patch, constraints = canonicalize_modification(
        modification,
        pending_ref_to_requirement={"t1": request.requirements[0].id},
        run=run,
        registry=registry,
    )

    assert requirement_id == request.requirements[0].id
    assert patch == {"geom_maxiter": 120, "method_profile": "pbe0_d3bj_def2svp"}
    assert constraints["method_resolution"]["request"] == "PBE0"
    assert constraints["method_resolution"]["status"] == "proposed"
