from __future__ import annotations

import pytest
from pydantic import ValidationError

from bg6022.semantic import (
    SemanticProposal,
    SemanticSubject,
    compact_method_catalog,
    compact_result_catalog,
    compact_tool_catalog,
    semantic_message,
    semantic_schema,
)
from bg6022.tools.registry import build_registry


def _compute_payload(**updates):
    value = {
        "mode": "compute",
        "subjects": [
            {
                "key": "subject_1",
                "query": "water",
                "input_kind": "name",
                "evidence": "water",
            }
        ],
        "tasks": [
            {
                "key": "t1",
                "subject_key": "subject_1",
                "capability": "optimize_geometry",
                "method_request": "PBE0",
                "parameters": {"geom_maxiter": 100},
                "requested_properties": ["energy"],
            }
        ],
        "relations": [],
    }
    value.update(updates)
    return value


def test_semantic_proposal_accepts_user_facing_intent_only():
    proposal = semantic_schema(build_registry()).model_validate(_compute_payload(), strict=True)

    assert isinstance(proposal, SemanticProposal)
    assert proposal.tasks[0].capability == "optimize_geometry"
    assert proposal.tasks[0].method_request == "PBE0"
    assert proposal.tasks[0].requested_properties == ["energy"]


def test_semantic_subject_rejects_untranslated_chinese_name_query():
    with pytest.raises(ValueError, match="reliable English PubChem lookup spelling"):
        SemanticSubject(
            key="ethanol",
            query="乙醇",
            input_kind="name",
            evidence="乙醇",
        )


def test_semantic_subject_accepts_english_lookup_with_chinese_evidence():
    subject = SemanticSubject(
        key="ethanol",
        query="ethanol",
        input_kind="name",
        evidence="乙醇",
    )

    assert subject.query == "ethanol"
    assert subject.evidence == "乙醇"
    assert subject.input_kind == "name"


@pytest.mark.parametrize(
    ("query", "input_kind"),
    [("702", "cid"), ("CCO", "smiles"), ("C2H6O", "formula")],
)
def test_semantic_subject_non_name_inputs_are_unchanged(query: str, input_kind: str):
    subject = SemanticSubject(
        key="subject_1",
        query=query,
        input_kind=input_kind,
        evidence=query,
    )

    assert subject.query == query
    assert subject.input_kind == input_kind


@pytest.mark.parametrize(
    "field",
    ["method_profile", "subject_id", "requirement_id", "step_id", "artifact_id"],
)
def test_semantic_task_rejects_internal_parameter_fields(field: str):
    payload = _compute_payload()
    payload["tasks"][0]["parameters"][field] = "hidden"

    with pytest.raises(ValidationError):
        semantic_schema(build_registry()).model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "capability",
    ["generate_geometry", "same_geometry_method_energy_difference"],
)
def test_dynamic_capability_schema_excludes_program_generated_tools(capability: str):
    payload = _compute_payload()
    payload["tasks"][0]["capability"] = capability

    with pytest.raises(ValidationError):
        semantic_schema(build_registry()).model_validate(payload, strict=True)
    assert capability not in {item["name"] for item in compact_tool_catalog(build_registry())}


def test_semantic_schema_rejects_unknown_result_query_pair():
    payload = {
        "mode": "context_query",
        "query_selection": {
            "status": "selected",
            "targets": [
                {
                    "subject_ref": "run_9",
                    "property": "electronic_energy",
                    "evidence": "the latest energy",
                }
            ],
        },
    }

    with pytest.raises(ValidationError):
        semantic_schema(
            build_registry(),
            result_catalog=[{"subject_ref": "run_1", "property": "electronic_energy"}],
        ).model_validate(payload, strict=True)


def test_semantic_query_accepts_nested_result_catalog_pair():
    payload = {
        "mode": "context_query",
        "query_selection": {
            "status": "selected",
            "targets": [
                {
                    "subject_ref": "t1",
                    "property": "electronic_energy",
                    "evidence": "the latest energy",
                }
            ],
        },
    }
    proposal = semantic_schema(
        build_registry(),
        result_catalog=[
            {
                "subject_ref": "t1",
                "step": {"method_profile": "internal-profile-id"},
                "result": {
                    "name": "opt_final_electronic_energy",
                    "kind": "field",
                    "property": "electronic_energy",
                    "label": "optimized electronic energy",
                },
            }
        ],
    ).model_validate(payload, strict=True)

    assert proposal.query_selection.targets[0].subject_ref == "t1"


def test_compact_semantic_catalogs_hide_internal_execution_identifiers():
    registry = build_registry()
    compact_results = compact_result_catalog(
        [
            {
                "subject_ref": "t1",
                "step": {"method_profile": "internal-profile-id"},
                "result": {
                    "name": "opt_final_electronic_energy",
                    "kind": "field",
                    "property": "electronic_energy",
                    "label": "optimized electronic energy",
                },
            }
        ]
    )

    assert compact_results == [
        {
            "subject_ref": "t1",
            "property": "electronic_energy",
            "label": "optimized electronic energy",
        }
    ]
    methods = compact_method_catalog(registry)
    serialized = str(methods)
    assert "pbe0_d3bj_def2svp" not in serialized
    assert "method_profile" not in serialized


def test_gibbs_calculation_is_stopped_as_known_unsupported_without_llm_call():
    class NoCallClient:
        def complete_json(self, *_args, **_kwargs):
            raise AssertionError("known unsupported request should be stopped in code")

    proposal = semantic_message(
        NoCallClient(),
        "计算这个分子的 Gibbs 自由能",
        registry=build_registry(),
    )

    assert proposal.mode == "unsupported"
    assert proposal.unsupported_requirements == ["Gibbs free energy"]


def test_global_conformer_search_is_stopped_as_known_unsupported_without_llm_call():
    class NoCallClient:
        def complete_json(self, *_args, **_kwargs):
            raise AssertionError("known unsupported request should be stopped in code")

    proposal = semantic_message(
        NoCallClient(),
        "帮我找乙醇全局最低能构象",
        registry=build_registry(),
    )

    assert proposal.mode == "unsupported"
    assert proposal.unsupported_requirements == ["global conformer search"]


def test_semantic_modify_uses_short_pending_ref_and_target_tool_contract():
    schema = semantic_schema(
        build_registry(),
        pending_tasks=[
            {
                "task_ref": "t1",
                "capability": "optimize_geometry",
                "parameters": {"geom_maxiter": 100},
            }
        ],
    )
    proposal = schema.model_validate(
        {
            "mode": "modify",
            "modification": {
                "target_task_ref": "t1",
                "parameters": {"geom_maxiter": 120},
            },
        },
        strict=True,
    )

    assert proposal.modification.target_task_ref == "t1"
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "mode": "modify",
                "modification": {
                    "target_task_ref": "t2",
                    "parameters": {"geom_maxiter": 120},
                },
            },
            strict=True,
        )


def test_semantic_proposal_mode_contracts_are_strict():
    with pytest.raises(ValidationError):
        SemanticProposal.model_validate({"mode": "compute"}, strict=True)
    with pytest.raises(ValidationError):
        SemanticProposal.model_validate({"mode": "clarify", "clarification": "  "}, strict=True)
