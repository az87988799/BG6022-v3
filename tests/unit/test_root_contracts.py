"""Behavior regressions for transient intent and registry bindings (test-only Tools)."""

import pytest

from bg6022.models import Tool
from bg6022.tools.registry import ToolRegistry, build_registry


def test_output_selector_rejects_name_property_collision():
    from bg6022.output_contracts import resolve_output_selector

    tool = Tool(
        name="collision",
        description="test",
        results={"a": "text", "b": "text"},
        result_properties={"a": "b", "b": "c"},
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_output_selector(tool, "b")


def test_dynamic_semantic_property_and_injected_registry_projection():
    from bg6022.canonicalize import canonicalize_semantic_request
    from bg6022.semantic import semantic_schema

    tool = Tool(
        name="test_property",
        description="test",
        results={"answer": "integer"},
        result_properties={"answer": "test_number"},
        planning_role="task",
        default_outputs=["answer"],
    )
    registry = ToolRegistry([tool])
    proposal = semantic_schema(registry).model_validate(
        {
            "mode": "compute",
            "subjects": [{"key": "water", "query": "water", "evidence": "water"}],
            "tasks": [
                {
                    "key": "t",
                    "subject_key": "water",
                    "capability": "test_property",
                    "requested_properties": ["test_number"],
                }
            ],
        }
    )
    request = canonicalize_semantic_request(
        "water test number", proposal, request_id="test", registry=registry
    )
    assert registry.operations_for_request(request) == []
    assert registry.result_targets_for_request(request)[0].field == "answer"


@pytest.mark.parametrize(
    "message",
    [
        "优化水，然后给出它的偶极矩",
        "优化水，然后告诉我偶极矩",
        "优化水并报告偶极矩",
        "优化水，然后计算它的偶极矩",
    ],
)
@pytest.mark.parametrize("semantic", [True, False])
def test_natural_attached_report_needs_no_output_keywords(message, semantic):
    from bg6022.planner import intake_message
    from bg6022.semantic import semantic_message

    intents = [
        {
            "kind": "compute",
            "evidence": "优化水",
            "task_keys": ["opt"],
            "requested_property": "geometry",
        },
        {
            "kind": "report",
            "evidence": "偶极矩",
            "task_keys": ["opt"],
            "requested_property": "dipole_moment",
        },
    ]

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            task = {
                "key": "opt",
                "subject_key": "water",
                "capability": "optimize_geometry",
                "report_queries": [{"evidence": "偶极矩", "search_terms": ["DIPOLE MOMENT"]}],
            }
            if semantic:
                task["requested_properties"] = ["geometry"]
                payload = {
                    "mode": "compute",
                    "tasks": [task],
                    "subjects": [{"key": "water", "query": "water", "evidence": "水"}],
                }
            else:
                task["outputs"] = ["optimized_geometry"]
                payload = {
                    "intent": "chemistry_compute",
                    "requirements": [task],
                    "subjects": {
                        "water": {
                            "key": "water",
                            "molecule_query": "water",
                            "molecule_input_kind": "name",
                            "molecule_name_evidence": "水",
                        }
                    },
                }
            return schema.model_validate(dict(payload, intent_items=intents))

    fn = semantic_message if semantic else intake_message
    result = fn(Client(), message, registry=build_registry())
    tasks = result.tasks if semantic else result.requirements
    assert len(tasks) == 1 and tasks[0].capability == "optimize_geometry"
    assert len(tasks[0].report_queries) == 1


def test_shared_intent_contract_blocks_uncovered_and_unresolved_targets():
    from bg6022.planner import IntentItem, RequirementProposal, validate_intent_items

    registry = build_registry()
    task = RequirementProposal(
        key="sp", capability="single_point", outputs=["sp_electronic_energy"]
    )
    for items in (
        [],
        [
            IntentItem(
                kind="compute",
                evidence="number",
                task_keys=["sp"],
                requested_property="test_number",
            )
        ],
        [IntentItem(kind="unresolved", evidence="number", requested_property="test_number")],
    ):
        with pytest.raises(ValueError):
            validate_intent_items(items, "number", [task], registry=registry)


@pytest.mark.parametrize("kind", ["exclude", "explain"])
def test_non_compute_intents_do_not_reject_unsupported_words(kind):
    from bg6022.semantic import semantic_message

    message = "Do not compute Gibbs; explain frequency"

    class Client:
        def complete_json(self, messages, schema, **kwargs):
            return schema.model_validate(
                {
                    "mode": "qa",
                    "intent_items": [
                        {
                            "kind": kind,
                            "evidence": message,
                            "requested_property": "gibbs_free_energy",
                        }
                    ],
                }
            )

    assert semantic_message(Client(), message, registry=build_registry()).mode == "qa"


def test_new_compute_missing_intents_uses_bounded_model_correction():
    import json

    from test_llm import _offline_client, _response

    from bg6022.semantic import semantic_message

    payload = {
        "mode": "compute",
        "subjects": [{"key": "w", "query": "water", "evidence": "water"}],
        "tasks": [{"key": "sp", "subject_key": "w", "capability": "single_point"}],
    }
    corrected = dict(
        payload, intent_items=[{"kind": "compute", "evidence": "SP water", "task_keys": ["sp"]}]
    )
    client, requests, http = _offline_client(
        [
            _response(json.dumps(payload), finish_reason="stop", usage={}),
            _response(json.dumps(corrected), finish_reason="stop", usage={}),
        ],
        structured_output_corrections=1,
    )
    try:
        result = semantic_message(client, "SP water", registry=build_registry())
        assert result.mode == "compute" and len(requests) == 2
        assert client.calls[0].category == "schema_error"
    finally:
        http.close()


def test_attached_report_promotes_to_registered_verified_output():
    from bg6022.planner import IntentItem, RequirementProposal, validate_intent_items

    base = build_registry().get("optimize_geometry")
    tool = base.model_copy(
        update={
            "results": {**base.results, "dipole": "record"},
            "result_properties": {**base.result_properties, "dipole": "dipole_moment"},
        }
    )
    registry = ToolRegistry([tool])
    task = RequirementProposal(
        key="opt",
        capability=tool.name,
        outputs=["optimized_geometry"],
        report_queries=[{"evidence": "dipole", "search_terms": ["DIPOLE MOMENT"]}],
    )
    items = [
        IntentItem(kind="compute", evidence="Optimize", task_keys=["opt"]),
        IntentItem(
            kind="report", evidence="dipole", task_keys=["opt"], requested_property="dipole_moment"
        ),
    ]
    validate_intent_items(items, "Optimize; dipole", [task], registry=registry)

    assert task.outputs == ["optimized_geometry", "dipole"]
    assert task.report_queries == []
    validate_intent_items(items, "Optimize; dipole", [task], registry=registry)
