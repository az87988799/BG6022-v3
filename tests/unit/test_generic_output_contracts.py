from __future__ import annotations

import pytest

from bg6022.answer import AnswerOutput, AnswerSection, render_answer_output, validate_result_answer
from bg6022.models import Request, Tool
from bg6022.output_contracts import (
    canonical_public_outputs,
    is_compatible_value,
    property_evidence_matches,
)
from bg6022.planner import (
    IntakeOutput,
    QuerySelection,
    QueryTarget,
    _intake_schema,
    _request_output_preferences,
    _validate_query_selection,
    normalize_user_explicit_parameters,
    strict_positive_index,
)
from bg6022.tools.registry import ToolRegistry, build_registry


def test_public_output_directory_keeps_fields_ports_and_checks_distinct() -> None:
    tool = Tool(
        name="generic_contract",
        description="Return generic outputs.",
        output_ports={"report": "text_file"},
        results={"records": "record_list"},
        scientific_checks={"quality_check": "A verified quality check."},
        result_metadata={
            "report": {"label": "原子坐标 CSV"},
            "records": {"label": "所选原子记录"},
            "quality_check": {"label": "质量检查"},
        },
        requires_compute_permission=False,
    )

    outputs = canonical_public_outputs(tool)
    assert [(item["kind"], item["name"]) for item in outputs] == [
        ("port", "report"),
        ("field", "records"),
        ("check", "quality_check"),
    ]
    assert outputs[0]["shape"] == "text_file"
    assert outputs[1]["shape"] == "record_list"
    assert outputs[2]["shape"] == "check"
    assert outputs[0]["property"] == "report"


def test_public_output_property_collision_is_rejected_at_registration() -> None:
    with pytest.raises(ValueError, match="ambiguous public property"):
        Tool(
            name="ambiguous_contract",
            description="Ambiguous outputs.",
            results={"left": "text", "right": "text"},
            result_properties={"left": "same_property", "right": "same_property"},
            requires_compute_permission=False,
        )


@pytest.mark.parametrize(
    ("value", "declared_type"),
    [
        (42, "record"),
        ("not a record list", "record_list"),
        ([1, 2], "record_list"),
        ([{"value": float("nan")}], "record_list"),
    ],
)
def test_record_contract_does_not_accept_scalar_or_non_record_rows(
    value: object, declared_type: str
) -> None:
    assert not is_compatible_value(value, declared_type)


def test_generic_property_evidence_accepts_declared_semantic_synonyms() -> None:
    assert property_evidence_matches(
        "angle",
        "角度",
        "请告诉我这三个原子的角度",
        metadata={"label": "原子夹角", "description": "以第二个原子为顶点的夹角"},
    )
    assert property_evidence_matches(
        "atom_report",
        "报告",
        "把原子坐标报告给我",
        metadata={"label": "原子坐标 CSV", "description": "所选原子的实际坐标表文件"},
    )


@pytest.mark.parametrize("raw", ["2.5", "2e0", "2/3", "0", "-1"])
def test_atom_index_assignment_is_parsed_as_a_whole_invalid_token(raw: str) -> None:
    normalized = normalize_user_explicit_parameters(f"atom_i={raw}", {"atom_i": 7})
    assert "atom_i" not in normalized.explicit_parameters
    assert "atom_i" in normalized.parameter_issues
    with pytest.raises(ValueError):
        strict_positive_index(raw)


def test_query_schema_rejects_subject_property_cross_binding() -> None:
    registry = build_registry()
    schema = _intake_schema(
        ("t1", "t2"),
        registry.result_capabilities(),
        result_catalog=[
            {"subject_ref": "t1", "result": {"property": "electronic_energy"}},
            {"subject_ref": "t2", "result": {"property": "molecular_geometry"}},
        ],
        registry=registry,
    )
    with pytest.raises(ValueError, match="subject/property pairs"):
        schema.model_validate(
            {
                "intent": "context_query",
                "query_selection": {
                    "status": "selected",
                    "targets": [
                        {
                            "subject_ref": "t1",
                            "property": "molecular_geometry",
                            "evidence": "xyz",
                        }
                    ],
                },
            },
            strict=True,
        )


def test_result_answer_rejects_free_prose_and_renders_verified_file_content() -> None:
    outputs = {
        "out_1": {
            "kind": "port",
            "type": "text_file",
            "fact": {
                "kind": "port",
                "name": "report",
                "expected_type": "text_file",
                "metadata": {"label": "原子坐标 CSV"},
            },
            "file": {
                "display_name": "原子坐标.csv",
                "filename": "原子坐标.csv",
                "path": r"C:\data\原子坐标.csv",
                "preview_text": "atom_index,element\n1,H\n",
                "preview_complete": True,
                "sha256": "a" * 64,
            },
        }
    }
    injected = AnswerOutput(
        action="respond",
        sections=[
            AnswerSection(
                format="auto",
                output_refs=["out_1"],
                text="优化已经完成，能量是 -999 Eh。",
            )
        ],
    )
    with pytest.raises(ValueError, match="free model prose"):
        validate_result_answer(injected, outputs, ["out_1"])

    safe = AnswerOutput(
        action="respond",
        sections=[AnswerSection(format="auto", output_refs=["out_1"], text=None)],
    )
    rendered = render_answer_output(
        safe,
        outputs_by_ref=outputs,
        required_refs=["out_1"],
        preferences={"file_content": "show", "detail": "normal"},
    )
    assert "atom_index,element" in rendered
    assert "-999" not in rendered


def test_request_output_preferences_are_small_and_non_scientific() -> None:
    request = Request(
        id="request_preferences",
        description="show a file",
        output_preferences={"layout": "table", "file_content": "show", "detail": "brief"},
    )
    assert request.output_preferences == {
        "layout": "table",
        "file_content": "show",
        "detail": "brief",
    }
    with pytest.raises(ValueError, match="unknown output preference"):
        Request(
            id="request_bad_preferences",
            description="bad",
            output_preferences={"run_tool": "yes"},
        )


def test_followup_query_requires_recent_delivery_and_conversational_reference() -> None:
    catalog = [
        {
            "subject_ref": "t1",
            "recently_delivered": True,
            "result": {
                "property": "angle",
                "label": "原子夹角",
                "description": "以第二个原子为顶点的夹角",
            },
        }
    ]
    output = IntakeOutput(
        intent="context_query",
        query_selection=QuerySelection(
            status="selected",
            targets=[
                QueryTarget(
                    subject_ref="t1",
                    property="angle",
                    evidence="刚才的内容",
                    reference_mode="followup",
                )
            ],
        ),
    )
    accepted = _validate_query_selection(
        output,
        ("t1",),
        message="把刚才的内容给我",
        result_catalog=catalog,
    )
    assert accepted.query_selection is not None
    assert accepted.query_selection.status == "selected"

    stale = _validate_query_selection(
        output,
        ("t1",),
        message="把刚才的内容给我",
        result_catalog=[{**catalog[0], "recently_delivered": False}],
    )
    assert stale.query_selection is not None
    assert stale.query_selection.status == "clarify"

    explicit = _validate_query_selection(
        output,
        ("t1",),
        message="己烷的 xyz 文件是什么",
        result_catalog=catalog,
    )
    assert explicit.query_selection is not None
    assert explicit.query_selection.status == "clarify"


def test_output_views_preserve_context_and_require_explicit_link_only() -> None:
    outputs = {
        "out_1": {
            "kind": "field",
            "type": "integer",
            "fact": {
                "kind": "field",
                "name": "atom_count",
                "expected_type": "integer",
                "step_tool": "generate_geometry",
                "system": "水",
                "metadata": {
                    "label": "结构原子数",
                    "caveat": "这是初始猜测结构，不代表已完成几何优化",
                },
                "value": 3,
            },
        },
        "out_2": {
            "kind": "port",
            "type": "text_file",
            "fact": {
                "kind": "port",
                "name": "report",
                "expected_type": "text_file",
                "step_tool": "generate_geometry",
                "system": "水",
                "metadata": {"label": "原子坐标 CSV"},
            },
            "file": {
                "display_name": "geometry.csv",
                "path": r"C:\data\geometry.csv",
                "role": "initial_geometry",
                "source": "test:geometry",
                "preview_text": "atom_index,element\n1,O\n",
                "preview_complete": True,
                "sha256": "a" * 64,
            },
        },
    }
    scalar = AnswerOutput(
        action="respond",
        sections=[AnswerSection(format="table", output_refs=["out_1"], text=None)],
    )
    rendered_scalar = render_answer_output(
        scalar, outputs_by_ref=outputs, required_refs=["out_1"], preferences={"layout": "table"}
    )
    assert "| 字段 | 值 |" in rendered_scalar
    assert "初始猜测结构" in rendered_scalar

    link = AnswerOutput(
        action="respond",
        sections=[AnswerSection(format="link", output_refs=["out_2"], text=None)],
    )
    with pytest.raises(ValueError, match="link-only"):
        render_answer_output(link, outputs_by_ref=outputs, required_refs=["out_2"])
    rendered_link = render_answer_output(
        link,
        outputs_by_ref=outputs,
        required_refs=["out_2"],
        preferences={"file_content": "link_only"},
    )
    assert "atom_index,element" not in rendered_link
    assert "geometry.csv" in rendered_link
    assert "来源：初始结构" in rendered_link


def test_index_normalization_uses_declared_tool_fields() -> None:
    normalized = normalize_user_explicit_parameters(
        "atom_k=2.5",
        {"atom_k": 3},
        parameter_names=("atom_k",),
    )
    assert "atom_k" not in normalized.explicit_parameters
    assert "atom_k" in normalized.parameter_issues

    registry = build_registry()
    assert registry.request_index_parameter_fields([], ["interatomic_distance"]) == {
        "atom_i",
        "atom_j",
    }


def test_registry_rejects_incompatible_cross_tool_property_contracts() -> None:
    first = Tool(
        name="first_property_tool",
        description="first",
        results={"first": "degree"},
        result_properties={"first": "shared_property"},
        requires_compute_permission=False,
    )
    second = Tool(
        name="second_property_tool",
        description="second",
        results={"second": "angstrom"},
        result_properties={"second": "shared_property"},
        requires_compute_permission=False,
    )
    with pytest.raises(ValueError, match="incompatible public property contract"):
        ToolRegistry([first, second])


def test_model_cannot_force_link_only_without_user_request() -> None:
    assert (
        _request_output_preferences("show the file", {"file_content": "link_only"})["file_content"]
        == "auto"
    )
    assert (
        _request_output_preferences("只给文件路径", {"file_content": "link_only"})["file_content"]
        == "link_only"
    )
