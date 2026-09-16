from __future__ import annotations

import pytest

from bg6022.answer import AnswerOutput, AnswerSection, render_answer_output, validate_result_answer
from bg6022.models import Request, Tool
from bg6022.output_contracts import (
    canonical_public_outputs,
    is_compatible_value,
    property_evidence_matches,
)
from bg6022.planner import _intake_schema, normalize_user_explicit_parameters, strict_positive_index
from bg6022.tools.registry import build_registry


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
