"""Bounded delivery tests: values stay immutable and prose cannot add science."""

import json

import pytest

from bg6022.agent import Agent
from bg6022.answer import AnswerOutput, AnswerSection, validate_result_answer


@pytest.mark.parametrize(
    "text", ["No imaginary frequencies; stable minimum confirmed.", "Total energy is 1 Eh."]
)
def test_verified_channel_rejects_free_scientific_assertions(text):
    fact = {"kind": "field", "expected_type": "record_list", "value": [{"atom": 1}]}
    with pytest.raises(ValueError):
        validate_result_answer(
            AnswerOutput(
                action="respond", sections=[AnswerSection(output_refs=["out_1"], text=text)]
            ),
            {"out_1": {"kind": "field", "type": "record_list", "fact": fact}},
            ["out_1"],
        )


def test_large_record_list_is_paged_before_model_serialization():
    from bg6022.planner import OutputView

    rows = [{"atom": n, "charge": n / 100} for n in range(10000)]
    fact = {
        "output_ref": "out_1",
        "kind": "field",
        "expected_type": "record_list",
        "value": rows,
        "view": OutputView(offset=20, limit=20, columns=["atom"]).model_dump(),
    }
    outputs = Agent._public_answer_outputs([fact], [])
    assert outputs[0]["verified_value"] == [{"atom": n} for n in range(20, 40)]
    assert outputs[0]["view_metadata"]["total_rows"] == 10000
    assert outputs[0]["view_metadata"]["requested_scope_complete"]
    assert not outputs[0]["view_metadata"]["full_value_shown"]
    assert len(json.dumps(outputs).encode()) < 16384
    assert len(rows) == 10000 and rows[20] == {"atom": 20, "charge": 0.2}


def test_complete_table_request_remains_partial_after_one_page():
    from bg6022.output_contracts import bounded_verified_value

    value, meta = bounded_verified_value([{"a": i} for i in range(100)], {"scope": "all"})
    assert len(value) == 30
    assert not meta["requested_scope_complete"] and meta["has_more"]


def test_display_precision_preserves_original_token():
    from bg6022.answer import format_verified_value

    value = {"value": -76.123456, "token": "-76.123456000", "unit": "Eh"}
    assert format_verified_value(value) == "-76.123456000"
    assert format_verified_value(value, precision=3) == "-76.123"
    assert value["token"] == "-76.123456000"


@pytest.mark.parametrize(
    "view",
    [{"offset": -1}, {"offset": True}, {"limit": 51}, {"precision": 13}, {"columns": ["x"] * 13}],
)
def test_view_bounds_reject_invalid_selectors(view):
    from bg6022.planner import OutputView

    with pytest.raises(ValueError):
        OutputView.model_validate(view)


def test_entire_model_output_catalog_including_metadata_fits_turn_budget():
    facts = [
        {
            "output_ref": f"out_{n}",
            "kind": "field",
            "expected_type": "record_list",
            "metadata": {"label": "测试" * 100},
            "value": [{"long": "a" * 1000} for _ in range(1000)],
        }
        for n in range(4)
    ]
    outputs = Agent._public_answer_outputs(facts, [])
    assert len(json.dumps(outputs, ensure_ascii=False).encode("utf-8")) <= 16384
    assert all(len(json.dumps(o["verified_value"]).encode()) <= 8192 for o in outputs)


def test_table_precision_is_programmatic_and_does_not_mutate_values():
    from bg6022.answer import _render_record_value

    rows = [{"charge": 0.123456789}]
    assert "0.1235" in _render_record_value("charges", rows, precision=4)
    assert rows[0]["charge"] == 0.123456789


def test_unrequested_rounding_never_turns_small_nonzero_value_into_zero():
    from bg6022.answer import format_verified_value

    assert float(format_verified_value(1e-15)) == 1e-15


def test_full_requested_column_subset_is_complete_without_claiming_full_value():
    from bg6022.output_contracts import bounded_verified_value

    value, metadata = bounded_verified_value([{"a": 1, "b": 2}], {"columns": ["a"], "scope": "all"})
    assert value == [{"a": 1}]
    assert metadata["requested_scope_complete"] and not metadata["full_value_shown"]


@pytest.mark.parametrize("view", [{"offset": 1}, {"columns": ["invented"]}])
def test_view_rejects_out_of_range_rows_and_unknown_columns(view):
    from bg6022.output_contracts import bounded_verified_value

    with pytest.raises(ValueError):
        bounded_verified_value([{"atom": 1}], view)


def test_preview_counts_multibyte_keys_inside_byte_budget():
    from bg6022.output_contracts import bounded_verified_value

    rows = [{("科学" * 80) + str(i): "\x00" * 1000 for i in range(12)} for _ in range(50)]
    value, metadata = bounded_verified_value(rows, budget_bytes=1024)
    assert len(json.dumps(value, ensure_ascii=False).encode("utf-8")) <= 1024
    assert not metadata["requested_scope_complete"]
