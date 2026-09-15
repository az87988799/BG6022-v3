from __future__ import annotations

import pytest

from bg6022.orca.frequency_parser import parse_vibrational_frequencies


def _frequency_output(
    rows: list[str],
    *,
    scale: str = "Scaling factor for frequencies = 1.0 (already applied!)",
    ending: str = "NORMAL MODES\n-------------",
) -> str:
    lines = ["VIBRATIONAL FREQUENCIES", "-----------------------", *rows]
    if scale:
        lines.append(scale)
    if ending:
        lines.extend(ending.splitlines())
    return "\n".join(lines)


def test_parses_all_signed_modes_and_already_applied_scale_from_bytes() -> None:
    rows = [
        "  0:     0.00 cm**-1",
        "  1:    -0.25 cm**-1",
        "  2:  1.234D+03 cm**-1",
        "  3:   12.50 cm**-1",
        "  4:   13.50 cm**-1",
        "  5:   14.50 cm**-1",
        "  6:   15.50 cm**-1",
        "  7:   16.50 cm**-1",
        "  8:   17.50 cm**-1",
    ]

    section = parse_vibrational_frequencies(
        _frequency_output(rows).replace("\n", "\r\n").encode(), expected_atom_count=3
    )

    assert section.complete
    assert section.error is None
    assert section.scaling_factor == 1.0
    assert section.scaling_applied
    assert len(section.modes) == 9
    assert section.modes[0].value == 0.0
    assert section.modes[1].value == -0.25
    assert section.modes[2].value == 1234.0
    assert section.modes[1].unit == "cm^-1"


def test_parses_orca_imaginary_mode_annotation_without_losing_negative_sign() -> None:
    rows = [
        "0: 0.00 cm**-1",
        "1: 0.00 cm**-1",
        "2: 0.00 cm**-1",
        "3: 0.00 cm**-1",
        "4: 0.00 cm**-1",
        "5: 0.00 cm**-1",
        "6: -83.22 cm**-1 ***imaginary mode***",
        "7: 435.79 cm**-1",
        "8: 544.50 cm**-1",
    ]

    section = parse_vibrational_frequencies(_frequency_output(rows), expected_atom_count=3)

    assert section.complete
    assert [(mode.index, mode.value) for mode in section.modes[6:]] == [
        (6, -83.22),
        (7, 435.79),
        (8, 544.5),
    ]


def test_scale_without_applied_marker_is_retained_as_unscaled() -> None:
    output = _frequency_output(
        ["0: 0.0 cm**-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"],
        scale="Scaling factor for frequencies = 0.987",
    )

    section = parse_vibrational_frequencies(output, expected_atom_count=1)

    assert section.complete
    assert section.scaling_factor == 0.987
    assert section.scaling_applied is False


def test_missing_heading_and_scaling_factor_are_incomplete() -> None:
    no_heading = parse_vibrational_frequencies("no frequency table", expected_atom_count=1)
    no_scale = parse_vibrational_frequencies(
        _frequency_output(["0: 0.0 cm**-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"], scale=""),
        expected_atom_count=1,
    )

    assert not no_heading.complete
    assert "section is missing" in (no_heading.error or "")
    assert not no_scale.complete
    assert "scaling factor" in (no_scale.error or "")


def test_truncated_block_reports_missing_mode_indices() -> None:
    section = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm**-1", "1: 1.0 cm**-1"],
            ending="",
        ),
        expected_atom_count=1,
    )

    assert not section.complete
    assert len(section.modes) == 2
    assert "missing indices: [2]" in (section.error or "")
    assert "expected 3 frequency modes" in (section.error or "")


def test_duplicate_and_out_of_range_indices_are_rejected() -> None:
    section = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm**-1", "1: 1.0 cm**-1", "1: 2.0 cm**-1"],
        ),
        expected_atom_count=1,
    )

    assert not section.complete
    assert len(section.modes) == 3
    assert "duplicate frequency mode indices: [1]" in (section.error or "")
    assert "missing indices: [2]" in (section.error or "")


def test_nonfinite_mode_and_scale_values_are_rejected() -> None:
    bad_mode = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm**-1", "1: NaN cm**-1", "2: 2.0 cm**-1"],
        ),
        expected_atom_count=1,
    )
    bad_scale = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm**-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"],
            scale="Scaling factor for frequencies = Infinity",
        ),
        expected_atom_count=1,
    )

    assert not bad_mode.complete
    assert "non-finite frequency value" in (bad_mode.error or "")
    assert not bad_scale.complete
    assert "non-finite scaling factor" in (bad_scale.error or "")


@pytest.mark.parametrize("factor", ["0", "-1.2"])
def test_nonpositive_scaling_factor_is_rejected(factor: str) -> None:
    section = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm**-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"],
            scale=f"Scaling factor for frequencies = {factor}",
        ),
        expected_atom_count=1,
    )

    assert not section.complete
    assert "scaling factor must be positive" in (section.error or "")


def test_unsupported_frequency_row_layout_or_unit_is_rejected() -> None:
    section = parse_vibrational_frequencies(
        _frequency_output(
            ["0: 0.0 cm^-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"],
        ),
        expected_atom_count=1,
    )

    assert not section.complete
    assert "unsupported frequency unit" in (section.error or "")


def test_multiple_frequency_sections_are_rejected() -> None:
    table = _frequency_output(["0: 0.0 cm**-1", "1: 1.0 cm**-1", "2: 2.0 cm**-1"])
    section = parse_vibrational_frequencies(
        table + "\n" + table,
        expected_atom_count=1,
    )

    assert not section.complete
    assert "multiple VIBRATIONAL FREQUENCIES sections" in (section.error or "")
