from __future__ import annotations

import json

import pytest

from bg6022.orca.checks import evaluate_success
from bg6022.orca.input import OrcaInputSpec, render_input
from bg6022.orca.parser import EnergyObservation, inspect_attempt
from bg6022.tools.molecule import parse_xyz_bytes
from bg6022.tools.orca import _electronic_energy_value, _external_mode_count

GEOMETRY = parse_xyz_bytes(b"1\nhydrogen\nH 0.0 0.0 0.0\n")
_DEFAULT_HESSIAN = object()


def _stdout(*, frequencies: list[str] | None = None, scf: str = "SCF CONVERGED AFTER 1 CYCLES\n"):
    rows = frequencies or [
        "0: 0.00 cm**-1",
        "1: -0.25 cm**-1",
        "2: 123.45 cm**-1",
    ]
    return (
        "Program Version 6.1.1\n"
        f"{scf}"
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        + "\n".join(rows)
        + "\nScaling factor for frequencies = 1.0 (already applied!)\n"
        "NORMAL MODES\n"
        "-------------\n"
        "****ORCA TERMINATED NORMALLY****\n"
        "TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 0 msec\n"
    )


def _hessian(
    *,
    dimension: int = 3,
    matrix_rows: list[str] | None = None,
    atom_rows: list[str] | None = None,
    frequencies: list[float] | None = None,
) -> bytes:
    columns = " ".join(str(index) for index in range(dimension))
    rows = matrix_rows or [
        f"{index} " + " ".join("1.0" if index == col else "0.0" for col in range(dimension))
        for index in range(dimension)
    ]
    atoms = atom_rows or ["H 1.007825 0.0 0.0 0.0"]
    mode_values = frequencies or [0.0, -0.25, 123.45]
    modes = "\n".join(f"{index} {value}" for index, value in enumerate(mode_values))
    return (
        "$orca_hessian_file\n"
        "$hessian\n"
        f"{dimension}\n"
        f"{columns}\n"
        + "\n".join(rows)
        + f"\n$vibrational_frequencies\n{len(mode_values)}\n{modes}\n"
        "$frequency_scale_factor\n1.000000\n"
        "$atoms\n"
        f"{len(atoms)}\n" + "\n".join(atoms) + "\n$end\n"
    ).encode()


def _blocked_hessian() -> bytes:
    dimension = 6
    lines = ["$orca_hessian_file", "$hessian", str(dimension)]
    for first_column in range(0, dimension, 5):
        columns = list(range(first_column, min(first_column + 5, dimension)))
        lines.append(" ".join(str(index) for index in columns))
        for row_index in range(dimension):
            row_values = ["1.0" if row_index == column else "0.0" for column in columns]
            lines.append(f"{row_index} " + " ".join(row_values))
    lines.extend(
        [
            "$vibrational_frequencies",
            "6",
            "0 0.0",
            "1 1.0",
            "2 2.0",
            "3 3.0",
            "4 4.0",
            "5 5.0",
            "$frequency_scale_factor",
            "1.000000",
            "$atoms",
            "2",
            "H 1.007825 0.0 0.0 0.0",
            "H 1.007825 0.0 0.0 0.0",
            "$end",
        ]
    )
    return ("\n".join(lines) + "\n").encode()


def _facts(
    *,
    stdout: str | None = None,
    hessian: bytes | None | object = _DEFAULT_HESSIAN,
    input_geometry=GEOMETRY,
    expected_atom_count: int | None = 1,
    **kwargs,
):
    return inspect_attempt(
        operation="Freq",
        stdout=(_stdout() if stdout is None else stdout).encode(),
        stderr=b"",
        exit_code=0,
        runner_status="succeeded",
        input_geometry=input_geometry,
        hessian=_hessian() if hessian is _DEFAULT_HESSIAN else hessian,
        expected_atom_count=expected_atom_count,
        **kwargs,
    )


def test_frequency_input_is_a_standalone_orca_operation() -> None:
    rendered = render_input(OrcaInputSpec("Freq", "r2scan3c", "gas", 0, 1, 4, 192))

    assert b"! r2SCAN-3c TightSCF Freq" in rendered
    assert b" Opt" not in rendered
    assert b" Freq Opt" not in rendered


def test_frequency_input_rejects_geometry_optimization_controls() -> None:
    with pytest.raises(ValueError, match="geom_maxiter"):
        render_input(OrcaInputSpec("Freq", "r2scan3c", "gas", 0, 1, 4, 192, geom_maxiter=100))


def test_energy_results_keep_opt_sp_and_frequency_categories_distinct() -> None:
    energy = EnergyObservation(
        token="-1.234",
        value=-1.234,
        line=20,
        offset=500,
        label="FINAL SINGLE POINT ENERGY",
    )

    assert _electronic_energy_value("Opt", energy)["opt_final_electronic_energy"]["value"] == -1.234
    assert _electronic_energy_value("SP", energy)["sp_electronic_energy"]["value"] == -1.234
    assert _electronic_energy_value("Freq", energy) == {}


def test_external_mode_count_depends_on_geometry_linearity() -> None:
    linear = parse_xyz_bytes(b"3\nlinear\nH -1 0 0\nC 0 0 0\nH 1 0 0\n")
    nonlinear = parse_xyz_bytes(b"3\nwater\nO 0 0 0\nH 0.75 0 0.5\nH -0.75 0 0.5\n")
    monatomic = parse_xyz_bytes(b"1\nhydrogen atom\nH 0 0 0\n")

    assert _external_mode_count(linear) == 5
    assert _external_mode_count(nonlinear) == 6
    assert _external_mode_count(monatomic) == 3


def test_complete_frequency_and_matching_hessian_succeed_without_energy() -> None:
    facts = _facts()
    outcome = evaluate_success(facts, operation="Freq")

    assert outcome.success
    assert facts.final_energy is None
    assert facts.frequency_section is not None
    assert facts.frequency_section.complete
    assert [mode.value for mode in facts.frequency_section.modes] == [0.0, -0.25, 123.45]
    assert facts.hessian_present
    assert facts.hessian_valid is True
    assert facts.hessian_dimension == 3
    assert outcome.checks["frequency_mode_indices_match_hessian"] is True


def test_hessian_frequency_values_must_match_stdout_modes() -> None:
    hessian = _hessian().replace(b"2 123.45", b"2 123.55")
    facts = _facts(hessian=hessian)
    outcome = evaluate_success(facts, operation="Freq")

    assert facts.hessian_valid is False
    assert "frequency values do not match stdout at mode 2" in (facts.hessian_error or "")
    assert not outcome.success


def test_hessian_frequency_scale_must_match_stdout_scale() -> None:
    hessian = _hessian().replace(
        b"$frequency_scale_factor\n1.000000", b"$frequency_scale_factor\n0.980000"
    )
    facts = _facts(hessian=hessian)
    outcome = evaluate_success(facts, operation="Freq")

    assert facts.hessian_valid is False
    assert "scaling factor does not match stdout" in (facts.hessian_error or "")
    assert not outcome.success


def test_hessian_frequency_count_must_match_stdout_modes() -> None:
    hessian = _hessian().replace(
        b"$vibrational_frequencies\n3\n0 0.0\n1 -0.25\n2 123.45",
        b"$vibrational_frequencies\n2\n0 0.0\n1 -0.25",
    )
    facts = _facts(hessian=hessian)

    assert facts.hessian_valid is False
    assert "Hessian has 2 frequencies; stdout reports 3" in (facts.hessian_error or "")


def test_annotated_imaginary_mode_is_complete_frequency_evidence_with_sign_preserved() -> None:
    facts = _facts(
        stdout=_stdout(
            frequencies=[
                "0: 0.00 cm**-1",
                "1: -0.25 cm**-1 ***imaginary mode***",
                "2: 123.45 cm**-1",
            ]
        ),
    )
    outcome = evaluate_success(facts, operation="Freq")

    assert outcome.success
    assert facts.frequency_section is not None
    assert facts.frequency_section.complete
    assert facts.frequency_section.modes[1].value == -0.25


def test_frequency_facts_serialize_as_inspectable_evidence() -> None:
    facts = _facts()

    serialized = json.dumps(facts.to_dict())
    payload = json.loads(serialized)

    assert payload["frequency_section"]["complete"] is True
    assert payload["frequency_section"]["modes"][1]["value"] == -0.25
    assert payload["hessian_present"] is True
    assert payload["hessian_valid"] is True
    assert payload["hessian_dimension"] == 3


def test_truncated_frequency_section_cannot_succeed() -> None:
    facts = _facts(
        stdout=_stdout(frequencies=["0: 0.0 cm**-1", "1: 1.0 cm**-1"]),
    )
    outcome = evaluate_success(facts, operation="Freq")

    assert not outcome.success
    assert facts.frequency_section is not None
    assert not facts.frequency_section.complete
    assert "missing indices: [2]" in (facts.frequency_section.error or "")
    assert outcome.failure_category == "invalid_output"


def test_missing_hessian_cannot_succeed() -> None:
    facts = _facts(hessian=None)
    outcome = evaluate_success(facts, operation="Freq")

    assert not facts.hessian_present
    assert facts.hessian_valid is False
    assert facts.hessian_error == "frequency Hessian file is missing"
    assert not outcome.success
    assert outcome.failure_category == "invalid_output"


def test_hessian_atom_distances_are_compared_in_bohr_and_ignore_origin_translation() -> None:
    geometry = parse_xyz_bytes(b"2\nhydrogen pair\nH 0.0 0.0 0.0\nH 0.529177210903 0.0 0.0\n")
    facts = _facts(
        input_geometry=geometry,
        expected_atom_count=2,
        hessian=_hessian(
            dimension=6,
            atom_rows=["H 1.007825 5.0 6.0 7.0", "H 1.007825 6.0 6.0 7.0"],
            frequencies=[0, 1, 2, 3, 4, 5],
        ),
        stdout=_stdout(frequencies=[f"{index}: {index:.2f} cm**-1" for index in range(6)]),
    )

    assert facts.hessian_valid is True


def test_hessian_column_blocks_are_fully_parsed() -> None:
    geometry = parse_xyz_bytes(b"2\nhydrogen pair\nH 0 0 0\nH 0 0 0\n")
    frequencies = [f"{index}: {float(index):.1f} cm**-1" for index in range(6)]
    facts = _facts(
        input_geometry=geometry,
        expected_atom_count=2,
        stdout=_stdout(frequencies=frequencies),
        hessian=_blocked_hessian(),
    )

    assert facts.frequency_section is not None
    assert facts.frequency_section.complete
    assert facts.hessian_valid is True
    assert facts.hessian_dimension == 6


def test_hessian_dimension_must_match_three_coordinates_per_atom() -> None:
    facts = _facts(hessian=_hessian(dimension=2, matrix_rows=["0 1.0 0.0", "1 0.0 1.0"]))
    outcome = evaluate_success(facts, operation="Freq")

    assert facts.hessian_valid is False
    assert facts.hessian_dimension == 2
    assert "expected 3" in (facts.hessian_error or "")
    assert not outcome.success


def test_hessian_interatomic_distances_must_match_input_geometry() -> None:
    geometry = parse_xyz_bytes(b"2\nhydrogen pair\nH 0.0 0.0 0.0\nH 1.0 0.0 0.0\n")
    facts = _facts(
        input_geometry=geometry,
        expected_atom_count=2,
        hessian=_hessian(
            dimension=6,
            atom_rows=["H 1.007825 0.0 0.0 0.0", "H 1.007825 0.1 0.0 0.0"],
        ),
    )
    outcome = evaluate_success(facts, operation="Freq")

    assert facts.hessian_valid is False
    assert "interatomic distances do not match input geometry" in (facts.hessian_error or "")
    assert not outcome.success


def test_nonfinite_hessian_matrix_values_are_rejected() -> None:
    facts = _facts(
        hessian=_hessian(matrix_rows=["0 nan 0.0 0.0", "1 0.0 1.0 0.0", "2 0.0 0.0 1.0"]),
    )
    outcome = evaluate_success(facts, operation="Freq")

    assert facts.hessian_valid is False
    assert "non-finite values" in (facts.hessian_error or "")
    assert not outcome.success
