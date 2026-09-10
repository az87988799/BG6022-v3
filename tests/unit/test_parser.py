from __future__ import annotations

from pathlib import Path

from bg6022.orca.checks import evaluate_success
from bg6022.orca.parser import inspect_attempt
from bg6022.tools.molecule import format_xyz, parse_xyz_bytes

FIXTURE = Path(__file__).parents[1] / "fixtures" / "orca_6_1_1_water_opt"


def test_real_crlf_fixture_is_observed_and_bound() -> None:
    initial = parse_xyz_bytes((FIXTURE / "geometry.xyz").read_bytes())
    facts = inspect_attempt(
        operation="Opt",
        stdout=(FIXTURE / "stdout.out").read_bytes(),
        stderr=b"",
        exit_code=0,
        runner_status="succeeded",
        input_geometry=initial,
        process_tree_empty=True,
        output_xyz=(FIXTURE / "input.xyz").read_bytes(),
    )
    outcome = evaluate_success(facts, operation="Opt")
    assert outcome.success
    assert facts.normal_termination is True
    assert facts.scf_converged is True
    assert facts.optimization_converged is True
    assert facts.geometry_consistent is True
    assert facts.final_energy is not None
    assert facts.final_energy.token == "-76.418938721015"


def test_failed_output_keeps_evidence_and_does_not_succeed() -> None:
    initial = parse_xyz_bytes(b"1\nhydrogen\nH 0 0 0\n")
    facts = inspect_attempt(
        operation="SP",
        stdout=b"Program Version 6.1.1\nSCF NOT CONVERGED after 100 cycles\n",
        stderr=b"SCF failed\r\n",
        exit_code=1,
        runner_status="failed",
        stop_reason="nonzero_exit",
        input_geometry=initial,
        process_tree_empty=True,
    )
    outcome = evaluate_success(facts, operation="SP")
    assert not outcome.success
    assert facts.error_category == "scf_not_converged"
    assert facts.stderr_preview == "SCF failed\r\n"
    assert facts.final_energy is None


def test_failed_opt_keeps_a_consistent_restart_candidate_geometry() -> None:
    initial = parse_xyz_bytes(b"3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n")
    output_xyz = format_xyz(
        initial.symbols,
        ((0.0, 0.1, 0.0), (0.75, -0.2, 0.0), (-0.75, -0.2, 0.0)),
        comment="failed Opt last geometry",
    )
    stdout = (
        b"Program Version 6.1.1\n"
        b"* GEOMETRY OPTIMIZATION CYCLE 1 *\n"
        b"CARTESIAN COORDINATES (ANGSTROEM)\n"
        b"------------------------------\n"
        b"O 0.000000 0.100000 0.000000\n"
        b"H 0.750000 -0.200000 0.000000\n"
        b"H -0.750000 -0.200000 0.000000\n"
        b"CARTESIAN COORDINATES (A.U.)\n"
        b"SCF CONVERGED AFTER 10 CYCLES\n"
        b"FINAL SINGLE POINT ENERGY -1.234000000000\n"
        b"MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED\n"
    )
    facts = inspect_attempt(
        operation="Opt",
        stdout=stdout,
        stderr=b"",
        exit_code=1,
        runner_status="failed",
        stop_reason="nonzero_exit",
        input_geometry=initial,
        process_tree_empty=True,
        output_xyz=output_xyz,
    )
    outcome = evaluate_success(facts, operation="Opt")
    assert not outcome.success
    assert facts.error_category == "opt_not_converged"
    assert facts.output_geometry is not None
    assert facts.stdout_geometry is not None
    assert facts.geometry_consistent is True
