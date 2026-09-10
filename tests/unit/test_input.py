from __future__ import annotations

import pytest

from bg6022.orca.input import OrcaInputSpec, render_input


def test_sp_input_is_closed_and_uses_resource_budget() -> None:
    rendered = render_input(
        OrcaInputSpec("SP", "r2scan3c", "gas", 0, 1, cores=4, maxcore_mb=384)
    ).decode("ascii")
    assert rendered.startswith("! r2SCAN-3c TightSCF SP\n")
    assert "%pal\n  nprocs 4\nend" in rendered
    assert "%maxcore 384" in rendered
    assert "* xyzfile 0 1 geometry.xyz" in rendered
    assert rendered.endswith("\n\n")
    assert "gas" not in rendered


def test_opt_input_has_typed_iteration_blocks() -> None:
    rendered = render_input(
        OrcaInputSpec(
            "Opt",
            "r2scan3c",
            "gas",
            0,
            1,
            cores=4,
            maxcore_mb=384,
            scf_maxiter=300,
            geom_maxiter=1,
        )
    ).decode("ascii")
    assert "%scf\n  MaxIter 300\nend" in rendered
    assert "%geom\n  MaxIter 1\nend" in rendered


def test_sp_rejects_optimization_parameter() -> None:
    with pytest.raises(ValueError, match="geom_maxiter"):
        render_input(OrcaInputSpec("SP", "r2scan3c", "gas", 0, 1, 4, 384, geom_maxiter=1))
