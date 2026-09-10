from __future__ import annotations

import os

import pytest

from bg6022.cli import main


@pytest.mark.live_orca
@pytest.mark.skipif(os.name != "nt", reason="real ORCA M0 acceptance is Windows-only")
def test_real_water_sp_and_opt(pytestconfig, tmp_path) -> None:
    config = pytestconfig.getoption("--orca-config")
    if not config:
        pytest.fail("--orca-config is required with --run-live-orca")
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nBG6022 live water\nO -0.00066957 0.37761498 -0.00000000\n"
        "H 0.80285178 -0.18859766 0.00000000\n"
        "H -0.80218221 -0.18901732 -0.00000000\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--config",
                config,
                "run-tool",
                "single_point",
                "--xyz",
                str(xyz),
                "--charge",
                "0",
                "--multiplicity",
                "1",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--config",
                config,
                "run-tool",
                "optimize_geometry",
                "--xyz",
                str(xyz),
                "--charge",
                "0",
                "--multiplicity",
                "1",
                "--execute",
            ]
        )
        == 0
    )
