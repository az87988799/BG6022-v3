from __future__ import annotations

from pathlib import Path

from bg6022.cli import main


def test_preview_does_not_create_a_run(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """[orca]
executable = 'E:\\orca\\orca.exe'

[runtime]
data_root = 'data'

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n", encoding="utf-8")
    code = main(
        [
            "--config",
            str(config),
            "run-tool",
            "single_point",
            "--xyz",
            str(xyz),
            "--charge",
            "0",
            "--multiplicity",
            "1",
        ]
    )
    assert code == 0
    assert "no calculation executed" in capsys.readouterr().out
    assert not (tmp_path / "data" / "runs").exists()
