from __future__ import annotations

from pathlib import Path

import pytest

from bg6022.cli import main


def test_preview_does_not_create_a_run(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false

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


def test_execute_with_missing_orca_is_rejected_without_runner_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("runner must not be called")

    monkeypatch.setattr("bg6022.tools.orca.run_orca", fail_if_called)
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
            "--execute",
        ]
    )
    assert code == 2
    assert "existing .exe" in capsys.readouterr().err
    assert not (tmp_path / "data" / "runs").exists()
