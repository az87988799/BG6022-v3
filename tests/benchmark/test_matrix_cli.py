from __future__ import annotations

import pytest

from bg6022.benchmark.__main__ import build_parser


def test_matrix_cli_requires_explicit_live_orca_permission() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["matrix", "benchmarks/scientific_v1", "--config", "config.toml"])

    assert error.value.code == 2


def test_matrix_cli_accepts_explicit_live_orca_and_required_config() -> None:
    args = build_parser().parse_args(
        ["matrix", "benchmarks/scientific_v1", "--live-orca", "--config", "config.toml"]
    )

    assert args.command == "matrix"
    assert args.live_orca is True
    assert args.config == "config.toml"
