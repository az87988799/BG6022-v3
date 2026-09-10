from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-orca",
        action="store_true",
        default=False,
        help="enable explicitly marked live ORCA tests",
    )
    parser.addoption(
        "--orca-config",
        action="store",
        default=None,
        help="configuration file for live ORCA tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live-orca"):
        return
    skip = pytest.mark.skip(reason="live_orca disabled; pass --run-live-orca explicitly")
    for item in items:
        if "live_orca" in item.keywords:
            item.add_marker(skip)
