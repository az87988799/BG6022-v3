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
    parser.addoption(
        "--run-live-llm",
        action="store_true",
        default=False,
        help="enable explicitly marked live DeepSeek tests",
    )
    parser.addoption(
        "--run-live-pubchem",
        action="store_true",
        default=False,
        help="enable explicitly marked live PubChem tests",
    )
    parser.addoption(
        "--enable-socket",
        action="store_true",
        default=False,
        help="allow network sockets for explicitly enabled live tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--enable-socket"):
        config.option.disable_socket = False
        config.option.force_enable_socket = True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "live_orca" in item.keywords and not config.getoption("--run-live-orca"):
            item.add_marker(
                pytest.mark.skip(reason="live_orca disabled; pass --run-live-orca explicitly")
            )
        if "live_llm" in item.keywords and not config.getoption("--run-live-llm"):
            item.add_marker(
                pytest.mark.skip(reason="live_llm disabled; pass --run-live-llm explicitly")
            )
        if "live_pubchem" in item.keywords and not config.getoption("--run-live-pubchem"):
            item.add_marker(
                pytest.mark.skip(reason="live_pubchem disabled; pass --run-live-pubchem explicitly")
            )
