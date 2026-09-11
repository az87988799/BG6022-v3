from __future__ import annotations

import os

import pytest

from bg6022.config import load_config
from bg6022.llm import LlmClient
from bg6022.planner import intake_message
from bg6022.tools.pubchem import fetch_pubchem


@pytest.mark.live_pubchem
def test_live_pubchem_water_and_ethanol(pytestconfig) -> None:
    config_path = pytestconfig.getoption("--orca-config") or "config.example.toml"
    config = load_config(config_path)
    water = fetch_pubchem("water", "name", config=config)
    ethanol = fetch_pubchem("ethanol", "name", config=config)
    assert water.candidates[0].get("CID") == 962
    assert ethanol.candidates[0].get("CID") == 702


@pytest.mark.live_llm
def test_live_deepseek_intake(pytestconfig) -> None:
    key_name = os.environ.get("DEEPSEEK_API_KEY")
    if not key_name:
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config_path = pytestconfig.getoption("--orca-config") or "config.example.toml"
    config = load_config(config_path)
    value = intake_message(LlmClient(config), "What is an Opt calculation?")
    assert value.intent in {"chemistry_qa", "daily_qa", "context_query", "chemistry_compute"}
