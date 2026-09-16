from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.llm import LlmClient
from bg6022.tools.registry import ToolRegistry, build_registry
from tests.support.geometry_angle_tool import make_geometry_angle_tool


@pytest.mark.live_llm
def test_live_llm_geometry_angle_and_csv_followup(pytestconfig) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live_llm")
    config_path = pytestconfig.getoption("--orca-config")
    if not config_path:
        pytest.fail("--orca-config is required for live_llm")
    config = load_config(config_path)
    base = build_registry(config)
    registry = ToolRegistry(
        [base.get(name) for name in base.names()] + [make_geometry_angle_tool(config)]
    )
    client = LlmClient(config)
    agent = Agent(config, registry, llm=client)
    xyz = "3\nright angle\nH 1.0 0.0 0.0\nO 0.0 0.0 0.0\nH 0.0 1.0 0.0\n"
    first = agent.handle_message(
        "请调用已注册的 geometry_angle 工具，使用下面的 XYZ，以第 2 个原子为顶点测量 "
        "第 1、2、3 个原子的夹角，同时返回所选原子记录和真实 CSV 文件。\n\n" + xyz
    )
    assert first.run is not None
    assert first.run.status == "succeeded"
    assert first.result is not None
    assert first.result.values["angle_value"]["value"] == pytest.approx(90.0)
    assert len(first.result.values["selected_atoms"]) == 3
    assert len(first.files) == 1
    report = first.files[0]
    report_hash = hashlib.sha256(Path(report["path"]).read_bytes()).hexdigest()
    assert report_hash == report["sha256"]
    assert "atom_index,element,x,y,z,raw_line" in first.text

    calls_after_first = len(client.calls)
    second = agent.handle_message("把刚才的 CSV 内容给我")
    assert second.run is not None
    assert second.run.id == first.run.id
    assert len(second.files) == 1
    assert second.files[0]["artifact_id"] == report["artifact_id"]
    assert second.files[0]["sha256"] == report["sha256"]
    assert "atom_index,element,x,y,z,raw_line" in second.text
    assert len(client.calls) > calls_after_first

    evidence = {
        "case": "V21-geometry-angle",
        "config": str(config_path),
        "run_id": first.run.id,
        "result_status": first.result.status,
        "angle": first.result.values["angle_value"],
        "artifact_id": report["artifact_id"],
        "sha256": report["sha256"],
        "first_delivery": first.delivery,
        "second_delivery": second.delivery,
        "call_metadata": [call.__dict__ for call in client.calls],
    }
    evidence_root = Path(config.data_root_path) / "m3-llm-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "V21-geometry-angle.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
