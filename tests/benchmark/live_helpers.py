from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bg6022.config import load_config
from bg6022.llm import LlmCall, LlmClient


def make_live_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[orca]\nexecutable = "orca.exe"\n'
        '[runtime]\ndata_root = "runtime-data"\n'
        "cores = 4\nmemory_mb = 1024\nmaxcore_mb = 192\n"
        "max_concurrent_jobs = 1\nconfirm_before_compute = false\n",
        encoding="utf-8",
    )
    return load_config(config_path)


def install_json_handler(
    monkeypatch,
    handler: Callable[[str, list[dict[str, str]]], dict[str, Any]],
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def complete_json(
        client: LlmClient,
        messages,
        schema,
        *,
        purpose="json",
        example=None,
        cancel=None,
        remaining_timeout_seconds=None,
    ):
        materialized = list(messages)
        if materialized and isinstance(materialized[-1].get("content"), str):
            try:
                context = json.loads(materialized[-1]["content"])
            except json.JSONDecodeError:
                context = {}
        else:
            context = {}
        captured.append({"purpose": purpose, "context": context})
        payload = handler(purpose, materialized)
        client.calls.append(
            LlmCall(
                purpose=purpose,
                request_model=client.settings.model,
                schema_version=None,
                elapsed_seconds=0.0,
                usage={},
            )
        )
        return schema.model_validate(payload, strict=True)

    monkeypatch.setattr(LlmClient, "complete_json", complete_json)
    return captured
