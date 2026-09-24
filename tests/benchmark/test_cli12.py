from __future__ import annotations

import json
from pathlib import Path

from bg6022.benchmark import __main__ as benchmark_cli
from bg6022.benchmark.models import Benchmark12Observation

DATASET = Path(__file__).parents[2] / "benchmarks" / "benchmark1.2"


def test_bench12_cli_is_registered_with_explicit_live_flags() -> None:
    args = benchmark_cli.build_parser().parse_args(
        [
            "bench12",
            str(DATASET),
            "--profile",
            "core",
            "--live-llm",
            "--live-orca",
            "--live-pubchem",
            "--seed",
            "20260924",
        ]
    )
    assert args.command == "bench12"
    assert args.profile == "core"
    assert args.live_llm and args.live_orca and args.live_pubchem
    assert args.seed == 20260924


def test_no_live_flags_produce_blocked_report_without_running_agent(tmp_path, monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("Agent must not run without explicit live flags")

    monkeypatch.setattr(benchmark_cli, "run_item", unexpected_run)
    output = tmp_path / "report"
    result = benchmark_cli.main(
        ["bench12", str(DATASET), "--profile", "smoke", "--output-dir", str(output)]
    )
    assert result == 3
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 5
    assert summary["failed_cases"] == 5
    assert summary["cost"]["calls"] == 0
    assert summary["cost"]["orca_attempts"] == 0
    assert all(case["error_category"] == "live_dependency_missing" for case in summary["cases"])


def test_core_without_live_flags_does_not_run_identity_or_scientific_cases(
    tmp_path, monkeypatch
) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("Agent must not run without explicit live flags")

    monkeypatch.setattr(benchmark_cli, "run_item", unexpected_run)
    output = tmp_path / "report"
    result = benchmark_cli.main(
        ["bench12", str(DATASET), "--profile", "core", "--output-dir", str(output)]
    )
    assert result == 3
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_items"] == []
    assert summary["case_count"] == 18
    assert summary["cost"]["calls"] == 0


def test_identity_cases_are_skipped_when_only_pubchem_permission_is_missing(
    tmp_path, monkeypatch
) -> None:
    def offline_stub(item, *, run_index, **_kwargs):
        return Benchmark12Observation(
            item_id=item.id,
            run_index=run_index,
            final_status="stubbed",
            error_category="product_failure",
        )

    monkeypatch.setattr(benchmark_cli, "run_item", offline_stub)
    output = tmp_path / "report"
    result = benchmark_cli.main(
        [
            "bench12",
            str(DATASET),
            "--profile",
            "core",
            "--config",
            str(DATASET.parents[1] / "config.example.toml"),
            "--live-llm",
            "--live-orca",
            "--output-dir",
            str(output),
        ]
    )
    assert result == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["skipped_items"]) == {"I001_water_name_opt", "I002_ethanol_name_opt"}


def test_live_flags_without_config_do_not_start_agent(tmp_path, monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("A config is required before the Agent can start")

    monkeypatch.setattr(benchmark_cli, "run_item", unexpected_run)
    output = tmp_path / "report"
    result = benchmark_cli.main(
        [
            "bench12",
            str(DATASET),
            "--profile",
            "smoke",
            "--live-llm",
            "--output-dir",
            str(output),
        ]
    )
    assert result == 3
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["cost"]["calls"] == 0
