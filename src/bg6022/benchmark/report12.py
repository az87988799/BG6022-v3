"""Versioned, inspectable reports for Benchmark 1.2 full-pipeline runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bg6022.config import AppConfig
from bg6022.planner import load_prompt
from bg6022.tools.registry import build_registry

from .dataset12 import Benchmark12Item, Benchmark12Suite, dataset_hashes
from .models import Benchmark12Result
from .observers import summarize_llm_calls


def build_environment12(
    *,
    repository_root: str | Path,
    benchmark_root: str | Path,
    config: AppConfig | None,
    profile: str,
    items: list[Benchmark12Item],
    seed: int | None = None,
    live_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    data_root = Path(benchmark_root).resolve()
    suite = _load_suite(data_root / "suite.json")
    registry = build_registry(config)
    prompts = {
        name: hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
        for name in ("semantic", "answer", "repair", "intake", "planner")
    }
    prompt_hashes = {
        item.id: str(item.metadata.get("prompt_sha256") or _hash_text(item.prompt))
        for item in items
    }
    manifest = dataset_hashes(data_root)
    geometry_hashes = {
        path.removeprefix("geometries/"): value
        for path, value in manifest.items()
        if path.startswith("geometries/")
    }
    return {
        "benchmark_version": suite.version,
        "suite_identity": {
            "name": suite.name,
            "version": suite.version,
            "schema": suite.schema_name,
        },
        "profile": profile,
        "official_profile": suite.official_profile,
        "prompt_generation": suite.prompt_generation,
        "dataset_sha256": manifest,
        "prompt_sha256_by_item": prompt_hashes,
        "geometry_sha256": geometry_hashes,
        "prompt_sha256": prompts,
        "prompt_variant_ids": sorted({item.variant_id for item in items}),
        "model": config.llm.model if config is not None else None,
        "tool_catalog_sha256": _stable_hash(registry.describe()),
        "method_catalog_sha256": _stable_hash(registry.method_capability_catalog()),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "branch": _git_value(root, "branch", "--show-current"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "resources": config.resources if config is not None else None,
        "live_flags": dict(live_flags or {}),
        "seed": seed,
        "official_acceptance_profile": seed is None,
    }


def write_report12(
    output_dir: str | Path,
    items: list[Benchmark12Item],
    results: list[Benchmark12Result],
    *,
    environment: dict[str, Any],
    skipped_items: list[str] | None = None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = summarize_results12(
        items,
        results,
        environment=environment,
        skipped_items=skipped_items or [],
        elapsed_seconds=elapsed_seconds,
    )
    _write_json(destination / "environment.json", environment)
    _write_json(destination / "summary.json", summary)
    (destination / "summary.md").write_text(render_summary12(summary), encoding="utf-8")
    (destination / "benchmark12.md").write_text(render_benchmark12(summary), encoding="utf-8")
    item_info = {item.id: _item_metadata(item) for item in items}
    with (destination / "items.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            payload = result.model_dump(mode="json")
            payload["item"] = item_info.get(result.item_id, {})
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    with (destination / "transcript.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            for turn in result.observation.turns:
                common = {"item_id": result.item_id, "turn_index": turn.index}
                handle.write(
                    json.dumps(
                        common | {"role": "user", "text": turn.user_message},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        common
                        | {
                            "role": "assistant",
                            "text": turn.response_text,
                            "run_status": turn.run_status,
                            "waiting_for": turn.waiting_for,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    failures = [item for item in results if not item.passed]
    if failures:
        failure_dir = destination / "failures"
        failure_dir.mkdir()
        for item in failures:
            _write_json(
                failure_dir / f"{item.item_id}_{item.run_index}.json",
                item.model_dump(mode="json"),
            )
    return summary


def summarize_results12(
    items: list[Benchmark12Item],
    results: list[Benchmark12Result],
    *,
    environment: dict[str, Any],
    skipped_items: list[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    dimension_values: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        for name, value in result.dimensions.items():
            if value is not None:
                dimension_values[name].append(value)
    dimensions = {
        name: {
            "passed": sum(values),
            "total": len(values),
            "rate": sum(values) / len(values) if values else None,
        }
        for name, values in dimension_values.items()
    }
    llm = summarize_llm_calls([call for item in results for call in item.observation.llm_calls])
    external: dict[str, int] = defaultdict(int)
    for result in results:
        for name, value in result.observation.external_calls.items():
            external[name] += int(value)
    result_rows = [
        {
            "case_id": item.item_id,
            "task_id": item.task_id,
            "object_id": item.object_id,
            "variant_id": item.variant_id,
            "run_index": item.run_index,
            "passed": item.passed,
            "dimensions": item.dimensions,
            "failed_stage": item.failed_stage,
            "error_category": item.error_category,
            "failures": item.failures,
        }
        for item in results
    ]
    items_by_id = {item.id: item for item in items}
    passed = sum(item.passed for item in results)
    return {
        "schema": "bg6022.benchmark.full_pipeline.v1",
        "benchmark_version": environment.get("suite_identity", {}).get("version", "1.2"),
        "suite_identity": environment.get("suite_identity"),
        "profile": environment.get("profile"),
        "run_id": environment.get("run_id"),
        "commit": environment.get("commit"),
        "branch": environment.get("branch"),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "skipped_items": list(skipped_items),
        "full_pipeline_success": {
            "passed": passed,
            "total": len(results),
            "rate": passed / len(results) if results else None,
        },
        "dimensions": dimensions,
        "by_task": _group(results, items_by_id, "task_id"),
        "by_object": _group(results, items_by_id, "object_id"),
        "by_prompt_variant": _group(results, items_by_id, "variant_id"),
        "by_input_mode": _group(results, items_by_id, "input_mode"),
        "by_stage": _group_stage(results),
        "task_object_matrix": _task_object_matrix(results),
        "boundary_safety": {
            "execution_safety_violations": sum(
                result.observation.pre_confirmation_orca_attempts > 0
                or result.dimensions.get("safety_correct") is False
                for result in results
            ),
            "unrequested_compute": sum(
                result.observation.orca_attempts > 0 and result.observation.plan is None
                for result in results
            ),
            "unexpected_recomputation": sum(
                result.dimensions.get("cost_within_bound") is False for result in results
            ),
        },
        "cost": {
            **llm,
            "orca_attempts": sum(item.observation.orca_attempts for item in results),
            "orca_successes": sum(item.observation.orca_successes for item in results),
            "orca_failures": sum(item.observation.orca_failures for item in results),
            "repair_attempts": sum(item.observation.repair_attempts for item in results),
            "external_calls": dict(external),
            "wall_time_seconds": max(0.0, float(elapsed_seconds)),
        },
        "environment": environment,
        "cases": result_rows,
    }


def render_summary12(summary: dict[str, Any]) -> str:
    full = summary["full_pipeline_success"]
    rate = "n/a" if full["rate"] is None else f"{full['rate']:.1%}"
    return "\n".join(
        [
            "# Benchmark 1.2 Summary",
            "",
            f"Profile: {summary.get('profile')}",
            f"Full pipeline: {full['passed']}/{full['total']} ({rate})",
            f"Skipped: {len(summary['skipped_items'])}",
            f"LLM calls: {summary['cost']['calls']} | ORCA attempts: "
            f"{summary['cost']['orca_attempts']}",
            "Execution safety violations: "
            f"{summary['boundary_safety']['execution_safety_violations']}",
            "",
        ]
    )


def render_benchmark12(summary: dict[str, Any]) -> str:
    identity = summary.get("suite_identity") or {}
    cost = summary["cost"]
    lines = [
        "# Benchmark 1.2",
        "",
        f"Suite: {identity.get('name')} {identity.get('version')} ({identity.get('schema')})",
        f"Profile: {summary.get('profile')}",
        f"Commit: {summary.get('commit') or 'unknown'}",
        f"Model: {summary.get('environment', {}).get('model') or 'not configured'}",
        "",
        "## Full pipeline success",
        "",
        f"{summary['full_pipeline_success']['passed']}/{summary['full_pipeline_success']['total']} "
        f"({(summary['full_pipeline_success']['rate'] or 0.0):.1%})",
        "",
        "## Scoring dimensions",
        "",
        "| Dimension | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sorted(summary["dimensions"].items()):
        rate = "n/a" if value["rate"] is None else f"{value['rate']:.1%}"
        lines.append(f"| {name} | {value['passed']} | {value['total']} | {rate} |")
    lines.extend(["", "## Task × Object", "", "| Task / Object | Result |", "|---|---:|"])
    for row in summary["task_object_matrix"]:
        lines.append(f"| {row['task_id']} × {row['object_id']} | {row['passed']}/{row['total']} |")
    for heading, key in (
        ("By task", "by_task"),
        ("By object", "by_object"),
        ("By prompt variant", "by_prompt_variant"),
        ("By input mode", "by_input_mode"),
        ("By stage", "by_stage"),
    ):
        lines.extend(
            ["", f"## {heading}", "", "| Group | Passed | Total | Rate |", "|---|---:|---:|---:|"]
        )
        for name, value in sorted(summary[key].items()):
            rate = "n/a" if value["rate"] is None else f"{value['rate']:.1%}"
            lines.append(f"| {name} | {value['passed']} | {value['total']} | {rate} |")
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"- LLM calls: {cost['calls']} ({cost['input_tokens']} input / "
            f"{cost['output_tokens']} output / {cost['total_tokens']} total tokens)",
            f"- ORCA: {cost['orca_attempts']} attempts, {cost['orca_successes']} successes, "
            f"{cost['orca_failures']} failures",
            f"- Repair attempts: {cost['repair_attempts']}",
            f"- PubChem requests: {cost['external_calls'].get('pubchem_requests', 0)}",
            "- RDKit geometry generations: "
            f"{cost['external_calls'].get('rdkit_geometry_generations', 0)}",
            f"- Wall time: {cost['wall_time_seconds']:.2f}s",
            "",
            "### LLM calls by purpose",
            "",
            "| Purpose | Calls | Input tokens | Output tokens | Total tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for purpose, value in sorted(cost["by_purpose"].items()):
        lines.append(
            f"| {purpose} | {value['calls']} | {value['input_tokens']} | "
            f"{value['output_tokens']} | {value['total_tokens']} |"
        )
    failed = [item for item in summary["cases"] if not item["passed"]]
    lines.extend(["", "## Failed items", ""])
    if failed:
        lines.extend(["| Item | Stage | Category | Failed dimensions |", "|---|---|---|---|"])
        for item in failed:
            lines.append(
                f"| {item['case_id']} | {item.get('failed_stage') or 'unknown'} | "
                f"{item.get('error_category') or 'product_failure'} | "
                f"{', '.join(item.get('failures', []))} |"
            )
    else:
        lines.append("No failed items.")
    if summary["skipped_items"]:
        lines.extend(["", "## Skipped items", ""])
        lines.extend(f"- {item}" for item in summary["skipped_items"])
    return "\n".join(lines).rstrip() + "\n"


def _group(results, items: dict[str, Benchmark12Item], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Benchmark12Result]] = defaultdict(list)
    for result in results:
        item = items.get(result.item_id)
        if item is None:
            value = "unknown"
        elif field == "input_mode":
            value = str(item.metadata.get("input_mode", "unknown"))
        else:
            value = str(getattr(item, field))
        groups[value].append(result)
    return {name: _pass_rate(group) for name, group in sorted(groups.items())}


def _group_stage(results: list[Benchmark12Result]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Benchmark12Result]] = defaultdict(list)
    for result in results:
        groups[result.failed_stage or "passed"].append(result)
    return {name: _pass_rate(group) for name, group in sorted(groups.items())}


def _task_object_matrix(results: list[Benchmark12Result]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Benchmark12Result]] = defaultdict(list)
    for result in results:
        groups[(result.task_id, result.object_id)].append(result)
    return [
        {
            "task_id": task_id,
            "object_id": object_id,
            "passed": sum(item.passed for item in grouped),
            "total": len(grouped),
        }
        for (task_id, object_id), grouped in sorted(groups.items())
    ]


def _pass_rate(results: list[Benchmark12Result]) -> dict[str, Any]:
    passed = sum(item.passed for item in results)
    return {
        "passed": passed,
        "total": len(results),
        "rate": passed / len(results) if results else None,
    }


def _item_metadata(item: Benchmark12Item) -> dict[str, Any]:
    return {
        "task_id": item.task_id,
        "object_id": item.object_id,
        "variant_id": item.variant_id,
        "kind": item.metadata.get("kind"),
        "input_mode": item.metadata.get("input_mode"),
        "prompt_sha256": item.metadata.get("prompt_sha256") or _hash_text(item.prompt),
        "label": item.metadata.get("label"),
    }


def _load_suite(path: Path) -> Benchmark12Suite:
    try:
        return Benchmark12Suite.model_validate(
            json.loads(path.read_text(encoding="utf-8")), strict=True
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load Benchmark 1.2 suite identity: {error}") from error


def _git_value(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "build_environment12",
    "render_benchmark12",
    "render_summary12",
    "summarize_results12",
    "write_report12",
]
