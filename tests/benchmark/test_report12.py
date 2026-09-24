from __future__ import annotations

import json

from bg6022.benchmark.grader12 import grade_item
from bg6022.benchmark.report12 import write_report12
from tests.support.benchmark12 import t003_fixture


def test_report_writes_environment_summary_items_and_transcript_without_ground_truth(
    tmp_path,
) -> None:
    item, observation = t003_fixture()
    result = grade_item(item, observation)
    output = tmp_path / "report"
    summary = write_report12(
        output,
        [item],
        [result],
        environment={
            "run_id": "bench12_test",
            "suite_identity": {
                "name": "benchmark1.2",
                "version": "1.2",
                "schema": "bg6022.benchmark.full_pipeline.v1",
            },
            "profile": "smoke",
            "commit": "abc123",
            "model": "offline-test",
        },
        elapsed_seconds=1.5,
    )
    assert summary["full_pipeline_success"]["passed"] == 1
    assert summary["by_task"]["T003"]["total"] == 1
    assert summary["by_object"]["water"]["passed"] == 1
    assert summary["by_prompt_variant"]["canonical"]["passed"] == 1
    assert summary["cost"]["orca_attempts"] == 2
    for name in (
        "environment.json",
        "summary.json",
        "summary.md",
        "benchmark12.md",
        "items.jsonl",
        "transcript.jsonl",
    ):
        assert (output / name).is_file()
    item_record = json.loads((output / "items.jsonl").read_text(encoding="utf-8"))
    assert item_record["passed"] is True
    assert "ground_truth" not in item_record
    transcript = [
        json.loads(line)
        for line in (output / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["role"] for entry in transcript] == ["user", "assistant", "user", "assistant"]
    assert transcript[2]["text"] == "确认"


def test_report_writes_failure_detail_for_failed_item(tmp_path) -> None:
    item, observation = t003_fixture(bad_dependency=True)
    result = grade_item(item, observation)
    output = tmp_path / "report"
    write_report12(
        output,
        [item],
        [result],
        environment={"suite_identity": {"name": "benchmark1.2", "version": "1.2", "schema": "v1"}},
    )
    failure = output / "failures" / "T003__water__canonical_1.json"
    assert failure.is_file()
    assert "plan_correct" in json.loads(failure.read_text(encoding="utf-8"))["failures"]
