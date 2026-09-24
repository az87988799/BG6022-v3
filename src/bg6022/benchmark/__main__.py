"""Command-line interface for repeatable, zero-cost-by-default benchmarks."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bg6022.config import load_config

from .compare import BenchmarkReportError, compare_reports, render_comparison_markdown
from .dataset12 import Benchmark12Item, collect_items
from .e2e12 import Benchmark12DriverError, failed_observation, run_item
from .grader import grade_case
from .grader12 import grade_item
from .loader import BenchmarkConfigurationError, load_cases
from .matrix import expand_matrix
from .models import Benchmark12Observation, CaseObservation
from .report import build_environment, write_matrix_report, write_report
from .report12 import build_environment12, write_report12
from .runner import BenchmarkModeDisabled, run_case

_DEPENDENCY_CATEGORIES = {
    "missing_api_key",
    "live_dependency_missing",
    "auth",
    "network_error",
    "rate_limited",
    "temporary_failure",
    "executable_missing",
    "environment_error",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bg6022.benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run offline/replay cases and explicitly enabled live cases"
    )
    run.add_argument("benchmark_dir", nargs="?", default="benchmarks/v1")
    run.add_argument("--config", help="TOML configuration for live cases")
    run.add_argument("--live-llm", action="store_true", help="allow real LLM calls")
    run.add_argument("--live-orca", action="store_true", help="allow real ORCA calculations")
    run.add_argument("--live-pubchem", action="store_true", help="allow real PubChem requests")
    run.add_argument("--include-holdout", action="store_true", help="include frozen holdout cases")
    run.add_argument("--output-dir", help="write results to this directory")
    run.add_argument("--data-root", help="isolated root for live ORCA data")

    compare = commands.add_parser("compare", help="compare two benchmark reports")
    compare.add_argument("baseline")
    compare.add_argument("candidate")

    matrix = commands.add_parser(
        "matrix", help="run the explicitly enabled real-ORCA scientific matrix"
    )
    matrix.add_argument("matrix_dir", nargs="?", default="benchmarks/scientific_v1")
    matrix.add_argument(
        "--live-orca",
        action="store_true",
        required=True,
        help="allow real ORCA calculations for every enabled matrix cell",
    )
    matrix.add_argument("--config", required=True, help="TOML configuration for ORCA")
    matrix.add_argument("--output-dir", help="write results to this directory")
    matrix.add_argument("--data-root", help="isolated root for live ORCA data")

    bench12 = commands.add_parser(
        "bench12", help="run the frozen natural-language full-pipeline Benchmark 1.2"
    )
    bench12.add_argument("benchmark_dir", nargs="?", default="benchmarks/benchmark1.2")
    bench12.add_argument("--profile", choices=("smoke", "core", "full"), default="smoke")
    bench12.add_argument("--config", help="TOML configuration for live Agent and ORCA runs")
    bench12.add_argument("--live-llm", action="store_true", help="allow real LLM calls")
    bench12.add_argument("--live-orca", action="store_true", help="allow real ORCA calculations")
    bench12.add_argument("--live-pubchem", action="store_true", help="allow real PubChem requests")
    bench12.add_argument("--output-dir", help="write the report to this directory")
    bench12.add_argument("--data-root", help="isolated root for live Agent/ORCA data")
    bench12.add_argument(
        "--include-holdout", action="store_true", help="include the frozen holdout"
    )
    bench12.add_argument(
        "--sample-variants", type=int, default=0, help="seeded development sampling"
    )
    bench12.add_argument(
        "--seed", type=int, help="fixed seed for development-only variant sampling"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        try:
            result = compare_reports(args.baseline, args.candidate)
        except BenchmarkReportError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(render_comparison_markdown(result), end="")
        return 0
    try:
        if args.command == "matrix":
            return _run_matrix(args)
        if args.command == "bench12":
            return _run_bench12(args)
        return _run(args)
    except (BenchmarkConfigurationError, BenchmarkReportError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    benchmark_dir = Path(args.benchmark_dir).resolve()
    repository_root = benchmark_dir.parent.parent
    cases = load_cases(benchmark_dir, include_holdout=args.include_holdout)
    config = load_config(args.config) if args.config else None
    selected = [case for case in cases if _case_selected(case, args)]
    skipped = [case.id for case in cases if case not in selected]
    environment = build_environment(
        repository_root=repository_root,
        config=config,
        benchmark_dir=benchmark_dir,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    commit = str(environment.get("commit") or "unknown")[:8]
    run_id = f"{timestamp}_{commit}"
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = benchmark_dir.parent / "results" / run_id
    data_root = (
        Path(args.data_root).resolve()
        if args.data_root
        else repository_root.parent / f"{repository_root.name}-benchmark-data" / f"bench_{run_id}"
    )

    started = time.monotonic()
    case_results = []
    for case in selected:
        try:
            observations = run_case(
                case,
                config=config,
                allow_live_llm=args.live_llm,
                allow_live_orca=args.live_orca,
                allow_live_pubchem=args.live_pubchem,
                benchmark_dir=benchmark_dir,
                data_root=data_root,
            )
        except BenchmarkModeDisabled as error:
            observation = CaseObservation(
                case_id=case.id,
                run_index=1,
                status="blocked",
                stage="complete",
                error_category="live_dependency_missing",
                error_message=str(error),
            )
            observations = [observation]
        for observation in observations:
            case_results.append(grade_case(case, observation))

    summary = write_report(
        output_dir,
        case_results,
        environment=environment,
        skipped_cases=skipped,
        elapsed_seconds=time.monotonic() - started,
    )
    print(f"Benchmark report: {output_dir}")
    print(f"Passed: {summary['passed_cases']}/{summary['case_count']}")
    print(
        f"LLM calls: {summary['cost']['calls']} | ORCA attempts: "
        f"{summary['cost']['orca_attempts']} | Critical assertion failures: "
        f"{summary['critical_assertion_failure_count']} | Execution safety violations: "
        f"{summary['execution_safety_violation_count']}"
    )
    if any(
        not result.passed and result.observation.error_category in _DEPENDENCY_CATEGORIES
        for result in case_results
    ):
        return 3
    return 0 if all(result.passed for result in case_results) else 1


def _case_selected(case, args: argparse.Namespace) -> bool:
    if case.mode in {"offline", "replay"}:
        return True
    if case.requires_llm and not args.live_llm:
        return False
    if case.requires_orca and not args.live_orca:
        return False
    if case.requires_pubchem and not args.live_pubchem:
        return False
    return case.mode in {"live_llm", "live_orca", "live_e2e"}


def _run_matrix(args: argparse.Namespace) -> int:
    matrix_dir = Path(args.matrix_dir).resolve()
    repository_root = matrix_dir.parents[1]
    expanded_cases = expand_matrix(matrix_dir)
    config = load_config(args.config)
    environment = build_environment(
        repository_root=repository_root,
        config=config,
        benchmark_dir=matrix_dir,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    commit = str(environment.get("commit") or "unknown")[:8]
    run_id = f"{timestamp}_{commit}"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else matrix_dir.parent / "results" / run_id
    )
    data_root = (
        Path(args.data_root).resolve()
        if args.data_root
        else repository_root.parent / f"{repository_root.name}-benchmark-data" / f"bench_{run_id}"
    )

    started = time.monotonic()
    case_results = []
    for expanded in expanded_cases:
        try:
            observations = run_case(
                expanded.case,
                config=config,
                allow_live_orca=args.live_orca,
                benchmark_dir=matrix_dir,
                data_root=data_root,
                fixture_override=expanded.fixture_override,
            )
        except BenchmarkModeDisabled as error:
            observations = [
                CaseObservation(
                    case_id=expanded.case.id,
                    run_index=1,
                    status="blocked",
                    stage="complete",
                    error_category="live_dependency_missing",
                    error_message=str(error),
                )
            ]
        case_results.extend(grade_case(expanded.case, item) for item in observations)

    elapsed_seconds = time.monotonic() - started
    summary = write_report(
        output_dir,
        case_results,
        environment=environment,
        elapsed_seconds=elapsed_seconds,
    )
    matrix_summary = write_matrix_report(output_dir, expanded_cases, case_results)
    print(f"Scientific matrix report: {output_dir}")
    print(f"Cells: {matrix_summary['cell_count']}")
    print(f"Passed runs: {matrix_summary['passed_runs']}/{matrix_summary['run_count']}")
    print(
        f"LLM calls: {summary['cost']['calls']} | ORCA attempts: {matrix_summary['orca_attempts']}"
    )
    print(f"Wall time: {matrix_summary['wall_time_seconds']:.2f}s")
    if any(
        not result.passed and result.observation.error_category in _DEPENDENCY_CATEGORIES
        for result in case_results
    ):
        return 3
    return 0 if all(result.passed for result in case_results) else 1


def _run_bench12(args: argparse.Namespace) -> int:
    benchmark_dir = Path(args.benchmark_dir).resolve()
    repository_root = benchmark_dir.parent.parent
    if args.include_holdout and args.profile != "full":
        raise BenchmarkConfigurationError("--include-holdout requires --profile full")
    if args.sample_variants and args.seed is None:
        raise BenchmarkConfigurationError("--sample-variants requires --seed")
    items = collect_items(
        benchmark_dir,
        profile=args.profile,
        include_holdout=args.include_holdout,
        sample_variants=args.sample_variants,
        seed=args.seed,
    )
    config = load_config(args.config) if args.config else None
    if args.live_orca and config is not None:
        _validate_live_benchmark_budget(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    commit = _git_short_hash(repository_root)
    run_id = f"benchmark12_{timestamp}_{commit}"
    environment = build_environment12(
        repository_root=repository_root,
        benchmark_root=benchmark_dir,
        config=config,
        profile=args.profile,
        items=items,
        seed=args.seed,
        live_flags={
            "live_llm": args.live_llm,
            "live_orca": args.live_orca,
            "live_pubchem": args.live_pubchem,
        },
    )
    environment["run_id"] = run_id
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else benchmark_dir.parent / "results" / run_id
    )
    data_root = (
        Path(args.data_root).resolve()
        if args.data_root
        else repository_root.parent / f"{repository_root.name}-benchmark-data" / run_id
    )

    started = time.monotonic()
    results = []
    skipped: list[str] = []
    dependency_failures = False
    for item in items:
        missing = _missing_live_requirements(item, args, config)
        if missing == ["--live-pubchem"]:
            skipped.append(item.id)
            continue
        for run_index in range(1, item.repeat + 1):
            if missing:
                observation = Benchmark12Observation(
                    item_id=item.id,
                    run_index=run_index,
                    final_status="blocked",
                    failed_stage="preflight",
                    error_category="live_dependency_missing",
                    final_response_text="Missing explicit live prerequisites: "
                    + ", ".join(missing),
                )
                dependency_failures = True
            else:
                try:
                    observation = run_item(
                        item,
                        config=config,
                        data_root=data_root,
                        run_index=run_index,
                    )
                except Benchmark12DriverError as error:
                    observation = failed_observation(
                        item, run_index, error=error, failed_stage="harness"
                    )
                except (OSError, ValueError) as error:
                    observation = failed_observation(
                        item, run_index, error=error, failed_stage="harness"
                    )
            result = grade_item(item, observation)
            results.append(result)
            if result.error_category in {
                "live_dependency_missing",
                "external_dependency_failure",
                "execution_environment_failure",
            }:
                dependency_failures = True

    environment["run_id"] = run_id
    summary = write_report12(
        output_dir,
        items,
        results,
        environment=environment,
        skipped_items=skipped,
        elapsed_seconds=time.monotonic() - started,
    )
    print(f"Benchmark 1.2 report: {output_dir}")
    print(
        f"Full pipeline: {summary['passed_cases']}/{summary['case_count']} | "
        f"Skipped: {len(skipped)}"
    )
    print(
        f"LLM calls: {summary['cost']['calls']} | ORCA attempts: "
        f"{summary['cost']['orca_attempts']} | Repairs: {summary['cost']['repair_attempts']}"
    )
    print(
        f"Execution safety violations: {summary['boundary_safety']['execution_safety_violations']}"
    )
    if dependency_failures:
        return 3
    return 0 if all(item.passed for item in results) and not skipped else 1


def _missing_live_requirements(
    item: Benchmark12Item, args: argparse.Namespace, config
) -> list[str]:
    required: list[str] = []
    if item.requires_llm and not args.live_llm:
        required.append("--live-llm")
    if item.requires_orca and not args.live_orca:
        required.append("--live-orca")
    if item.requires_pubchem and not args.live_pubchem:
        required.append("--live-pubchem")
    if required == ["--live-pubchem"]:
        return required
    if config is None:
        required.append("--config")
    return required


def _validate_live_benchmark_budget(config) -> None:
    expected = {"cores": 4, "memory_mb": 1024, "maxcore_mb": 192, "max_concurrent_jobs": 1}
    actual = {name: getattr(config.runtime, name) for name in expected}
    if actual != expected:
        raise BenchmarkConfigurationError(
            f"real ORCA benchmark budget must be {expected}; configured values are {actual}"
        )


def _git_short_hash(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
