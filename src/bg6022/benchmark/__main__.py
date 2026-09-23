"""Command-line interface for repeatable, zero-cost-by-default benchmarks."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bg6022.config import load_config

from .compare import BenchmarkReportError, compare_reports, render_comparison_markdown
from .grader import grade_case
from .loader import BenchmarkConfigurationError, load_cases
from .models import CaseObservation
from .report import build_environment, write_report
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


if __name__ == "__main__":
    raise SystemExit(main())
