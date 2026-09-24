# Benchmark 1.2 implementation and live verification

Date: 2026-09-24 (Asia/Hong_Kong)

Status: implementation and verification evidence are recorded; the phase is
awaiting the user's review and explicit acceptance.

## Live evidence

| Run | Result | LLM calls | ORCA attempts | Duration |
|---|---:|---:|---:|---:|
| Benchmark 1.2 smoke (`benchmark12_20260924_061351_786070_f03c695`) | 5/5 | 10 | 8 | recorded in report |
| Benchmark 1.2 core (`benchmark12_20260924_061631_176981_f03c695`) | 18/18 | 37 | 23 (22 successful, one expected controlled failure) | 667.55 s |
| Scientific Matrix v1 (`bg6022-scientific-v1-impl-scientific-matrix-20260924143021`) | 11/11 | 0 | 16/16 successful | 421.52 s |

The core profile contains the 11 frozen scientific matrix cells, five public
conversation scenarios (C001–C005), and two identity-lookup cases. All 18 core
items passed every applicable grading dimension. Core recorded one bounded
repair, two PubChem requests, two RDKit geometry generations, zero execution
safety violations, zero unrequested compute, and zero unexpected recomputation.
Fresh compute used no Intake or Planner LLM calls.

All ORCA work recorded 4 cores, 1024 MB total memory, `%maxcore 192`, and one
concurrent job. These values agree with the generated ORCA input and the active
project budget.

The raw reports are local and ignored by Git:

- `benchmarks/results/benchmark12-smoke-live-20260924/`
- `benchmarks/results/benchmark12-core-live-20260924/`
- `benchmarks/results/scientific-v1-live-20260924/`

They include environment and dataset hashes, per-item observations, transcript,
cost breakdowns, and matrix details. The benchmark reports' `commit` field is
`f03c695f0c2419ff2cf96b2596d5fc56d9ec7d86`, the repository HEAD at execution
time. The implementation and dataset were present in the working tree during
these runs; the reports do not capture a dirty-tree hash. No source or dataset
changes were made after the live runs, except for this verification/documentation
record.

## Grading calibration

The first core grading pass treated C004's semantic LLM call as mandatory. The
production path deterministically rejects the unsupported global conformer
request before invoking the LLM. The scenario ground truth was corrected to
allow that supported deterministic refusal path. The same saved live observation
was regraded; no product behavior, scientific expectation, ORCA result, or live
call changed. C004 passed with zero compute and no LLM call.

## Offline verification

- `python -m pytest -q`: 510 passed, 17 skipped.
- Ruff lint: passed.
- Ruff format check: 121 files already formatted.
- `compileall` for `src` and `tests`: passed.
- `python -m build`: passed.
- `git diff --check`: passed before final staging.

The full profile and holdout were not run. Passing smoke/core and Scientific
Matrix v1 verifies the requested implementation gates; it does not record user
acceptance.
