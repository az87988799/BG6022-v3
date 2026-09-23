# BG6022-v3 Benchmarks

Benchmark v1 is a regression tool for Request normalization, Tool planning, geometry provenance, scientific result boundaries, query/replay behavior, and repair. It is a test-only package under `src/bg6022/benchmark/`; the production Agent does not import it.

## Default run

From the repository root, run:

```powershell
.venv\Scripts\python.exe -m bg6022.benchmark run benchmarks/v1
```

This runs the offline and replay cases only. It makes no LLM, PubChem, or ORCA calls. Reports are written to `benchmarks/results/<timestamp>_<commit>/` and are ignored by Git.

## Explicit live runs

Live work requires a TOML configuration and explicit flags:

```powershell
.venv\Scripts\python.exe -m bg6022.benchmark run benchmarks/v1 --live-llm --config config.toml
.venv\Scripts\python.exe -m bg6022.benchmark run benchmarks/v1 --live-orca --config config.toml
.venv\Scripts\python.exe -m bg6022.benchmark run benchmarks/v1 --live-llm --live-orca --config config.toml
```

Cases marked `requires_pubchem` also require `--live-pubchem`. ORCA cases run under `E:\BG6022-v3-benchmark-data\bench_<run-id>\` by default (or `--data-root`); their runtime budget must match 4 cores, 1024 MB total memory, `%maxcore 192`, and one concurrent job. The first repair case is intentionally controlled and only passes when its first real optimization attempt fails before a bounded repair succeeds.

Holdout cases stay out of routine runs. Include them explicitly with `--include-holdout` after selecting the required live modes.

## Reports and comparison

Each run writes `environment.json`, `summary.json`, `summary.md`, `cases.jsonl`, and per-case failure records. Environment metadata contains prompt/tool/method hashes and resource settings, never API credentials.

Compare two runs with:

```powershell
.venv\Scripts\python.exe -m bg6022.benchmark compare benchmarks/results/run_A benchmarks/results/run_B
```

The summary reports separate capability dimensions, live/replay pass rates, LLM usage, ORCA attempts, critical violations, and computation/recomputation counts. It does not collapse results into one score.
