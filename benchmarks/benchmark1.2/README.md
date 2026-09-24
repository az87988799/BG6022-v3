# BG6022 Benchmark 1.2

Benchmark 1.2 measures the production path from a frozen natural-language
prompt through `Agent.handle_message()`, public confirmation, registered Tools,
scientific Results and Artifacts, and deterministic ground-truth grading.
The harness never supplies a production Request or Plan and never invokes
`Agent.execute_plan()`.

## Dataset

- `objects.json` defines the six neutral, closed-shell structures.
- `tasks/` contains frozen prompt variants and scientific expectations for
  single-point, optimization, optimization-plus-frequency, same-geometry
  energy differences, and independent optimization comparisons.
- `matrix.json` defines the 11 canonical scientific cells.
- `scenarios.json` defines the five public conversation scenarios.
- `suite.json` freezes the official profile and two identity-lookup cases.
- `holdout.json` contains separate prompts and is included only on request.
- `geometries/` contains the same XYZ inputs used by `scientific_v1`; these are
  input structures, not a second ORCA output-fixture collection.

Prompt rendering accepts only `{object_id}`, `{label}`, `{xyz_text}`,
`{charge}`, and `{multiplicity}`. Every resolved geometry and data file stays
inside this dataset directory; symlinks and escaping paths are rejected.

## Profiles

- `smoke`: water with T001–T005, canonical variants.
- `core`: the 11 scientific cells, C001–C005, and two PubChem identity cases.
- `full`: core plus the frozen natural and compact water variants. Add
  `--include-holdout` to run the separate holdout set.

Official profiles are deterministic. Development-only prompt sampling requires
both `--sample-variants N` and a fixed `--seed`; the report records the seed.

## Run

No external dependency is called unless its live flag is supplied. A config is
required whenever an Agent is run. Real ORCA runs must use the repository
budget: 4 cores, 1024 MB total memory, `%maxcore 192`, and one concurrent job.

```powershell
uv run python -m bg6022.benchmark bench12 benchmarks\\benchmark1.2 `
  --profile smoke `
  --config config.toml `
  --live-llm `
  --live-orca
```

Add `--live-pubchem` to run identity-lookup cases. Reports are written below
`benchmarks/results/` unless `--output-dir` is supplied. Each report includes
the dataset and prompt hashes, environment, per-item observations, transcript,
costs by LLM purpose, task/object/variant/stage breakdowns, and failure details.

The harness sends the public message `确认` only after the Agent reports
`waiting_for == "confirmation"`. A clarification or rejected unchanged plan
does not receive automatic confirmation.
