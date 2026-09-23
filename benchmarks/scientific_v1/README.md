# Scientific Matrix v1

Scientific Matrix v1 measures real ORCA execution for a fixed set of canonical
Request/Plan fixtures. It does not call an LLM or PubChem and disables repair so
that each cell records the behavior of the scientific execution path itself.

## Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m bg6022.benchmark matrix `
    benchmarks\scientific_v1 `
    --live-orca `
    --config config.toml
```

The `--live-orca` flag is mandatory. The runner also checks the configured
compute budget against the project limit of 4 cores, 1024 MB total memory,
`%maxcore 192`, and one concurrent job. No API key is needed.

The selected matrix is deliberately not the full Cartesian product:

| Task | H2 | H2O | NH3 | CO2 | CH4 | ethanol |
|---|---:|---:|---:|---:|---:|---:|
| T001 r2SCAN-3c SP | ✓ | ✓ |  |  |  |  |
| T002 r2SCAN-3c Opt |  | ✓ | ✓ |  | ✓ | ✓ |
| T003 Opt → Freq |  | ✓ | ✓ | ✓ |  |  |
| T004 dual-SP energy difference |  | ✓ |  |  |  |  |
| T005 independent dual-Opt comparison |  | ✓ |  |  |  |  |

This is 11 cells and about 16 ORCA attempts if every cell reaches all its
planned operations without an ORCA-level failure. CO2 frequency behavior is
reported as observed; its expectation is not weakened to force a passing cell.

## Files

- `objects.json` declares neutral singlet molecules and their geometry files.
- `tasks.json` declares task templates, cost class, and bounded assertions.
- `matrix.json` lists the enabled task/object cells and optional repeat count.
- `task_templates/` contains ordinary intake and plan-proposal fixture data.
- `geometries/` is the only source of XYZ coordinates for each object.

Templates support only exact one-key markers of the form
`{"$object": "xyz_text"}`, `geometry`, `charge`, `multiplicity`, `id`, or
`label`. There is no expression language or arbitrary code evaluation. Each
expanded Request's inline XYZ must match the corresponding geometry file.

Each cell runs through the existing live ORCA benchmark path with a separate
data root. The output directory contains the standard `environment.json`,
`summary.json`, `summary.md`, `cases.jsonl`, and `failures/` files, plus:

- `matrix.json` with per-cell, per-task, and per-object counts, ORCA attempts,
  wall time, failed stage, and error category.
- `matrix.md` with the Task × Scientific Object table and the same breakdowns.

The first real matrix run establishes a baseline; scientifically meaningful
failures are retained rather than rewritten as expected successes.
