# M2 final minimal-fix implementation and acceptance evidence

- **Code revision:** `cc11043` on `codex/v3-m2-implementation`
- **Final cleanup commit:** `01ceed4` (test isolation and evidence correction)
- **Windows CI:** [offline run 35051882558](https://github.com/az87988799/BG6022-v3/actions/runs/35051882558), commit `01ceed43447cba48fb2f8b68c888f0c9e286e3f2` — **success**
- **Date:** 2026-09-16 (Asia/Hong_Kong)
- **Live evidence root:** `E:\BG6022-v3-data\acceptance_live_20260916_finalplan`
- **Machine-readable file index:** [`M2-final-live-run-index.json`](M2-final-live-run-index.json)
- **Credential policy:** the live harness loaded the configured key privately; no key, prompt secret, or response body was written to the repository evidence.

This is the final candidate evidence for the supplied M2 minimal-fix plan. The
live checks used production `Agent.handle_message` and `Agent.confirm`, real
`deepseek-flash`, and ORCA 6.1.1. Each preview was checked before confirmation.
The active budget remained 4 cores, 1024 MB total, `%maxcore 192`, one
concurrent job, 1200 seconds per attempt, and 3600 seconds per Run.

## Implemented boundaries

- LLM JSON classification now safely handles objects, nested arrays, nulls, and
  other malformed shapes while retaining the existing single correction budget.
- Intake parameter names and result targets come from the live Tool registry;
  `geom_maxiter` is scoped to Opt and unknown names are rejected before Run
  creation.
- Empty parameter-only patches remain on the current pending Run and do not
  repeat molecule or geometry preparation. New operations, targets, sources,
  and non-empty bindings leave that fast path.
- Unsupported results such as Gibbs free energy are preserved as unresolved
  requirements and stop before Planner/Run creation.
- Inline XYZ is removed from the Intake model input, retained locally byte for
  byte, and seeded as the immutable initial geometry. `Agent.handle_message`
  preserves the message's trailing newline so the exact block hash survives.
- The result catalog exposes preparation outputs while keeping the old
  `geometry` alias compatible for saved Requests. Historical geometry selection
  is separate from new molecule/XYZ input and is verified before copying.

## Final live confirmation checks

| Check | Final evidence | Result |
|---|---|---|
| P1: three independent requests for water Opt + Freq with “几何优化迭代上限为100” | `records/P1-1-preview.json`, `P1-2-preview.json`, `P1-3-preview.json` and canceled snapshots | Passed. All three previews waited for confirmation; Opt had `geom_maxiter=100`, Freq did not; no ORCA started. |
| P2: water SP electronic energy plus Gibbs free energy | `records/P2-unsupported-gibbs.json` | Passed. The unsupported free-energy requirement was shown explicitly; Planner calls and new Runs were zero. |
| P3: new SP while an Opt preview was pending | `records/P3-cancelled-runs.json` | Passed. The SP was a distinct SP-only Run; the pending Opt plan was unchanged; both were canceled before confirmation. |
| P4: SP on the just-optimized historical structure | `records/P4-history-sp-canceled.json` and `P4-history-sp-explicit-state-preview.json` | Passed. After a safe charge/multiplicity clarification, the preview was SP-only and its copied input SHA-256 equaled the successful Opt artifact; it was canceled before confirmation. |

The first P4 wording produced a clarification because charge and multiplicity
were not explicit. A subsequent request supplied neutral charge 0 and singlet
multiplicity 1. The clarification and the earlier malformed-model safe stop
remain in the records and are not counted as successful calculations.

## Final live ORCA Runs

### R1 — water Opt to independent SP

`records/R1-science-checks.json` records Run
`run_8af28d593f034d0ea29ae933c88adf15`. The real model created the requested
resolve → geometry → Opt → independent SP plan. The confirmation-time patch
changed only Opt `geom_maxiter` from 50 to 100, kept the same Run and target
set, and made zero Planner calls. After confirmation, both science attempts
succeeded. The initial geometry artifact and optimized geometry artifact have
different SHA-256 values; the SP input artifact is exactly the successful Opt
output. SP energy is `-76.418938721008 Eh`, within `1e-6 Eh` of the fixed
same-profile reference.

### R2 — ethanol bounded failure, repair, Freq, and independent SP

`records/R2-science-checks.json` records Run
`run_0f3504469c0740e5867fdf31eefa94a8`. The supplied 9-atom XYZ is retained as
the initial artifact with SHA-256
`54a62568f2c998ad703ad4d2cea17fe95be994802d3bf99a186143d1dbab463b`.

- Opt attempt 1 ended as `opt_not_converged` and exposed no successful geometry
  or energy.
- The real Repair stage used one bounded schema correction and selected
  `restart_optimization`; the validated patch changed only `geom_maxiter` from
  1 to 100 and used the hashed restart candidate.
- Opt attempt 2 succeeded. Freq and SP both consumed its
  `optimized_geometry` artifact, whose geometry SHA-256 is
  `7550520a8bf67d1b9a2b90ed4c0d40cfa2864f2cb34d5aea4759a6ed5f9a6902`.
- Freq returned 27 modes and a valid Hessian. SP returned
  `-155.002350059646 Eh`. One extra ORCA execution was used, within the Run
  budget. Generated `.inp` files contain `%pal` with `nprocs 4` and
  `%maxcore 192`.

The punctuation variant with `multiplicity=1.` was safely rejected before Run
creation because the value token was invalid; its record is
`records/R2-intake-punctuation-rejected.json`. The final R2 request uses a
semicolon boundary and preserves the exact XYZ bytes.

## Offline gates

The final Windows frozen-dependency checks passed:

```
245 passed, 2 skipped, 1 deselected
ruff check .                         passed
ruff format --check .                passed (68 files)
python -m compileall -q src          passed
uv build                             passed
git diff --check                     passed
```

The focused root-cause, parameter-scoping, plan-composition, and LLM tests
also pass (`73 passed`). The full file-level index contains 417 credential-free
records and raw artifacts with size and SHA-256 metadata.

## Acceptance state

Implementation, offline validation, and final-candidate live evidence are
recorded. M2 is **awaiting user acceptance** and is not marked complete. M1
acceptance remains a separate pending decision.
