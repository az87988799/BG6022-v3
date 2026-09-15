# M0 acceptance record

This file records facts only. The user accepted M0 on 2026-09-11 after reviewing
the pushed implementation and the readable evidence package. The acceptance
record and the requested VS Code settings are included in the acceptance commit.

| Item | Evidence | Status |
|---|---|---|
| User acceptance | User reviewed the pushed implementation and explicitly requested M0 acceptance on 2026-09-11 | accepted |
| Independent install | Python 3.11.4 package build and package-outside-source help | passed |
| Implementation commit | `b1ba4e50dbaf1e14e8484db3e4aa521af80c0ac8` closes the parser, residual-process, and probe defects | passed |
| Offline regression | `46 passed, 1 deselected` with live ORCA disabled, including the new M0 minimal-fix regressions | passed |
| Original fixture | Three expected SHA-256 values and byte-preserving fixture copy | passed |
| Windows controls | Cancellation and parent-exits-before-child integration tests passed; residual-child case is failed with confirmed empty tree | passed |
| Probe | ORCA 6.1.1 banner probe returned `ok=true` with expected exit code 2, `nonzero_exit`, no stop request, and no guard | passed |
| Real SP | Run `run_5ec571c4d0d44a9fa8e4c173af2ab584`, `-76.417246084177 Eh`, ORCA 6.1.1, 4-core input/output evidence, active time persisted | passed |
| Real Opt | Run `run_17053fc626634282a6b090c14eeadd43`, `-76.418938721015 Eh`, converged final geometry, unchanged initial hash, active time persisted | passed |
| Failure facts | Existing run `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` replayed with the final parser; retained `opt_not_converged` and no success values | passed |
| Architecture boundary | No v2 runtime dependency, no fixed three-step protocol, two public tools | passed |

## M0 minimal repair evidence

The repair pass keeps the same seven durable objects and two Tool implementations.
It adds explicit execution permission and an accepted execution fingerprint, a
single lock-scoped execution guard, bounded runner teardown, a real active-time
clock, final-input hash checks, single-source Tool metadata, and pure preview/query
configuration loading. The final minimal fix additionally rejects a later
unpaired SCF convergence, revokes natural success when residual Job members must
be killed, and preserves probe timeout/output-limit failures. Malformed final
records remain diagnostics rather than success values.

The repaired real runs used Python 3.11.4, ORCA 6.1.1 at `E:\orca\orca.exe`,
`r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`, `%pal nprocs 4`,
`%maxcore 384`, a 2048 MB Job Object limit, and concurrency `1`. Both runs
reported `process_tree_empty=true`, `stop_confirmed=true`, and no remaining
`execution_guard.json`.

The historical fixture has these expected hashes:

| File | SHA-256 |
|---|---|
| `stdout.out` | `d05245e18d406d3d59e7a80ef607eaef889291b085cd2ba6691b17231bbd77af` |
| `geometry.xyz` | `9ad6fbb7bcc183d55ff471d122a4551907003fd53f9060bf2b4065057cc5a433` |
| `input.xyz` | `b5cb73f44717bf84cb411507be4b6d9b0921814da39a2c86dcbfbc1870db8027` |

The M0 real commands are:

```powershell
uv run python -m bg6022 --config config.toml doctor --probe-orca
uv run python -m bg6022 --config config.toml run-tool single_point --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
uv run python -m bg6022 --config config.toml run-tool optimize_geometry --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
```

The historical one-core fixture, fake processes, and offline parser tests never
count as the two required four-core real calculations.

## M1 implementation status

M1 code is present on the M1 implementation branch and has offline regression
coverage, but it is not marked accepted. The user must review the pushed commit
and run the live DeepSeek/PubChem/ORCA sequence described in
[`docs/milestones/M1.md`](milestones/M1.md) and
[`docs/evidence/M1-agent-evidence.md`](evidence/M1-agent-evidence.md) before
M1 is recorded as complete.

The 2026-09-15 parameter/query repair follow-up was pushed as
`09ccd60d9b9456a33fa55019dc93e3158458c80a`; its offline suite passed (104
passed, 3 live tests deselected). The local DeepSeek credential is unavailable,
so this follow-up adds no real Agent/ORCA acceptance evidence and does not
change M1's pending status.

## Recorded real runs

The local configuration used ORCA `E:\orca\orca.exe`, Program Version 6.1.1,
`r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`, `%pal nprocs 4`,
`%maxcore 384`, and a 2048 MB Job Object limit.

- SP: `run_a50fa185c4ca4870baa96aecfff46cb9`; parsed
  `sp_electronic_energy = -76.417246084177 Eh` (pre-repair baseline).
- Repair SP: `run_0d66e2666d4c44f680c888a2ded130bf`; parsed
  `sp_electronic_energy = -76.417246084177 Eh` and active time was persisted.
- Final-fix SP: `run_5ec571c4d0d44a9fa8e4c173af2ab584`; parsed
  `sp_electronic_energy = -76.417246084177 Eh`, with `process_tree_empty=true`
  and `stop_confirmed=true`.
- Opt: `run_9d169f9d6bb540c8a67c603b6ac8898a`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh` (pre-repair baseline).
- Repair Opt: `run_5a42ae6d30ef47499e3c794d15112f46`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh`; the final `input.xyz`
  was bound as `optimized_geometry`, and stdout geometry comparison passed.
- Final-fix Opt: `run_17053fc626634282a6b090c14eeadd43`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh`, with converged and
  consistent final geometry, `process_tree_empty=true`, and `stop_confirmed=true`.
- Controlled failure: `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` with
  `geom_maxiter=1`; ORCA exited normally but optimization did not converge.
  The Result has no successful geometry port and the Run retains a
  `restart_candidate` artifact for M1. Final-code replay remains
  `opt_not_converged` with no published value.

The initial geometry SHA-256 for both successful runs and their attempt copies
is `c4dd083de426421a719338bd0f0e17dc1057a88982e8cf245fb115556525ffd2`.

## 2026-09-15 M1 acceptance-review follow-up

Review `BG6022_v3_M1_Acceptance_Review_ba840e1.md` identified two regressions.
Code commit `8d1eb24ed706e38007007bec319d2507bb803b33` fixes both: q/M values
are now extracted only from a value expression bound to that field, and an
explicit unchanged statement cannot inherit an iteration-count number. Query
history now retains six distinct Runs (active plus up to five earlier Runs),
allocates references only for valid declared properties, prioritizes requested
Plan results, and keeps Step identity. The candidate catalog is bounded at 24
facts; the existing three-target per-query limit remains separate.

Offline validation on this commit: `115 passed, 3 live tests deselected`; Ruff
lint/format, `compileall`, `git diff --check`, and `uv build` passed. New
regressions include a full Agent parameter-continuation path whose actual
RDKit-generated water geometry retains charge 0/multiplicity 1 in the Request,
Opt Step and confirmation preview after changing only `geom_maxiter`; two
complete three-step stored Runs for current/history/comparison queries; bounded
distinct-Run retention; non-substitution across tasks; and distinct Opt/SP
Step subjects.

Live subchecks were run separately from L1/L2:

| Check | Evidence | Status |
|---|---|---|
| PubChem | `tests/live/test_m1_live.py -m live_pubchem`; real water CID 962 and ethanol CID 702 | passed |
| ORCA SP | Run `run_dd5c92654c9b4db89b64af5d2846bc78`; `-76.417246084177 Eh`; ORCA 6.1.1 | passed |
| ORCA Opt | Run `run_855cc60f6a8a4b538473be3db704e4f5`; `-76.418938721015 Eh`; convergence and geometry checks | passed |
| Budget and cleanup | Both Runs record 4 cores / 1024 MB / MaxCore 192 / concurrency 1, matching `%pal nprocs 4` and `%maxcore 192`; both have `process_tree_empty=true` | passed |
| ORCA probe | `doctor --probe-orca`; version 6.1.1, expected missing-input exit 2, clean process tree | passed |
| Natural-language L1/L2 | No DeepSeek request was made; `DEEPSEEK_API_KEY` is unset in this environment | pending |

The ORCA runs above used the direct Tool command in the live test, not the
DeepSeek Agent path. They do not satisfy L1 or L2. Full live evidence details
and raw paths are recorded in
[`docs/evidence/M1-agent-evidence.md`](evidence/M1-agent-evidence.md). M1
remains pending the real-LLM L1/L2 sequence and the user's explicit acceptance.
