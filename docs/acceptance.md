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

## Current milestone acceptance (2026-09-16)

The user explicitly requested “验收M1和M2” after reviewing the pushed
implementations and evidence. Both milestones are accepted on 2026-09-16
(Asia/Hong_Kong). Their detailed records are [`docs/milestones/M1.md`](milestones/M1.md),
[`docs/milestones/M2.md`](milestones/M2.md), and the linked evidence indexes.
M3 has not started.

## M1 implementation status

M1 implementation and evidence were verified, and the real Agent-path L1/L2
DeepSeek/PubChem/ORCA evidence was collected on 2026-09-15. The user reviewed
the pushed commit and evidence described in
[`docs/milestones/M1.md`](milestones/M1.md) and
[`docs/evidence/M1-agent-evidence.md`](evidence/M1-agent-evidence.md) and
explicitly accepted M1 on 2026-09-16.

The 2026-09-15 parameter/query repair follow-up was pushed as
`09ccd60d9b9456a33fa55019dc93e3158458c80a`; its offline suite passed (104
passed, 3 live tests deselected). Its environment check observed that
`DEEPSEEK_API_KEY` was absent from the directly launched Python process; it did
not inspect `.env`. The project-root `.env` did contain the key, and the user's
active chat was launched with `uv run --env-file .env`, which passes it to the
application. The earlier “credential unavailable” conclusion was incorrect.

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
| Natural-language L1/L2 | See the 2026-09-15 live Agent-path evidence below; real DeepSeek plans and ORCA runs passed | passed; accepted by user on 2026-09-16 |

The ORCA runs in the preceding table used direct Tool commands and are separate
from L1/L2. Full real-Agent evidence and raw paths are recorded in
[`docs/evidence/M1-agent-evidence.md`](evidence/M1-agent-evidence.md). M1 was
pending the user's explicit acceptance until the acceptance recorded above.

## 2026-09-15 live Agent-path acceptance evidence

The root `.env` contained a non-empty `DEEPSEEK_API_KEY`; the key value was not
printed or saved in evidence. The earlier check ran `.venv\Scripts\python.exe`
directly, while the application only reads the process environment and does not
load `.env` itself. The user's active command uses `uv --env-file .env`, and the
live Agent checks used that same environment-file mechanism.

The user's already-running `start_chat.py` held the configured
`E:\BG6022-v3-data\.chat.lock`. To avoid interrupting or bypassing that session,
the actual production `Agent.handle_message` / `/confirm` flow ran with the same
`config.toml`, ORCA executable, and resource limits but an isolated data root at
`E:\BG6022-v3-data\acceptance_live_20260915`. The CLI wrapper itself was not
started concurrently.

| Check | Evidence | Status |
|---|---|---|
| Live DeepSeek smoke test | `tests/live/test_m1_live.py -m live_llm`; real intake request | passed |
| L1 normal Opt | Run `run_fee3692d4fa34244b4f68452062cb2e2`; PubChem water CID 962, 3-atom RDKit ETKDGv3 seed 61453 geometry, real LLM Plan, confirmation, ORCA 6.1.1; `-76.418938720831 Eh`; convergence, geometry consistency, input hashes, and process cleanup verified | passed |
| L2 controlled failure and repair | Run `run_24854da4f43346cabe67a730bfa00891`; PubChem ethanol CID 702, 9-atom structure; real ORCA attempt 1 used `geom_maxiter=1`, exited normally but did not converge; validated restart candidate and clean process; real LLM selected the allowed `restart_optimization` patch to `geom_maxiter=100`; attempt 2 in the same Run converged at `-155.002350062010 Eh` with consistent geometry and clean process | passed |
| Resource agreement | Both Runs: 4 cores, 1024 MB total, `%maxcore 192`, concurrency 1; generated inputs agree (`nprocs 4`, `%maxcore 192`) | passed |
| M1 status | L1/L2 evidence exists and was accepted by the user on 2026-09-16 | accepted |

The L2 artifacts retain the initial failed result, restart-candidate hash,
LLM-selected repair record, and both ORCA attempts. Earlier non-passing live
attempts in the same isolated evidence root are retained but are not counted as
acceptance: `run_87598d747ff8468088159f395ad0bba8` exposed the ORCA “maximum
number of optimization cycles” parser gap, and `run_af02fdbc9b9d4887a093d21dfe8cecbe` exposed repair-Plan topological-order replay. Both defects are fixed and covered by regression tests.

## 2026-09-15 M1 verification record

This record documents verification evidence for the M1 baseline; it does not
record user acceptance. The verified baseline is branch `codex/v3-m1-agent` at
`7cfd814d9ee64672b92d0a4301c2b079a97f6b1c`.

| Check | Evidence | Status |
|---|---|---|
| GitHub Actions | `offline` workflow run `34969764361` completed successfully | passed |
| Local offline validation | `119 passed, 3 live tests deselected`; Ruff lint and format, `compileall`, and `uv build` passed | passed |
| Chat entry point | With an isolated temporary data root, the chat command reached its interactive prompt and `/exit` returned code 0 | passed |
| ORCA installation | `E:\\orca\\orca.exe` exists; recorded run outputs identify ORCA 6.1.1 | verified |
| L1 artifact integrity | For water Run `run_fee3692d4fa34244b4f68452062cb2e2`, every artifact's size and SHA-256 match its raw `run.json` index; the optimization succeeded, energy and geometry checks passed, and resources agree at 4 cores / 1024 MB / `%maxcore 192` / concurrency 1 | passed |
| L2 artifact integrity and repair | For ethanol Run `run_24854da4f43346cabe67a730bfa00891`, every artifact's size and SHA-256 match its raw `run.json` index. Attempt 1 with `geom_maxiter=1` failed without success outputs or values and retained a hashed `restart_candidate`; the bounded restart changed only `geom_maxiter` to 100; attempt 2 succeeded and `current_results` binds to attempt 2. Resources agree at 4 cores / 1024 MB / `%maxcore 192` / concurrency 1 | passed |
| M1 acceptance | Evidence was verified and the user explicitly accepted M1 on 2026-09-16 | accepted |
| Resource baseline | The master plan's 2048 MB / `%maxcore 384` is superseded for current work by the user's active project instructions: 1024 MB / `%maxcore 192`; Run snapshots and generated inputs agree | verified |

## M2 status

M2 has offline O1–O8 and final-candidate real Agent/ORCA recertification. The
root-cause repairs pass 245 offline tests; P1–P4 and the final R1/R2 chains are
recorded with credential-free file hashes. Safe stops for malformed state
punctuation, unsupported targets, and ambiguous historical geometry remain in
the evidence and are not counted as passes. The user reviewed these records and
explicitly accepted M2 on 2026-09-16; see
[`docs/milestones/M2.md`](milestones/M2.md) and
[`docs/evidence/M2-final-minimal-fix-acceptance.md`](evidence/M2-final-minimal-fix-acceptance.md),
plus the [machine-readable run index](evidence/M2-final-live-run-index.json).

## 2026-09-24 lightweight control-plane L1–L8

Implementation and local/live validation evidence are recorded in
[`docs/evidence/L1-L8-lightweight-control-plane.md`](evidence/L1-L8-lightweight-control-plane.md).
The live Benchmark v1.1 passed 41/41 and Scientific Matrix v1 passed 11/11 with
16 successful ORCA attempts. GitHub Actions status is pending the implementation
push. **The phase is awaiting the user's review and explicit acceptance; it is
not marked accepted or complete.**
