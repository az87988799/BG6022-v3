# M2 composition evidence index

**Status:** implementation and validation are ready for review; **user
acceptance is pending**. This record distinguishes offline/simulated evidence
from actual DeepSeek, PubChem/RDKit, and ORCA Runs.

## Candidate and environment

| Field | Observed value | Status |
|---|---|---|
| Branch | `codex/v3-m2-implementation` | pushed to `origin` |
| Code commit | `e7e300a227973a815c6213f1a4a36996fa3f96f7` (`Implement M2 frequency and plan composition`) | pushed |
| Evidence commit | `dd0ee54eb0d6b4fed714b73a065f0505b18ff0e1` (`Record M1 review and M2 evidence`) | pushed |
| Python / OS | Python 3.11.4 / Windows 10 build 26200 | observed |
| ORCA | `E:\orca\orca.exe`, Program Version 6.1.1 | observed in Run records |
| Method and scope | r2SCAN-3c, gas phase, neutral singlet H2O and C2H6O | exercised |
| Active real-compute budget | 4 cores / 1024 MB total / `%maxcore 192` / concurrency 1 | active project instruction |
| Run/input agreement | All four current-candidate Runs snapshot 4/1024/192/1; each generated ORCA input has `nprocs 4` and `%maxcore 192` | verified |
| Local offline validation | `ruff check src tests`; `ruff format --check src tests`; `python -m compileall -q src start_chat.py tests`; `pytest -q -m "not live_llm and not live_pubchem and not live_orca"` → 181 passed, 3 deselected; `uv build` | passed |
| GitHub Actions | [offline run 34993113284](https://github.com/az87988799/BG6022-v3/actions/runs/34993113284), tested commit `dd0ee54eb0d6b4fed714b73a065f0505b18ff0e1` | passed |
| User acceptance | M2 acceptance owner and date | pending user review |

Evidence root: `E:\BG6022-v3-data\m2_final_candidate_20260915_freqfix`.
Machine-readable run index and file hashes:
`E:\BG6022-v3-data\m2_final_candidate_20260915_freqfix\m2_live_summary.json`.
Each `runs/<run_id>` directory retains the request/Plan/Run, per-Step Results,
ORCA input, stdout/stderr, geometry/Hessian artifacts, and SHA-256 inventory.

The master plan specifies 2048 MB / `%maxcore 384`; the current user-provided
project instructions specify 1024 MB / `%maxcore 192`. This candidate follows
the current project instruction. The Run snapshots and generated inputs agree
with that budget; no runtime resource increase was made.

## Offline acceptance evidence

All cases below are simulated/offline except where a real Run is identified in
the next section.

| ID | Evidence | Result | Status |
|---|---|---|---|
| O1 | `tests/unit/test_plan_composition.py`: Opt, SP, Opt/Freq, Opt/Freq/SP, and Freq-only on a supplied XYZ; requested-operation/result/check coverage | no omitted/extra operation; standalone Freq needs no Opt/local-minimum claim; SP and Opt energy categories remain distinct | passed |
| O2 | `tests/unit/test_frequency_parser.py`, `tests/unit/test_frequency_support.py`: complete/truncated output, missing Hessian, dimensional checks, `.hess` frequency count/value/scaling comparison | incomplete or mismatched evidence cannot succeed | passed |
| O3 | `test_parses_orca_imaginary_mode_annotation_without_losing_negative_sign`, unsupported-layout tests, and local-minimum tests | preserve negative signs; unsupported layouts remain incomplete/unverified; frequency completion is separate from local-minimum support | passed |
| O4 | `test_history_geometry_is_verified_copied_and_bound_to_the_new_run`, invalid binding parameterization, `test_model_cannot_choose_between_multiple_history_geometries` | source role/attempt/hash are verified; altered, fabricated, missing, and ambiguous references stop before ORCA | passed |
| O5 | `test_recovered_opt_geometry_flows_to_frequency_and_sp_without_repeating_preparation` | Opt retry geometry feeds Freq and SP; preparation is performed once | passed |
| O6 | `test_failed_local_minimum_gate_never_calls_the_sp_executor`, `test_passed_check_unblocks_sp_and_upstream_invalidation_cascades`, geometry-reference mismatch regressions | dependent SP is blocked without a passed check on the same geometry; stale downstream Results are invalidated | passed |
| O7 | cumulative science-attempt/extra-call budget, expired active-time, and `test_cancel_request_stops_process_tree_and_clears_execution_guard` | limits are cumulative; cancellation stops the process tree and clears its guard | passed |
| O8 | `test_energy_results_keep_opt_sp_and_frequency_categories_distinct`, plan target/operation tests, invalid-reference tests | no unsupported result or energy substitution can be reported as success | passed |

## Real current-candidate Runs

R1 and R3 use fresh live DeepSeek plans. R2 and R4 replay a previously
successful, real DeepSeek-generated Plan through the current candidate after
current validation; the current Agent copied and verified R2's history input,
and the current Agent executed R4's full ORCA chain. R4 repair selection used a
live DeepSeek call during the current-candidate Run. This distinction is
intentional: unsuccessful fresh R2/R4 intake attempts are retained but are not
represented as successful planner calls.
The standalone Freq-on-supplied-XYZ path is covered by offline Plan validation
and standalone input-generation tests; it is not a separate R1–R4 live Run.

| ID | Run and saved evidence | Observed result | Status |
|---|---|---|---|
| R1 | `run_ed127474c45844da813c7a9bd997faf6`; `runs/run_ed127474c45844da813c7a9bd997faf6` | Fresh Plan contains resolve → geometry → Opt only. `opt_final_electronic_energy = -76.418938720831 Eh`. No Freq or SP. Optimized geometry SHA-256 `81b32f5e147c25463e86a63b282b0ed5e11b3d5fb8287ff8ff97ecbb466ce9ed`. | passed |
| R2 | `run_ef6c4591cd2d466dab45fdacbc283338`; `runs/run_ef6c4591cd2d466dab45fdacbc283338` | SP only. Source R1 Run `run_ed127474c45844da813c7a9bd997faf6`, artifact `artifact_229d1ce373f14215b9f8c0aba30a7238`, and copied input artifact `artifact_8e71dcdafb774c9386214927da34356b` all bind SHA-256 `81b32f5e147c25463e86a63b282b0ed5e11b3d5fb8287ff8ff97ecbb466ce9ed`. `sp_electronic_energy = -76.418938721008 Eh`. | passed |
| R3 | `run_3ff91d6a68f5402384ab54489b8cc6e5`; `runs/run_3ff91d6a68f5402384ab54489b8cc6e5` | Fresh Plan contains Opt/Freq only. Freq consumes optimized artifact `artifact_ded8998c68e7460a90dd8380611f75d0`, geometry SHA-256 `49f33dcf2a539e016d7f4df33b5ed8ba2fb86025f6784846b3dab7ec08c51b0c`; Hessian artifact `artifact_b5fb2318e6f34d4d9a3bf1f2679b555d`, SHA-256 `06f199c234957f11819b98534591c927747d22351e5b9e7ee6a40d34f616ac56`. All 9 modes and both frequency/local-minimum checks passed. The frequency Result exposes only `vibrational_frequencies`. | passed |
| R4 | `run_9822756723c74606809eefcb6cbae761`; `runs/run_9822756723c74606809eefcb6cbae761` | Captured live plan source Run `run_30b95f6a4bc2496f9795a4a81f9f6253`, revalidated and executed on current code. First Opt (`geom_maxiter=1`) failed as `opt_not_converged` with no success port/value. Same-Run `restart_optimization` changed only `geom_maxiter` to 100; live DeepSeek selected the validated action. Retry succeeded at `-155.002350062010 Eh`. Freq and SP both use `artifact_57baf83a4afb4f41b86f7b020cdcb623`, geometry SHA-256 `5192bf4095346bd1ec8114d6e11221da10843e80c17a66b6b2bd4d30e3099dd0`. Freq has 27/27 modes, matching Hessian artifact `artifact_1f8f004dd9644464a2e624a41e4f4b5f` (SHA-256 `1cafc0142ccdb25328ae9a9976f64ad33569bf9edf64f6debaa6905070a5a0ed`), and `local_minimum_supported` passed before SP. SP energy is `-155.002350059646 Eh`. Molecule/geometry preparation was not repeated. | passed |

All current-candidate ORCA attempts report `process_tree_empty=true`; generated
inputs were checked for `%pal nprocs 4` and `%maxcore 192`. R3 and R4 Hessian
files were replayed with the final parser, which verified Hessian/stdout frequency
and scale agreement. Earlier successful live R3/R4 raw evidence was also replayed
with the final parser; the current-candidate R3/R4 Runs supersede it for candidate
evidence.

### Non-passing live attempts retained, not counted

- Fresh R2 intake returned an empty model response before Run creation. The
  subsequent saved-plan replay passed; no SP was started by the failed intake.
- Two fresh R4 attempts stopped before Run creation: one Plan placed the
  Opt-only `geom_maxiter` on Freq; another intake treated multiplicity as an
  unsupported value. Neither started ORCA. The saved-plan replay passed with a
  live, validated repair selection.
- An R1/R2 harness attempt used a noncanonical case label, so its evidence join
  canceled the Plan before ORCA. The successful canonical R1 and subsequent R2
  replay are recorded above.
- The prior evidence root retains earlier planner corrections and an R4
  preflight stopped before ORCA because available memory was below the required
  1 GiB floor. None of these attempts is counted as a scientific-path pass.

## Acceptance record

| Item | Status |
|---|---|
| M2 implementation and offline/live verification | ready for review |
| Candidate code commit | pending commit |
| Evidence commit / pushed branch head | pending commit and push |
| M2 user acceptance owner and date | pending user review |
| M1 user acceptance | remains separately pending |

Do not mark M2 accepted until the user has reviewed the pushed candidate and
explicitly accepts it.
