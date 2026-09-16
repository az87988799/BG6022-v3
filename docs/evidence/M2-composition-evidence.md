# M2 composition evidence index

**Status:** the 2026-09-16 root-cause fixes are implemented and offline-verified;
fresh post-fix real-entry recertification and **user acceptance are pending**.
This record distinguishes the pre-fix DeepSeek, PubChem/RDKit, and ORCA Runs
from the post-fix offline evidence.

## Candidate and environment

| Field | Observed value | Status |
|---|---|---|
| Branch | `codex/v3-m2-implementation` at the final candidate (two commits: `cc11043` plus the evidence commit) | pushed to `origin` |
| Pre-fix baseline code commits | `e7e300a227973a815c6213f1a4a36996fa3f96f7` (M2 implementation) and `9da71b0d8adae70a812a00704eb6544911d53e1b` (standalone Freq Agent regression) | pushed; historical baseline |
| Pre-fix evidence commits | `dd0ee54eb0d6b4fed714b73a065f0505b18ff0e1` (M1/M2 records) and `03a421438af09154b567d0c49d7c50c2aacc0b30` (CI result) | pushed; historical baseline |
| Root-cause repair candidate | `cc11043` | pushed; final live evidence in [`M2-final-minimal-fix-acceptance.md`](M2-final-minimal-fix-acceptance.md) |
| Python / OS | Python 3.11.4 / Windows 10 build 26200 | observed |
| ORCA | `E:\orca\orca.exe`, Program Version 6.1.1 | observed in Run records |
| Method and scope | r2SCAN-3c, gas phase, neutral singlet H2O and C2H6O | exercised |
| Active real-compute budget | 4 cores / 1024 MB total / `%maxcore 192` / concurrency 1 | active project instruction |
| Run/input agreement | All four historical pre-fix Agent Runs snapshot 4/1024/192/1; each generated ORCA input has `nprocs 4` and `%maxcore 192` | verified for those Runs |
| Pre-fix local offline validation | `ruff check src tests`; `ruff format --check src tests`; `python -m compileall -q src start_chat.py tests`; `pytest -q -m "not live_llm and not live_pubchem and not live_orca"` → 182 passed, 3 deselected; `uv build` | passed on pre-fix baseline |
| Pre-fix GitHub Actions | [offline run 34994186571](https://github.com/az87988799/BG6022-v3/actions/runs/34994186571), tested commit `9da71b0d8adae70a812a00704eb6544911d53e1b` | passed on baseline |
| Post-fix GitHub Actions | [offline run 35007233667](https://github.com/az87988799/BG6022-v3/actions/runs/35007233667), tested commit `f6547f76315c27cd72a2899db279fe90799abdcc` | passed |
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
| O1 | `tests/unit/test_plan_composition.py` and `test_chat_freq_only_request_runs_on_supplied_xyz_without_preparation`: Opt, SP, Opt/Freq, Opt/Freq/SP, and Freq-only on supplied XYZ; request-to-confirmation runtime coverage | no omitted/extra operation; standalone Freq binds the user XYZ without Opt/preparation and makes no local-minimum claim; SP and Opt energy categories remain distinct | passed |
| O2 | `tests/unit/test_frequency_parser.py`, `tests/unit/test_frequency_support.py`: complete/truncated output, missing Hessian, dimensional checks, `.hess` frequency count/value/scaling comparison | incomplete or mismatched evidence cannot succeed | passed |
| O3 | `test_parses_orca_imaginary_mode_annotation_without_losing_negative_sign`, unsupported-layout tests, and local-minimum tests | preserve negative signs; unsupported layouts remain incomplete/unverified; frequency completion is separate from local-minimum support | passed |
| O4 | `test_history_geometry_is_verified_copied_and_bound_to_the_new_run`, invalid binding parameterization, `test_model_cannot_choose_between_multiple_history_geometries` | source role/attempt/hash are verified; altered, fabricated, missing, and ambiguous references stop before ORCA | passed |
| O5 | `test_recovered_opt_geometry_flows_to_frequency_and_sp_without_repeating_preparation` | Opt retry geometry feeds Freq and SP; preparation is performed once | passed |
| O6 | `test_failed_local_minimum_gate_never_calls_the_sp_executor`, `test_passed_check_unblocks_sp_and_upstream_invalidation_cascades`, geometry-reference mismatch regressions | dependent SP is blocked without a passed check on the same geometry; stale downstream Results are invalidated | passed |
| O7 | cumulative science-attempt/extra-call budget, expired active-time, and `test_cancel_request_stops_process_tree_and_clears_execution_guard` | limits are cumulative; cancellation stops the process tree and clears its guard | passed |
| O8 | `test_energy_results_keep_opt_sp_and_frequency_categories_distinct`, plan target/operation tests, invalid-reference tests | no unsupported result or energy substitution can be reported as success | passed |

## Post-fix root-cause regression evidence

On the repair candidate, `pytest -q -rs` passed **224 tests, with 3 skipped**
(the opt-in live PubChem, DeepSeek, and ORCA tests are disabled by default).
`ruff check .`, `ruff format --check .`, `python -m compileall -q src
start_chat.py tests`, and `git diff --check` passed. The new regressions cover
Tool-derived target catalogs and Intake gating, ambiguous energy targets,
geometry-source binding, bounded empty-response correction and diagnostics,
per-Tool parameter scope and atomic edits, preparation failure state, and
history-query routing. These tests use local fixtures and mocked HTTP; they do
not count as live model or ORCA evidence.

Fresh live Agent/ORCA entry scenarios from the repair plan have not been run on
this candidate. The local Windows environment lacks `uv`, and a local `pip
wheel` fallback could not run because it lacks `bdist_wheel`; GitHub Actions did
run `uv build` successfully on this commit.

## Historical real Runs (pre-root-cause fix)

The Runs in this section were created before the root-cause repair candidate.
They remain valid records of their actual Tool executions, but they do not
verify post-fix Intake/Planner behavior. Fresh real-entry recertification is
still required before M2 acceptance review.

R1 and R3 use fresh live DeepSeek plans. R2 and R4 replay a previously
successful, real DeepSeek-generated Plan through the then-current pre-fix
candidate; its Agent copied and verified R2's history input and executed R4's
full ORCA chain. R4 repair selection used a live DeepSeek call during that
pre-fix Run. This distinction is intentional: unsuccessful fresh R2/R4 intake
attempts are retained but are not represented as successful planner calls.
The standalone Freq-on-supplied-XYZ path is covered by offline Plan and Agent
validation, plus the production Tool Run recorded below; it is not a separate
R1–R4 live Agent Run.

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

An additional standalone Tool run covers the supplied-XYZ Freq path: Run
`run_1ce81c9b55a540028d1980f39beb12fb` used `examples/water.xyz` with the
production `run-tool frequency` command and an isolated data root at
`E:\BG6022-v3-data\m2_freq_only_smoke_20260916`. Its Run contains only the
`frequency` Tool and binds input geometry artifact
`artifact_63702fd5535741fb9d71f4ab8e793ca1`. ORCA returned 9/9 finite modes
(six external modes printed as zero, plus 1638.86, 3525.78, and 3684.07 cm⁻¹);
the frequency section, Hessian, Hessian/stdout mode matching, SCF convergence,
input hashes, normal termination, and process cleanup checks passed. The Run
and input agree at 4 cores / 1024 MB / `%maxcore 192` / concurrency 1. No
`local_minimum_supported` claim was requested. This real Tool check complements
the offline Freq-only Agent test and does not replace any R1–R4 Agent scenario.

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
| Pre-fix M2 implementation and offline/live evidence | historical baseline; not post-fix recertification |
| Root-cause repair candidate commit | `cc11043` (final candidate; push pending in this turn) |
| Post-fix offline verification | 245 passed, 3 deselected; lint, format, compile, package build, and diff checks passed |
| Post-fix fresh real-entry recertification | final candidate recorded in [`M2-final-minimal-fix-acceptance.md`](M2-final-minimal-fix-acceptance.md); awaiting user acceptance |
| Evidence commits | `dd0ee54eb0d6b4fed714b73a065f0505b18ff0e1`, `03a421438af09154b567d0c49d7c50c2aacc0b30`, `e7b33d74d1e59f045ec66dbabcf2865bcdb0cda7` (post-fix live recertification; pushed) |
| M2 user acceptance owner and date | pending user review |
| M1 user acceptance | remains separately pending |

Do not mark M2 accepted until the user has reviewed the pushed candidate and
explicitly accepts it.

## Post-fix live recertification

Fresh DeepSeek and ORCA results for P1–P4 and the final R1/R2 chains are
recorded in [`M2-final-minimal-fix-acceptance.md`](M2-final-minimal-fix-acceptance.md),
with every evidence-root file indexed by SHA-256 in
[`M2-final-live-run-index.json`](M2-final-live-run-index.json). The earlier
post-fix document remains historical evidence. This record does not constitute
user acceptance of M2.
