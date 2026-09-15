# M2 post-fix live Agent and ORCA recertification

- **Date:** 2026-09-16 (Asia/Hong_Kong)
- **Code under test:** `codex/v3-m2-implementation` at `f6547f76315c27cd72a2899db279fe90799abdcc`
- **Evidence-index parent:** `820c58b140c2334e56be3d30391091b911640241`
- **Model:** `deepseek-flash` (request and response metadata)
- **ORCA:** 6.1.1
- **Evidence root:** `E:\BG6022-v3-data\acceptance_live_20260916_postfix`

This is fresh post-fix evidence from production `Agent.handle_message` and
`Agent.confirm`, not saved-Plan replay. The local harness loaded the configured
DeepSeek key from `.env` without printing or storing its value. Each compute
preview was checked before confirmation. An isolated data root was used so the
configured default data root and any active chat session were not touched; the
ORCA executable, method profile, and resource limits were unchanged. The
plan-only cases were canceled after their previews and did not start ORCA.

All compute Runs record 4 cores, 1024 MB total, `%maxcore 192`, concurrency 1,
1200-second attempt timeout, and 3600-second Run activity limit. ORCA inputs
record `%pal nprocs 4` and `%maxcore 192`. Every ORCA result has matching input
hashes and a clean process tree. Artifact file sizes and SHA-256 digests match
the Run indexes in the tested Runs, including failed-attempt and restart
candidate artifacts. Raw Run JSON, `result.json`, `.inp`, `.out`, Hessian, and
geometry files are retained below the evidence root.

## Fresh compute paths

| Path | Run | Fresh request and validated Plan | Real result |
|---|---|---|---|
| Water Opt | `run_878e339bf4e741bdae49d4809269a4a4` | PubChem water CID 962 → RDKit ETKDGv3 geometry → Opt only; target `opt_final_electronic_energy` | Converged at `-76.418938720831 Eh`; the initial and optimized geometry artifacts remain separate. |
| Water SP on the initial geometry | `run_e88b422a2dc44b168db6dce797fcfe63` | Resolve → generate → SP only; the Request requires `initial_geometry`, with no Opt/Freq step | `-76.415240803144 Eh`; ORCA input geometry hash equals the initial geometry artifact hash. |
| Water optimized SP, same-session history reuse | `run_f652b9daf733419cac15ebeef6e0d825` | After the successful Opt above, a fresh request planned `single_point` only, bound to the verified historical optimized geometry; no resolve, geometry generation, or Opt | `-76.418938721008 Eh`; copied input and source optimized Artifact both hash to `b9c984dee7565417449c48410bea106134900267e5884672ed677e086c90283d`. |
| Water Opt → independent SP | `run_5375134df1f446bc807022e70e574213` | Fresh request canonicalized to Opt + SP and `sp_electronic_energy`; the Request requires `SP.geometry = Opt.optimized_geometry` | Opt converged at `-76.418938720831 Eh`; independent SP returned `-76.418938721008 Eh` from the successful Opt output. |
| Water Opt → Freq with request-level `geom_maxiter=100` | `run_bfcf9bfd590b491485b672d837a29c3e` | Fresh Request preserved `geom_maxiter=100` only on Opt; Freq has no Opt-only parameter and consumes `Opt.optimized_geometry` | Opt converged. Freq returned all 9 modes; Hessian was valid and matched the frequency section. `frequency_complete` and `local_minimum_supported` both passed for the verified optimized geometry. Vibrational modes: 1653.24, 3813.58, and 3932.74 cm⁻¹. |
| Ethanol controlled failure → repair → Freq → SP | `run_63d6b37367d645bb9fead3a31d4493a9`; telemetry-capture repeat: `run_7a255dd20e0d4ecf8415a96b10e9d0a4` | Fresh Request sets `geom_maxiter=1` on Opt only and requires both downstream geometry bindings to the successful Opt output. Both Runs used the real repair path in the same Run. | In each Run, Opt attempt 1 stopped normally but failed `opt_not_converged`, with no success geometry/value. The real model-selected `restart_optimization` changed only `geom_maxiter` from 1 to 100 using the hashed, validated restart candidate. Attempt 2 converged at `-155.002350062010 Eh`. Freq and SP both consumed that successful geometry; Freq returned 27/27 modes with a valid Hessian, and SP returned `-155.002350059646 Eh`. Each Run used 1 extra ORCA execution, below the limit of 3. |

For the telemetry-capture repeat, the restart candidate hash is
`d11ed713befc488bd20972ce446c799874773b466a781a3f942ef84c434aaa5f`; the
successful optimized geometry hash used by both later steps is
`079750f303a18c2997c5c3f031ec1c01138d9bdb705347a3f3222b614d6f1b27`. The
first controlled Run records the same geometry flow and repair values, with
its own attempt artifacts and hashes. Neither successful Run reran molecule
resolution or initial geometry generation after the Opt failure.

## Planning-only and waiting-confirmation continuity

| Path | Run(s) | Observation |
|---|---|---|
| Benzene Opt → Freq | `run_1b4c847e34414e72b954150a4ccbca8e` | Fresh model request produced resolve → generate → Opt → Freq, with Freq bound to `Opt.optimized_geometry`. It was canceled at confirmation; no ORCA calculation was started. |
| Hexane Freq on the initial structure | `run_33813b288c2544e4924a29ca0873566d` | Fresh model request produced resolve → generate → Freq only and an explicit initial-geometry binding. No Opt was inserted. It was canceled at confirmation; no ORCA calculation was started. |
| New request while Opt confirmation was pending | `run_3783b04b4999487f8e4496f78a950fe2` and `run_9c42ebbaea4f41c2b4f2f49aaa77d2d0` | In one session, a water Opt preview was followed by a separately worded water SP request. Intake created a distinct SP Request/Run (`single_point` only); it did not patch the pending Opt Run. Both unconfirmed previews were canceled without ORCA. |

## Live LLM diagnostic evidence

All successful Intake/Planner calls report `deepseek-flash` for both requested
and responding model. On the water Opt → Freq request, the first Intake call
was categorized `truncated` with `finish_reason=length` and an empty body. One
bounded structured correction produced a valid Intake response, and the Plan
then passed local validation. On the telemetry-capture ethanol Run, the first
repair response had `schema_error`; one bounded correction produced a valid
`restart_optimization` proposal, which passed deterministic repair validation.
Credential-free metadata for the actual Intake, Planner, and both repair calls
is saved at:

`E:\BG6022-v3-data\acceptance_live_20260916_postfix\retest_records\ethanol_repair_chain_live_m2_20260916_ethanol_repair_capture.json`

The metadata records purpose, requested/responding model, category,
`finish_reason`, body length, elapsed time, token usage, and correction counts;
it does not contain credentials, prompts, or response bodies.

## Additional safe stop

An exploratory request phrased only as “几何优化迭代上限为100” led Intake to
return the unrecognized parameter name `max_iterations`. The program stopped
before Run creation because no calculation Tool accepts that field; no ORCA
process started. The same case with the canonical key `geom_maxiter=100`
produced the valid Opt → Freq preview and passed as recorded above. This safe
stop is retained as an input-vocabulary limitation, not counted as a pass for
parameter extraction from that colloquial wording.

## Status

The post-fix real-entry scenarios in the M2 repair plan have been exercised on
the candidate. This document records implementation and test evidence only;
**M2 remains awaiting the user's acceptance**. M1 acceptance remains separately
pending.
