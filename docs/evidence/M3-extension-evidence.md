# M3 extension evidence

**Candidate status:** awaiting user acceptance; this document does not mark M3
complete or accepted.

## Candidate and environment

| Field | Observed value |
|---|---|
| Branch | `codex/v3-m3-extension` |
| Implementation candidate | `a6830a57492bf2d5879602994e5987d6f4816e3b` |
| Baseline | `f6904328e4227fb60f0438268c222a8a04f8ca6b` |
| Python | 3.11.4 |
| Key packages | pydantic 2.13.5; httpx 0.28.1; pytest 9.1.1; rdkit 2026.03.6 |
| Lock hash | `uv.lock` SHA-256 `4ebcff69853f57680c368e061b57e1ea0a1cd2171ebfb8e426c95a5b40128bf3` |
| ORCA | 6.1.1, `E:\orca\orca.exe` |
| LLM | configured model `deepseek-flash`; live calls not executed because `DEEPSEEK_API_KEY` was absent |
| Active resources | 4 cores; 1024 MB total; `%maxcore 192`; one concurrent job |
| Live data root | `E:\BG6022-v3-data-m3-live-20260916` |
| Config record | `config.example.toml` SHA-256 `1935185b5da3bdb83792021281fbe804adc5f7e8760ed506ef1d1a2be7d09736`; only non-secret path/model/resource settings were recorded |
| Prompt hashes | `intake.md` `8717035f102e879f25b9c7d3b4b910f0d58f3268234c39aad609c3819660f92`; `planner.md` `ca3a15d284a306d1675991fad8d523d0ebdec077e136044dbd4284dc1393201e` |

The machine did not have `uv`. The locked offline workflow was therefore run
with the existing `.venv`; after the initial build attempt reported missing
`bdist_wheel`, `wheel` was installed into that virtual environment and the
package wheel was built successfully. No production fallback or fake execution
was enabled.

## Offline acceptance evidence

The M3-specific suite has 24 passing tests. It covers exact `1.0` and
`sqrt(5)` measurements, reversed atom order, strict index types and distinctness,
out-of-range failure without a success value, operation-free chat Intake to
Plan, one-distance-Step cardinality, Opt-to-distance Tool binding, request
parameter precedence, immutable input bytes/hash, strict angstrom query values,
B3LYP rendering, profile-aware repair scope, configured default precedence, and
result presentation.

The complete offline regression command passed 272 tests with 9 opt-in live
tests deselected. Ruff lint and format checks, Python compilation, package wheel
build, and `git diff --check` passed. `uv sync`/`uv build` were not available on
this host; this environment limitation is recorded rather than hidden.

## Real ORCA evidence

All four final ORCA cases below used the isolated data root and the active
4/1024/192/1 resource contract. The first full live invocation had a test-only
path-joining defect for the L2/L3 assertions after their ORCA Runs had already
succeeded; the helper was fixed, and the affected L2/L3 assertions were rerun
successfully. The final records below are the corrected runs, not the test-only
failure attempts.

| ID | Run | Scenario and observed facts | Status |
|---|---|---|---|
| L1 | `run_c2392fc441f34f90a694bfbfaa55b8a1` | B3LYP water SP; energy `-76.320237678723 Eh`; ORCA version and SCF/normal-termination checks passed | passed |
| L2 | `run_65ee6e4eff0d45838253c459a6ab578e` | B3LYP water Opt→Freq→independent SP; Opt energy `-76.321846986879 Eh`; SP energy `-76.321843874159 Eh`; 9/9 finite modes; matching Hessian; `frequency_complete=passed`; `local_minimum_supported=passed` | passed |
| L3 | `run_3abe260e4e06474cb6b474d27f659221` | B3LYP ethanol Opt→distance; Opt energy `-154.835566700447 Eh`; distance between XYZ atoms 1 C and 3 O is `2.449244495068977 Å`; direct remeasurement of the saved optimized XYZ matched within `1e-10 Å` | passed |
| L4 | `run_06e718c86e274a48ab9609ed0d739caa` | B3LYP ammonia SP, covering N; energy `-56.470443697465 Eh`; ORCA version and SCF/normal-termination checks passed | passed |

Every successful ORCA Result includes the selected method, actual ORCA
version, input hash checks, and raw output paths. L3's distance Result also
stores the actual geometry Artifact ID and SHA-256. L2's Freq Result stores the
verified Hessian and geometry-bound scientific checks. No repair record was used
in L1–L4; B3LYP's candidate repair scope is empty.

The complete file size/hash inventory, including `run.json`, Result JSON,
input `.inp`, input/output `.xyz`, `.out`, and the L2 Hessian, is maintained in
[`M3-extension-live-index.json`](M3-extension-live-index.json). The raw files
remain available for independent inspection at the external data root.

## Live LLM evidence

`tests/live/test_m3_extension_live.py` defines two opt-in matrices:

- three distance phrasings, each repeated twice, asserting `operations=[]`,
  exact indices 1 and 2, one `geometry_distance` Step, and no ORCA operation;
- three B3LYP phrasing variants, each repeated twice, asserting the canonical
  method profile, Opt operation, result target, and inline geometry source.

These calls were not run because the required DeepSeek API key was not present.
They are not counted as planning passes, and no live model output is stored in
the repository.

## Remaining acceptance items

The following are intentionally not represented as completed M3 evidence:

- live DeepSeek formulation/repetition matrix;
- a new live r2SCAN-3c controlled Opt failure → already-authorized repair →
  distance chain (the existing M2 repair evidence remains separate);
- any claim that B3LYP repair, solvents, unsupported combinations, or versions
  outside ORCA 6.1.1 are implemented or scientifically validated.

The user must review the candidate and explicitly accept it before the milestone
can be marked accepted.
