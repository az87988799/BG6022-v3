# M3 extension evidence

**Candidate status:** awaiting user acceptance; this document does not mark M3
complete or accepted.

## Candidate and environment

| Field | Observed value |
|---|---|
| Branch | `codex/v3-m3-extension` |
| Implementation candidate | `f84f02d` (`fix: close identity resolution closure gaps`) |
| Baseline | `1772cc4cea23bdc001b2c36ef1f7bbd5d15591f9` |
| Python | 3.14.6 for current offline validation; project package gate requires Python 3.11.x |
| Key packages | pydantic 2.13.5; httpx 0.28.1; pytest 9.0.3; rdkit 2026.03.5 |
| Lock hash | `uv.lock` SHA-256 `4ebcff69853f57680c368e061b57e1ea0a1cd2171ebfb8e426c95a5b40128bf3` |
| ORCA | 6.1.1, `E:\orca\orca.exe` |
| LLM | configured model `deepseek-flash`; live LLM suite passed with the key injected only into the current process |
| Active resources | 4 cores; 1024 MB total; `%maxcore 192`; one concurrent job |
| Live data root | `E:\BG6022-v3-data` |
| Config record | `config.example.toml` SHA-256 `53f75bde62ae3fa6d464abbbb28368438248f68c9734fc2a5c008c4577985e07`; only non-secret path/model/resource settings were recorded |
| Prompt hashes | `intake.md` `a216cdf4e5628eb356105c5a83f4504d44d10a608fc3a09d398488ee7d72edd5`; `planner.md` `c88dc578545b2ebbc2add3b80fb73d70183ed56934e8a9fed984671b5b8deee7` |

The current machine did not have `uv`. Its Python 3.11 interpreter lacked the
project's dev dependencies and `wheel`, while Python 3.14 correctly rejected
the package's `>=3.11,<3.12` requirement; therefore the package-build gate was
not rerun for this candidate. The wheel and 3.11.4 environment recorded in
earlier M3 evidence remain historical evidence for the earlier candidate, not
a claim about this identity-root commit. No production fallback or fake
execution was enabled.

## Offline acceptance evidence

The M3-specific suite has 32 passing tests. It covers exact `1.0` and
`sqrt(5)` measurements, reversed atom order, strict index types and distinctness,
out-of-range failure without a success value, operation-free chat Intake to
Plan, one-distance-Step cardinality, Opt-to-distance Tool binding, request
parameter precedence, immutable input bytes/hash, strict angstrom query values,
B3LYP rendering, profile-aware repair scope, configured default precedence, and
result presentation. The M3 plus parameter-scoping targeted regression has 45
passing tests.

The complete offline regression command passed 280 tests with 15 opt-in live
tests deselected. Ruff lint and format checks, Python compilation, package wheel
build, and `git diff --check` passed. `uv sync`/`uv build` were not available on
this host; this environment limitation is recorded rather than hidden.

## Formula input and boundary-fix evidence

This candidate also implements the detailed formula-input and M3 boundary plan
without adding a formula-specific execution loop. Program-owned identity
constraints are stored on the Request; formula input accepts ordinary ASCII or
Unicode-subscript formulas, keeps `H20` distinct from `H2O`, and rejects model
invention of identity facts. PubChem formula resolution uses the bounded
`fastformula -> CID list -> batched properties` path: one shared 20-second
budget, at most three total attempts, at most 32 CIDs by default, and at most
five displayed candidates. An explicit configured value of 20 remains legal and
correctly reports an incomplete search when more than 20 CIDs are returned.
RDKit-computed composition, component count, isotope state, and
formal charge are authoritative; source metadata is checked and cannot replace
those facts. Raw source responses are retained as molecule-source artifacts.

The dedicated formula/boundary suite passed 54 tests. It covers Unicode and
literal formula preservation, narrow case normalization, all labelled SMILES
forms (including a label followed by spaces), no-hydrogen
structures and legacy `H:0` compatibility, explicit-hydrogen accounting,
selected-CID/selected-structure binding, wrong-isomer and wrong-CID rejection,
candidate collision handling, local state preservation after an invalid choice,
candidate classification and structure de-duplication, incomplete/unverified
source evidence, formula suffix rejection, name evidence, and saved candidate
and name-lookup continuations. The complete offline regression for this
candidate passed 370 tests with 16 opt-in live tests skipped. Real PubChem
formula resolution checked all 15 CO₂ and 23 H₂O records in one bounded
two-request lookup; the resolver succeeded with one ordinary structure and
excluded 14 and 22 verified out-of-scope records respectively. Real name
lookups returned ethane CID 6324, butane CID 7843, and isobutane CID 6360;
the name evidence path retained the original Chinese text separately from the
English lookup spelling. These are source/identity smoke tests only and are
not ORCA scientific-success claims.

### Identity-root fix T01–T16 evidence

The detailed identity-root plan is scoped to the five existing identity,
planning, and agent files; it does not add a runtime object, a second molecule
framework, or a molecule-specific execution loop. The following offline
evidence is tied to implementation candidate
`6504ec8b62baccdd503d16ff81d24e971d3aaef8`:

| Case | Offline evidence and result |
|---|---|
| T01 | CO₂, N₂, and CCl₄ facts contain no artificial H count; source metadata agrees; a CO₂ geometry is generated successfully, including a legacy H:0 record. **passed** |
| T02 | Explicit H is counted once; charged legacy SMILES remains charged; formula-only neutrality limits do not apply to name input. **passed** |
| T03 | The five labelled SMILES spellings, including no-space and full-width-colon forms, remain `smiles` and are excluded from formula scanning. **passed** |
| T04 | `C4h10`/`c4h10` normalize only through the narrow unambiguous rule; Unicode subscripts, `H20`, `CO`, `Cl`, and `Br` retain their meanings; `Co` is not guessed. **passed** |
| T05 | A real `handle_message` continuation over fixture PubChem responses selects `candidate_1`, executes resolve and RDKit geometry, then reaches the original Opt confirmation with no ORCA call. **passed offline; real LLM/ORCA acceptance remains pending** |
| T06 | Candidate number, CID, title, tagged SMILES, and equivalent bare SMILES resolve to the current snapshot while retaining candidate CID/title/source. **passed** |
| T07 | `CCCCCCC` while waiting for C₆H₁₄ is rejected locally; the original identity, plan, and waiting state are unchanged. **passed** |
| T08 | Invalid raw `cccccc` is never replaced by model-proposed `CCCCCC`; the full `handle_message` path keeps the same Run waiting, and a later valid candidate remains selectable. **passed** |
| T09 | Planner validation and direct Tool validation reject selected CID 8058 bound to query 7892 before source lookup. **passed** |
| T10 | Missing/wrong CID and same-formula wrong-isomer candidates do not publish a molecule output. **passed** |
| T11 | selected CID/SMILES checks still run when no formula constraint exists. **passed** |
| T12 | Saved candidate continuation skips Intake and Planner and still reaches the deterministic confirmation fallback. **passed** |
| T13 | Candidate/CID numeric collisions, out-of-range numbers, and stale snapshot choices are rejected; existing `/new` and cancellation regressions remain green. **passed** |
| T14 | Existing request-routing regressions keep explicit molecule replacement separate from parameter continuation and new complete requests; affected confirmation/results are invalidated on replacement. **passed** |
| T15 | Locally invalid choices leave Run state unchanged; source/network failures retain bounded attempts and raw response artifacts. **passed** |
| T16 | The saved-run continuation test reloads the active Run from its session record; ordinary historical identity records remain readable without migration. **passed** |

This table records controlled offline behavior only. It does not mark the
phase accepted: the user still needs to perform the requested real interactive
candidate-selection and small-molecule ORCA acceptance before this phase can be
accepted.

The active resource contract remains 4 cores, 1024 MB total, `%maxcore 192`,
and one concurrent job. The candidate remains awaiting user acceptance.

### Identity-closure follow-up evidence

The resolver now keeps four separate outcomes: verified structures, verified
formula-scope exclusions, unverified source records, and explicit identity
rejections. Accepted records are grouped by canonical isomeric SMILES while
their source CIDs remain in the candidate evidence; a single verified neutral
structure is therefore sufficient even when other returned records are
isotopic, charged, multicomponent, or composition-mismatched. An incomplete
bounded search or any unverified record still pauses for identity input.

The full interaction regression confirms that a model-proposed replacement
SMILES cannot overwrite a waiting formula Run. Labelled `SMILES` and `formula`
values accept horizontal spaces, while `C6H14+`, `C6H14.Cl`, and
`C6H14(OH)` are rejected as complete unsupported tokens rather than truncated.
For names, `molecule_name_evidence` is verified against the original user
message; a not-found name becomes a same-Run identity clarification, and a
later English name updates only `lookup_query` while retaining the original
name and calculation target.

## Minimal repairs applied

- Distance indices are checked against a locally parsed, trusted XYZ atom count
  before confirmation and before a parameter-update mutation. Opt geometry edges
  declare their geometry-preserving input so the same check works downstream.
- Explicit ORCA Plans now apply configured method/environment defaults before
  schema defaults while preserving explicit step values.
- Explicit `atom_i`/`atom_j` assignments are parsed locally for waiting-task
  updates; rejected out-of-range changes leave the Run, Plan, and attempts
  unchanged.
- L3 independently re-reads the saved optimized XYZ and checks the Result's
  geometry Artifact ID and SHA-256 binding.

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

`tests/live/test_m3_extension_live.py` defines eight opt-in real-LLM tests. The
final run passed all eight tests and 54 cases, with every expression repeated
twice:

- direct distance: 6 cases;
- B3LYP Opt planning: 6 cases;
- M2-compatible Opt→SP and Opt→Freq composition: 12 cases;
- Opt→distance binding: 6 cases;
- verified historical-geometry→distance binding: 6 cases;
- confirmation, valid `atom_j` update, and rejected out-of-range update: 6 cases;
- completed-distance query with no recomputation: 6 cases;
- missing atom indices and unsupported Gibbs-free-energy requests: 6 cases.

The tests assert that distance is operation-free when requested, that Opt output
is bound through `optimized_geometry`, that saved geometry queries do not append
results or start ORCA, and that unsupported/missing requirements remain blocked.
Credential-free JSON evidence (request text, structured intake/plan, redacted
call metadata, and state assertions) is stored outside the repository at
`E:\BG6022-v3-data\m3-llm-evidence`.

## Remaining acceptance items

The following are intentionally not represented as completed M3 evidence:

- a new live r2SCAN-3c controlled Opt failure → already-authorized repair →
  distance chain (the existing M2 repair evidence remains separate);
- any claim that B3LYP repair, solvents, unsupported combinations, or versions
  outside ORCA 6.1.1 are implemented or scientifically validated.

The user must review the candidate and explicitly accept it before the milestone
can be marked accepted.
