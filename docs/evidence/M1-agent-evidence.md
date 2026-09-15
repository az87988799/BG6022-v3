# M1 Agent evidence index

This is an implementation-stage evidence index. It does not record user
acceptance and does not claim a new live DeepSeek, PubChem, or ORCA run.

## Offline evidence

- The minimal-repair regression suite contains 65 passing tests and 3 skipped
  opt-in live/integration tests. It covers Request/Plan target preservation,
  old `optimized_geometry` field-to-port conversion, parser evidence for the
  geometry-settings `MaxIter` (line 239 in the preserved ORCA fixture), input
  versus output iteration-limit conflicts, q/M authority and O2 clarification,
  complete waiting previews, atomic rejection of an unsupported method update,
  and an explicit no-iteration-increase repair scope.
- Static checks pass with Ruff lint and format verification. The new default
  resource snapshot is 4 cores / 1024 MB / MaxCore 192 / concurrency 1; the
  ignored local `config.toml` was synchronized without adding it to Git.
- `tests/unit/test_m1_tools.py` covers old-config compatibility, explicit
  SMILES resolution, RDKit ETKDGv3 artifact metadata, bounded PubChem retry and
  URL encoding, one JSON-format correction, parameter-source precedence, and
  deterministic Opt repair admission.
- The same test module replays a controlled Opt failure with a mock Tool-side
  runner and a mock repair response. It verifies two attempts, one Run, a
  validated `restart_candidate`, a derived bounded parameter change, and a
  successful `optimized_geometry` port only on the second attempt. This is
  explicitly offline/fake evidence.
- The accepted M0 fixture and process-tree tests remain in the repository and
  are included in the M1 regression command.

## Implementation smoke observations (not M1 acceptance)

On 2026-09-11, with the local `config.toml`, the unchanged deterministic CLI
path completed real ORCA smoke runs after the M1 changes:

- SP Run `run_305a7de860cd4275b2bea1d750ae28b4`: `-76.417246084177 Eh`,
  normal termination, `input_hashes_match=true`, and an empty process tree.
- Opt Run `run_31e4ad4a8bd144aa8ca8ae9c86618df8`: `-76.418938721015 Eh`,
  optimization convergence and output/stdout geometry consistency passed, with
  the same resource budget and cleanup facts.
- A separate read-only PubChem smoke resolved water to CID 962 and ethanol to
  CID 702. The responses were not used as fake fixtures.

These observations verify that the M0 direct path and live PubChem transport
survived the M1 implementation. They are not the required natural-language
DeepSeek/confirmation/automatic-repair acceptance sequence.

## Required live evidence before acceptance

The live evidence package must contain the original chat text, model and use,
the validated Plan and accepted snapshot, real PubChem source response, RDKit
version/seed, all Run/Result JSON files, every ORCA attempt input/geometry/stdout
/stderr, candidate and final geometry hashes, repair proposal and validation
record, resource budgets, and process cleanup facts. It must distinguish real
artifacts from offline fixtures.

The required acceptance sequence is:

1. Start `python -m bg6022 --config config.toml chat` and prepare real water and
   ethanol from natural-language requests.
2. Confirm a real Opt request and verify the deterministic result explanation.
3. Run a controlled real `geom_maxiter=1` Opt, verify the retained failure
   facts, allow the real DeepSeek repair proposal to select
   `restart_optimization`, and verify a successful second ORCA attempt in the
   same Run.

Until that sequence is reviewed and accepted by the user, M1 remains
“implementation complete; live acceptance pending”.

## 2026-09-15 parameter/query repair follow-up

Implementation commit: `09ccd60d9b9456a33fa55019dc93e3158458c80a` (pushed to
`codex/v3-m1-agent`). The q/M intake now keeps per-turn `absent`, `set`,
`ambiguous`, and `invalid` states, checks model candidate quotes against the
current user text and complete value token, and sends invalid/ambiguous claims
to clarification without changing a pending Request. Regression cases include
`charge 0`, triplet selection, `charge 0 -> +1`, rejecting `1.5`, `1e2`,
booleans and non-positive multiplicity, discarding model-only q/M, preserving
the omitted field, and stopping a conflicting update against a waiting Run.

Saved-result queries now bind a program-generated `subject_ref` to a
machine-readable property. Tool property mappings are validated against
declared result fields/ports. Coverage uses `(subject_ref, property)` only;
presentation labels and descriptions are not used to classify a scientific
fact. Regression cases cover “这个结构的能量是多少？”, energy plus geometry,
missing zero-point energy, cross-task non-substitution, unsupported-property
evidence, and a structure mentioned only as the query subject.

Offline validation against this commit: `104 passed, 3 deselected` with live
ORCA/LLM/PubChem tests excluded; Ruff check and format check passed,
`compileall` passed, and `uv build` succeeded. The resource baseline was not
changed: 4 cores, 1024 MB total, `%maxcore 192`, concurrency 1.

Environment check on 2026-09-15: the effective local config names
`DEEPSEEK_API_KEY`, and that environment variable is unset. An ORCA executable
path is configured, but no LLM, PubChem, or ORCA acceptance sequence was
attempted for this follow-up. No L1/L2 live evidence is added here; M1 remains
pending user review and acceptance.
