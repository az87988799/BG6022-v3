# M1 Agent evidence index

This is an implementation-stage evidence index. It does not record user
acceptance and does not claim a new live DeepSeek, PubChem, or ORCA run.

## Offline evidence

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
