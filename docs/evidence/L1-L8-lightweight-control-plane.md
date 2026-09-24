# Lightweight control-plane L1–L8 verification record

**Status:** implementation and local gates verified; awaiting explicit user
acceptance. This record does not mark the phase accepted.

**Plan:** `E:\chrome\BG6022_Lightweight_Control_Plane_Implementation_Plan.md`  
**Base commit:** `bfa02cfdf5048a71175cff22f65c786bd99b8a07` on
`codex/v3-m3-extension`.

## Implementation

- Tools declare validated default public outputs. Preparation tools keep an
  empty default list.
- The Semantic Proposal is a strict, transient request schema. It exposes only
  user-level task intent, compact Tool/method/result catalogs, and ephemeral
  task references. Program-derived energy-difference Tools are inserted by the
  canonicalizer and are not selectable by the model.
- The canonicalizer reuses the existing Intake-to-Request checks, resolves
  method wording, derives output targets, and creates typed relations for
  comparisons, differences, and explicit output reuse.
- The deterministic PlanBuilder orders Requirement dependencies, reuses each
  subject's initial geometry, adds only required preparation steps, and builds
  ResultTargets from Tool contracts. The Agent has no Planner LLM call on this
  path; the legacy route remains available behind the feature flag.
- Inline XYZ input, historical geometry selection, scoped method updates,
  ambiguous iteration edits, and known unsupported Gibbs/free-energy and global
  conformer-search requests retain bounded behavior.
- Benchmark reports include LLM call and token totals by `purpose`.

## Validation

| Check | Result |
|---|---|
| Offline test suite | 462 passed, 16 skipped |
| Ruff | `ruff check .` passed |
| Bytecode compilation | `compileall -q src tests` passed |
| Whitespace validation | `git diff --check` passed |
| GitHub Actions offline | Run `35956751600` on implementation commit `37f639f53eafbfca1f7a4246a4ba1b16dab043b9` passed: pytest, Ruff lint/format, compileall, and `uv build` |
| Live Benchmark v1.1 | 41/41 passed; boundary accuracy 11/11; critical assertion failures 0; execution-safety violations 0 |
| Live LLM cost route | Fresh compute cases B001, B004, B008, B009 (3 runs), and B018 (3 runs) each used one Semantic call and zero Planner calls |
| Live LLM totals | 15 Semantic calls, 3 Answer calls, 0 Intake/Planner/Repair calls; 62,126 tokens |
| Scientific Matrix v1 | 11/11 cells passed; 16/16 ORCA attempts succeeded; 0 failed cells |
| Matrix T003/T004 | Water, ammonia, and CO₂ Opt→Freq checks passed; dual-SP energy difference used the same initial geometry |
| Active resource agreement | Every Run snapshot used 4 cores / 1024 MB / `%maxcore 192` / one concurrent job; all 32 stored ORCA input copies matched `%pal nprocs 4` and `%maxcore 192` |

The live benchmark used an isolated data root because the configured project
data root had an active chat lock. The raw reports are retained locally at:

- `E:\BG6022-v3-data\acceptance_lightweight_control_plane_20260924\benchmark_v1_live_final4`
- `E:\BG6022-v3-data\acceptance_lightweight_control_plane_20260924\scientific_matrix_live`

The reports record the repository base commit because these checks ran against
the implementation worktree before its commits were created. The phase remains
pending the user's review and explicit acceptance.
