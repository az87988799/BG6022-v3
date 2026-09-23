# Extensibility audit and root-fix implementation record

Date: 2026-09-23
Source plan: `E:\chrome\BG6022_V3_Extensibility_Audit_and_Root_Fix.md`
Master plan copy: [`../reference/BG6022-v3-Migration-Rebuild-Plan-2.0.md`](../reference/BG6022-v3-Migration-Rebuild-Plan-2.0.md)

**Status: implementation and validation recorded; awaiting user acceptance.** This record does not mark the phase accepted or complete.

## Changes

- Kept the seven runtime objects. Tool-local preparation, result verification, scientific-check input declarations, repair options, and execution budget categories now drive the shared Agent path. Scientific check prerequisites are declared by the producing Tool contract and validated against matching input references.
- Request requirements can contain repeated instances of one capability with requirement-scoped parameters and results. Completion now resolves those targets by their requirement IDs, including when several Steps use the same Tool. Geometry-source requirements can name a consumer Tool or requirement.
- Replaced the generic ORCA execution-budget key with Tool execution categories while retaining read compatibility for saved legacy Runs and configuration. Defaults remain 4 cores, 1024 MB, `%maxcore 192`, one concurrent job.
- Added production `geometry_angle` and `energy_difference` Tools. Energy outputs are hash-bound to the input or optimized geometry, method, electronic state, observation, producer Result, attempt, and current output port. The comparison Tool requires two different registered method profiles on the same geometry and electronic state; its sign convention is B minus A.
- Added the PBE0-D3(BJ)/def2-SVP profile for the existing SP, Opt, and Freq adapters. Existing B3LYP-D3(BJ)/def2-SVP behavior remains.
- Answer and confirmation context now use registered Tool labels and result metadata. `AGENTS.md` points to the byte-identical master-plan copy in the repository instead of an author-machine absolute path.

## Validation

- `ruff check src tests`: passed.
- `pytest -q`: **383 passed, 16 skipped**. Live tests stayed skipped unless explicitly selected.
- Real ORCA PBE0 water SP: Run `run_f26a4637350a4639975b9d23ccafc207`, succeeded at `-76.274794332234 Eh`; input had `nprocs 4` and `%maxcore 192`.
- Real ORCA B3LYP water Opt → Freq → independent SP after the generic check-gate change: Run `run_bd8cb6c86560469bb390c1e1322b4069`, all three Steps succeeded; complete-frequency and local-minimum checks passed.
- Real same-geometry r²SCAN-3c/PBE0 comparison through two SP output ports and `energy_difference`: Run `run_ccf36fa371ab475f86464c160f53f18b`, all three Steps succeeded. The Tool reported B − A = `0.142451751943 Eh` on the shared geometry hash. This is a method difference, not an accuracy ranking.

The live controlled Opt-failure → LLM repair selection → successful retry chain was not run because `DEEPSEEK_API_KEY` is unavailable in this environment. The bounded repair path and continuation are covered by simulated tests; they are separate evidence from real ORCA runs. The real runs above used the project configuration's 4/1024/192/1 budget and wrote only new Runs under `E:\BG6022-v3-data`.

## Acceptance

Implementation, code validation, and the available real-compute checks are recorded. The phase remains **awaiting user acceptance**.
