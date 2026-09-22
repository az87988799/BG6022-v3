# BG6022-v3 implementation constraints

1. The first product goal is a real ORCA calculation with bounded failure facts;
   abstractions must directly serve that tool chain.
2. Long-lived runtime domain objects are only `Request`, `Plan`, `Step`, `Tool`,
   `Run`, `Result`, and `Artifact`.
3. Milestone names never enter runtime filenames, classes, or states.
4. A Tool is the execution boundary. Plans compose Steps and do not encode a
   fixed Opt/Freq/SP protocol.
5. There is one production implementation for ORCA input, runner, parser, and
   scientific checks.
6. Do not migrate the v2 database, Kernel/Reducer, outbox, launch ticket, or P7
   turn schema into v3.
7. The current default real-compute budget is 4 cores, 1024 MB total, `%maxcore 192`,
   and one concurrent job. Configured and actual values must agree.
8. The initial geometry is never overwritten. A successful output port cannot
   expose an unconverged geometry or an unverified number.
9. Expected failures retain structured diagnostics and raw files; there are no
   unbounded or unconditional retries.
10. Future models may suggest plans and repair actions, but cannot execute
    arbitrary commands, alter source code, or declare scientific success.
11. Fakes are test-only. Production has no silent fake fallback; real and
    simulated evidence remain separate.
12. New capability should add a Tool and its local adapter/tests, not another
    Kernel, Service, or Record framework.

## Master plan and phase workflow

- The repository-local v3 implementation master plan is `docs/reconstruction/master_plan.md`. It is the checked-in copy of the user-designated `E:\chrome\MASTER_PLAN (1).md`; read it at the start of every phase task and use it to check the phase goals, dependencies, engineering principles, and acceptance evidence.
- The current detailed phase plan is `docs/reconstruction/stages/R0.md`. R0 is limited to baseline, environment/data isolation, behavior-contract mapping, bounded gap reproduction, evidence, and handoff. It does not authorize R1 schema or execution-loop changes.
- `docs/reconstruction/preflight.md` is the single current preparation record; update it with observed facts rather than treating the plans or templates as already-verified evidence.
- The master plan is project guidance, not authorization to implement every phase or every aspirational item it describes. The detailed phase plan supplied by the user defines the current task's scope. Do not mark a phase complete until the user accepts the implementation.
- If the master plan conflicts with this file or the user's current instructions, follow this file and the current user instructions. In particular, the active compute budget is 4 cores, 1024 MB total, `%maxcore 192`, and one concurrent job; the master plan's older 2048 MB / `%maxcore 384` values do not apply unless the user explicitly changes this constraint.
- After completing and validating each user-provided phase plan, commit the corresponding changes and push them to `https://github.com/az87988799/BG6022-v3.git`. The user's current push instruction supersedes the master plan's older instruction to push only on explicit request.
- Once implementation, validation, commit, and push are done, report the phase as awaiting user acceptance. Do not call it complete until the user accepts it.
