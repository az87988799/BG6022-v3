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
7. The default real-compute budget is 4 cores, 2048 MB total, `%maxcore 384`,
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
