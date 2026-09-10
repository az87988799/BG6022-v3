# M0 migration record

The v3 source tree is independent of `E:\BG6022-v2`. The local v2 checkout was
read-only during this implementation and remains on its existing branch with
its working tree preserved.

Reference baselines from the implementation plan:

- BG6022-v2 repair baseline: `89f69de430ab4e42cc0cca3fc5798612eafdad82`
- BG6022-v2 main: `c64f547956f9ede2c0db02757199d1e2c0dc86f0`
- TritonDFT main: `d9ac1ee5035baa07bf36a5cdc8a0f7c12bc21d79`

M0 reimplements the small input, process, and parser boundary needed by v3.
It does not copy the v2 application layer, database, launch tickets, Kernel,
Reducer, outbox, or session records. The original ORCA fixture is preserved
byte-for-byte under `tests/fixtures/orca_6_1_1_water_opt/` and is regression
evidence only; its historical one-core run is not the v3 four-core acceptance.

The exact companion master plan currently present in the working environment is
`E:\chrome\BG6022_v3_Migration_Rebuild_Plan.md`. The implementation request
for this stage is `E:\chrome\BG6022_v3_M0_Implementation_Plan.md`; attached
documents provide architecture and stage requirements, while the user's
current stage request authorizes implementation of M0.
