# M0 repair evidence index

Final implementation commit: `b1ba4e50dbaf1e14e8484db3e4aa521af80c0ac8`
(`fix: close M0 parser and runner edge cases`). Earlier M0 repair work is in
`c3a3f27`; this index covers the final minimal-fix pass as well.

Environment used for the repair-version real checks:

- Python 3.11.4
- ORCA 6.1.1 at `E:\orca\orca.exe`
- `r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`
- 4 cores, `%maxcore 384`, 2048 MB Job Object limit, concurrency `1`

The final code also enforces the following facts: a later SCF without a paired
energy invalidates the attempt; a residual Job member that needs forced
termination changes natural parent success to `failed` while preserving confirmed
tree cleanup; and probe timeout/output-limit/startup/cleanup failures cannot be
reported as `ok=true`. A natural nonzero exit with a recognized ORCA banner remains
valid for probing.

The compact raw evidence remains outside the source tree under
`E:\BG6022-v3-data\runs`:

- Final-fix SP: `run_5ec571c4d0d44a9fa8e4c173af2ab584`
- Final-fix Opt: `run_17053fc626634282a6b090c14eeadd43`
- Earlier repair SP: `run_0d66e2666d4c44f680c888a2ded130bf`
- Earlier repair Opt: `run_5a42ae6d30ef47499e3c794d15112f46`
- Controlled failure replay source: `run_7ea3b32a1b7e4f27a0e62a88367f0c8f`

Each final-fix Run directory contains `run.json`, its Result index, the attempt
input and geometry, stdout, stderr, and the registered artifact index. The Opt
attempt also contains the final `input.xyz`. The controlled failure was replayed
by the final parser without changing historical raw files; the replay result was
`opt_not_converged` with no result value.

The compact, readable evidence package is
`E:\BG6022-v3-data\evidence\BG6022-v3-M0-fix-b1ba4e5.zip`. It contains the
final-fix Run JSON/Result JSON, selected raw input/output files, referenced
artifact contents, and the controlled-failure raw evidence plus replay judgment;
it omits GBW files, the virtual environment, and the ORCA installation.

Validation result: `46 passed, 1 deselected`; both Windows process-tree
integration tests passed; `ruff check`, format check, `compileall`, `uv build`,
and whitespace checks passed. The M0 acceptance record remains awaiting the
user's review and acceptance.

Validation commands and the observed acceptance record are in
[`docs/acceptance.md`](../acceptance.md).
