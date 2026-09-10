# M0 repair evidence index

Repair commit: `c3a3f27` (`fix: harden M0 execution and result validation`)

Environment used for the repair-version real checks:

- Python 3.11.4
- ORCA 6.1.1 at `E:\orca\orca.exe`
- `r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`
- 4 cores, `%maxcore 384`, 2048 MB Job Object limit, concurrency `1`

The compact raw evidence remains outside the source tree under
`E:\BG6022-v3-data\runs`:

- SP: `run_0d66e2666d4c44f680c888a2ded130bf`
- Opt: `run_5a42ae6d30ef47499e3c794d15112f46`
- Controlled failure replay source: `run_7ea3b32a1b7e4f27a0e62a88367f0c8f`

Each Run directory contains `run.json`, its Result index, the attempt input and
geometry, stdout, stderr, and the registered artifact index. The Opt attempt
also contains the final `input.xyz`. The replay was performed by the repaired
parser without changing the historical raw files.

Validation commands and the observed acceptance record are in
[`docs/acceptance.md`](../acceptance.md).
