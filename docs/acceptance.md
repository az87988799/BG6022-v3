# M0 acceptance record

This file records facts only. The user accepted M0 on 2026-09-11 after reviewing
the pushed implementation and the readable evidence package. The acceptance
record and the requested VS Code settings are included in the acceptance commit.

| Item | Evidence | Status |
|---|---|---|
| User acceptance | User reviewed the pushed implementation and explicitly requested M0 acceptance on 2026-09-11 | accepted |
| Independent install | Python 3.11.4 package build and package-outside-source help | passed |
| Implementation commit | `b1ba4e50dbaf1e14e8484db3e4aa521af80c0ac8` closes the parser, residual-process, and probe defects | passed |
| Offline regression | `46 passed, 1 deselected` with live ORCA disabled, including the new M0 minimal-fix regressions | passed |
| Original fixture | Three expected SHA-256 values and byte-preserving fixture copy | passed |
| Windows controls | Cancellation and parent-exits-before-child integration tests passed; residual-child case is failed with confirmed empty tree | passed |
| Probe | ORCA 6.1.1 banner probe returned `ok=true` with expected exit code 2, `nonzero_exit`, no stop request, and no guard | passed |
| Real SP | Run `run_5ec571c4d0d44a9fa8e4c173af2ab584`, `-76.417246084177 Eh`, ORCA 6.1.1, 4-core input/output evidence, active time persisted | passed |
| Real Opt | Run `run_17053fc626634282a6b090c14eeadd43`, `-76.418938721015 Eh`, converged final geometry, unchanged initial hash, active time persisted | passed |
| Failure facts | Existing run `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` replayed with the final parser; retained `opt_not_converged` and no success values | passed |
| Architecture boundary | No v2 runtime dependency, no fixed three-step protocol, two public tools | passed |

## M0 minimal repair evidence

The repair pass keeps the same seven durable objects and two Tool implementations.
It adds explicit execution permission and an accepted execution fingerprint, a
single lock-scoped execution guard, bounded runner teardown, a real active-time
clock, final-input hash checks, single-source Tool metadata, and pure preview/query
configuration loading. The final minimal fix additionally rejects a later
unpaired SCF convergence, revokes natural success when residual Job members must
be killed, and preserves probe timeout/output-limit failures. Malformed final
records remain diagnostics rather than success values.

The repaired real runs used Python 3.11.4, ORCA 6.1.1 at `E:\orca\orca.exe`,
`r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`, `%pal nprocs 4`,
`%maxcore 384`, a 2048 MB Job Object limit, and concurrency `1`. Both runs
reported `process_tree_empty=true`, `stop_confirmed=true`, and no remaining
`execution_guard.json`.

The historical fixture has these expected hashes:

| File | SHA-256 |
|---|---|
| `stdout.out` | `d05245e18d406d3d59e7a80ef607eaef889291b085cd2ba6691b17231bbd77af` |
| `geometry.xyz` | `9ad6fbb7bcc183d55ff471d122a4551907003fd53f9060bf2b4065057cc5a433` |
| `input.xyz` | `b5cb73f44717bf84cb411507be4b6d9b0921814da39a2c86dcbfbc1870db8027` |

The M0 real commands are:

```powershell
uv run python -m bg6022 --config config.toml doctor --probe-orca
uv run python -m bg6022 --config config.toml run-tool single_point --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
uv run python -m bg6022 --config config.toml run-tool optimize_geometry --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
```

The historical one-core fixture, fake processes, and offline parser tests never
count as the two required four-core real calculations.

## Recorded real runs

The local configuration used ORCA `E:\orca\orca.exe`, Program Version 6.1.1,
`r2SCAN-3c`, gas phase, charge `0`, multiplicity `1`, `%pal nprocs 4`,
`%maxcore 384`, and a 2048 MB Job Object limit.

- SP: `run_a50fa185c4ca4870baa96aecfff46cb9`; parsed
  `sp_electronic_energy = -76.417246084177 Eh` (pre-repair baseline).
- Repair SP: `run_0d66e2666d4c44f680c888a2ded130bf`; parsed
  `sp_electronic_energy = -76.417246084177 Eh` and active time was persisted.
- Final-fix SP: `run_5ec571c4d0d44a9fa8e4c173af2ab584`; parsed
  `sp_electronic_energy = -76.417246084177 Eh`, with `process_tree_empty=true`
  and `stop_confirmed=true`.
- Opt: `run_9d169f9d6bb540c8a67c603b6ac8898a`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh` (pre-repair baseline).
- Repair Opt: `run_5a42ae6d30ef47499e3c794d15112f46`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh`; the final `input.xyz`
  was bound as `optimized_geometry`, and stdout geometry comparison passed.
- Final-fix Opt: `run_17053fc626634282a6b090c14eeadd43`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh`, with converged and
  consistent final geometry, `process_tree_empty=true`, and `stop_confirmed=true`.
- Controlled failure: `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` with
  `geom_maxiter=1`; ORCA exited normally but optimization did not converge.
  The Result has no successful geometry port and the Run retains a
  `restart_candidate` artifact for M1. Final-code replay remains
  `opt_not_converged` with no published value.

The initial geometry SHA-256 for both successful runs and their attempt copies
is `c4dd083de426421a719338bd0f0e17dc1057a88982e8cf245fb115556525ffd2`.
