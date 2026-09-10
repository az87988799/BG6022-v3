# M0 acceptance record

This file records facts only. M0 is not marked complete until the user has
accepted the implementation after reviewing the pushed commit.

| Item | Evidence | Status |
|---|---|---|
| Independent install | Python 3.11.4 package build and package-outside-source help | passed |
| Offline regression | `18 passed, 1 deselected` with live ORCA disabled | passed |
| Original fixture | Three expected SHA-256 values and byte-preserving fixture copy | passed |
| Windows controls | Windows parent/child cancellation integration test passed | passed |
| Real SP | Run `run_a50fa185c4ca4870baa96aecfff46cb9`, ORCA 6.1.1, 4-core input/output evidence | passed |
| Real Opt | Run `run_9d169f9d6bb540c8a67c603b6ac8898a`, converged energy/final geometry, unchanged initial hash | passed |
| Failure facts | Run `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` retained `opt_not_converged` and `restart_candidate` | passed |
| Architecture boundary | No v2 runtime dependency, no fixed three-step protocol, two public tools | passed |

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
  `sp_electronic_energy = -76.417246084177 Eh`.
- Opt: `run_9d169f9d6bb540c8a67c603b6ac8898a`; parsed
  `opt_final_electronic_energy = -76.418938721015 Eh`; the final `input.xyz`
  was bound as `optimized_geometry`, and stdout geometry comparison passed.
- Controlled failure: `run_7ea3b32a1b7e4f27a0e62a88367f0c8f` with
  `geom_maxiter=1`; ORCA exited normally but optimization did not converge.
  The Result has no successful geometry port and the Run retains a
  `restart_candidate` artifact for M1.

The initial geometry SHA-256 for both successful runs and their attempt copies
is `c4dd083de426421a719338bd0f0e17dc1057a88982e8cf245fb115556525ffd2`.
