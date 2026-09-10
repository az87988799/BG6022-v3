# BG6022-v3

BG6022-v3 is being rebuilt as a Windows-local ORCA agent. M0 contains the
deterministic execution foundation only: typed requests and plans, two registered
ORCA tools, controlled process execution, source-backed results, and durable run
artifacts. Natural-language planning, PubChem/RDKit, and bounded repair are M1
work and are intentionally not part of this package yet.

## Development

The supported runtime is Python 3.11. With `uv` installed:

```powershell
uv venv --python 3.11
uv sync --group dev
Copy-Item config.example.toml config.toml
uv run python -m bg6022 --help
uv run pytest -m "not live_orca"
```

Edit the ignored `config.toml` only on the local machine. It must point to an
installed ORCA executable and a data directory outside the source tree.

## M0 commands

```powershell
uv run python -m bg6022 --config config.toml doctor --probe-orca
uv run python -m bg6022 --config config.toml tools
uv run python -m bg6022 --config config.toml run-tool single_point --xyz examples/water.xyz --charge 0 --multiplicity 1
uv run python -m bg6022 --config config.toml run-tool single_point --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
uv run python -m bg6022 --config config.toml run-tool optimize_geometry --xyz examples/water.xyz --charge 0 --multiplicity 1 --execute
```

The preview command never starts ORCA. Real SP and Opt runs are separate runs;
they share the public `Request -> Plan -> Step -> Tool.execute` path but are not
combined into a fixed workflow.

## Evidence and limitations

Every attempt keeps its input, initial geometry, stdout, stderr, result, and
registered artifacts. A failed Opt may publish a `restart_candidate`, but never
as the successful `optimized_geometry` port. M0 does not claim frequency
stability, global-minimum status, automatic repair, or natural-language
planning. See `docs/acceptance.md` for the local acceptance record.
