# BG6022-v3

BG6022-v3 is being rebuilt as a Windows-local ORCA agent. M1 adds the first
natural-language path on the accepted M0 foundation: a bounded DeepSeek client,
real PubChem/RDKit molecule preparation, one confirmation point, the existing
ORCA Tools, and evidence-driven bounded Opt repair. The four production Tools
are `resolve_molecule`, `generate_geometry`, `single_point`, and
`optimize_geometry`; frequency analysis is not part of M1.

## Development

The supported runtime is Python 3.11. With `uv` installed:

```powershell
uv venv --python 3.11
uv sync --group dev
Copy-Item config.example.toml config.toml
uv run python -m bg6022 --help
uv run pytest -m "not live_orca and not live_llm and not live_pubchem"
```

Edit the ignored `config.toml` only on the local machine. It must point to an
installed ORCA executable and a data directory outside the source tree. For
the current M1 repair baseline, set `[runtime]` to `cores = 4`,
`memory_mb = 1024`, `maxcore_mb = 192`, and `max_concurrent_jobs = 1`.
Existing local configurations are not overwritten by the example file; update
those four keys explicitly before creating a new Run. Existing or running Runs
retain the resource snapshot with which they were created.

## Deterministic commands

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

## Chat

```powershell
uv run python -m bg6022 --config config.toml chat
```

Set the environment variable named by `[llm].api_key_env` (by default
`DEEPSEEK_API_KEY`) before sending a message that requires the model. Chat can
prepare explicit SMILES without network access; names/CAS/CID use the real
PubChem endpoint. `/confirm`, `/status`, `/cancel`, `/new`, and `/exit` are
available. The model cannot grant execution permission, edit source code, or
declare scientific success.

## Evidence and limitations

Every attempt keeps its input, initial geometry, stdout, stderr, result, and
registered artifacts. A failed Opt may publish a `restart_candidate`, but never
as the successful `optimized_geometry` port. M1 Opt results are local optimized
electronic energies only; they do not claim frequency stability or a global
minimum. Live M1 acceptance is still pending and is tracked in
`docs/milestones/M1.md`; see `docs/acceptance.md` for the accepted M0 record.
