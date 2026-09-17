# M3 root repair evidence

This document records the implementation and validation evidence for the root
repair plan. It is a candidate record only; user acceptance is still pending.

## Implemented scope

- Waiting identity turns now pass a bounded `pending_context` into Intake.
- Intake has a temporary `pending_action` contract for identity supplement,
  identity replacement, and clarification. Identity-only actions cannot carry
  a new operation, result target, parameter patch, or geometry binding.
- Complete messages such as `优化水，并计算频率` are routed as a new Request
  and new Run. The old waiting Run remains unchanged.
- Name lookup proposals must use a bounded non-Han lookup spelling and retain
  exact source evidence in `molecule_name_evidence`. A Chinese lookup token is
  rejected locally with `NAME_LOOKUP_NOT_NORMALIZED` and uses the existing one
  structured-correction budget.
- Candidate/CID/exact-title/labelled-SMILES/formula shortcuts remain bounded;
  arbitrary pending names and bare unverified SMILES no longer mutate a Run.
- Same-Run identity updates reset execution permission and the accepted
  execution snapshot. Name-not-found messages display both the raw name and
  the actual lookup query.
- RDKit logging keeps the known hydrogen warning in `logs/rdkit.log`, filters
  only that warning from the console, and preserves helper stderr beside the
  geometry attempt when present.

## Validation

| Check | Command or evidence | Result |
|---|---|---|
| Offline regression | `.venv\Scripts\python.exe -m pytest -q` | `375 passed, 16 skipped` |
| New routing/diagnostics/Intake tests | `test_turn_routing.py`, `test_rdkit_diagnostics.py`, `test_llm.py` | passed |
| Static checks | Ruff format check and Ruff check over the repository | passed |
| Compile check | `.venv\Scripts\python.exe -m compileall -q src` | passed |
| Package build | `python -m pip wheel . --no-deps --no-build-isolation` | passed; `bg6022-0.1.0` |
| Live PubChem | `test_live_pubchem_water_and_ethanol` with sockets enabled | passed; water CID 962 and ethanol CID 702 |
| Live DeepSeek | `test_live_deepseek_intake` with the local key injected only into the child process | passed |
| Live Chinese Intake | `优化水，并计算频率` through the real Intake client | passed; `water`, `Opt + Freq`, original evidence `水`, `pending_action=none` |
| Real ORCA H₂ | CLI single-point smoke run `run_2446d84a7ad34dc097ee700fd34fef9c` | passed; `-1.169380035769 Eh`, normal termination, SCF convergence, input hashes match |

The H₂ input used for the smoke run was temporary and was removed after the
run. The raw ORCA evidence remains under
`E:\BG6022-v3-data\runs\run_2446d84a7ad34dc097ee700fd34fef9c`.

## Acceptance state

Implementation and validation are recorded for this candidate. The repair is
awaiting the Git handoff and then the user's review and acceptance; it is not
marked complete by this record.
