# ORCA output query and attached reports — implementation evidence

Status: awaiting user acceptance. Implementation follows sections 3–8 of
`E:\chrome\BG6022_ORCA_Output_Query_Minimal_Plan.md`, against baseline
`611316191c0680ec77ed2ab1dca9feea94b14b4a`. The eight structured property families
and ORCA JSON export in section 9 are outside this delivery.

## Delivered behavior

- Both Semantic and legacy Intake select `context_query` entries tagged `raw_output`.
  The private `inspect_orca_output` Tool is absent from the compute registry.
- One verified read per selected stdout, grouped across questions. Ownership binds
  session, Run, producer Step, finished attempt, Result fingerprint and Artifact hash.
  Failed attempts remain readable and explicitly labeled. Only confirmed success
  remains eligible for the existing scientific output contracts.
- Report questions pass through temporary models into the reserved
  `Requirement.constraints.report_queries` key. Empty requests retain their prior
  serialized shape. Creation, resumed execution and report delivery validate the key.
- Terminal delivery selects each Requirement's current actual producer, including
  the latest ended failed attempt when there is no current successful result.
  Missing reports produce partial delivery without changing Run status or retrying.
- Verified delivery locators survive a partial raw report. Raw followups persist
  only bounded source locators/questions, never excerpt bodies. Only an explicit
  repeat with one unambiguous recorded question can reuse old search terms.

## Limits and interpretation

The whole raw query stage shares three questions/files, at most 64 MiB or the smaller
runtime output limit, a five-second cooperative deadline, three excerpts, 120 lines
and 8 KiB of text. Limits are program-owned constants. Reads, hashes and searches
check cancellation; they do not claim to interrupt blocking operating-system IO.
Catalog construction and existing verified-result validation are outside this new
raw query-stage budget.

Search is case-insensitive literal text with normalized matching whitespace. It
retains bounded candidates independently per term, merges overlapping windows per
question and never chooses the last match as a scientific final value. At most
4 KiB of each input line is searched/displayed; skipped long-line content is labeled
truncated. Control characters are visibly escaped. Multiple candidates, truncation
and missing text prevent a claim of complete unambiguous delivery. A header hit is
only raw evidence, not a verified number or a complete table.

## Acceptance coverage

| Plan cases | Executable evidence |
|---|---|
| Q01 | Real historical water stdout: exact source lines and `1.861363429` Debye copied from verified bytes |
| Q02, Q07, Q22 | Unknown synthetic title; three questions share one read with independent found/not-found states |
| Q03 | Undeclared scientific Result values are still rejected before publication |
| Q04–Q06 | Failed-only source; both entrypoints and their second routing calls; mixed verified-energy/raw delivery |
| Q08–Q11 | Requirement persistence/preview; missing first report stays partial; separate method producers; current and explicitly historical attempts |
| Q12–Q13 | Historical Gibbs query allowed; new Gibbs computation stays unsupported; fabricated evidence, demotion and reserved-key collisions rejected |
| Q14 | 30-field verified source still reserves a raw entry within 24 items; unknown historical alias cannot select a different source |
| Q15–Q16 | Ownership/session/path/reparse/hash checks; pre-open size rejection; shared byte limits; long lines; deadline and mid-read cancellation |
| Q17 | Multiple positions and title-only text remain raw evidence, with no final-state inference |
| Q18 | Checked-in acceptance snapshot generated with the unmodified baseline code; stored original hash remains valid after loading |
| Q19–Q21 | Verified focus retained after partial delivery; new-property/ambiguous followups rejected; Markdown/control text safely rendered and absent from subsequent model context |

New tests are in `tests/unit/test_orca_output_query.py` and
`tests/unit/test_output_reports.py`. The no-write tests compare file inventories and
hashes for source Run directories and replace compute/publication/context-write
entrypoints with forbidden-call assertions. Test-only sources and model clients are
explicitly synthetic.

`tests/fixtures/output_query_old_acceptance.json` was produced by exporting the
baseline `src` tree into a temporary directory, invoking its original
`Agent._acceptance_snapshot()` and `execution_fingerprint()`, and saving the resulting
synthetic no-compute Run. The regression loads its stored fingerprint without
regenerating the expected hash.

The real stdout fixture is the existing ORCA 6.1.1 water optimization from
2026-09-07, at 1 core / 2048 MB, formerly rejected for CRLF parsing. This change
tests retrieval of those bytes, not a new computation at today's budget. No new
ORCA, DeepSeek or PubChem live acceptance is claimed. Production compute settings
remain 4 cores / 1024 MB total / `%maxcore 192` / one concurrent job.

## Validation record

Executed on Windows with the repository's `.venv/Scripts/uv.exe` and frozen lock:

| Check | Result |
|---|---|
| `uv sync --frozen --group dev` | passed |
| Two new query/report test modules | 51 passed |
| `uv run --frozen python -m pytest -m "not live_orca" --disable-socket -q` | 561 passed, 12 skipped, 5 deselected |
| `uv run --frozen ruff check .` | passed |
| `uv run --frozen ruff format --check .` | only the pre-existing untracked `docs/plans/ORCA-Results-and-Output-Query-Implementation-Plan.md` has a Python-code-block formatting issue; left unchanged and excluded from this commit |
| Same format check excluding that one unrelated untracked file | 152 files already formatted |
| `uv run --frozen python -m compileall -q src` | passed |
| `git diff --check` | passed |
| `uv build` | source distribution and wheel built successfully, including both new production modules |

No user acceptance has been recorded. Existing untracked plan documents and
`docs/evidence/benchmark1.2-summary.png` are preserved and excluded from this change.
