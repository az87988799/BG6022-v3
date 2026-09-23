# Benchmark v1 case set

`cases.jsonl` contains the 32 core B001–B032 cases, six explicitly enabled ORCA cases O001–O006, and one Result-publisher replay contract R001. `holdout.jsonl` is loaded only with `--include-holdout`.

The 20 routine cases use curated Intake/Plan fixtures and execute the production Request normalization, Plan construction, validation, Artifact registration, Result publication, and deterministic grader paths that their fixture exercises. These fixtures are hand-authored contract inputs; they are not stored outputs from a live LLM run. R001 uses a clearly labeled synthetic failed Result solely to exercise failure publication and does not represent an ORCA calculation or scientific observation.

`fixtures/geometries/` holds stable geometry inputs. Planner fixtures are JSON snapshots of expected structured inputs. `fixtures/result/` contains replay fixtures. Empty `intake/` and `repair/` directories reserve space for future captured fixtures. Live LLM and ORCA cases are never inferred from these static inputs: each must be explicitly selected and runs the current production adapters.

Every grader assertion uses one of the finite assertion types in `BenchmarkAssertion`; expression strings and arbitrary evaluation are rejected. A blocked boundary case can pass when its expected refusal or clarification is observed.
