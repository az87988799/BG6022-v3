"""Transient source bindings and bounded orchestration for raw output reports."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Event

from bg6022.models import InputReference, Result, Step
from bg6022.orca.profiles import get_profile
from bg6022.output_contracts import public_source_context
from bg6022.session import (
    ArtifactReadError,
    load_run,
    new_id,
    new_metadata_budget,
    read_metadata_json,
    read_registered_artifact_bytes,
    registered_read_path,
)
from bg6022.tools.molecule import ATOMIC_NUMBERS, parse_xyz_bytes
from bg6022.tools.orca_output import (
    MAX_FILE_BYTES,
    MAX_LINES,
    MAX_QUERIES,
    MAX_SECONDS,
    MAX_SNIPPETS,
    MAX_TEXT_BYTES,
    OutputQuerySpec,
    dump_query_spec,
    make_orca_output_tool,
    validate_output_evidence,
)
from bg6022.tools.runtime import ToolCallContext

ORCA_CAPABILITIES = {"optimize_geometry", "single_point", "frequency"}
PROPERTY_SEARCH_TERMS = {
    "dipole_moment": ["DIPOLE MOMENT"],
    "homo_energy": ["ORBITAL ENERGIES"],
    "lumo_energy": ["ORBITAL ENERGIES"],
    "homo_lumo_gap": ["ORBITAL ENERGIES"],
    "frontier_orbitals": ["ORBITAL ENERGIES"],
    "orca_printed_electron_count": ["Number of Electrons"],
    "orca_printed_alpha_electrons": ["Number of Alpha Electrons"],
    "orca_printed_beta_electrons": ["Number of Beta Electrons"],
    "orca_printed_correlated_electrons": ["Number of Correlated Electrons"],
}


def pending_query_targets(pending, hint, bindings):
    """Restore goals from program-owned context, never from a short reply's words."""
    targets = []
    for source in pending["source_bindings"]:
        run_id, step_id = source.get("run_id"), source.get("source_step_id", source.get("step_id"))
        formal = [
            (key, b)
            for key, b in bindings.items()
            if key[1] == hint and b.get("run_id") == run_id and b.get("step_id") == step_id
        ]
        observation = [
            (key, b)
            for key, b in bindings.items()
            if b.get("access") == "readonly_observation"
            and b.get("derivation_id") == hint
            and same_source(b["source_binding"], source)
        ]
        raw = [
            (key, b)
            for key, b in bindings.items()
            if b.get("access") == "raw_output" and same_source(b, source)
        ]
        matches = formal or observation or raw
        if len(matches) != 1:
            raise ValueError("pending source is no longer uniquely bound")
        (ref, prop), binding = matches[0]
        target = {"subject_ref": ref, "property": prop, "evidence": pending["origin_evidence"]}
        if binding.get("access") == "raw_output":
            target["queries"] = [
                {
                    "evidence": pending["origin_evidence"],
                    "property_hint": hint,
                    "question_key": pending["pending_id"],
                    "search_terms": PROPERTY_SEARCH_TERMS.get(hint)
                    or pending.get("search_terms")
                    or [hint.replace("_", " ")],
                }
            ]
        targets.append(target)
    return targets


def normalize_report_queries(queries, message, *, capability=None):
    if not isinstance(queries, list) or len(queries) > MAX_QUERIES:
        raise ValueError("at most three report queries are allowed")
    if queries and capability not in ORCA_CAPABILITIES:
        raise ValueError("output reports require an ORCA producer")
    normalized = []
    for value in queries:
        query = OutputQuerySpec.model_validate(value, strict=True)
        if query.evidence not in message:
            raise ValueError("report evidence must quote the user's message")
        normalized.append(dump_query_spec(query))
    return normalized


def merge_report_constraints(constraints, queries, message, capability):
    result = dict(constraints)
    if "report_queries" in result:
        raise ValueError("report_queries is reserved; use the temporary report_queries field")
    normalized = normalize_report_queries(queries, message, capability=capability)
    if normalized:
        result["report_queries"] = normalized
    return result


def read_report_queries(request):
    result = []
    for requirement in request.requirements:
        queries = normalize_report_queries(
            requirement.constraints.get("report_queries", []),
            request.original_text,
            capability=requirement.capability,
        )
        if queries:
            result.append((requirement, queries))
    if sum(len(queries) for _, queries in result) > MAX_QUERIES:
        raise ValueError("a Request may contain at most three output report questions")
    return result


def validate_raw_query_targets(targets, message, catalog, *, evidence_context=None):
    entries = {
        (item.get("subject_ref"), item.get("result", item).get("property")): item
        for item in catalog
    }
    total = 0
    for target in targets:
        origin = (evidence_context or {}).get("origin_question")
        origin_evidence = (evidence_context or {}).get("origin_evidence")
        resumed = bool(
            origin
            and origin_evidence
            and target.get("evidence") == origin_evidence
            and origin.count(origin_evidence) == 1
        )
        item = entries.get((target["subject_ref"], target["property"]), {})
        queries = [dump_query_spec(q) for q in target.get("queries", [])]
        if item.get("access") != "raw_output":
            if item.get("access") == "readonly_observation":
                total += 1
            if queries:
                raise ValueError("verified targets cannot contain raw queries")
            continue
        if not queries:
            raise ValueError("raw targets require explicit questions")
        specs = [OutputQuerySpec.model_validate(q, strict=True) for q in queries]
        total += len(specs)
        if target.get("reference_mode", "explicit") == "followup":
            recent = [dump_query_spec(q) for q in item.get("recent_queries", [])]
            recent_sources = [x for x in catalog if x.get("recent_queries")]
            explicit_new_goal = all(message.count(q.evidence) == 1 for q in specs)
            if (
                not target.get("evidence", "").strip()
                or message.count(target["evidence"]) != 1
                or (
                    not explicit_new_goal
                    and (
                        len(recent_sources) != 1
                        or len(recent) != 1
                        or queries != recent
                        or not item.get("recently_delivered")
                    )
                )
            ):
                raise ValueError("raw followup needs one unambiguous, unchanged recorded question")
        elif not resumed and (
            not target.get("evidence", "").strip()
            or target["evidence"] not in message
            or any(q.evidence not in message for q in specs)
        ):
            raise ValueError("raw evidence must quote the current message")
        if resumed and any(q.evidence != origin_evidence for q in specs):
            raise ValueError("resumed evidence must match the stored origin question")
    if total > MAX_QUERIES:
        raise ValueError("at most three readonly questions per selection")


def _load_source_result(data_root, run, relative, metadata_budget=None):
    path = registered_read_path(data_root, run, relative)
    if path.name != "result.json" or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("invalid source result")
    result = Result.model_validate(
        read_metadata_json(path, metadata_budget or new_metadata_budget()), strict=True
    )
    if (
        relative != f"{result.attempt_relative_path}/result.json"
        or result.run_id != run.id
        or result.attempt_relative_path != f"{result.step_id}/attempt-{result.attempt:02d}"
    ):
        raise ValueError("source result identity mismatch")
    return result


def _fingerprint(result):
    return hashlib.sha256(result.model_dump_json().encode()).hexdigest()


def _executed_parameters(step, result):
    fingerprint = hashlib.sha256(
        json.dumps(
            step.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return step.parameters if result.step_fingerprint == fingerprint else {}


def new_content_budget(cancel=None, output_limit_bytes=MAX_FILE_BYTES):
    return {
        "cancel": cancel or Event(),
        "deadline": time.monotonic() + MAX_SECONDS,
        "remaining": min(MAX_FILE_BYTES, output_limit_bytes),
        "limits": {"snippets": MAX_SNIPPETS, "lines": MAX_LINES, "bytes": MAX_TEXT_BYTES},
        "cache": {},
    }


def _raw_sources_for_result(run, result, relative, session_id):
    sources = []
    step = next((step for step in run.plan.steps if step.id == result.step_id), None)
    if step is None or step.tool not in ORCA_CAPABILITIES:
        return []
    attempts = [
        a
        for a in run.attempts
        if a.get("step_id") == result.step_id and a.get("attempt") == result.attempt
    ]
    if len(attempts) != 1:
        return []
    attempt = attempts[0]
    if (
        attempt.get("phase") != "finished"
        or attempt.get("status") != result.status
        or (
            attempt.get("result_relative_path") != result.attempt_relative_path
            and not (
                "result_relative_path" not in attempt
                and attempt.get("relative_path") == f"runs/{run.id}/{result.attempt_relative_path}"
            )
        )
    ):
        return []
    parameters = _executed_parameters(step, result)
    try:
        method_label = get_profile(parameters.get("method_profile", "")).display_name
    except ValueError:
        method_label = "unknown"
    subject = run.request.subjects.get(
        step.subject_id
        or next((r.subject_id for r in run.request.requirements if r.id == step.requirement_id), "")
    )
    context = public_source_context(
        {
            "subject_label": (subject.molecule_query or subject.key) if subject else None,
            "method_label": method_label,
            "operation_label": step.tool,
            "task_label": run.request.description,
            "attempt": result.attempt,
            "source_status": result.status,
        }
    )
    for artifact in run.artifact_index:
        if (
            artifact.artifact_type != "orca_output"
            or artifact.role != "stdout"
            or artifact.run_id != run.id
            or artifact.step_id != result.step_id
            or artifact.attempt != result.attempt
            or artifact.id not in result.artifact_ids
            or artifact.id not in attempt.get("artifact_ids", [])
        ):
            continue
        sources.append(
            {
                "access": "raw_output",
                "source_context": context,
                "session_id": session_id,
                "run_id": run.id,
                "source_step_id": result.step_id,
                "attempt": result.attempt,
                "source_result_path": relative,
                "source_result_fingerprint": _fingerprint(result),
                "artifact_id": artifact.id,
                "artifact_sha256": artifact.sha256,
                "artifact_size": artifact.size_bytes,
                "artifact_role": artifact.role,
                "source_status": result.status,
                "is_current_attempt": (
                    run.current_results.get(step.id) == relative
                    if result.status == "succeeded"
                    else result.attempt
                    == max(
                        (a.get("attempt", 0) for a in run.attempts if a.get("step_id") == step.id),
                        default=0,
                    )
                    and step.id not in run.current_results
                ),
                "task_description": run.request.description[:240],
                "tool": step.tool,
                "run_status": run.status,
            }
        )
    return sources


def collect_raw_output_sources(data_root, run, session_id, *, metadata_budget=None, cancel=None):
    if run.session_id not in {None, session_id}:
        return []
    budget = metadata_budget or new_metadata_budget(cancel)
    sources = []
    for relative in reversed(run.result_index):
        try:
            result = _load_source_result(data_root, run, relative, budget)
        except ArtifactReadError as error:
            budget["diagnostic"] = error.category
            break
        except (OSError, ValueError):
            continue
        sources.extend(_raw_sources_for_result(run, result, relative, session_id))
    return sources


def same_source(left, right):
    return all(
        left.get(key) == right.get(key)
        for key in (
            "run_id",
            "source_step_id",
            "attempt",
            "artifact_id",
            "artifact_sha256",
            "source_result_fingerprint",
        )
    )


def build_raw_catalog_entries(sources, recent, *, limit=6):
    entries, bindings = [], {}
    focus = recent
    recent = [r for r in recent if r.get("access") == "raw_output"]
    sources = sorted(
        sources,
        key=lambda s: (
            not any(same_source(s, r) for r in focus),
            not s["is_current_attempt"],
        ),
    )[:limit]
    for index, source in enumerate(sources, 1):
        ref = f"raw_{index}_a{source['attempt']}"
        previous = next((r for r in recent if same_source(source, r)), {})
        entries.append(
            {
                "subject_ref": ref,
                "access": "raw_output",
                "source_context": public_source_context(
                    {
                        **source.get("source_context", {}),
                        "is_current_attempt": source["is_current_attempt"],
                    }
                ),
                "task": {"description": source["task_description"], "status": source["run_status"]},
                "step": {
                    "tool": source["tool"],
                    "attempt": source["attempt"],
                    "source_status": source["source_status"],
                    "is_current_attempt": source["is_current_attempt"],
                    "method": source.get("source_context", {}).get("method_label", "unknown"),
                },
                "result": {
                    "property": "orca_output",
                    "label": "ORCA 原始输出：可按内容查询",
                    "validity": "raw",
                },
                "recently_delivered": bool(previous),
                "recent_queries": previous.get("queries", [])[:MAX_QUERIES],
            }
        )
        bindings[(ref, "orca_output")] = source
    return entries, bindings


def build_readonly_observation_entries(
    sources, data_root, *, metadata_budget=None, limit=6, recent=(), raw_bindings=None
):
    """Advertise bounded derivations from recorded inputs, verified only on selection."""
    entries, bindings = [], {}
    budget = metadata_budget or new_metadata_budget()
    if raw_bindings is not None:
        sources = list(raw_bindings.values())
    for index, source in enumerate(sources[:limit], 1):
        try:
            run = load_run(data_root, source["run_id"], metadata_budget=budget)
            result = _load_source_result(data_root, run, source["source_result_path"], budget)
            step = next(s for s in run.plan.steps if s.id == result.step_id)
            if (
                type(_executed_parameters(step, result).get("charge")) is not int
                or not result.input_bindings.get("geometry")
                or not step.parameters
                or not result.step_fingerprint
            ):
                continue
        except (OSError, ValueError, StopIteration):
            continue
        ref = next(
            (key[0] for key, value in (raw_bindings or {}).items() if same_source(value, source)),
            f"obs_{index}_a{source['attempt']}",
        )
        prop = "chemical_total_electrons"
        entries.append(
            {
                "subject_ref": ref,
                "access": "readonly_observation",
                "derivation_id": prop,
                "source_context": source["source_context"],
                "recently_delivered": any(
                    same_source(r, source)
                    and r.get("access") == "readonly_observation"
                    and any(q.get("property_hint") == prop for q in r.get("queries", []))
                    for r in recent
                ),
                "result": {
                    "property": prop,
                    "label": "由组成与实际总电荷计数的总电子数",
                    "validity": "metadata_only",
                },
            }
        )
        bindings[(ref, prop)] = {
            "access": "readonly_observation",
            "derivation_id": prop,
            "source_binding": source,
            "session_id": source["session_id"],
        }
    return entries, bindings


def _bound_geometry(run, result, step):
    ref = step.inputs.get("geometry")
    artifact_id = result.input_bindings.get("geometry")
    if ref is None or artifact_id not in result.input_artifact_ids:
        raise ValueError("geometry is not bound to executed inputs")
    if ref.artifact_id is not None and ref.artifact_id != artifact_id:
        raise ValueError("geometry input binding changed")
    artifact = next(a for a in run.artifact_index if a.id == artifact_id)
    if artifact.run_id != run.id or artifact.artifact_type != "molecular_geometry":
        raise ValueError("geometry has wrong identity or type")
    return artifact


def query_readonly_observations(
    data_root,
    selections,
    *,
    session_id,
    indexed_ids,
    cancel=None,
    output_limit_bytes=MAX_FILE_BYTES,
    read_budget=None,
):
    """Dispatch advertised derivations without publishing scientific Results."""
    cancel = cancel or Event()
    read_budget = read_budget or new_content_budget(cancel, output_limit_bytes)
    deadline, remaining = read_budget["deadline"], read_budget["remaining"]
    reports, budget = [], new_metadata_budget(cancel)
    for binding, evidence in selections:
        source = binding["source_binding"]
        query = {
            "evidence": evidence,
            "search_terms": ["Number of Electrons"],
            "property_hint": binding["derivation_id"],
        }
        base = {**source, "queries": [query], "access": "readonly_observation"}
        try:
            run, result, _ = resolve_raw_output_source(
                data_root, source, session_id, indexed_ids, metadata_budget=budget
            )
            step = next(s for s in run.plan.steps if s.id == result.step_id)
            geometry = _bound_geometry(run, result, step)
            ref = step.inputs["geometry"]
            if ref.step_id is not None:
                relative = run.current_results.get(ref.step_id)
                upstream = _load_source_result(data_root, run, relative, budget)
                if (
                    upstream.status != "succeeded"
                    or upstream.output_ports.get(ref.port) != geometry.id
                    or geometry.step_id != upstream.step_id
                    or geometry.attempt != upstream.attempt
                ):
                    raise ValueError("geometry producer binding changed")
            payload = read_registered_artifact_bytes(
                data_root, run, geometry, max_bytes=remaining, deadline=deadline, cancel=cancel
            )
            remaining -= len(payload)
            parsed = parse_xyz_bytes(payload)
            charge = _executed_parameters(step, result).get("charge")
            if type(charge) is not int or binding["derivation_id"] != "chemical_total_electrons":
                raise ValueError("derivation lacks executed charge")
            numbers = [ATOMIC_NUMBERS[s] for s in parsed.symbols]
            count = sum(numbers) - charge
            if count < 0:
                raise ValueError("negative chemical electron count")
            observation = {
                "property_hint": binding["derivation_id"],
                "view_kind": "derived_value",
                "token": str(count),
                "unit": "electrons",
                "binding_status": "selected_stage",
                "operands": {"atomic_numbers": numbers, "charge": charge},
                "formula": "sum(Z) - charge",
                "geometry_artifact_id": geometry.id,
                "geometry_sha256": geometry.sha256,
                "source_lines": [],
                "limitations": [],
                "required_scope_complete": True,
            }
            reports.append(
                {
                    **base,
                    "bytes_read": len(payload),
                    "evidence": [
                        {
                            "query_index": 0,
                            "lookup_status": "found",
                            "snippets": [],
                            "ambiguous": False,
                            "truncated": False,
                            "search_status": "complete",
                            "candidate_status": "unique",
                            "binding_status": "selected_stage",
                            "excerpt_complete": True,
                            "observations": [observation],
                            "required_scope_complete": True,
                            "limitations": [],
                        }
                    ],
                }
            )
        except (ValueError, OSError, StopIteration, KeyError) as error:
            reports.append(
                {**base, "error": getattr(error, "category", "invalid_source"), "evidence": []}
            )
        finally:
            read_budget["remaining"] = remaining
    return reports


def _stage_context(run, result, source, geometry_verified):
    step = next(s for s in run.plan.steps if s.id == result.step_id)
    facts = result.diagnostics.get("facts", {})
    return {
        "source_locations": facts.get("source_locations", {}),
        "multiplicity": step.parameters.get("multiplicity"),
        "verified_stage": (
            geometry_verified
            and bool(_executed_parameters(step, result))
            and result.status == "succeeded"
            and facts.get("input_hashes_match") is True
            and facts.get("normal_termination") is True
            and facts.get("scf_converged") is True
            and source["is_current_attempt"]
        ),
    }


def resolve_raw_output_source(data_root, binding, session_id, indexed_ids, *, metadata_budget=None):
    if binding.get("session_id") != session_id or binding.get("run_id") not in indexed_ids:
        raise ValueError("source is outside this session")
    budget = metadata_budget or new_metadata_budget()
    run = load_run(data_root, binding["run_id"], metadata_budget=budget)
    if (
        run.session_id not in {None, session_id}
        or binding["source_result_path"] not in run.result_index
    ):
        raise ValueError("source is outside this session")
    result = _load_source_result(data_root, run, binding["source_result_path"], budget)
    current = next(
        (
            s
            for s in _raw_sources_for_result(run, result, binding["source_result_path"], session_id)
            if same_source(s, binding)
        ),
        None,
    )
    if current != binding:
        raise ValueError("source binding has changed")
    artifact = next(a for a in run.artifact_index if a.id == binding["artifact_id"])
    return run, result, artifact


def query_output_sources(
    data_root,
    selections,
    *,
    session_id,
    indexed_ids,
    cancel=None,
    output_limit_bytes=MAX_FILE_BYTES,
    read_budget=None,
):
    cancel = cancel or Event()
    read_budget = read_budget or new_content_budget(cancel, output_limit_bytes)
    deadline, remaining, limits = (
        read_budget["deadline"],
        read_budget["remaining"],
        read_budget["limits"],
    )
    grouped = {}
    for binding, queries in selections:
        key = (binding["run_id"], binding["artifact_id"])
        group = grouped.setdefault(key, [binding, []])
        group[1].extend(queries)
    if sum(len(q) for _, q in grouped.values()) > MAX_QUERIES:
        raise ValueError("at most three raw questions per turn")
    reports = []
    metadata_budget = new_metadata_budget(cancel)
    for binding, queries in grouped.values():
        base = {**binding, "queries": queries}
        try:
            if cancel.is_set():
                raise ArtifactReadError("cancelled")
            if time.monotonic() >= deadline:
                raise ArtifactReadError("deadline")
            run, result, artifact = resolve_raw_output_source(
                data_root,
                binding,
                session_id,
                indexed_ids,
                metadata_budget=metadata_budget,
            )
            before_limits = dict(limits)
            geometry_verified = False
            step = next(s for s in run.plan.steps if s.id == result.step_id)
            geometry_id = (
                result.output_ports.get("optimized_geometry")
                if step.tool == "optimize_geometry"
                else result.input_bindings.get("geometry")
            )
            geometry = next((a for a in run.artifact_index if a.id == geometry_id), None)
            if geometry is not None and geometry.artifact_type == "molecular_geometry":
                geometry_bytes = read_registered_artifact_bytes(
                    data_root,
                    run,
                    geometry,
                    max_bytes=remaining,
                    deadline=deadline,
                    cancel=cancel,
                )
                parse_xyz_bytes(geometry_bytes)
                remaining -= len(geometry_bytes)
                geometry_verified = (
                    step.tool != "optimize_geometry"
                    or result.diagnostics.get("facts", {}).get("geometry_consistent") is True
                )
            tool = make_orca_output_tool(
                remaining_file_bytes=remaining,
                deadline=deadline,
                remaining_excerpt_limits=limits,
                stage_context=_stage_context(run, result, binding, geometry_verified),
            )
            parameters = tool.validate_parameters({"queries": queries})
            step = Step(
                id=new_id("query"),
                tool=tool.name,
                parameters=parameters,
                inputs={"source": InputReference(artifact_id=artifact.id)},
            )
            context = ToolCallContext(
                data_root=Path(data_root),
                run=run.model_copy(deep=True),
                step=step,
                attempt=result.attempt,
                cancel=cancel,
                workdir=Path(data_root) / "runs" / run.id / result.attempt_relative_path,
                relative_attempt_path=result.attempt_relative_path,
                frozen_inputs={"source": artifact.model_copy(deep=True)},
            )
            queried = tool.execute(step, context)
            if (
                queried.run_id,
                queried.step_id,
                queried.attempt,
                queried.attempt_relative_path,
            ) != (
                run.id,
                step.id,
                result.attempt,
                result.attempt_relative_path,
            ):
                raise ValueError("query Tool identity mismatch")
            # Reserve the file's full allowance even if a failed read returned no bytes.
            remaining -= min(remaining, artifact.size_bytes)
            if queried.status != "succeeded":
                raise ArtifactReadError(queried.diagnostics.get("category", "invalid_source"))
            evidence = queried.values.get("evidence")
            usage = validate_output_evidence(evidence, len(queries), before_limits)
            if queried.artifact_ids or queried.output_ports:
                raise ValueError("a query cannot publish artifacts")
            if queried.diagnostics.get("bytes_read") != artifact.size_bytes:
                raise ValueError("query byte accounting mismatch")
            for key in limits:
                limits[key] = min(limits[key], before_limits[key] - usage[key])
            reports.append(
                {
                    **base,
                    "evidence": evidence,
                    "file_path": str(Path(data_root) / "runs" / run.id / artifact.relative_path),
                    "bytes_read": queried.diagnostics["bytes_read"],
                }
            )
        except (ValueError, OSError) as error:
            reports.append(
                {
                    **base,
                    "error": getattr(error, "category", "invalid_source"),
                    "evidence": [
                        {
                            "query_index": i,
                            "lookup_status": "unavailable",
                            "snippets": [],
                            "truncated": False,
                            "ambiguous": False,
                            "search_status": "unavailable",
                            "candidate_status": "no_match",
                            "binding_status": "unavailable",
                            "limitations": [getattr(error, "category", "invalid_source")],
                            "excerpt_complete": False,
                            "required_scope_complete": False,
                            "observations": [],
                        }
                        for i in range(len(queries))
                    ],
                }
            )
        finally:
            read_budget["remaining"] = remaining
    return reports


def reports_for_run(data_root, run, *, session_id, cancel, output_limit_bytes):
    report_specs = read_report_queries(run.request)
    if not report_specs:
        return []
    if run.status == "cancelled" or (cancel is not None and cancel.is_set()):
        return [
            {"queries": queries, "error": "cancelled", "evidence": []}
            for _, queries in report_specs
        ]
    sources = collect_raw_output_sources(data_root, run, session_id)
    selections, missing = [], []
    for requirement, queries in report_specs:
        producers = [
            s
            for s in run.plan.steps
            if s.requirement_id == requirement.id and s.tool == requirement.capability
        ]
        candidates = [
            s
            for s in sources
            if any(p.id == s["source_step_id"] for p in producers) and s["is_current_attempt"]
        ]
        if len(candidates) == 1:
            selections.append((candidates[0], queries))
        else:
            missing.append({"queries": queries, "error": "invalid_source", "evidence": []})
    return (
        query_output_sources(
            data_root,
            selections,
            session_id=session_id,
            indexed_ids=[run.id],
            cancel=cancel,
            output_limit_bytes=output_limit_bytes,
        )
        + missing
    )
