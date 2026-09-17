"""Bounded PubChem lookup and the real molecule-resolution Tool."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, StrictStr

from bg6022.config import AppConfig
from bg6022.models import Result, Run, Step, Tool
from bg6022.molecule_identity import (
    SUPPORTED_FORMULA_ELEMENTS,
    MoleculeInputKind,
    canonical_formula,
    identity_matches_facts,
    identity_mismatch_code,
    parse_formula_counts,
    validate_resolve_binding,
)
from bg6022.session import (
    register_bytes_artifact,
    run_directory,
    save_run,
)

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
InputKind = MoleculeInputKind


class ResolveMoleculeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: StrictStr
    input_kind: InputKind = "name"


@dataclass(frozen=True)
class PubChemLookup:
    query: str
    input_kind: str
    url: str
    candidates: tuple[dict[str, Any], ...]
    raw_bytes: bytes
    attempts: int
    source_responses: tuple[dict[str, Any], ...] = ()
    search_complete: bool = True
    returned_cid_count: int = 0
    candidates_truncated: bool = False


class PubChemError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool = False,
        source_responses: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.source_responses = source_responses


def make_resolve_molecule_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, run: Run, cancel: Event) -> Result:
        if config is None:
            raise RuntimeError("tool 'resolve_molecule' is a description-only Tool")
        return execute_resolve_molecule(config, step=step, run=run, cancel=cancel)

    return Tool(
        name="resolve_molecule",
        description=(
            "Resolve a name, CAS, CID, formula, or explicit SMILES into a verified "
            "molecule artifact."
        ),
        parameter_model=ResolveMoleculeParameters.__name__,
        parameter_schema=ResolveMoleculeParameters.model_json_schema(),
        parameter_type=ResolveMoleculeParameters,
        output_ports={"molecule": "molecule"},
        results={"molecule_formula": "text", "formal_charge": "integer"},
        result_properties={
            "molecule": "molecular_identity",
            "molecule_formula": "molecular_formula",
            "formal_charge": "formal_charge",
        },
        result_metadata={
            "molecule": {
                "label": "已验证分子结构记录",
                "description": "来自结构来源并经解析校验的分子身份与 SMILES 记录",
            },
            "molecule_formula": {
                "label": "分子式",
                "description": "已验证分子结构的分子式",
            },
            "formal_charge": {
                "label": "形式电荷",
                "description": "已验证分子结构记录中的形式电荷",
            },
        },
        success_conditions=["structure parsed and identity source recorded"],
        repair_capabilities=[],
        requires_compute_permission=False,
        execute_function=execute if config is not None else None,
    )


def execute_resolve_molecule(config: AppConfig, *, step: Step, run: Run, cancel: Event) -> Result:
    parameters = ResolveMoleculeParameters.model_validate(step.parameters, strict=True)
    attempt = _next_attempt(run, step.id)
    relative = f"{step.id}/attempt-{attempt:02d}"
    (run_directory(config.data_root_path, run.id) / relative).mkdir(parents=True, exist_ok=True)
    identity = run.request.structure_input.get("molecule_identity")
    identity = identity if isinstance(identity, Mapping) else None
    source_responses: tuple[dict[str, Any], ...] = ()
    source_artifact_ids: list[str] = []
    source_url = ""
    lookup_attempts = 0

    try:
        if cancel.is_set():
            return _result(
                run,
                step,
                attempt,
                "cancelled",
                diagnostics={"category": "cancelled", "reason": "cancelled before molecule lookup"},
                relative=relative,
            )
        try:
            validate_resolve_binding(
                identity,
                parameters.model_dump(mode="python"),
            )
        except ValueError as error:
            _record_attempt(
                run,
                step,
                attempt,
                "failed",
                artifact_ids=source_artifact_ids,
            )
            save_run(config.data_root_path, run)
            return _result(
                run,
                step,
                attempt,
                "failed",
                diagnostics={"category": "identity_binding", "reason": str(error)},
                relative=relative,
            )
        if parameters.input_kind == "smiles":
            source_url = "explicit:smiles"
            source_responses = (
                {
                    "url": source_url,
                    "raw_bytes": json.dumps(
                        {"input_kind": "smiles", "query": parameters.query},
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8"),
                },
            )
            facts = _facts_from_smiles(parameters.query)
            raw_bytes = json.dumps(
                {"input_kind": "smiles", "query": parameters.query, "facts": facts},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            source_responses = ({"url": source_url, "raw_bytes": raw_bytes},)
            lookup_attempts = 0
            source_artifact_ids = _register_source_responses(
                config, run, step, attempt, source_responses
            )
            matches, reason = identity_matches_facts(identity, facts)
            if not matches:
                _record_attempt(
                    run,
                    step,
                    attempt,
                    "failed",
                    artifact_ids=source_artifact_ids,
                )
                save_run(config.data_root_path, run)
                return _result(
                    run,
                    step,
                    attempt,
                    "failed",
                    diagnostics={
                        "category": "identity_mismatch",
                        "reason": reason or "explicit SMILES does not match the requested identity",
                        "requested_formula": identity.get("formula") if identity else None,
                        "calculated_formula": facts.get("formula"),
                    },
                    artifact_ids=source_artifact_ids,
                    relative=relative,
                )
        else:
            lookup = fetch_pubchem(
                parameters.query,
                parameters.input_kind,
                config=config,
                cancel=cancel,
                remaining_timeout_seconds=_remaining_active_seconds(run, config),
            )
            source_responses = lookup.source_responses or (
                {"url": lookup.url, "raw_bytes": lookup.raw_bytes},
            )
            source_artifact_ids = _register_source_responses(
                config, run, step, attempt, source_responses
            )
            if cancel.is_set():
                _record_attempt(
                    run,
                    step,
                    attempt,
                    "cancelled",
                    artifact_ids=source_artifact_ids,
                )
                save_run(config.data_root_path, run)
                return _result(
                    run,
                    step,
                    attempt,
                    "cancelled",
                    diagnostics={
                        "category": "cancelled",
                        "reason": "cancelled after molecule lookup",
                    },
                    artifact_ids=source_artifact_ids,
                    relative=relative,
                )
            accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
            excluded: list[dict[str, Any]] = []
            unverified: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            checked_cid_count = 0
            has_explicit_selection = bool(
                identity
                and (
                    identity.get("selected_cid") is not None
                    or identity.get("selected_smiles") is not None
                )
            )
            formula_filter = bool(
                identity
                and identity.get("element_counts") is not None
                and not has_explicit_selection
            )
            for candidate in lookup.candidates:
                checked_cid_count += 1
                candidate_reference = {
                    "cid": candidate.get("CID"),
                    "title": candidate.get("Title"),
                }
                try:
                    candidate_facts = _facts_from_pubchem_candidate(
                        candidate,
                        parameters.query,
                        lookup.url,
                        strict_formula_metadata=(
                            identity is not None and identity.get("element_counts") is not None
                        ),
                    )
                except PubChemError as error:
                    diagnostic = {
                        **candidate_reference,
                        "reason": str(error),
                        "category": error.category,
                    }
                    if has_explicit_selection:
                        rejected.append(diagnostic)
                    else:
                        unverified.append(diagnostic)
                    continue
                matches, reason = identity_matches_facts(identity, candidate_facts)
                if matches:
                    accepted.append((candidate, candidate_facts))
                elif formula_filter:
                    excluded.append(
                        {
                            "cid": candidate_facts.get("cid"),
                            "title": candidate_facts.get("title"),
                            "formula": candidate_facts.get("formula"),
                            "reason": reason or "candidate is outside the formula scope",
                            "reason_code": identity_mismatch_code(identity, candidate_facts)
                            or "excluded_identity_mismatch",
                        }
                    )
                else:
                    rejected.append(
                        {
                            "cid": candidate_facts.get("cid"),
                            "title": candidate_facts.get("title"),
                            "formula": candidate_facts.get("formula"),
                            "reason": reason
                            or "candidate does not satisfy the identity constraint",
                            "category": "identity_mismatch",
                        }
                    )

            grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for candidate, candidate_facts in accepted:
                structure_key = str(
                    candidate_facts.get("isomeric_smiles")
                    or candidate_facts.get("canonical_smiles")
                )
                grouped.setdefault(structure_key, []).append((candidate, candidate_facts))
            groups = list(grouped.values())
            for group in groups:
                group.sort(key=lambda item: _candidate_cid_sort_key(item[0], item[1]))
            candidate_views = [
                _candidate_public_view(
                    representative_candidate,
                    representative_facts,
                    index=index,
                    source_cids=[
                        item_facts.get("cid")
                        for _item_candidate, item_facts in group
                        if isinstance(item_facts.get("cid"), int)
                    ],
                )
                for index, group in enumerate(groups, start=1)
                for representative_candidate, representative_facts in [group[0]]
            ]
            complete = lookup.search_complete and not lookup.candidates_truncated
            resolution_diagnostics = {
                "returned_cid_count": lookup.returned_cid_count or len(lookup.candidates),
                "checked_cid_count": checked_cid_count,
                "accepted_structure_count": len(groups),
                "excluded_candidates": excluded,
                "unverified_candidates": unverified,
                "rejected_candidates": rejected,
                "search_complete": lookup.search_complete,
                "candidates_truncated": lookup.candidates_truncated,
                "source_url": lookup.url,
            }

            def pause_for_identity(category: str, reason: str) -> Result:
                _record_attempt(
                    run,
                    step,
                    attempt,
                    "needs_input",
                    artifact_ids=source_artifact_ids,
                )
                save_run(config.data_root_path, run)
                return _result(
                    run,
                    step,
                    attempt,
                    "needs_input",
                    diagnostics={
                        **resolution_diagnostics,
                        "category": category,
                        "reason": reason,
                        "requested_formula": identity.get("formula") if identity else None,
                        "input_kind": parameters.input_kind,
                        "lookup_query": parameters.query,
                        "raw_query": identity.get("raw_query") if identity else parameters.query,
                        "candidates": candidate_views,
                    },
                    clarification={
                        "question": _resolution_clarification(
                            category,
                            input_kind=parameters.input_kind,
                            raw_query=(identity.get("raw_query") if identity else parameters.query),
                            has_candidates=bool(candidate_views),
                        ),
                        "candidates": candidate_views[
                            : config.molecule.pubchem_formula_max_display_candidates
                        ],
                        "input_requirement": "molecule_identity",
                    },
                    artifact_ids=source_artifact_ids,
                    relative=relative,
                )

            if len(groups) > 1:
                return pause_for_identity(
                    "ambiguous_molecule",
                    "PubChem returned multiple verified structures for this identity query",
                )
            if not complete:
                return pause_for_identity(
                    "molecule_search_incomplete",
                    "the bounded PubChem search did not verify all returned records",
                )
            if unverified:
                return pause_for_identity(
                    "molecule_source_unverified",
                    "one or more PubChem records could not be verified as structures",
                )
            if len(groups) == 1:
                _, facts = groups[0][0]
            else:
                category = (
                    "identity_mismatch"
                    if has_explicit_selection or rejected
                    else "molecule_identity_not_found"
                )
                if category == "molecule_identity_not_found":
                    return pause_for_identity(
                        category,
                        "no verified structure satisfies the requested molecule identity",
                    )
                _record_attempt(
                    run,
                    step,
                    attempt,
                    "failed",
                    artifact_ids=source_artifact_ids,
                )
                save_run(config.data_root_path, run)
                return _result(
                    run,
                    step,
                    attempt,
                    "failed",
                    diagnostics={
                        **resolution_diagnostics,
                        "category": category,
                        "reason": "no PubChem candidate satisfies the requested molecule identity",
                        "requested_formula": identity.get("formula") if identity else None,
                        "candidates": candidate_views,
                    },
                    artifact_ids=source_artifact_ids,
                    relative=relative,
                )
            raw_bytes = lookup.raw_bytes
            source_url = lookup.url
            lookup_attempts = lookup.attempts

        if cancel.is_set():
            return _result(
                run,
                step,
                attempt,
                "cancelled",
                diagnostics={"category": "cancelled", "reason": "cancelled before artifact write"},
                artifact_ids=source_artifact_ids,
                relative=relative,
            )
        molecule_bytes = json.dumps(
            {
                "schema_version": 1,
                "facts": facts,
                "query": parameters.query,
                "input_kind": parameters.input_kind,
                "molecule_identity": dict(identity) if identity is not None else None,
                "source": source_url,
                "lookup_attempts": lookup_attempts,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        molecule_artifact = register_bytes_artifact(
            config.data_root_path,
            run,
            molecule_bytes,
            artifact_type="molecule",
            role="resolved_molecule",
            source=source_url,
            extension=".json",
            step_id=step.id,
            attempt=attempt,
            metadata={
                "canonical_smiles": facts["canonical_smiles"],
                "source": source_url,
                "cid": facts.get("cid"),
            },
        )
        raw_artifact_ids = source_artifact_ids or _register_source_responses(
            config,
            run,
            step,
            attempt,
            ({"url": source_url, "raw_bytes": raw_bytes},),
        )
        artifact_ids = [molecule_artifact.id, *raw_artifact_ids]
        values = {
            "molecule_formula": facts["formula"],
            "formal_charge": facts["formal_charge"],
        }
        run.attempts.append(
            {
                "step_id": step.id,
                "attempt": attempt,
                "phase": "finished",
                "status": "succeeded",
                "artifact_ids": artifact_ids,
                "output_ports": {"molecule": molecule_artifact.id},
            }
        )
        save_run(config.data_root_path, run)
        return _result(
            run,
            step,
            attempt,
            "succeeded",
            values=values,
            diagnostics=resolution_diagnostics if parameters.input_kind != "smiles" else {},
            artifact_ids=artifact_ids,
            output_ports={"molecule": molecule_artifact.id},
            parameter_sources={"structure": source_url},
            relative=relative,
        )
    except PubChemError as error:
        source_artifact_ids = _register_source_responses(
            config,
            run,
            step,
            attempt,
            error.source_responses or source_responses,
        )
        status = "cancelled" if error.category == "cancelled" else "failed"
        if error.category == "not_found" and parameters.input_kind in {"name", "cas", "formula"}:
            status = "needs_input"
        if status == "needs_input" and parameters.input_kind == "name":
            diagnostic_category = "molecule_name_not_found"
        elif status == "needs_input":
            diagnostic_category = "molecule_identity_not_found"
        else:
            diagnostic_category = error.category
        diagnostics = {"category": diagnostic_category, "reason": str(error)}
        if status == "needs_input":
            raw_query = identity.get("raw_query") if identity else parameters.query
            diagnostics.update(
                {
                    "input_requirement": "molecule_identity",
                    "requested_formula": identity.get("formula") if identity else None,
                    "input_kind": parameters.input_kind,
                    "lookup_query": parameters.query,
                    "raw_query": raw_query,
                }
            )
        _record_attempt(
            run,
            step,
            attempt,
            status,
            artifact_ids=source_artifact_ids,
        )
        save_run(config.data_root_path, run)
        return _result(
            run,
            step,
            attempt,
            status,
            diagnostics=diagnostics,
            clarification=(
                {
                    "question": _resolution_clarification(
                        diagnostic_category,
                        input_kind=parameters.input_kind,
                        raw_query=(identity.get("raw_query") if identity else parameters.query),
                        has_candidates=False,
                    ),
                    "input_requirement": "molecule_identity",
                }
                if status == "needs_input"
                else None
            ),
            artifact_ids=source_artifact_ids,
            relative=relative,
        )
    except (OSError, ValueError) as error:
        _record_attempt(
            run,
            step,
            attempt,
            "failed",
            artifact_ids=source_artifact_ids,
        )
        save_run(config.data_root_path, run)
        return _result(
            run,
            step,
            attempt,
            "failed",
            diagnostics={"category": "invalid_molecule", "reason": str(error)},
            artifact_ids=source_artifact_ids,
            relative=relative,
        )


def fetch_pubchem(
    query: str,
    input_kind: str,
    *,
    config: AppConfig,
    cancel: Event | None = None,
    client: httpx.Client | None = None,
    remaining_timeout_seconds: float | None = None,
) -> PubChemLookup:
    if input_kind not in {"name", "cas", "cid", "formula"}:
        raise PubChemError("PubChem does not accept this input kind", category="invalid_query")
    if input_kind == "cid" and (not str(query).isdecimal() or int(query) <= 0):
        raise PubChemError("CID query must be a positive integer", category="invalid_query")
    attempts_limit = config.molecule.pubchem_max_attempts
    request_timeout = float(config.molecule.pubchem_timeout_seconds)
    if remaining_timeout_seconds is not None:
        request_timeout = min(request_timeout, max(0.0, remaining_timeout_seconds))
    if request_timeout <= 0:
        raise PubChemError("PubChem time budget is exhausted", category="timeout")
    deadline = time.monotonic() + request_timeout
    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(request_timeout)
        client = httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        if input_kind == "formula":
            counts = parse_formula_counts(str(query))
            formula = canonical_formula(counts)
            encoded_formula = quote(formula, safe="")
            cid_url = f"{PUBCHEM_BASE_URL}/compound/fastformula/{encoded_formula}/cids/JSON"
            cid_payload, _cid_raw, attempts, cid_sources = _request_json_with_budget(
                client,
                cid_url,
                query=str(query),
                cancel=cancel,
                deadline=deadline,
                attempts_used=0,
                attempts_limit=attempts_limit,
            )
            all_cids = _extract_cids(cid_payload)
            if not all_cids:
                raise PubChemError(
                    f"PubChem found no structure for formula {formula!r}",
                    category="not_found",
                    source_responses=cid_sources,
                )
            max_cids = config.molecule.pubchem_formula_max_cids
            selected_cids = all_cids[:max_cids]
            candidates_truncated = len(all_cids) > len(selected_cids)
            cid_path = ",".join(str(cid) for cid in selected_cids)
            if attempts >= attempts_limit:
                raise PubChemError(
                    "PubChem attempt bound was exhausted before fetching formula properties",
                    category="temporary_failure",
                    retryable=True,
                    source_responses=cid_sources,
                )
            encoded_cids = quote(cid_path, safe=",")
            property_url = (
                f"{PUBCHEM_BASE_URL}/compound/cid/{encoded_cids}/property/"
                "CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,Title,"
                "MolecularFormula,Charge,InChIKey/JSON"
            )
            property_payload, _property_raw, attempts, property_sources = _request_json_with_budget(
                client,
                property_url,
                query=formula,
                cancel=cancel,
                deadline=deadline,
                attempts_used=attempts,
                attempts_limit=attempts_limit,
            )
            candidates = _extract_candidates(property_payload)
            if not candidates:
                raise PubChemError(
                    "PubChem formula properties contain no usable structure",
                    category="not_found",
                    source_responses=cid_sources + property_sources,
                )
            raw_bytes = json.dumps(
                {
                    "formula": formula,
                    "cid_response": cid_payload,
                    "property_response": property_payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            return PubChemLookup(
                query=str(query),
                input_kind=input_kind,
                url=property_url,
                candidates=tuple(candidates),
                raw_bytes=raw_bytes,
                attempts=attempts,
                source_responses=cid_sources + property_sources,
                search_complete=(
                    not candidates_truncated
                    and {
                        int(candidate["CID"])
                        for candidate in candidates
                        if str(candidate.get("CID", "")).isdigit()
                    }
                    >= set(selected_cids)
                ),
                returned_cid_count=len(all_cids),
                candidates_truncated=candidates_truncated,
            )

        encoded = quote(str(query), safe="")
        namespace = "cid" if input_kind == "cid" else "name"
        url = (
            f"{PUBCHEM_BASE_URL}/compound/{namespace}/{encoded}/property/"
            "CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,Title,MolecularFormula,Charge/JSON"
        )
        payload, raw, attempts, sources = _request_json_with_budget(
            client,
            url,
            query=str(query),
            cancel=cancel,
            deadline=deadline,
            attempts_used=0,
            attempts_limit=attempts_limit,
        )
        candidates = _extract_candidates(payload)
        if not candidates:
            raise PubChemError(
                "PubChem response contains no usable structure",
                category="not_found",
                source_responses=sources,
            )
        return PubChemLookup(
            query=str(query),
            input_kind=input_kind,
            url=url,
            candidates=tuple(candidates),
            raw_bytes=raw,
            attempts=attempts,
            source_responses=sources,
            returned_cid_count=len(candidates),
        )
    finally:
        if owns_client:
            client.close()


def _request_json_with_budget(
    client: httpx.Client,
    url: str,
    *,
    query: str,
    cancel: Event | None,
    deadline: float,
    attempts_used: int,
    attempts_limit: int,
) -> tuple[Any, bytes, int, tuple[dict[str, Any], ...]]:
    """Fetch one endpoint while sharing the caller's total attempt budget."""

    source_responses: list[dict[str, Any]] = []
    while attempts_used < attempts_limit:
        if cancel is not None and cancel.is_set():
            raise PubChemError(
                "PubChem lookup cancelled",
                category="cancelled",
                source_responses=tuple(source_responses),
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PubChemError(
                "PubChem lookup timed out",
                category="timeout",
                retryable=True,
                source_responses=tuple(source_responses),
            )
        attempts_used += 1
        try:
            response = client.get(url, timeout=max(0.001, remaining))
            if cancel is not None and cancel.is_set():
                raise PubChemError("PubChem lookup cancelled", category="cancelled")
            raw = response.content
            source_responses.append({"url": url, "raw_bytes": raw})
            if len(raw) > MAX_RESPONSE_BYTES:
                raise PubChemError(
                    "PubChem response exceeds the response size bound",
                    category="response_too_large",
                )
            if response.status_code == 404:
                raise PubChemError(
                    f"PubChem found no structure for {query!r}", category="not_found"
                )
            if response.status_code in {401, 403}:
                raise PubChemError("PubChem rejected the request", category="auth")
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempts_used >= attempts_limit:
                    raise PubChemError(
                        f"PubChem temporary failure after {attempts_used} attempts: "
                        f"HTTP {response.status_code}",
                        category="temporary_failure",
                        retryable=True,
                    )
                _wait_retry(response, cancel, deadline=deadline)
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise PubChemError(
                    f"PubChem request failed: HTTP {response.status_code}",
                    category="http_error",
                )
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise PubChemError(
                    "PubChem returned invalid JSON", category="invalid_response"
                ) from error
            return payload, raw, attempts_used, tuple(source_responses)
        except PubChemError as error:
            if not error.source_responses:
                error.source_responses = tuple(source_responses)
            raise
        except httpx.TimeoutException as error:
            if attempts_used >= attempts_limit:
                raise PubChemError(
                    f"PubChem request timed out after {attempts_used} attempts",
                    category="timeout",
                    retryable=True,
                    source_responses=tuple(source_responses),
                ) from error
            _wait_retry(None, cancel, deadline=deadline)
        except httpx.RequestError as error:
            if attempts_used >= attempts_limit:
                raise PubChemError(
                    f"PubChem network request failed after {attempts_used} attempts",
                    category="network_error",
                    retryable=True,
                    source_responses=tuple(source_responses),
                ) from error
            _wait_retry(None, cancel, deadline=deadline)
    raise PubChemError(
        "PubChem lookup exhausted its attempt bound",
        category="temporary_failure",
        retryable=True,
        source_responses=tuple(source_responses),
    )


def _wait_retry(
    response: httpx.Response | None,
    cancel: Event | None,
    *,
    deadline: float | None = None,
) -> None:
    delay = 0.0
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                delay = min(5.0, max(0.0, float(value)))
            except ValueError:
                delay = 0.0
    if deadline is not None:
        delay = min(delay, max(0.0, deadline - time.monotonic()))
    if cancel is not None:
        if cancel.wait(delay):
            raise PubChemError("PubChem lookup cancelled", category="cancelled")
    elif delay:
        time.sleep(delay)


def _remaining_active_seconds(run: Run, config: AppConfig) -> float:
    limit = float(
        run.resources.get("run_active_timeout_seconds", config.runtime.run_active_timeout_seconds)
    )
    return limit - run.current_active_seconds()


def _extract_candidates(payload: Any) -> list[dict[str, Any]]:
    properties = (
        payload.get("PropertyTable", {}).get("Properties", []) if isinstance(payload, dict) else []
    )
    if not isinstance(properties, list):
        return []
    # Keep records without a structure in the local candidate set.  They are
    # evidence that was returned by PubChem and must be classified as
    # unverified, rather than silently disappearing before completeness is
    # evaluated.
    return [dict(item) for item in properties if isinstance(item, dict)]


def _has_smiles(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("IsomericSMILES")
        or candidate.get("CanonicalSMILES")
        or candidate.get("ConnectivitySMILES")
        or candidate.get("SMILES")
    )


def _facts_from_pubchem_candidate(
    candidate: dict[str, Any],
    query: str,
    source_url: str,
    *,
    strict_formula_metadata: bool = False,
) -> dict[str, Any]:
    smiles = str(
        candidate.get("IsomericSMILES")
        or candidate.get("SMILES")
        or candidate.get("CanonicalSMILES")
        or candidate.get("ConnectivitySMILES")
    )
    facts = _facts_from_smiles(smiles)
    _validate_remote_metadata(candidate, facts, strict_formula=strict_formula_metadata)
    raw_cid = candidate.get("CID")
    if raw_cid is None:
        raise PubChemError(
            "PubChem property record is missing its CID",
            category="invalid_response",
        )
    elif type(raw_cid) is int and raw_cid > 0:
        cid = raw_cid
    elif isinstance(raw_cid, str) and raw_cid.isdecimal() and int(raw_cid) > 0:
        cid = int(raw_cid)
    else:
        raise PubChemError("PubChem returned an invalid CID", category="invalid_response")
    facts.update(
        {
            "cid": cid,
            "title": candidate.get("Title") or query,
            "source_url": source_url,
        }
    )
    return facts


def _validate_remote_metadata(
    candidate: Mapping[str, Any], facts: Mapping[str, Any], *, strict_formula: bool
) -> None:
    """Treat PubChem formula/charge fields as evidence, never as calculated facts."""

    remote_formula = candidate.get("MolecularFormula")
    if strict_formula and remote_formula is None:
        raise PubChemError(
            "PubChem formula property record is missing molecular formula",
            category="invalid_response",
        )
    if remote_formula is not None:
        try:
            remote_counts = _metadata_formula_counts(str(remote_formula))
        except ValueError as error:
            if strict_formula:
                raise PubChemError(
                    "PubChem returned an invalid molecular formula", category="invalid_response"
                ) from error
            remote_counts = None
        if remote_counts is not None and dict(remote_counts) != dict(
            facts.get("element_counts") or {}
        ):
            raise PubChemError(
                "PubChem formula metadata does not match the RDKit structure",
                category="identity_mismatch",
            )
    remote_charge = candidate.get("Charge")
    if strict_formula and remote_charge is None:
        raise PubChemError(
            "PubChem formula property record is missing formal charge",
            category="invalid_response",
        )
    if remote_charge is not None:
        try:
            parsed_charge = int(remote_charge)
        except (TypeError, ValueError) as error:
            raise PubChemError(
                "PubChem returned an invalid formal charge", category="invalid_response"
            ) from error
        if parsed_charge != facts.get("formal_charge"):
            raise PubChemError(
                "PubChem charge metadata does not match the RDKit structure",
                category="identity_mismatch",
            )


def _facts_from_smiles(smiles: str) -> dict[str, Any]:
    try:
        from rdkit import Chem, rdBase
    except ImportError as error:
        raise PubChemError(
            "RDKit is not installed; molecule resolution is unavailable", category="dependency"
        ) from error
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PubChemError("SMILES is not a valid RDKit structure", category="invalid_structure")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unsupported = sorted(set(atom_symbols) - SUPPORTED_FORMULA_ELEMENTS)
    if unsupported:
        raise PubChemError(
            f"structure contains unsupported elements: {unsupported}",
            category="unsupported_element",
        )
    element_counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        element_counts[atom.GetSymbol()] += 1
        element_counts["H"] += int(atom.GetTotalNumHs())
    element_counts = Counter(
        {symbol: count for symbol, count in element_counts.items() if count > 0}
    )
    return {
        "canonical_smiles": canonical,
        "isomeric_smiles": canonical,
        "formula": canonical_formula(element_counts),
        "element_counts": dict(element_counts),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "radical_electrons": int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())),
        "atom_symbols": atom_symbols,
        "atom_count": mol.GetNumAtoms(),
        "component_count": len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)),
        "isotopic": any(atom.GetIsotope() != 0 for atom in mol.GetAtoms()),
        "rdkit_version": getattr(rdBase, "rdkitVersion", "unknown"),
    }


def _metadata_formula_counts(raw: str) -> dict[str, int]:
    """Parse common PubChem formula labels without changing calculated facts."""

    text = re.sub(r"\s+", "", raw).strip()
    text = re.sub(r"[+-]\d*$", "", text)
    if not text:
        raise ValueError("unsupported metadata formula")
    total: Counter[str] = Counter()
    for component in text.split("."):
        total.update(parse_formula_counts(component))
    return dict(total)


def _candidate_public_view(
    candidate: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    index: int,
    source_cids: list[int] | None = None,
) -> dict[str, Any]:
    view = {
        "choice_id": f"candidate_{index}",
        "cid": facts.get("cid") or candidate.get("CID"),
        "title": facts.get("title") or candidate.get("Title"),
        "formula": facts.get("formula"),
        "canonical_smiles": facts.get("canonical_smiles"),
        "isomeric_smiles": facts.get("isomeric_smiles"),
        "source_url": facts.get("source_url"),
    }
    if source_cids and len(source_cids) > 1:
        view["source_cids"] = source_cids
    return view


def _candidate_cid_sort_key(
    candidate: Mapping[str, Any], facts: Mapping[str, Any]
) -> tuple[int, int | str]:
    raw_cid = facts.get("cid", candidate.get("CID"))
    try:
        cid = int(raw_cid)
    except (TypeError, ValueError):
        return (1, str(facts.get("canonical_smiles") or ""))
    if cid <= 0:
        return (1, str(facts.get("canonical_smiles") or ""))
    return (0, cid)


def _resolution_clarification(
    category: str,
    *,
    input_kind: str,
    raw_query: Any,
    has_candidates: bool,
) -> str:
    if category == "ambiguous_molecule":
        return "已核验出多个不同结构，请回复候选编号、CID，或明确的 SMILES；确认后继续原计算任务。"
    if category == "molecule_search_incomplete":
        suffix = "当前已有可选候选。" if has_candidates else "当前没有可安全展示的完整候选。"
        return f"{suffix}当前检索尚未完整，不能确认唯一结构；请回复候选编号、CID 或明确的 SMILES。"
    if category == "molecule_source_unverified":
        return "部分来源记录无法可靠核验；请明确提供 CID 或 SMILES 后继续原计算任务。"
    if category == "molecule_name_not_found":
        return f"来源未识别名称“{raw_query}”，原计算任务已保留。请补充英文名称、CID 或明确 SMILES。"
    if category == "molecule_identity_not_found":
        if input_kind == "formula":
            return "没有找到符合当前分子式约束的结构；请提供明确的 CID 或 SMILES。"
        return "没有找到可验证的结构；请补充明确的名称、CID 或 SMILES。"
    return "候选结构与用户给出的分子身份约束不一致；请提供明确的 CID 或 SMILES。"


def _extract_cids(payload: Any) -> list[int]:
    values = payload.get("IdentifierList", {}).get("CID", []) if isinstance(payload, dict) else []
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            cid = int(value)
        except (TypeError, ValueError):
            continue
        if cid > 0 and cid not in result:
            result.append(cid)
    return result


def _register_source_responses(
    config: AppConfig,
    run: Run,
    step: Step,
    attempt: int,
    responses: tuple[dict[str, Any], ...],
) -> list[str]:
    artifact_ids: list[str] = []
    for index, response in enumerate(responses, start=1):
        raw = response.get("raw_bytes")
        url = response.get("url")
        if not isinstance(raw, bytes) or not isinstance(url, str) or not url:
            continue
        artifact = register_bytes_artifact(
            config.data_root_path,
            run,
            raw,
            artifact_type="molecule_source",
            role="source_response",
            source=url,
            extension=".json",
            step_id=step.id,
            attempt=attempt,
            metadata={"source_sequence": index},
        )
        artifact_ids.append(artifact.id)
    return artifact_ids


def _record_attempt(
    run: Run,
    step: Step,
    attempt: int,
    status: str,
    *,
    artifact_ids: list[str] | None = None,
) -> None:
    run.attempts.append(
        {
            "step_id": step.id,
            "attempt": attempt,
            "phase": "finished",
            "status": status,
            "artifact_ids": list(artifact_ids or []),
            "output_ports": {},
        }
    )


def _result(
    run: Run,
    step: Step,
    attempt: int,
    status: str,
    *,
    values: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    clarification: dict[str, Any] | None = None,
    artifact_ids: list[str] | None = None,
    output_ports: dict[str, str] | None = None,
    parameter_sources: dict[str, str] | None = None,
    relative: str,
) -> Result:
    return Result(
        run_id=run.id,
        step_id=step.id,
        attempt=attempt,
        status=status,  # type: ignore[arg-type]
        values=values or {},
        diagnostics=diagnostics or {},
        clarification=clarification or {},
        artifact_ids=artifact_ids or [],
        output_ports=output_ports or {},
        parameter_sources=parameter_sources or {},
        attempt_relative_path=relative,
    )


def _next_attempt(run: Run, step_id: str) -> int:
    attempts = [
        int(item.get("attempt", 0)) for item in run.attempts if item.get("step_id") == step_id
    ]
    return max(attempts, default=0) + 1


__all__ = [
    "MAX_RESPONSE_BYTES",
    "PubChemError",
    "PubChemLookup",
    "ResolveMoleculeParameters",
    "fetch_pubchem",
    "make_resolve_molecule_tool",
]
