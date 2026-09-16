"""Bounded PubChem lookup and the real molecule-resolution Tool."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, StrictStr

from bg6022.config import AppConfig
from bg6022.models import Result, Run, Step, Tool
from bg6022.session import (
    register_bytes_artifact,
    run_directory,
    save_run,
)

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
InputKind = Literal["name", "cas", "cid", "smiles"]


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


class PubChemError(ValueError):
    def __init__(self, message: str, *, category: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


def make_resolve_molecule_tool(config: AppConfig | None = None) -> Tool:
    def execute(step: Step, run: Run, cancel: Event) -> Result:
        if config is None:
            raise RuntimeError("tool 'resolve_molecule' is a description-only Tool")
        return execute_resolve_molecule(config, step=step, run=run, cancel=cancel)

    return Tool(
        name="resolve_molecule",
        description=(
            "Resolve a name, CAS, CID, or explicit SMILES into a verified molecule artifact."
        ),
        parameter_model=ResolveMoleculeParameters.__name__,
        parameter_schema=ResolveMoleculeParameters.model_json_schema(),
        parameter_type=ResolveMoleculeParameters,
        output_ports={"molecule": "molecule"},
        results={"molecule_formula": "text", "formal_charge": "integer"},
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
        if parameters.input_kind == "smiles":
            facts = _facts_from_smiles(parameters.query)
            raw_bytes = json.dumps(
                {"input_kind": "smiles", "query": parameters.query, "facts": facts},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            source_url = "explicit:smiles"
            lookup_attempts = 0
        else:
            lookup = fetch_pubchem(
                parameters.query,
                parameters.input_kind,
                config=config,
                cancel=cancel,
                remaining_timeout_seconds=_remaining_active_seconds(run, config),
            )
            if cancel.is_set():
                return _result(
                    run,
                    step,
                    attempt,
                    "cancelled",
                    diagnostics={
                        "category": "cancelled",
                        "reason": "cancelled after molecule lookup",
                    },
                    relative=relative,
                )
            if len(lookup.candidates) != 1:
                raw_artifact = register_bytes_artifact(
                    config.data_root_path,
                    run,
                    lookup.raw_bytes,
                    artifact_type="molecule_source",
                    role="source_response",
                    source=lookup.url,
                    extension=".json",
                    step_id=step.id,
                    attempt=attempt,
                )
                run.attempts.append(
                    {
                        "step_id": step.id,
                        "attempt": attempt,
                        "phase": "finished",
                        "status": "needs_input",
                        "artifact_ids": [raw_artifact.id],
                        "output_ports": {},
                    }
                )
                save_run(config.data_root_path, run)
                return _result(
                    run,
                    step,
                    attempt,
                    "needs_input",
                    diagnostics={
                        "category": "ambiguous_molecule",
                        "reason": "PubChem returned multiple candidate structures",
                        "candidates": list(lookup.candidates),
                        "source_url": lookup.url,
                    },
                    clarification={
                        "question": "Which exact molecule/CID should be used?",
                        "candidates": list(lookup.candidates),
                    },
                    artifact_ids=[raw_artifact.id],
                    relative=relative,
                )
            candidate = lookup.candidates[0]
            facts = _facts_from_pubchem_candidate(candidate, parameters.query, lookup.url)
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
                relative=relative,
            )
        molecule_bytes = json.dumps(
            {
                "schema_version": 1,
                "facts": facts,
                "query": parameters.query,
                "input_kind": parameters.input_kind,
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
        raw_artifact = register_bytes_artifact(
            config.data_root_path,
            run,
            raw_bytes,
            artifact_type="molecule_source",
            role="source_response",
            source=source_url,
            extension=".json",
            step_id=step.id,
            attempt=attempt,
        )
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
                "artifact_ids": [molecule_artifact.id, raw_artifact.id],
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
            artifact_ids=[molecule_artifact.id, raw_artifact.id],
            output_ports={"molecule": molecule_artifact.id},
            parameter_sources={"structure": source_url},
            relative=relative,
        )
    except PubChemError as error:
        return _result(
            run,
            step,
            attempt,
            "failed",
            diagnostics={"category": error.category, "reason": str(error)},
            relative=relative,
        )
    except (OSError, ValueError) as error:
        return _result(
            run,
            step,
            attempt,
            "failed",
            diagnostics={"category": "invalid_molecule", "reason": str(error)},
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
    if input_kind not in {"name", "cas", "cid"}:
        raise PubChemError("PubChem does not accept this input kind", category="invalid_query")
    encoded = quote(str(query), safe="")
    namespace = "cid" if input_kind == "cid" else "name"
    url = (
        f"{PUBCHEM_BASE_URL}/compound/{namespace}/{encoded}/property/"
        "CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,Title,MolecularFormula,Charge/JSON"
    )
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
        for attempt in range(1, attempts_limit + 1):
            if cancel is not None and cancel.is_set():
                raise PubChemError("PubChem lookup cancelled", category="cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PubChemError("PubChem lookup timed out", category="timeout", retryable=True)
            try:
                response = client.get(url, timeout=max(0.001, remaining))
                if cancel is not None and cancel.is_set():
                    raise PubChemError("PubChem lookup cancelled", category="cancelled")
                raw = response.content
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise PubChemError(
                        "PubChem response exceeds the response size bound",
                        category="response_too_large",
                    )
                if response.status_code == 404:
                    raise PubChemError(
                        f"PubChem found no structure for {query!r}", category="not_found"
                    )
                if response.status_code == 401 or response.status_code == 403:
                    raise PubChemError("PubChem rejected the request", category="auth")
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    if attempt >= attempts_limit:
                        raise PubChemError(
                            f"PubChem temporary failure after {attempt} attempts: "
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
                candidates = _extract_candidates(payload)
                if not candidates:
                    raise PubChemError(
                        "PubChem response contains no usable structure", category="not_found"
                    )
                if cancel is not None and cancel.is_set():
                    raise PubChemError("PubChem lookup cancelled", category="cancelled")
                return PubChemLookup(
                    query=str(query),
                    input_kind=input_kind,
                    url=url,
                    candidates=tuple(candidates),
                    raw_bytes=raw,
                    attempts=attempt,
                )
            except PubChemError:
                raise
            except httpx.TimeoutException as error:
                if attempt >= attempts_limit:
                    raise PubChemError(
                        f"PubChem request timed out after {attempt} attempts",
                        category="timeout",
                        retryable=True,
                    ) from error
                _wait_retry(None, cancel, deadline=deadline)
            except httpx.RequestError as error:
                if attempt >= attempts_limit:
                    raise PubChemError(
                        f"PubChem network request failed after {attempt} attempts",
                        category="network_error",
                        retryable=True,
                    ) from error
                _wait_retry(None, cancel, deadline=deadline)
        raise PubChemError(
            "PubChem lookup exhausted its attempt bound", category="temporary_failure"
        )
    finally:
        if owns_client:
            client.close()


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
    return [dict(item) for item in properties if isinstance(item, dict) and _has_smiles(item)]


def _has_smiles(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("ConnectivitySMILES")
        or candidate.get("SMILES")
        or candidate.get("CanonicalSMILES")
    )


def _facts_from_pubchem_candidate(
    candidate: dict[str, Any], query: str, source_url: str
) -> dict[str, Any]:
    smiles = str(
        candidate.get("IsomericSMILES")
        or candidate.get("SMILES")
        or candidate.get("CanonicalSMILES")
        or candidate.get("ConnectivitySMILES")
    )
    facts = _facts_from_smiles(smiles)
    candidate_charge = candidate.get("Charge")
    if candidate_charge is None:
        candidate_charge = facts["formal_charge"]
    try:
        candidate_charge = int(candidate_charge)
    except (TypeError, ValueError) as error:
        raise PubChemError(
            "PubChem returned an invalid formal charge", category="invalid_structure"
        ) from error
    facts.update(
        {
            "cid": candidate.get("CID"),
            "title": candidate.get("Title") or query,
            "formula": candidate.get("MolecularFormula") or facts["formula"],
            "formal_charge": candidate_charge,
            "source_url": source_url,
        }
    )
    return facts


def _facts_from_smiles(smiles: str) -> dict[str, Any]:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdMolDescriptors
    except ImportError as error:
        raise PubChemError(
            "RDKit is not installed; molecule resolution is unavailable", category="dependency"
        ) from error
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PubChemError("SMILES is not a valid RDKit structure", category="invalid_structure")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unsupported = sorted(set(atom_symbols) - {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
    if unsupported:
        raise PubChemError(
            f"structure contains unsupported elements: {unsupported}",
            category="unsupported_element",
        )
    return {
        "canonical_smiles": canonical,
        "isomeric_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "radical_electrons": int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())),
        "atom_symbols": atom_symbols,
        "atom_count": mol.GetNumAtoms(),
        "rdkit_version": getattr(rdBase, "rdkitVersion", "unknown"),
    }


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
