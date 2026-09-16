from __future__ import annotations

from pathlib import Path
from threading import Event

import httpx
import pytest

from bg6022.agent import Agent, _pending_molecule_selection
from bg6022.config import load_config
from bg6022.llm import LlmError
from bg6022.models import Plan, Request, ResultTarget, Run, Step
from bg6022.molecule_identity import (
    build_identity_constraint,
    formula_token_from_text,
    parse_formula_counts,
)
from bg6022.planner import (
    IntakeOutput,
    QuerySelection,
    QueryTarget,
    _request_output_preferences,
    _validate_query_selection,
    request_from_intake,
    validate_request_plan,
)
from bg6022.session import create_run, utc_now
from bg6022.tools.pubchem import (
    PubChemError,
    PubChemLookup,
    _facts_from_pubchem_candidate,
    _facts_from_smiles,
    execute_resolve_molecule,
    fetch_pubchem,
)
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
""",
        encoding="utf-8",
    )
    return load_config(path)


def _run_for_resolve(tmp_path: Path, identity: dict[str, object]) -> tuple[object, Step, Run]:
    config = _config(tmp_path)
    request = Request(
        id="request_formula",
        description="formula lookup",
        source="chat",
        structure_input={"molecule_identity": identity},
    )
    step = Step(
        id="molecule",
        tool="resolve_molecule",
        parameters={"query": identity["lookup_query"], "input_kind": identity["input_kind"]},
    )
    plan = Plan(id="plan_formula", request_id=request.id, steps=[step])
    run = Run(
        id="run_formula",
        request=request,
        plan=plan,
        resources=config.resources,
        status="running",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    return config, step, run


def _candidate(cid: int, title: str, smiles: str, formula: str = "C2H6O") -> dict[str, object]:
    return {
        "CID": cid,
        "Title": title,
        "ConnectivitySMILES": smiles,
        "MolecularFormula": formula,
        "Charge": 0,
    }


def test_formula_parser_preserves_unicode_and_literal_h20() -> None:
    assert parse_formula_counts("C₆H₆") == {"C": 6, "H": 6}
    assert formula_token_from_text("优化 H20 的结构") == "H20"
    assert formula_token_from_text("SMILES:CO") is None
    identity = build_identity_constraint(
        message="优化 H20 的结构",
        query="water",
        input_kind="name",
    )
    assert identity is not None
    assert identity["input_kind"] == "formula"
    assert identity["lookup_query"] == "H20"
    assert identity["formula"] == "H20"
    with pytest.raises(ValueError, match="multiple formula inputs"):
        build_identity_constraint(
            message="优化 C3H6O 和 C4H8O",
            query="C3H6O",
            input_kind="formula",
        )


def test_formula_identity_is_program_owned_and_planner_preserves_it(tmp_path: Path) -> None:
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        molecule_query="water",
        molecule_input_kind="name",
        requested_results=["sp_electronic_energy"],
    )
    request = request_from_intake(
        "对 H₂O 做 SP",
        intake,
        request_id="request_formula_intake",
        registry=build_registry(_config(tmp_path)),
    )
    identity = request.structure_input["molecule_identity"]
    assert identity["input_kind"] == "formula"
    assert identity["raw_query"] == "H₂O"
    assert identity["element_counts"] == {"H": 2, "O": 1}


def test_planner_cannot_replace_unselected_formula_with_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["SP"],
        molecule_query="C3H6O",
        molecule_input_kind="formula",
        requested_results=["sp_electronic_energy"],
    )
    request = request_from_intake(
        "SP C3H6O",
        intake,
        request_id="request_formula_guard",
        registry=registry,
    )
    plan = Plan(
        id="plan_formula_guard",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "acetone", "input_kind": "name"},
            )
        ],
        requested_results=[ResultTarget(field="sp_electronic_energy")],
    )
    with pytest.raises(ValueError):
        validate_request_plan(request, plan, registry)


def test_formula_pubchem_uses_fastformula_then_batched_properties(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "/fastformula/" in str(request.url):
            return httpx.Response(200, json={"IdentifierList": {"CID": [1, 2]}})
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [
                        _candidate(1, "ethyl alcohol", "CCO"),
                        _candidate(2, "dimethyl ether", "COC"),
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lookup = fetch_pubchem("C2H6O", "formula", config=config, client=client)
    client.close()
    assert lookup.attempts == 2
    assert len(calls) == 2
    assert "/fastformula/C2H6O/cids/JSON" in calls[0]
    assert "/compound/cid/1,2/property/" in calls[1]
    assert lookup.returned_cid_count == 2
    assert len(lookup.source_responses) == 2


def test_formula_pubchem_shares_retry_budget_across_endpoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if "/fastformula/" in str(request.url):
            return httpx.Response(200, json={"IdentifierList": {"CID": [1]}})
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [_candidate(1, "ethyl alcohol", "CCO")]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lookup = fetch_pubchem("C2H6O", "formula", config=config, client=client)
    client.close()
    assert lookup.attempts == 3
    assert len(calls) == 3


def test_remote_formula_metadata_cannot_override_rdkit_facts() -> None:
    with pytest.raises(PubChemError, match="formula metadata") as error:
        _facts_from_pubchem_candidate(
            _candidate(1, "wrong", "O", formula="C2H6O"),
            "C2H6O",
            "test:pubchem",
        )
    assert error.value.category == "identity_mismatch"


def test_existing_charged_smiles_input_keeps_rdkit_facts() -> None:
    facts = _facts_from_smiles("[NH4+]")
    assert facts["formula"] == "H4N"
    assert facts["formal_charge"] == 1
    checked = _facts_from_pubchem_candidate(
        {
            "CID": 999,
            "Title": "ammonium",
            "ConnectivitySMILES": "[NH4+]",
            "MolecularFormula": "H4N+",
            "Charge": 1,
        },
        "ammonium",
        "test:pubchem",
    )
    assert checked["formula"] == "H4N"
    assert checked["formal_charge"] == 1


def test_formula_resolution_returns_bounded_candidates_and_raw_sources(
    tmp_path: Path, monkeypatch
) -> None:
    identity = build_identity_constraint(
        message="formula C2H6O",
        query="C2H6O",
        input_kind="formula",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    lookup = PubChemLookup(
        query="C2H6O",
        input_kind="formula",
        url="test:properties",
        candidates=(
            _candidate(1, "ethyl alcohol", "CCO"),
            _candidate(2, "dimethyl ether", "COC"),
        ),
        raw_bytes=b"combined",
        attempts=2,
        source_responses=(
            {"url": "test:fastformula", "raw_bytes": b"cids"},
            {"url": "test:properties", "raw_bytes": b"properties"},
        ),
        returned_cid_count=2,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "needs_input"
    assert result.diagnostics["category"] == "ambiguous_molecule"
    assert len(result.clarification["candidates"]) == 2
    assert len(result.artifact_ids) == 2
    assert all(item.role == "source_response" for item in run.artifact_index)


def test_formula_resolution_marks_no_match_as_identity_failure(tmp_path: Path, monkeypatch) -> None:
    identity = build_identity_constraint(
        message="formula C2H6O",
        query="C2H6O",
        input_kind="formula",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    lookup = PubChemLookup(
        query="C2H6O",
        input_kind="formula",
        url="test:properties",
        candidates=(_candidate(1, "water", "O", formula="C2H6O"),),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=1,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "failed"
    assert result.diagnostics["category"] == "identity_mismatch"
    assert result.output_ports == {}
    assert result.artifact_ids


def test_pending_formula_candidate_selection_is_snapshot_bound() -> None:
    pending = {
        "input_requirement": "molecule_identity",
        "candidates": [
            {"choice_id": "candidate_1", "cid": 962, "title": "water"},
            {"choice_id": "candidate_2", "cid": 702, "title": "ethanol"},
        ],
    }
    selected = _pending_molecule_selection("2", pending)
    assert selected is not None and selected["candidate"]["cid"] == 702
    invalid = _pending_molecule_selection("3", pending)
    assert invalid is not None and "不在当前候选快照" in invalid["invalid"]
    replacement = _pending_molecule_selection("CID 999", pending)
    assert replacement == {
        "query": "999",
        "input_kind": "cid",
        "explicit_new": True,
        "preserve_identity": True,
    }


def test_followup_subject_binding_rejects_other_molecule() -> None:
    catalog = [
        {
            "subject_ref": "m1",
            "recently_delivered": True,
            "system": {"formula": "H2O", "title": "water", "query": "water"},
            "task": {"description": "water geometry"},
            "result": {"property": "molecular_geometry", "label": "XYZ structure"},
        },
        {
            "subject_ref": "m2",
            "recently_delivered": True,
            "system": {"formula": "C2H6O", "title": "ethanol", "query": "ethanol"},
            "task": {"description": "ethanol geometry"},
            "result": {"property": "molecular_geometry", "label": "XYZ structure"},
        },
    ]
    output = IntakeOutput(
        intent="context_query",
        query_selection=QuerySelection(
            status="selected",
            targets=[
                QueryTarget(
                    subject_ref="m1",
                    property="molecular_geometry",
                    evidence="乙醇的 XYZ",
                    reference_mode="followup",
                )
            ],
        ),
    )
    rejected = _validate_query_selection(
        output,
        ("m1", "m2"),
        message="乙醇的 XYZ 给我",
        result_catalog=catalog,
    )
    assert rejected.query_selection is not None
    assert rejected.query_selection.status == "clarify"


def test_output_preference_negation_does_not_cross_clauses() -> None:
    assert (
        _request_output_preferences("不要链接，把正文给我", {"file_content": "link_only"})[
            "file_content"
        ]
        == "show"
    )
    assert (
        _request_output_preferences("不要正文，只给路径", {"file_content": "show"})[
            "file_content"
        ]
        == "link_only"
    )


def test_answer_cancellation_after_provider_call_is_not_complete(
    tmp_path: Path, monkeypatch
) -> None:
    config, _step, run = _run_for_resolve(tmp_path, build_identity_constraint(
        message="water", query="water", input_kind="name"
    ) or {})
    agent = Agent(config, build_registry(config), llm=object())
    cancel = Event()

    def cancelled_draft(*_args, **_kwargs):
        cancel.set()
        raise LlmError("cancelled", category="cancelled")

    monkeypatch.setattr(agent, "_compose_answer_draft", cancelled_draft)
    _text, delivery = agent._render_verified_delivery(
        run,
        [
            {
                "output_ref": "out_1",
                "kind": "field",
                "expected_type": "integer",
                "name": "atom_count",
                "value": 3,
                "metadata": {},
            }
        ],
        [],
        unavailable=[],
        cancel=cancel,
    )
    assert delivery["status"] == "cancelled"
