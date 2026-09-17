from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import httpx
import pytest

from bg6022.agent import Agent, _pending_molecule_selection
from bg6022.config import load_config
from bg6022.llm import LlmError
from bg6022.models import InputReference, Plan, Request, ResultTarget, Run, Step
from bg6022.molecule_identity import (
    build_identity_constraint,
    canonical_structure,
    extract_explicit_smiles,
    formula_token_from_text,
    identity_matches_facts,
    normalize_formula_token,
    parse_formula_counts,
    validate_resolve_binding,
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
from bg6022.session import (
    create_run,
    register_bytes_artifact,
    save_run,
    save_session,
    utc_now,
)
from bg6022.tools.molecule import execute_generate_geometry
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


@pytest.mark.parametrize(
    ("smiles", "formula"),
    [("O=C=O", "CO2"), ("N#N", "N2"), ("ClC(Cl)(Cl)Cl", "CCl4")],
)
def test_no_hydrogen_structures_keep_rdkit_and_source_facts_consistent(
    smiles: str, formula: str
) -> None:
    facts = _facts_from_smiles(smiles)
    assert facts["formula"] == formula
    assert "H" not in facts["element_counts"]
    checked = _facts_from_pubchem_candidate(
        {
            "CID": 9000,
            "Title": formula,
            "ConnectivitySMILES": smiles,
            "MolecularFormula": formula,
            "Charge": 0,
        },
        formula,
        "test:pubchem",
    )
    assert checked["element_counts"] == facts["element_counts"]
    assert checked["formula"] == formula


def test_explicit_hydrogens_are_not_double_counted_and_non_formula_scope_stays_open() -> None:
    facts = _facts_from_smiles("[H]O[H]")
    assert facts["formula"] == "H2O"
    assert facts["element_counts"] == {"H": 2, "O": 1}
    name_identity = {
        "input_kind": "name",
        "raw_query": "ammonium",
        "lookup_query": "ammonium",
    }
    charged = _facts_from_smiles("[NH4+]")
    assert identity_matches_facts(name_identity, charged) == (True, None)


def test_co2_geometry_generation_uses_verified_no_hydrogen_composition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = Request(id="request_co2", description="CO2 geometry", source="chat")
    step = Step(id="geometry", tool="generate_geometry")
    run = Run(
        id="run_co2",
        request=request,
        plan=Plan(id="plan_co2", request_id=request.id, steps=[step]),
        resources=config.resources,
        status="running",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    facts = _facts_from_smiles("O=C=O")
    legacy_facts = dict(facts)
    legacy_facts["element_counts"] = {"C": 1, "O": 2, "H": 0}
    molecule = register_bytes_artifact(
        config.data_root_path,
        run,
        json.dumps({"facts": legacy_facts}, sort_keys=True).encode("utf-8"),
        artifact_type="molecule",
        role="resolved_molecule",
        source="test:co2",
        extension=".json",
    )
    step.inputs["molecule"] = InputReference(artifact_id=molecule.id)
    result = execute_generate_geometry(config, step=step, run=run, cancel=Event())
    assert result.status == "succeeded"
    geometry = next(
        item for item in run.artifact_index if item.id == result.output_ports["geometry"]
    )
    assert geometry.metadata["initial_guess_only"] is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SMILES:C1CCCCC1", "C1CCCCC1"),
        ("SMILES: C1CCCCC1", "C1CCCCC1"),
        ("优化 SMILES:C1CCCCC1", "C1CCCCC1"),
        ("优化SMILES:C1CCCCC1", "C1CCCCC1"),
        ("SMILES：C1CCCCC1", "C1CCCCC1"),
        ("SMILES： CCO", "CCO"),
        ("SMILES=C1CCCCC1", "C1CCCCC1"),
    ],
)
def test_explicit_smiles_forms_are_extracted_once_and_never_formula_normalized(
    text: str, expected: str
) -> None:
    extracted = extract_explicit_smiles(text)
    assert len(extracted) == 1
    assert extracted[0]["raw_query"] == expected
    assert formula_token_from_text(text) is None
    identity = build_identity_constraint(
        message=text,
        query=expected,
        input_kind="smiles",
    )
    assert identity is not None
    assert identity["input_kind"] == "smiles"
    assert identity["raw_query"] == expected


@pytest.mark.parametrize("text", ["formula: C6H14", "formula：\tC6H14", "分子式: C6H14"])
def test_formula_labels_skip_horizontal_space_and_keep_complete_value(text: str) -> None:
    assert formula_token_from_text(text) == "C6H14"
    identity = build_identity_constraint(message=text, query="C6H14", input_kind="formula")
    assert identity is not None
    assert identity["formula"] == "C6H14"


@pytest.mark.parametrize("text", ["formula:C6H14+", "formula:C6H14.Cl", "formula:C6H14(OH)"])
def test_formula_suffixes_are_rejected_as_whole_tokens(text: str) -> None:
    with pytest.raises(ValueError, match="unsupported formula"):
        formula_token_from_text(text)
    with pytest.raises(ValueError):
        build_identity_constraint(message=text, query="C6H14", input_kind="formula")


def test_name_evidence_keeps_original_name_separate_from_lookup_spelling(tmp_path: Path) -> None:
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt"],
        molecule_query="ethane",
        molecule_input_kind="name",
        molecule_name_evidence="乙烷",
        requested_results=["opt_final_electronic_energy"],
    )
    request = request_from_intake(
        "优化乙烷",
        intake,
        request_id="request_ethane_name",
        registry=build_registry(_config(tmp_path)),
    )
    identity = request.structure_input["molecule_identity"]
    assert identity["raw_query"] == "乙烷"
    assert identity["lookup_query"] == "ethane"
    fallback = build_identity_constraint(message="优化乙烷", query="ethane", input_kind="name")
    assert fallback is not None
    assert fallback["raw_query"] == "ethane"
    assert fallback["lookup_query"] == "ethane"


def test_conflicting_explicit_smiles_require_clarification() -> None:
    with pytest.raises(ValueError, match="conflicting SMILES"):
        build_identity_constraint(
            message="SMILES:CCO；SMILES:CCN",
            query="CCO",
            input_kind="smiles",
        )


def test_formula_case_normalization_is_narrow_and_preserves_literal_identity() -> None:
    assert normalize_formula_token("C4h10") == "C4H10"
    assert normalize_formula_token("c4h10") == "C4H10"
    assert normalize_formula_token("C₄H₁₀") == "C4H10"
    assert normalize_formula_token("H20") == "H20"
    assert normalize_formula_token("CO") == "CO"
    assert normalize_formula_token("Cl") == "Cl"
    assert normalize_formula_token("Br") == "Br"
    with pytest.raises(ValueError):
        normalize_formula_token("Co")
    identity = build_identity_constraint(
        message="优化 c4h10",
        query="c4h10",
        input_kind="formula",
    )
    assert identity is not None
    assert identity["raw_query"] == "c4h10"
    assert identity["lookup_query"] == "C4H10"
    assert identity["formula"] == "C4H10"


def test_structure_identity_binding_allows_selected_cid_and_rejects_other_queries(
    tmp_path: Path,
) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    identity.update({"selected_cid": 8058, "selected_smiles": "CCCCCC"})
    validate_resolve_binding(identity, {"query": "8058", "input_kind": "cid"})
    with pytest.raises(ValueError):
        validate_resolve_binding(identity, {"query": "7892", "input_kind": "cid"})
    with pytest.raises(ValueError):
        validate_resolve_binding(identity, {"query": "CCCCCC", "input_kind": "smiles"})
    request = Request(
        id="request_binding",
        description="selected hexane",
        source="chat",
        structure_input={"molecule_identity": identity},
    )
    plan = Plan(
        id="plan_binding",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "7892", "input_kind": "cid"},
            )
        ],
    )
    with pytest.raises(ValueError):
        validate_request_plan(request, plan, build_registry(_config(tmp_path)))


def test_identity_facts_check_selected_fields_without_formula() -> None:
    identity = {
        "input_kind": "name",
        "raw_query": "hexane",
        "lookup_query": "hexane",
        "selected_cid": 8058,
        "selected_smiles": "CCCCCC",
    }
    facts = _facts_from_smiles("CC(C)CCC")
    facts["cid"] = 8058
    matches, reason = identity_matches_facts(identity, facts)
    assert not matches
    assert reason and "selected SMILES" in reason
    facts["isomeric_smiles"] = canonical_structure("CCCCCC")
    matches, reason = identity_matches_facts(identity, facts)
    assert matches is True
    assert reason is None


def test_explicit_cid_identity_requires_the_returned_cid() -> None:
    identity = build_identity_constraint(
        message="对 CID 8058 做 SP",
        query="8058",
        input_kind="cid",
    )
    assert identity is not None
    assert identity["selected_cid"] == 8058
    facts = _facts_from_smiles("CCCCCC")
    facts["cid"] = 7892
    matches, reason = identity_matches_facts(identity, facts)
    assert not matches
    assert reason and "CID" in reason


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


def test_formula_pubchem_default_bound_covers_twenty_three_records(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "/fastformula/" in str(request.url):
            return httpx.Response(200, json={"IdentifierList": {"CID": list(range(1, 24))}})
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [
                        _candidate(cid, "water", "O", formula="H2O") for cid in range(1, 24)
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lookup = fetch_pubchem("H2O", "formula", config=config, client=client)
    client.close()
    assert config.molecule.pubchem_formula_max_cids == 32
    assert lookup.candidates_truncated is False
    assert lookup.search_complete is True
    assert len(lookup.candidates) == 23
    assert ",23/property/" in calls[1]


def test_formula_pubchem_explicit_bound_twenty_keeps_search_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'

[molecule]
pubchem_formula_max_cids = 20
""",
        encoding="utf-8",
    )
    config = load_config(path)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/fastformula/" in str(request.url):
            return httpx.Response(200, json={"IdentifierList": {"CID": list(range(1, 24))}})
        return httpx.Response(
            200,
            json={
                "PropertyTable": {
                    "Properties": [
                        _candidate(cid, "water", "O", formula="H2O") for cid in range(1, 21)
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    lookup = fetch_pubchem("H2O", "formula", config=config, client=client)
    client.close()
    assert config.molecule.pubchem_formula_max_cids == 20
    assert lookup.candidates_truncated is True
    assert lookup.search_complete is False


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
            json={"PropertyTable": {"Properties": [_candidate(1, "ethyl alcohol", "CCO")]}},
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
        candidates=(_candidate(1, "water", "O", formula="H2O"),),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=1,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "needs_input"
    assert result.diagnostics["category"] == "molecule_identity_not_found"
    assert result.output_ports == {}
    assert result.artifact_ids


def test_formula_resolution_ignores_verified_exclusions_when_one_structure_is_unique(
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
            _candidate(1, "ethanol", "CCO", formula="C2H6O"),
            _candidate(2, "charged ethanol", "CC[O-]", formula="C2H5O") | {"Charge": -1},
        ),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=2,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "succeeded"
    assert result.diagnostics["accepted_structure_count"] == 1
    assert result.diagnostics["excluded_candidates"][0]["reason_code"] == "excluded_charged"
    assert result.output_ports["molecule"]


def test_formula_resolution_deduplicates_same_verified_structure_and_keeps_source_cids(
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
            _candidate(20, "ethanol duplicate", "CCO"),
            _candidate(10, "ethanol", "CCO"),
        ),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=2,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "succeeded"
    assert result.diagnostics["accepted_structure_count"] == 1
    molecule = next(
        item for item in run.artifact_index if item.id == result.output_ports["molecule"]
    )
    assert molecule.metadata["cid"] == 10


def test_formula_resolution_does_not_auto_accept_unverified_source_record(
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
        candidates=(_candidate(1, "ethanol", "CCO"), {"CID": 2, "Title": "missing structure"}),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=2,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "needs_input"
    assert result.diagnostics["category"] == "molecule_source_unverified"
    assert result.diagnostics["unverified_candidates"][0]["cid"] == 2


def test_formula_resolution_reports_incomplete_search_even_with_one_candidate(
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
        candidates=(_candidate(1, "ethanol", "CCO"),),
        raw_bytes=b"properties",
        attempts=1,
        source_responses=({"url": "test:properties", "raw_bytes": b"properties"},),
        returned_cid_count=23,
        candidates_truncated=True,
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "needs_input"
    assert result.diagnostics["category"] == "molecule_search_incomplete"


def test_name_not_found_preserves_identity_task_for_name_supplement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = build_identity_constraint(
        message="优化乙烷",
        query="ethane",
        input_kind="name",
        name_evidence="乙烷",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)

    def missing(*_args, **_kwargs):
        raise PubChemError("PubChem found no structure", category="not_found")

    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", missing)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "needs_input"
    assert result.diagnostics["category"] == "molecule_name_not_found"
    assert result.diagnostics["raw_query"] == "乙烷"
    assert result.diagnostics["lookup_query"] == "ethane"
    assert result.clarification["input_requirement"] == "molecule_identity"


def test_direct_tool_rejects_selected_cid_binding_before_network(
    tmp_path: Path, monkeypatch
) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    identity.update({"selected_cid": 8058, "selected_smiles": "CCCCCC"})
    config, step, run = _run_for_resolve(tmp_path, identity)
    step.parameters = {"query": "7892", "input_kind": "cid"}

    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("identity binding must fail before PubChem")

    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", unexpected_fetch)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "failed"
    assert result.diagnostics["category"] == "identity_binding"
    assert result.output_ports == {}


@pytest.mark.parametrize(
    ("candidate_cid", "candidate_smiles"),
    [(7892, "CCCCCC"), (8058, "CC(C)CCC")],
)
def test_selected_cid_and_structure_are_both_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_cid: int,
    candidate_smiles: str,
) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    identity.update({"selected_cid": 8058, "selected_smiles": "CCCCCC"})
    config, step, run = _run_for_resolve(tmp_path, identity)
    step.parameters = {"query": "8058", "input_kind": "cid"}
    lookup = PubChemLookup(
        query="8058",
        input_kind="cid",
        url="test:cid",
        candidates=(_candidate(candidate_cid, "hexane", candidate_smiles, formula="C6H14"),),
        raw_bytes=b"cid",
        attempts=1,
        source_responses=({"url": "test:cid", "raw_bytes": b"cid"},),
    )
    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", lambda *args, **kwargs: lookup)
    result = execute_resolve_molecule(config, step=step, run=run, cancel=Event())
    assert result.status == "failed"
    assert result.diagnostics["category"] == "identity_mismatch"
    assert result.output_ports == {}


def test_wrong_bare_smiles_does_not_replace_waiting_formula_or_change_target(
    tmp_path: Path,
) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    run.status = "waiting"
    run.waiting_for = "clarification"
    run.pending_data = {
        "category": "ambiguous_molecule",
        "input_requirement": "molecule_identity",
        "step_id": step.id,
    }
    before_identity = dict(run.request.structure_input["molecule_identity"])
    before_parameters = dict(step.parameters)
    response = Agent(config, build_registry(config), llm=None)._apply_molecule_clarification(
        run,
        "CCCCCCC",
        "smiles",
        preserve_identity=True,
    )
    assert "rejected" in response.text
    assert run.request.structure_input["molecule_identity"] == before_identity
    assert run.plan.steps[0].parameters == before_parameters
    assert run.status == "waiting"


def test_invalid_original_smiles_is_not_silently_replaced_by_model_case() -> None:
    identity = build_identity_constraint(
        message="SMILES:cccccc",
        query="CCCCCC",
        input_kind="smiles",
    )
    assert identity is not None
    assert identity["raw_query"] == "cccccc"
    assert identity["lookup_query"] == "cccccc"
    pending = {
        "input_requirement": "molecule_identity",
        "candidates": [{"choice_id": "candidate_1", "cid": 8058, "title": "hexane"}],
    }
    assert _pending_molecule_selection("cccccc", pending) is None


def test_invalid_original_smiles_keeps_full_handle_message_waiting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    session_id = "session_invalid_raw_smiles"
    run.session_id = session_id
    run.status = "waiting"
    run.waiting_for = "clarification"
    run.pending_data = {
        "category": "ambiguous_molecule",
        "input_requirement": "molecule_identity",
        "step_id": step.id,
        "candidates": [{"choice_id": "candidate_1", "cid": 8058, "title": "hexane"}],
    }
    save_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
            "pending_prompt": None,
        },
    )
    intake = IntakeOutput(
        intent="chemistry_compute",
        operations=["Opt"],
        molecule_query="CCCCCC",
        molecule_input_kind="smiles",
        requested_results=["opt_final_electronic_energy"],
    )
    agent = Agent(config, build_registry(config), llm=object(), session_id=session_id)
    monkeypatch.setattr("bg6022.agent.intake_message", lambda *args, **kwargs: intake)
    monkeypatch.setattr(
        "bg6022.agent.plan_message",
        lambda *args, **kwargs: pytest.fail("invalid identity must not reach Planner"),
    )
    response = agent.handle_message("cccccc")
    assert response.run is not None
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "clarification"
    assert response.run.request.structure_input["molecule_identity"] == identity
    assert response.run.plan.steps[0].parameters == {
        "query": "C6H14",
        "input_kind": "formula",
    }


def test_name_not_found_supplement_updates_lookup_only_on_same_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = build_identity_constraint(
        message="优化乙烷",
        query="乙烷",
        input_kind="name",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    run.status = "waiting"
    run.waiting_for = "clarification"
    run.pending_data = {
        "category": "molecule_name_not_found",
        "input_requirement": "molecule_identity",
        "step_id": step.id,
        "raw_query": "乙烷",
    }
    agent = Agent(config, build_registry(config), llm=None)

    def fake_advance(current_run: Run, *, cancel=None):
        current_run.status = "waiting"
        current_run.waiting_for = "confirmation"
        current_run.pending_data = {"operation": "Opt"}
        return None

    monkeypatch.setattr(agent, "advance", fake_advance)
    selection = _pending_molecule_selection("ethane", run.pending_data)
    assert selection == {
        "query": "ethane",
        "input_kind": "name",
        "explicit_new": True,
        "preserve_identity": True,
    }
    response = agent._apply_molecule_clarification(
        run,
        "ethane",
        "name",
        preserve_identity=True,
        message="ethane",
    )
    assert response.run is run
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation"
    updated_identity = run.request.structure_input["molecule_identity"]
    assert updated_identity["raw_query"] == "乙烷"
    assert updated_identity["lookup_query"] == "ethane"
    assert run.plan.steps[0].parameters == {"query": "ethane", "input_kind": "name"}


def test_saved_candidate_selection_skips_intake_and_planner(monkeypatch, tmp_path: Path) -> None:
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    config, step, run = _run_for_resolve(tmp_path, identity)
    session_id = "session_saved_candidate"
    run.session_id = session_id
    run.status = "waiting"
    run.waiting_for = "clarification"
    run.pending_data = {
        "category": "ambiguous_molecule",
        "input_requirement": "molecule_identity",
        "step_id": step.id,
        "candidates": [
            {
                "choice_id": "candidate_1",
                "cid": 8058,
                "title": "hexane",
                "formula": "C6H14",
                "canonical_smiles": "CCCCCC",
                "isomeric_smiles": "CCCCCC",
            }
        ],
    }
    save_run(config.data_root_path, run)
    save_session(
        config.data_root_path,
        session_id,
        {
            "session_id": session_id,
            "active_run_id": run.id,
            "recent_messages": [],
            "recent_results": [],
            "last_delivery": [],
            "pending_prompt": None,
        },
    )
    agent = Agent(config, build_registry(config), llm=object(), session_id=session_id)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("saved candidate selection must not call intake or planner")

    monkeypatch.setattr("bg6022.agent.intake_message", fail_if_called)
    monkeypatch.setattr("bg6022.agent.plan_message", fail_if_called)

    def fake_advance(current_run: Run, *, cancel=None):
        current_run.status = "waiting"
        current_run.waiting_for = "confirmation"
        current_run.pending_data = {
            "operation": "SP",
            "request": {"description": "优化 C6H14"},
            "structure": {"formula": "C6H14"},
            "parameters": {"charge": 0, "multiplicity": 1},
            "resources": config.resources,
        }
        return None

    monkeypatch.setattr(agent, "advance", fake_advance)
    response = agent.handle_message("candidate_1")
    assert response.run is not None
    assert response.run.waiting_for == "confirmation"
    selected_identity = response.run.request.structure_input["molecule_identity"]
    assert selected_identity["formula"] == "C6H14"
    assert selected_identity["selected_cid"] == 8058
    assert selected_identity["selected_smiles"] == "CCCCCC"
    assert response.run.plan.steps[0].parameters == {"query": "8058", "input_kind": "cid"}


def test_handle_message_candidate_selection_reaches_confirmation_without_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    identity = build_identity_constraint(
        message="优化 C6H14",
        query="C6H14",
        input_kind="formula",
    )
    assert identity is not None
    request = Request(
        id="request_candidate_chain",
        description="优化 C6H14",
        original_text="优化 C6H14",
        source="chat",
        operations=["Opt"],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
        explicit_parameters={"charge": 0, "multiplicity": 1},
        structure_input={"molecule_identity": identity},
    )
    plan = Plan(
        id="plan_candidate_chain",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "C6H14", "input_kind": "formula"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={
                    "method_profile": "r2scan3c",
                    "environment": "gas",
                    "charge": 0,
                    "multiplicity": 1,
                },
                inputs={"geometry": InputReference(step_id="geometry", port="geometry")},
            ),
        ],
        requested_results=[ResultTarget(step_id="opt", field="opt_final_electronic_energy")],
    )
    plan = validate_request_plan(request, registry.validate_plan(plan), registry)
    candidate_one = _candidate(8058, "hexane", "CCCCCC", formula="C6H14")
    candidate_two = _candidate(7892, "branched hexane", "CC(C)CCC", formula="C6H14")
    formula_lookup = PubChemLookup(
        query="C6H14",
        input_kind="formula",
        url="test:formula",
        candidates=(candidate_one, candidate_two),
        raw_bytes=b"formula",
        attempts=2,
        source_responses=({"url": "test:formula", "raw_bytes": b"formula"},),
        returned_cid_count=2,
    )
    cid_lookup = PubChemLookup(
        query="8058",
        input_kind="cid",
        url="test:cid",
        candidates=(candidate_one,),
        raw_bytes=b"cid",
        attempts=1,
        source_responses=({"url": "test:cid", "raw_bytes": b"cid"},),
        returned_cid_count=1,
    )

    def fake_fetch(query, input_kind, **_kwargs):
        return cid_lookup if input_kind == "cid" else formula_lookup

    monkeypatch.setattr("bg6022.tools.pubchem.fetch_pubchem", fake_fetch)
    agent = Agent(config, registry, llm=object())
    run = agent._create_chat_run(request, plan)
    agent._session["active_run_id"] = run.id
    agent._save_session()
    first = agent.advance(run)
    assert first is not None and first.status == "needs_input"
    assert run.status == "waiting" and run.waiting_for == "clarification"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("candidate continuation must not call intake or planner")

    monkeypatch.setattr("bg6022.agent.intake_message", fail_if_called)
    monkeypatch.setattr("bg6022.agent.plan_message", fail_if_called)
    response = agent.handle_message("candidate_1")
    assert response.run is not None
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    selected = response.run.request.structure_input["molecule_identity"]
    assert selected["formula"] == "C6H14"
    assert selected["selected_cid"] == 8058
    assert selected["selected_smiles"] == "CCCCCC"
    assert response.run.plan.steps[0].parameters == {"query": "8058", "input_kind": "cid"}
    assert "molecule" in response.run.current_results
    assert "geometry" in response.run.current_results
    assert not any(item.get("step_id") == "opt" for item in response.run.attempts)


def test_candidate_choice_and_bare_equivalent_smiles_keep_original_formula() -> None:
    pending = {
        "input_requirement": "molecule_identity",
        "candidates": [
            {
                "choice_id": "candidate_1",
                "cid": 8058,
                "title": "hexane",
                "formula": "C6H14",
                "isomeric_smiles": "CCCCCC",
            }
        ],
    }
    selected = _pending_molecule_selection("C(C)CCCC", pending)
    assert selected is not None and selected.get("candidate") is not None
    assert selected["candidate"]["cid"] == 8058
    assert _pending_molecule_selection("candidate_8058", pending)["invalid"]


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
        _request_output_preferences("不要正文，只给路径", {"file_content": "show"})["file_content"]
        == "link_only"
    )


def test_answer_cancellation_after_provider_call_is_not_complete(
    tmp_path: Path, monkeypatch
) -> None:
    config, _step, run = _run_for_resolve(
        tmp_path, build_identity_constraint(message="water", query="water", input_kind="name") or {}
    )
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
