from __future__ import annotations

from bg6022.benchmark.loader import load_cases, load_fixture
from bg6022.benchmark.runner import _build_contract
from bg6022.tools.registry import build_registry


def _build_live_case_contract(case_id: str):
    benchmark_dir = "benchmarks/v1"
    case = next(case for case in load_cases(benchmark_dir) if case.id == case_id)
    fixture = load_fixture(case, benchmark_dir)
    _, request, plan = _build_contract(
        fixture,
        prompt=case.prompt,
        case_id=case.id,
        run_index=1,
        registry=build_registry(),
    )
    assert plan is not None
    return request, plan


def test_non_geometry_requirement_bindings_stay_out_of_legacy_geometry_view() -> None:
    request, plan = _build_live_case_contract("O005_same_geometry_energy_difference")

    difference = next(
        requirement
        for requirement in request.requirements
        if requirement.capability == "same_geometry_method_energy_difference"
    )
    assert set(difference.input_bindings) == {"energy_a", "energy_b"}
    assert len(plan.steps) == 3

    legacy_bindings = request.structure_input.get("required_bindings", [])
    assert not any(
        binding.get("input_port") in {"energy_a", "energy_b"} for binding in legacy_bindings
    )


def test_opt_to_freq_geometry_binding_still_projects_to_legacy_view() -> None:
    request, _ = _build_live_case_contract("O004_water_opt_freq")

    geometry_binding = next(
        binding
        for binding in request.structure_input["required_bindings"]
        if binding.get("consumer_operation") == "Freq"
    )
    assert geometry_binding["input_port"] == "geometry"
    assert geometry_binding["source_operation"] == "Opt"
    assert geometry_binding["source_port"] == "optimized_geometry"
