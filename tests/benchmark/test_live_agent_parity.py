from __future__ import annotations

import json
from pathlib import Path

import pytest
from live_helpers import install_json_handler, make_live_config

from bg6022.benchmark.loader import load_cases
from bg6022.benchmark.runner import _case_data_root, run_case

_WATER_XYZ = (
    "3\nwater geometry fixture; coordinates in angstrom\n"
    "O  0.000000  0.000000  0.000000\n"
    "H  0.758602  0.000000  0.504284\n"
    "H -0.758602  0.000000  0.504284\n"
)


def _case(case_id: str):
    return next(item for item in load_cases("benchmarks/v1") if item.id == case_id)


def test_live_llm_uses_real_agent_and_never_invokes_tools(monkeypatch, tmp_path: Path) -> None:
    from bg6022.benchmark import runner

    executed: list[str] = []
    original_build = runner.build_registry

    def guarded_registry(config):
        registry = original_build(config)

        def forbidden(*_args, **_kwargs):
            executed.append("tool")
            raise AssertionError("planning-only benchmark invoked a Tool")

        registry._tools = {
            name: tool.model_copy(update={"execute_function": forbidden})
            for name, tool in registry._tools.items()
        }
        return registry

    monkeypatch.setattr(runner, "build_registry", guarded_registry)

    def scripted(purpose, messages):
        if purpose == "intake":
            return {
                "intent": "chemistry_compute",
                "subjects": {
                    "water": {"key": "water", "inline_xyz": _WATER_XYZ},
                },
                "requirements": [
                    {
                        "key": "opt_water",
                        "subject_key": "water",
                        "capability": "optimize_geometry",
                        "parameters": {
                            "method_request": "r2scan3c",
                            "charge": 0,
                            "multiplicity": 1,
                        },
                        "outputs": ["opt_final_electronic_energy"],
                        "input_bindings": {},
                    }
                ],
            }
        context = json.loads(messages[-1]["content"])
        request_id = context["request"]["requirements"][0]["id"]
        return {
            "steps": [
                {
                    "key": "opt_water",
                    "tool": "optimize_geometry",
                    "requirement_id": request_id,
                    "parameters": {
                        "method_profile": "r2scan3c",
                        "environment": "gas",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    "inputs": {"geometry": {"artifact_alias": "request_geometry"}},
                }
            ],
            "requested_results": [
                {"step_key": "opt_water", "field": "opt_final_electronic_energy"}
            ],
        }

    install_json_handler(monkeypatch, scripted)
    config = make_live_config(tmp_path)
    observations = run_case(
        _case("B001_single_opt"),
        config=config,
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )

    observation = observations[0]
    assert observation.status == "completed"
    assert observation.stage == "complete"
    assert observation.run is not None
    assert observation.run["status"] == "waiting"
    assert observation.run["pending_data"]["benchmark_barrier"] is True
    assert observation.orca_attempts == 0
    assert observation.llm_structured_outputs[0]["purpose"] == "intake"
    assert executed == []
    isolated_session = _case_data_root(tmp_path / "benchmark-data", "B001_single_opt", 1)
    isolated_session = isolated_session / "sessions" / "bench_live_B001_single_opt_1.json"
    assert isolated_session.is_file()


@pytest.mark.parametrize(
    ("case_id", "unresolved"),
    [
        ("B023_gibbs_unsupported", "Gibbs 自由能不在 Tool 目录中"),
        ("B024_global_conformer_search", "没有全局构象搜索 Tool"),
    ],
)
def test_fresh_session_separates_results_from_capabilities_and_blocks_unsupported(
    monkeypatch, tmp_path: Path, case_id: str, unresolved: str
) -> None:
    captured = []

    def scripted(purpose, messages):
        context = json.loads(messages[-1]["content"])
        if purpose == "intake":
            captured.append(context)
            return {
                "intent": "chemistry_compute",
                "unresolved_requirements": [unresolved],
            }
        raise AssertionError("an unsupported requirement must not reach the Planner")

    install_json_handler(monkeypatch, scripted)
    case = _case(case_id).model_copy(update={"repeat": 1})
    observation = run_case(
        case,
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "blocked"
    assert observation.stage == "intake"
    assert observation.orca_attempts == 0
    assert captured[0]["result_catalog"] == []
    assert captured[0]["capability_catalog"]
    assert observation.intake["unresolved_requirements"]


def test_live_agent_retries_a_locally_invalid_planner_proposal(
    monkeypatch, tmp_path: Path
) -> None:
    planner_contexts = []

    def scripted(purpose, messages):
        context = json.loads(messages[-1]["content"])
        if purpose == "intake":
            return {
                "intent": "chemistry_compute",
                "subjects": {"water": {"key": "water", "inline_xyz": _WATER_XYZ}},
                "requirements": [
                    {
                        "key": "opt_water",
                        "subject_key": "water",
                        "capability": "optimize_geometry",
                        "parameters": {
                            "method_request": "r2scan3c",
                            "charge": 0,
                            "multiplicity": 1,
                        },
                        "outputs": ["opt_final_electronic_energy"],
                        "input_bindings": {},
                    }
                ],
            }
        if purpose == "planner":
            planner_contexts.append(context)
            request_id = context["request"]["requirements"][0]["id"]
            target = (
                {"step_key": "opt_water", "port": "opt_final_electronic_energy"}
                if len(planner_contexts) == 1
                else {"step_key": "opt_water", "field": "opt_final_electronic_energy"}
            )
            return {
                "steps": [
                    {
                        "key": "opt_water",
                        "tool": "optimize_geometry",
                        "requirement_id": request_id,
                        "parameters": {
                            "method_profile": "r2scan3c",
                            "environment": "gas",
                            "charge": 0,
                            "multiplicity": 1,
                        },
                        "inputs": {
                            "geometry": {"artifact_alias": "request_geometry"}
                        },
                    }
                ],
                "requested_results": [target],
            }
        raise AssertionError(f"planning correction unexpectedly called {purpose}")

    install_json_handler(monkeypatch, scripted)
    observation = run_case(
        _case("B001_single_opt").model_copy(update={"repeat": 1}),
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "completed"
    assert observation.stage == "complete"
    assert [item["purpose"] for item in observation.llm_structured_outputs] == [
        "intake",
        "planner",
        "planner",
    ]
    assert len(planner_contexts) == 2
    assert planner_contexts[1]["validation_feedback"]
    assert observation.orca_attempts == 0


def test_live_agent_observer_accepts_a_pre_intake_response(monkeypatch, tmp_path: Path) -> None:
    from bg6022.agent import AgentResponse
    from bg6022.benchmark import runner

    install_json_handler(monkeypatch, lambda *_args: pytest.fail("Intake should not run"))
    monkeypatch.setattr(
        runner._PlanningBarrierAgent,
        "handle_message",
        lambda _self, _message: AgentResponse("The Agent requested a deterministic clarification."),
    )
    observation = run_case(
        _case("B001_single_opt").model_copy(update={"repeat": 1}),
        config=make_live_config(tmp_path),
        allow_live_llm=True,
        benchmark_dir="benchmarks/v1",
        data_root=tmp_path / "benchmark-data",
    )[0]

    assert observation.status == "blocked"
    assert observation.stage == "intake"
    assert observation.error_category == "agent_route_short_circuit"
    assert observation.llm_calls == []
