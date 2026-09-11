from __future__ import annotations

from pathlib import Path

from bg6022.agent import Agent
from bg6022.config import load_config
from bg6022.planner import (
    InputBindingProposal,
    IntakeOutput,
    PlanProposal,
    PlanStepProposal,
    PlanTargetProposal,
)
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'

[defaults]
method_profile = 'r2scan3c'
environment = 'gas'
""",
        encoding="utf-8",
    )
    return load_config(path)


class FakePlanner:
    def complete_json(self, _messages, schema, **_kwargs):
        if schema is IntakeOutput:
            return IntakeOutput(
                intent="chemistry_compute",
                operation="Opt",
                molecule_query="O",
                molecule_input_kind="smiles",
                requested_results=["energy"],
            )
        return PlanProposal(
            steps=[
                PlanStepProposal(
                    key="molecule",
                    tool="resolve_molecule",
                    parameters={"query": "O", "input_kind": "smiles"},
                ),
                PlanStepProposal(
                    key="geometry",
                    tool="generate_geometry",
                    inputs={"molecule": InputBindingProposal(step_key="molecule", port="molecule")},
                ),
                PlanStepProposal(
                    key="opt",
                    tool="optimize_geometry",
                    parameters={
                        "method_profile": "r2scan3c",
                        "environment": "gas",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    inputs={"geometry": InputBindingProposal(step_key="geometry", port="geometry")},
                ),
            ],
            requested_results=[
                PlanTargetProposal(step_key="opt", field="opt_final_electronic_energy"),
                PlanTargetProposal(step_key="opt", port="optimized_geometry"),
            ],
        )


def test_chat_prepares_then_waits_for_one_confirmation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent = Agent(config, build_registry(config), llm=FakePlanner())
    response = agent.handle_message("optimize water")
    assert response.run is not None
    assert response.run.status == "waiting"
    assert response.run.waiting_for == "confirmation"
    assert response.run.pending_data["operation"] == "Opt"
    assert response.run.execution_permission is False

    repeated = agent.confirm(response.run)
    assert repeated.run is not None
    assert repeated.run.status == "failed"
    assert "execution boundary" in repeated.text
    again = agent.confirm(response.run)
    assert again.run is not None
    assert len(again.run.result_index) == 2
    assert "failed before" in again.text
