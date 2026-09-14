from __future__ import annotations

from pathlib import Path

from bg6022.agent import Agent
from bg6022.answer import (
    render_already_finished,
    render_clarification,
    render_confirmation,
    render_result,
)
from bg6022.config import load_config
from bg6022.models import InputReference, Plan, Request, Result, ResultTarget, Run, Step
from bg6022.session import utc_now
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


def _water_plan() -> tuple[Request, Plan]:
    request = Request(
        id="request_water",
        description="optimize water",
        operation="Opt",
        requested_results=[
            ResultTarget(step_id="opt", field="opt_final_electronic_energy"),
            ResultTarget(step_id="opt", port="optimized_geometry"),
        ],
        source="chat",
    )
    plan = Plan(
        id="plan_water",
        request_id=request.id,
        steps=[
            Step(
                id="molecule",
                tool="resolve_molecule",
                parameters={"query": "O", "input_kind": "smiles"},
            ),
            Step(
                id="geometry",
                tool="generate_geometry",
                inputs={"molecule": InputReference(step_id="molecule", port="molecule")},
            ),
            Step(
                id="opt",
                tool="optimize_geometry",
                parameters={"method_profile": "r2scan3c", "environment": "gas"},
                inputs={"geometry": InputReference(step_id="geometry", port="geometry")},
            ),
        ],
        requested_results=request.requested_results,
    )
    return request, plan


def test_confirmation_is_chinese_and_uses_the_bound_xyz_atom_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request, plan = _water_plan()
    plan = registry.validate_plan(plan)
    agent = Agent(config, registry, llm=None)
    run = agent._create_chat_run(request, plan)
    agent.advance(run)

    assert run.status == "waiting"
    assert run.pending_data["structure"]["atom_count"] == 3
    text = render_confirmation(run.pending_data)
    assert "水分子（H₂O）" in text
    assert "3 个原子" in text
    assert "结果目标：优化后的电子能、优化后的几何" in text
    assert "4 核" in text
    assert "1024 MB" in text
    assert "192 MB" in text
    assert "最多 1000" in text
    assert "20 分钟" in text
    assert "60 分钟" in text
    assert "{" not in text and "}" not in text


def test_result_and_repeat_confirmation_are_property_scoped_natural_language(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    request = Request(id="request", description="optimize water", operation="Opt")
    step = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
    )
    run = Run(
        id="run",
        request=request,
        plan=Plan(id="plan", request_id=request.id, steps=[step]),
        resources={"cores": 4, "memory_mb": 1024, "maxcore_mb": 192},
        status="succeeded",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    result = Result(
        run_id=run.id,
        step_id=step.id,
        attempt=1,
        status="succeeded",
        values={
            "opt_final_electronic_energy": {
                "value": -76.418938720831,
                "unit": "Eh",
                "token": "-76.418938720831",
            }
        },
    )

    text = render_result(run, result, registry)
    repeated = render_already_finished(run, result, registry)
    assert "优化后的电子能" in text
    assert "-76.418938720831 Eh" in text
    assert "尚未进行频率验证" in text
    assert "opt_final_electronic_energy" not in text
    assert "result.json" not in text
    assert "已经完成，无需再次确认" in repeated
    assert "{" not in text and "}" not in text


def test_clarification_and_missing_property_do_not_dump_internal_data() -> None:
    clarification = render_clarification(
        {"status": "clarify", "clarification": "请说明要查询哪个任务。", "refs": []}
    )
    unavailable = render_clarification(
        {"status": "unavailable", "missing_description": "零点能", "refs": []}
    )
    assert "哪个任务" in clarification
    assert "当前可查询范围内" in unavailable
    assert "从未进行过" in unavailable
    assert "{" not in clarification + unavailable
