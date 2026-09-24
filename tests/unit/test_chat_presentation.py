from __future__ import annotations

from pathlib import Path

import pytest

from bg6022.agent import Agent
from bg6022.answer import (
    render_already_finished,
    render_clarification,
    render_confirmation,
    render_result,
    render_selected_facts,
    select_facts_for_question,
)
from bg6022.config import load_config
from bg6022.models import (
    GoalCheckRequirement,
    InputReference,
    Plan,
    Request,
    Result,
    ResultTarget,
    Run,
    Step,
)
from bg6022.session import utc_now
from bg6022.tools.registry import build_registry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false
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


def _identity_confirmation_preview(structure: dict[str, str]) -> dict[str, object]:
    return {
        "request": {"description": "身份展示回归", "requested_results": []},
        "operation": "Opt",
        "parameters": {
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        "resources": {},
        "structure": structure,
    }


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
    assert "SMILES:O" in text
    assert "3 个原子" in text
    assert "优化后的电子能" in text
    assert "优化后的几何" in text
    assert "完整计算计划" in text
    assert "生成初始结构" in text
    assert "4 核" in text
    assert "1024 MB" in text
    assert "192 MB" in text
    assert "最多 1000" in text
    assert "20 分钟" in text
    assert "60 分钟" in text
    assert "{" not in text and "}" not in text


def test_confirmation_keeps_water_smiles_for_name_variants() -> None:
    for title in ("water", "Water", "O"):
        text = render_confirmation(
            _identity_confirmation_preview(
                {"title": title, "formula": "H2O", "isomeric_smiles": "O"}
            )
        )
        assert title in text
        assert "H₂O" in text
        assert "SMILES:O" in text


def test_confirmation_marks_missing_identity_fields_without_hiding_known_facts() -> None:
    missing_title = render_confirmation(
        _identity_confirmation_preview({"formula": "C6H14", "canonical_smiles": "CCCCCC"})
    )
    assert "名称未提供" in missing_title
    assert "C₆H₁₄" in missing_title
    assert "SMILES:CCCCCC" in missing_title

    missing_formula = render_confirmation(
        _identity_confirmation_preview({"title": "Hexane", "canonical_smiles": "CCCCCC"})
    )
    assert "Hexane" in missing_formula
    assert "分子式未提供" in missing_formula
    assert "SMILES:CCCCCC" in missing_formula

    missing_smiles = render_confirmation(
        _identity_confirmation_preview({"title": "Hexane", "formula": "C6H14"})
    )
    assert "Hexane" in missing_smiles
    assert "C₆H₁₄" in missing_smiles
    assert "SMILES:未提供" in missing_smiles


def test_confirmation_includes_verified_hexane_identity_in_plan_heading() -> None:
    text = render_confirmation(
        {
            "request": {
                "description": "计算己烷优化后的能量",
                "requested_results": [],
            },
            "operation": "Opt",
            "parameters": {
                "method_profile": "r2scan3c",
                "environment": "gas",
                "charge": 0,
                "multiplicity": 1,
            },
            "resources": {
                "cores": 4,
                "memory_mb": 1024,
                "maxcore_mb": 192,
                "attempt_timeout_seconds": 1200,
                "run_active_timeout_seconds": 3600,
            },
            "structure": {
                "title": "Hexane",
                "formula": "C6H14",
                "isomeric_smiles": "CCCCCC",
                "atom_count": 20,
            },
            "plan_steps": [
                {
                    "index": 1,
                    "tool": "resolve_molecule",
                    "tool_label": "解析分子",
                    "parameters": {"query": "hexane"},
                },
                {
                    "index": 2,
                    "tool": "generate_geometry",
                    "tool_label": "生成初始结构",
                    "parameters": {},
                    "inputs": [{"name": "molecule", "source_step": "步骤 1", "port": "molecule"}],
                },
                {
                    "index": 3,
                    "tool": "optimize_geometry",
                    "tool_label": "几何优化",
                    "parameters": {
                        "method_profile": "r2scan3c",
                        "environment": "gas",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    "inputs": [{"name": "geometry", "source_step": "步骤 2", "port": "geometry"}],
                    "requested_results": [{"label": "优化后的电子能"}],
                },
            ],
            "repair_scope": {"steps": {"s03_opt": {"actions": {"restart_optimization": {}}}}},
            "budget": {
                "max_attempts_per_science_step": 3,
                "max_extra_executions_by_category": {"electronic_structure": 3},
            },
        }
    )

    assert "准备对Hexane  C₆H₁₄  (SMILES:CCCCCC)  执行以下完整计算计划：" in text
    assert "1. 解析分子；分子查询“hexane”。" in text
    assert "2. 生成初始结构；molecule来自步骤 1.molecule。" in text
    assert (
        "3. 几何优化；方法 r²SCAN-3c；气相；电荷 0；多重度 1；"
        "geometry来自步骤 2.geometry；目标：优化后的电子能。"
    ) in text
    assert "使用 4 核，总内存上限 1024 MB，每核 MaxCore 为 192 MB。" in text
    assert (
        "若因优化迭代次数用尽而失败，可从经校验的中间结构继续优化，并将迭代上限提高至最多 1000。"
        in text
    )
    assert "输入 /confirm 开始，也可以先告诉我需要调整什么。" in text


def test_confirmation_discloses_the_full_ordered_opt_freq_sp_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_registry(config)
    agent = Agent(config, registry, llm=None, session_id="session_confirm_composed")
    opt = Step(
        id="opt",
        tool="optimize_geometry",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(artifact_id="request_geometry")},
    )
    freq = Step(
        id="freq",
        tool="frequency",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
    )
    sp = Step(
        id="sp",
        tool="single_point",
        parameters={
            "method_profile": "r2scan3c",
            "environment": "gas",
            "charge": 0,
            "multiplicity": 1,
        },
        inputs={"geometry": InputReference(step_id=opt.id, port="optimized_geometry")},
        goal_checks=[GoalCheckRequirement(source_step_id=freq.id, check="local_minimum_supported")],
    )
    targets = [
        ResultTarget(step_id=freq.id, field="vibrational_frequencies"),
        ResultTarget(step_id=freq.id, check="local_minimum_supported"),
        ResultTarget(step_id=sp.id, field="sp_electronic_energy"),
    ]
    request = Request(
        id="request_composed_confirmation",
        description="optimize water, calculate frequencies, then run an independent SP",
        operations=["Opt", "Freq", "SP"],
        requested_results=targets,
        explicit_parameters={"charge": 0, "multiplicity": 1},
        structure_input={
            "xyz_text": "3\nwater\nO 0 0 0\nH 0.758602 0 0.504284\nH -0.758602 0 0.504284\n"
        },
        source="chat",
    )
    plan = Plan(
        id="plan_composed_confirmation",
        request_id=request.id,
        steps=[opt, freq, sp],
        requested_results=targets,
    )
    run = agent._create_chat_run(request, plan)
    agent.advance(run)
    assert run.status == "waiting"
    assert run.waiting_for == "confirmation", run.pending_data
    text = render_confirmation(run.pending_data)

    assert "完整计算计划" in text
    assert text.index("几何优化") < text.index("频率计算")
    assert text.index("频率计算") < text.index("独立单点计算")
    assert "局部极小值检查（passed）" in text
    assert "振动频率" in text
    assert "单点电子能" in text
    assert "1024 MB" in text and "192 MB" in text
    assert "result.json" not in text
    assert "sha256" not in text


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
        {"status": "clarify", "clarification": "请说明要查询哪个任务。", "targets": []}
    )
    unavailable = render_clarification(
        {"status": "unavailable", "missing_description": "零点能", "targets": []}
    )
    assert "哪个任务" in clarification
    assert "当前可查询范围内" in unavailable
    assert "从未进行过" in unavailable
    assert "{" not in clarification + unavailable


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("ambiguous_molecule", "多个不同结构"),
        ("molecule_search_incomplete", "检索尚未完整"),
        ("molecule_source_unverified", "无法可靠核验"),
    ],
)
def test_molecule_clarifications_explain_the_verified_failure_reason(
    category: str, expected: str
) -> None:
    text = render_clarification(
        {
            "category": category,
            "candidates": [
                {
                    "choice_id": "candidate_1",
                    "cid": 1,
                    "title": "ethanol",
                    "formula": "C2H6O",
                    "isomeric_smiles": "CCO",
                }
            ],
        }
    )
    assert expected in text
    assert "candidate_1" in text


def test_structured_property_targets_cover_multiple_facts_for_one_task() -> None:
    energy = {
        "subject_ref": "t1",
        "result_property": "electronic_energy",
        "name": "opt_final_electronic_energy",
        "kind": "field",
        "expected_type": "Eh",
        "value": {"value": -76.4, "unit": "Eh", "token": "-76.4"},
        "metadata": {
            "label": "优化后的电子能",
            "description": "几何优化末态的电子能，不含零点能和热校正",
        },
    }
    geometry = {
        "subject_ref": "t1",
        "result_property": "molecular_geometry",
        "name": "optimized_geometry",
        "kind": "port",
        "expected_type": "molecular_geometry",
        "value": {"artifact_id": "private-artifact-id"},
        "metadata": {"label": "优化后的结构", "description": "已通过收敛检查"},
    }

    targets = [
        {"subject_ref": "t1", "property": "electronic_energy"},
        {"subject_ref": "t1", "property": "molecular_geometry"},
    ]
    selected, covered = select_facts_for_question(targets, [energy, geometry])
    assert covered
    assert selected == [energy, geometry]

    selected, covered = select_facts_for_question(targets, [energy])
    assert not covered
    assert selected == [energy]


def test_structured_targets_cannot_be_covered_by_a_different_task() -> None:
    expected = {
        "subject_ref": "t1",
        "result_property": "molecular_geometry",
        "name": "optimized_geometry",
        "kind": "port",
        "expected_type": "molecular_geometry",
        "value": {"artifact_id": "private-artifact-id"},
    }
    other_task_energy = {
        "subject_ref": "t2",
        "result_property": "electronic_energy",
        "name": "opt_final_electronic_energy",
        "kind": "field",
        "expected_type": "Eh",
        "value": {"value": -76.4, "unit": "Eh", "token": "-76.4"},
    }

    selected, covered = select_facts_for_question(
        [{"subject_ref": "t1", "property": "electronic_energy"}],
        [expected, other_task_energy],
    )

    assert not covered
    assert selected == []


def test_multi_task_result_context_identifies_each_request() -> None:
    common = {
        "system": "水分子（H₂O）",
        "step_tool": "optimize_geometry",
        "method_profile": "r2scan3c",
        "environment": "gas",
        "kind": "field",
        "expected_type": "Eh",
        "name": "opt_final_electronic_energy",
        "value": {"value": -76.4, "unit": "Eh", "token": "-76.4"},
        "metadata": {"label": "优化后的电子能"},
    }
    facts = [
        {
            **common,
            "task_key": "run-a:opt",
            "task_description": "第一轮水分子优化",
            "task_created_at": "2026-09-15T10:00:00+00:00",
        },
        {
            **common,
            "task_key": "run-b:opt",
            "task_description": "第二轮水分子优化",
            "task_created_at": "2026-09-15T11:00:00+00:00",
        },
    ]

    text = render_selected_facts(facts)

    assert "任务：第一轮水分子优化" in text
    assert "任务：第二轮水分子优化" in text
    assert "创建于 2026-09-15T10:00" in text
    assert "创建于 2026-09-15T11:00" in text
    assert "private-artifact-id" not in text
