"""Deterministic scientific facts rendered as bounded natural-language replies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from bg6022.llm import LlmClient
from bg6022.models import Result, Run
from bg6022.planner import load_prompt

if TYPE_CHECKING:
    from bg6022.tools.registry import ToolRegistry


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_PARAMETER_LABELS = {
    "charge": "电荷",
    "multiplicity": "自旋多重度",
    "method_profile": "计算方法",
    "environment": "计算环境",
    "geom_maxiter": "几何优化迭代上限",
    "scf_maxiter": "SCF 迭代上限",
}


def render_confirmation(preview: Mapping[str, Any]) -> str:
    """Render the exact pending preview without exposing its internal JSON."""

    request = _mapping(preview.get("request"))
    structure = _mapping(preview.get("structure"))
    parameters = _mapping(preview.get("parameters"))
    resources = _mapping(preview.get("resources"))
    operation = str(preview.get("operation") or "calculation")
    system = _system_label(structure, request.get("description"))
    task = _operation_label(operation, parameters, structure)
    method = _method_label(parameters.get("method_profile"))
    environment = _environment_label(parameters.get("environment"))
    atom_count = structure.get("atom_count")

    if operation == "Opt":
        task_phrase = f"进行{environment}几何优化"
    elif operation == "SP":
        task_phrase = f"进行{environment}单点计算"
    else:
        task_phrase = f"进行{task}"
    if atom_count is not None:
        system = f"{system}（{_format_scalar(atom_count)} 个原子）"
    lines = [f"准备对{system}{task_phrase}，采用 {method}。"]
    target_line = _result_target_sentence(preview.get("result_targets"), request)
    if target_line:
        lines.append(target_line)

    charge = parameters.get("charge")
    multiplicity = parameters.get("multiplicity")
    if charge is not None and multiplicity is not None:
        lines.append(
            f"电荷为 {_format_scalar(charge)}，自旋多重度为 {_format_scalar(multiplicity)}。"
        )
    else:
        lines.append("电荷和自旋多重度尚未完整确定。")

    resource_line = _resource_sentence(resources)
    if resource_line:
        lines.append(resource_line)
    repair_line = _repair_sentence(preview)
    if repair_line:
        lines.append(repair_line)
    budget_line = _budget_sentence(preview.get("budget"))
    if budget_line:
        lines.append(budget_line)
    timeout_line = _timeout_sentence(resources)
    if timeout_line:
        lines.append(timeout_line)
    lines.append("输入 /confirm 开始，也可以先告诉我需要调整什么。")
    return "\n".join(lines)


def render_clarification(pending_data: Mapping[str, Any] | None) -> str:
    """Render missing-input, ambiguous-selection, and unavailable states."""

    data = _mapping(pending_data)
    status = data.get("status")
    if status == "clarify":
        question = data.get("clarification") or data.get("question")
        if isinstance(question, str) and question.strip():
            return f"我找到了多个可能的结果。{question.strip()}"
        return "我找到了多个可能的结果，请说明要查询哪个任务或哪一种性质。"
    if status == "unavailable":
        description = data.get("missing_description") or data.get("question")
        suffix = f"（{description}）" if isinstance(description, str) and description else ""
        return f"当前可查询范围内没有找到所问结果{suffix}。这不表示该计算从未进行过。"

    category = data.get("category")
    if category == "ambiguous_molecule":
        candidates = data.get("candidates")
        details: list[str] = []
        if isinstance(candidates, list):
            for candidate in candidates[:5]:
                item = _mapping(candidate)
                title = item.get("Title") or item.get("title")
                cid = item.get("CID") or item.get("cid")
                if title and cid:
                    details.append(f"{title}（CID {cid}）")
                elif title or cid:
                    details.append(str(title or f"CID {cid}"))
        suffix = f"候选包括：{'、'.join(details)}。" if details else ""
        return f"分子结构存在歧义，请指定准确的分子或 CID。{suffix}".strip()

    missing = data.get("missing_fields")
    if isinstance(missing, list) and missing:
        labels = [_PARAMETER_LABELS.get(str(item), str(item)) for item in missing]
        return f"还需要补充：{'、'.join(labels)}。"
    question = data.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    if data.get("binding_invalid"):
        return "所选结果的文件或绑定已失效，无法安全读取；请明确要查询的任务。"
    return "还需要更多信息才能继续。"


def render_result(
    run: Run,
    result: Result,
    registry: ToolRegistry | None = None,
    *,
    structure: Mapping[str, Any] | None = None,
) -> str:
    """Render only values that passed the Tool's success contract."""

    if run.status == "waiting":
        if run.waiting_for == "confirmation":
            return "计算等待确认（waiting for confirmation）。\n" + render_confirmation(
                run.pending_data
            )
        return "计算正在等待补充信息。\n" + render_clarification(run.pending_data)
    if result.status != "succeeded":
        return _render_failed_result(result)

    facts = _facts_from_result(run, result, registry, structure=structure)
    if not facts:
        return "计算已完成，但没有可安全呈现的已验证结果。"
    return render_selected_facts(facts)


def render_selected_facts(facts: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> str:
    """Render facts selected from a validated query catalog.

    The function accepts the small internal fact dictionaries produced by the
    Agent.  It never serializes those dictionaries, so private Run IDs,
    relative paths, hashes, and parser diagnostics cannot leak into chat.
    """

    if isinstance(facts, Mapping):
        values = [facts]
    else:
        values = list(facts)
    if not values:
        return "当前可查询范围内没有找到所问结果。这不表示该计算从未进行过。"

    lines: list[str] = []
    task_keys: list[str] = []
    for fact in values:
        task_key = str(fact.get("task_key") or _system_label(fact, None))
        if task_key not in task_keys:
            task_keys.append(task_key)
        if fact.get("run_status") not in {None, "succeeded"}:
            notice = "该任务尚未全部完成；以下仅是其中已验证步骤的结果。"
            if notice not in lines:
                lines.append(notice)

    if len(task_keys) == 1:
        context = _fact_context(values[0])
        if context:
            lines.append(context)
    for fact in values:
        lines.append(_fact_sentence(fact))
        caveat = _string_or_none(_mapping(fact.get("metadata")).get("caveat"))
        if caveat:
            lines.append(f"说明：{caveat}。")
    return "\n".join(lines)


def render_run(
    run: Run,
    result: Result | None = None,
    registry: ToolRegistry | None = None,
    *,
    structure: Mapping[str, Any] | None = None,
) -> str:
    if run.status == "waiting":
        if run.waiting_for == "confirmation":
            return "计算等待确认（waiting for confirmation）。\n" + render_confirmation(
                run.pending_data
            )
        return "任务正在等待补充信息。\n" + render_clarification(run.pending_data)
    if run.status == "succeeded" and result is not None:
        return render_result(run, result, registry, structure=structure)
    if run.status == "failed":
        reason = run.pending_data.get("reason") if run.pending_data else None
        if not reason and result is not None and result.diagnostics:
            reason = result.diagnostics.get("reason")
        category = result.diagnostics.get("category") if result is not None else None
        category_text = f"，{category}" if category else ""
        suffix = f"原因：{reason}。" if reason else ""
        if result is not None and result.status == "succeeded":
            fact_text = render_result(run, result, registry, structure=structure)
            return (
                f"任务未全部完成（failed before all requested results）。{suffix}\n"
                f"最后一个已验证步骤的结果仍可参考，但不能宣告整个任务成功。\n{fact_text}"
            )
        return f"任务已失败（failed{category_text}）。{suffix}".rstrip()
    if run.status in {"cancelled", "interrupted"}:
        return f"任务已{_status_label(run.status)}（{run.status}）。未宣告整体科学成功。"
    if result is not None:
        return render_result(run, result, registry, structure=structure)
    return f"任务当前状态为 {_status_label(run.status)}。"


def render_already_finished(
    run: Run,
    result: Result | None,
    registry: ToolRegistry | None = None,
    *,
    structure: Mapping[str, Any] | None = None,
) -> str:
    """Explain that confirmation is idempotent and show the existing result."""

    if result is None:
        return "该计算已经结束，无需再次确认。当前没有可呈现的结果。"
    if result.status == "succeeded" and run.status == "succeeded":
        return "该计算已经完成，无需再次确认。\n" + render_result(
            run, result, registry, structure=structure
        )
    return "该计算已经结束，无需再次确认。\n" + render_run(
        run, result, registry, structure=structure
    )


def explain_result(
    client: LlmClient | None, *, run: Run, result: Result, cancel: Any = None
) -> str | None:
    """Optionally add prose; the deterministic result remains authoritative."""

    if client is None or result.status != "succeeded":
        return None
    try:
        return client.complete_text(
            [
                {"role": "system", "content": load_prompt("answer")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "deterministic_result": result.values,
                            "checks": result.checks,
                            "step": result.step_id,
                            "run_status": run.status,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            purpose="answer",
            cancel=cancel,
        )
    except Exception:
        return None


def context_answer(
    run: Run | None,
    result: Result | None,
    question: str,
    *,
    registry: ToolRegistry | None = None,
    facts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Compatibility helper for deterministic, property-scoped answers."""

    if facts is not None:
        return render_selected_facts(facts)
    if run is None or result is None:
        return "当前可查询范围内没有找到所问结果。这不表示该计算从未进行过。"
    candidates = _facts_from_result(run, result, registry)
    selected = [fact for fact in candidates if _question_matches_fact(question, fact)]
    if not selected:
        return "本次任务尚未得到所问性质；已保存的其他性质不能替代它。"
    return render_selected_facts(selected)


def fact_matches_question(question: str, fact: Mapping[str, Any]) -> bool:
    """Check a selected fact against broad property semantics, not a field name."""

    return _question_matches_fact(question, fact)


def _facts_from_result(
    run: Run,
    result: Result,
    registry: ToolRegistry | None,
    *,
    structure: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    step = next((item for item in run.plan.steps if item.id == result.step_id), None)
    tool = None
    if registry is not None and step is not None:
        try:
            tool = registry.get(step.tool)
        except ValueError:
            tool = None
    values: list[dict[str, Any]] = []
    declared = set(tool.results) if tool is not None else set(result.values)
    for name, value in result.values.items():
        if name not in declared or (tool is not None and name in tool.output_ports):
            continue
        metadata = _metadata(tool, name)
        values.append(
            _fact(
                run=run,
                result=result,
                step=step,
                name=name,
                kind="field",
                value=value,
                expected_type=(tool.results.get(name) if tool is not None else None),
                metadata=metadata,
                structure=structure,
            )
        )
    ports = set(tool.output_ports) if tool is not None else set(result.output_ports)
    for name, artifact_id in result.output_ports.items():
        if name not in ports:
            continue
        metadata = _metadata(tool, name)
        values.append(
            _fact(
                run=run,
                result=result,
                step=step,
                name=name,
                kind="port",
                value={"artifact_id": artifact_id},
                expected_type=(tool.output_ports.get(name) if tool is not None else None),
                metadata=metadata,
                structure=structure,
            )
        )
    return values


def _fact(
    *,
    run: Run,
    result: Result,
    step: Any,
    name: str,
    kind: str,
    value: Any,
    expected_type: str | None,
    metadata: Mapping[str, str],
    structure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = _mapping(step.parameters) if step is not None else {}
    return {
        "task_key": f"{run.id}:{result.step_id}",
        "system": _system_label(structure, run.request.description)
        if structure is not None
        else None,
        "run_status": run.status,
        "result_status": result.status,
        "step_id": result.step_id,
        "step_tool": step.tool if step is not None else None,
        "method_profile": params.get("method_profile"),
        "environment": params.get("environment"),
        "name": name,
        "kind": kind,
        "value": value,
        "expected_type": expected_type,
        "metadata": dict(metadata),
    }


def _metadata(tool: Any, name: str) -> dict[str, str]:
    label = name.replace("_", " ")
    description = tool.description if tool is not None else "已验证的计算结果"
    metadata = dict(tool.result_metadata.get(name, {})) if tool is not None else {}
    metadata.setdefault("label", label)
    metadata.setdefault("description", description)
    return metadata


def _fact_context(fact: Mapping[str, Any]) -> str:
    system = _string_or_none(fact.get("system"))
    tool = fact.get("step_tool")
    operation = "优化" if tool == "optimize_geometry" else "单点" if tool == "single_point" else ""
    method = _method_label(fact.get("method_profile"))
    if system and operation and method != "未指定方法":
        return f"{system}{operation}结果（{method}）"
    if system and operation:
        return f"{system}{operation}结果"
    if system:
        return f"{system}的已验证结果"
    return "已验证结果"


def _result_target_sentence(value: Any, request: Mapping[str, Any]) -> str:
    targets = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []
    labels: list[str] = []
    for target in targets:
        item = _mapping(target)
        label = _string_or_none(item.get("label"))
        if label is None:
            name = item.get("name") or item.get("field") or item.get("port")
            label = str(name).replace("_", " ") if name else None
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raw_targets = request.get("requested_results")
        if isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes)):
            for target in raw_targets:
                item = _mapping(target)
                name = item.get("field") or item.get("port")
                if name:
                    label = str(name).replace("_", " ")
                    if label not in labels:
                        labels.append(label)
    return f"结果目标：{'、'.join(labels)}。" if labels else ""


def _fact_sentence(fact: Mapping[str, Any]) -> str:
    metadata = _mapping(fact.get("metadata"))
    label = str(metadata.get("label") or fact.get("name") or "结果")
    if fact.get("kind") == "port":
        return f"{label}已生成并通过校验，可作为后续计算的结构输入。"
    value, unit = _display_value(fact.get("value"), fact.get("expected_type"))
    return f"{label}为 **{value}{unit}**。"


def _display_value(value: Any, expected_type: str | None) -> tuple[str, str]:
    if isinstance(value, Mapping) and "value" in value:
        token = value.get("token")
        raw = token if isinstance(token, str) and token.strip() else value.get("value")
        unit = value.get("unit") or _display_unit(expected_type)
        return _format_scalar(raw), f" {unit}" if unit else ""
    unit = f" {_display_unit(expected_type)}" if _display_unit(expected_type) else ""
    return _format_scalar(value), unit


def _display_unit(expected_type: str | None) -> str:
    if expected_type in {None, "integer", "text", "boolean", "molecular_geometry"}:
        return ""
    return str(expected_type)


def _render_failed_result(result: Result) -> str:
    diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    category = diagnostics.get("category") or "unknown_failure"
    reason = diagnostics.get("reason") or "没有得到已验证的科学结果"
    return f"本次计算未成功完成（{category}）：{reason}。"


def _question_matches_fact(question: str, fact: Mapping[str, Any]) -> bool:
    metadata = _mapping(fact.get("metadata"))
    haystack = " ".join(
        str(value).casefold()
        for value in (fact.get("name"), metadata.get("label"), metadata.get("description"))
        if value
    )
    question_terms = _property_terms(question)
    if not question_terms:
        return True
    fact_terms = _property_terms(haystack)
    if "zero_point" in question_terms:
        return "zero_point" in fact_terms and "不含零点" not in haystack
    if "free_energy" in question_terms:
        return "free_energy" in fact_terms
    if "frequency" in question_terms:
        return "frequency" in fact_terms
    return question_terms.issubset(fact_terms)


def _property_terms(text: str) -> set[str]:
    value = text.casefold()
    terms: set[str] = set()
    if any(token in value for token in ("能量", "电子能", "energy", "electronic")):
        terms.add("energy")
    if any(token in value for token in ("零点能", "零点", "zero-point", "zero point", "zpe")):
        terms.add("zero_point")
    if any(token in value for token in ("自由能", "free energy", "free_energy")):
        terms.add("free_energy")
    if any(token in value for token in ("频率", "振动频率", "frequency", "frequencies")):
        terms.add("frequency")
    if any(token in value for token in ("几何", "结构", "geometry", "structure")):
        terms.add("geometry")
    if any(token in value for token in ("原子数", "atom count", "atom_count", "number of atoms")):
        terms.add("atom_count")
    return terms


def _system_label(structure: Mapping[str, Any], description: Any) -> str:
    formula = _string_or_none(structure.get("formula"))
    title = _string_or_none(structure.get("title"))
    query = _string_or_none(structure.get("query"))
    if formula == "H2O":
        return "水分子（H₂O）"
    if title and title.casefold() not in {"o", "water"}:
        return title
    if formula:
        return _pretty_formula(formula)
    if query:
        return query
    if isinstance(description, str) and description.strip():
        return description.strip()
    return "该体系"


def _pretty_formula(value: str) -> str:
    return value.translate(_SUBSCRIPT_DIGITS)


def _operation_label(
    operation: str, parameters: Mapping[str, Any], structure: Mapping[str, Any]
) -> str:
    if operation == "Opt":
        return "几何优化"
    if operation == "SP":
        return "单点计算"
    return "计算任务"


def _method_label(value: Any) -> str:
    if not value:
        return "未指定方法"
    text = str(value)
    return {"r2scan3c": "r²SCAN-3c", "r2scan-3c": "r²SCAN-3c"}.get(text, text)


def _environment_label(value: Any) -> str:
    return {"gas": "气相", "vacuum": "气相"}.get(str(value), str(value or "未指定环境"))


def _resource_sentence(resources: Mapping[str, Any]) -> str:
    cores = resources.get("cores")
    memory = resources.get("memory_mb")
    maxcore = resources.get("maxcore_mb")
    if cores is None or memory is None or maxcore is None:
        return ""
    return (
        f"使用 {_format_scalar(cores)} 核，总内存上限 {_format_scalar(memory)} MB，"
        f"每核 MaxCore 为 {_format_scalar(maxcore)} MB。"
    )


def _repair_sentence(preview: Mapping[str, Any]) -> str:
    scope = _mapping(preview.get("repair_scope"))
    actions = {
        action
        for item in _mapping(scope.get("steps")).values()
        for action in _mapping(item).get("actions", {})
    }
    if "restart_optimization" in actions:
        maximum = _repair_maximum(scope, "restart_optimization", 1000)
        return (
            "若因优化迭代次数用尽而失败，可从经校验的中间结构继续优化，"
            f"并将迭代上限提高至最多 {maximum}。"
        )
    if "increase_scf_maxiter" in actions:
        maximum = _repair_maximum(scope, "increase_scf_maxiter", 1000)
        return (
            "若 SCF 因迭代次数用尽而失败，可在保留当前几何的前提下"
            f"将 SCF 迭代上限提高至最多 {maximum}。"
        )
    return "本次没有预先授权的自动修复范围。"


def _repair_maximum(scope: Mapping[str, Any], action: str, fallback: int) -> int:
    for item in _mapping(scope.get("steps")).values():
        details = _mapping(_mapping(item).get("actions")).get(action)
        maximum = _mapping(details).get("maximum")
        if isinstance(maximum, int):
            return maximum
    return fallback


def _budget_sentence(value: Any) -> str:
    budget = _mapping(value)
    attempts = budget.get("max_attempts_per_science_step")
    extra = budget.get("max_extra_orca_executions")
    if attempts is None and extra is None:
        return ""
    parts: list[str] = []
    if attempts is not None:
        parts.append(f"本步骤最多尝试 {_format_scalar(attempts)} 次（包含首次计算）")
    if extra is not None:
        parts.append(f"整个任务最多允许 {_format_scalar(extra)} 次额外计算")
    return "，".join(parts) + "，这些限制同时生效。"


def _timeout_sentence(resources: Mapping[str, Any]) -> str:
    attempt = resources.get("attempt_timeout_seconds")
    total = resources.get("run_active_timeout_seconds")
    if attempt is None and total is None:
        return ""
    parts: list[str] = []
    if attempt is not None:
        parts.append(f"单次计算最长 {_duration(attempt)}")
    if total is not None:
        parts.append(f"任务累计活动时间上限 {_duration(total)}")
    return "，".join(parts) + "。"


def _duration(seconds: Any) -> str:
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return f"{seconds} 秒"
    if value % 60 == 0:
        return f"{value // 60} 分钟"
    if value % 3600 == 0:
        return f"{value // 3600} 小时"
    return f"{value} 秒"


def _status_label(status: Any) -> str:
    return {
        "cancelled": "取消",
        "interrupted": "中断",
        "planned": "已计划",
        "running": "运行中",
        "waiting": "等待中",
    }.get(str(status), str(status))


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "不可用"
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "context_answer",
    "explain_result",
    "fact_matches_question",
    "render_already_finished",
    "render_clarification",
    "render_confirmation",
    "render_result",
    "render_run",
    "render_selected_facts",
]
