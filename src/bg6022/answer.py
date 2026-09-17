"""Deterministic scientific facts rendered as bounded natural-language replies."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from bg6022.llm import LlmClient
from bg6022.models import Result, Run
from bg6022.output_contracts import public_type_info
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
    "atom_i": "第一个原子索引",
    "atom_j": "第二个原子索引",
}


class AnswerSection(BaseModel):
    """Small, ephemeral public-answer section; not a runtime domain object."""

    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["auto", "plain", "table", "json", "code", "link"] = "auto"
    heading: Literal["results", "files", "checks", "notes"] | None = None
    output_refs: list[StrictStr] = Field(default_factory=list, max_length=24)
    detail: Literal["brief", "normal", "full"] = "normal"
    # Free prose is retained only for knowledge/clarification answers.  The
    # result/query renderer rejects it before any verified fact is displayed.
    text: StrictStr | None = None


class AnswerOutput(BaseModel):
    """Bounded common protocol for knowledge, result, and query answers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["respond", "needs_tools", "clarify"]
    requested_results: list[StrictStr] = Field(default_factory=list)
    clarification: StrictStr | None = None
    sections: list[AnswerSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_action(self) -> AnswerOutput:
        if self.action == "needs_tools" and not self.requested_results:
            raise ValueError("needs_tools answer must name at least one requested result")
        if self.action != "needs_tools" and self.requested_results:
            raise ValueError("requested_results are only valid for needs_tools")
        if self.action == "clarify" and not (
            (self.clarification and self.clarification.strip()) or self.sections
        ):
            raise ValueError("clarify answer needs a clarification or section")
        if len(self.sections) > 8:
            raise ValueError("answer may contain at most eight sections")
        return self


def compose_answer(
    client: LlmClient,
    *,
    question: str,
    mode: Literal["knowledge", "result", "query"],
    capability_catalog: Sequence[Mapping[str, Any]] = (),
    available_outputs: Sequence[Mapping[str, Any]] = (),
    required_outputs: Sequence[str] = (),
    context: Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> AnswerOutput:
    """Ask one bounded model call to organize a public answer.

    The model receives names and descriptions only.  Verified values, paths,
    and file bytes remain program-owned and are rendered by the caller.
    """

    payload = {
        "question": question,
        "mode": mode,
        "capability_catalog": [dict(item) for item in capability_catalog],
        "available_outputs": [dict(item) for item in available_outputs],
        "required_outputs": list(required_outputs),
        "context": dict(context or {}),
    }
    return client.complete_json(
        [
            {"role": "system", "content": load_prompt("answer")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
        AnswerOutput,
        purpose="answer",
        example=(
            {
                "action": "respond",
                "requested_results": [],
                "clarification": None,
                "sections": [
                    {
                        "format": "plain",
                        "heading": "notes",
                        "output_refs": [],
                        "detail": "normal",
                        "text": "下面是与用户目标直接相关的说明。",
                    }
                ],
            }
            if mode == "knowledge"
            else {
                "action": "respond",
                "requested_results": [],
                "clarification": None,
                "sections": [
                    {
                        "format": "auto",
                        "heading": "results",
                        "output_refs": list(required_outputs)[:1],
                        "detail": "normal",
                        "text": None,
                    }
                ],
            }
        ),
        cancel=cancel,
    )


def validate_result_answer(
    output: AnswerOutput | None,
    outputs_by_ref: Mapping[str, Mapping[str, Any]],
    required_refs: Sequence[str],
    preferences: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate the reference-only result/query presentation contract."""

    if output is None or output.action != "respond" or output.clarification:
        raise ValueError("completed result presentation must use respond")
    used: list[str] = []
    for section in output.sections:
        if section.text is not None:
            raise ValueError("free model prose is not allowed in result/query mode")
        if not section.output_refs:
            raise ValueError("a result section must reference verified outputs")
        for ref in section.output_refs:
            if ref not in outputs_by_ref:
                raise ValueError(f"unknown output reference: {ref}")
            _check_supported_view(outputs_by_ref[ref], section.format)
            if (
                section.format == "link"
                and str(_mapping(preferences).get("file_content") or "auto") != "link_only"
            ):
                raise ValueError("link view requires an explicit link-only file request")
            used.append(ref)
    if len(used) != len(set(used)):
        raise ValueError("duplicate output rendering")
    required = {str(ref) for ref in required_refs}
    if not required <= set(used):
        raise ValueError("answer omits a requested output")
    return tuple(used)


def render_answer_output(
    output: AnswerOutput | None,
    *,
    outputs_by_ref: Mapping[str, Mapping[str, Any]] | None = None,
    required_refs: Sequence[str] = (),
    preferences: Mapping[str, Any] | None = None,
) -> str:
    """Render knowledge prose or reference-only verified result outputs.

    In result/query mode all visible facts are read from ``outputs_by_ref``;
    the model chooses only section order and supported view, never the text.
    """

    if output is None:
        return ""
    if outputs_by_ref is not None:
        used = validate_result_answer(
            output,
            outputs_by_ref,
            required_refs,
            preferences=preferences,
        )
        section_by_ref: dict[str, AnswerSection] = {
            ref: section for section in output.sections for ref in section.output_refs
        }
        lines: list[str] = []
        heading_labels = {
            "results": "结果",
            "files": "文件",
            "checks": "科学检查",
            "notes": "说明",
        }
        emitted_headings: set[str] = set()
        for ref in used:
            section = section_by_ref[ref]
            if section.heading and section.heading not in emitted_headings:
                lines.append(f"{heading_labels[section.heading]}：")
                emitted_headings.add(section.heading)
            entry = outputs_by_ref[ref]
            rendered = _render_public_output(
                entry,
                section.format,
                detail=section.detail,
                preferences=preferences,
            )
            if rendered:
                lines.append(rendered)
        return "\n".join(lines)
    sections = [
        section.text.strip()
        for section in output.sections
        if section.text is not None and section.text.strip()
    ]
    if output.clarification and output.clarification.strip():
        sections.insert(0, output.clarification.strip())
    return "\n\n".join(sections)


def _check_supported_view(entry: Mapping[str, Any], requested: str) -> None:
    if requested == "auto":
        return
    fact = _mapping(entry.get("fact"))
    kind = str(entry.get("kind") or fact.get("kind") or "")
    expected_type = str(
        entry.get("type") or entry.get("expected_type") or fact.get("expected_type") or ""
    )
    declared = entry.get("supported_views")
    if not isinstance(declared, list):
        declared = public_type_info(expected_type, kind=kind).get("supported_views", [])
    if requested not in declared:
        raise ValueError(
            f"view {requested!r} is unsupported for {kind or expected_type!r}; "
            f"supported views: {', '.join(str(item) for item in declared)}"
        )


def _render_public_output(
    entry: Mapping[str, Any],
    view: str,
    *,
    detail: str,
    preferences: Mapping[str, Any] | None,
) -> str:
    fact = _mapping(entry.get("fact"))
    file_info = _mapping(entry.get("file"))
    if not fact:
        fact = entry
    kind = str(fact.get("kind") or entry.get("kind") or "field")
    expected_type = fact.get("expected_type") or entry.get("type")
    if view == "auto":
        layout = str(_mapping(preferences).get("layout") or "auto")
        supported = public_type_info(str(expected_type or ""), kind=kind).get("supported_views", [])
        if layout in {"plain", "table"} and layout in supported:
            view = layout
        elif expected_type in {"record", "record_list"} and "table" in supported:
            view = "table"
        elif expected_type == "json" and "json" in supported:
            view = "json"
    label = str(_mapping(fact.get("metadata")).get("label") or fact.get("name") or "结果")
    if kind == "port":
        body = _render_file_output(
            label,
            file_info,
            fact=fact,
            view=view,
            preferences=preferences,
            detail=detail,
        )
    elif kind == "check":
        body = _fact_sentence(fact)
    else:
        value = fact.get("value")
        if view == "json" or (view == "code" and isinstance(value, (dict, list))):
            body = f"{label}：\n```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```"
        elif view == "table":
            body = (
                _render_record_value(label, value)
                if expected_type in {"record_list", "record"}
                else _render_scalar_table(label, value)
            )
        elif expected_type in {"record_list", "record"}:
            body = _render_record_value(label, value)
        else:
            body = _fact_sentence(fact)
    context = _fact_context(fact, include_task_identity=bool(fact.get("_include_task_identity")))
    lines = [f"{context}：", body] if context else [body]
    caveat = _string_or_none(_mapping(fact.get("metadata")).get("caveat"))
    if caveat:
        lines.append(f"说明：{caveat}。")
    return "\n".join(lines)


def _render_file_output(
    label: str,
    file_info: Mapping[str, Any],
    *,
    fact: Mapping[str, Any],
    view: str,
    preferences: Mapping[str, Any] | None,
    detail: str,
) -> str:
    if not file_info:
        return f"{label}：文件入口暂不可用。"
    display_name = str(file_info.get("display_name") or file_info.get("filename") or label)
    path = str(file_info.get("path") or "")
    lines = [f"{label}（{display_name}）"]
    file_content = str(_mapping(preferences).get("file_content") or "auto")
    show_content = view != "link" and file_content != "link_only"
    preview = file_info.get("preview_text")
    if show_content and isinstance(preview, str) and preview:
        fence = _code_fence(preview)
        lines.extend([f"{fence[0]}", preview.rstrip("\r\n"), fence[1]])
        if file_info.get("preview_complete") is False:
            lines.append("（以上为预览前缀，完整文件见下方入口。）")
    elif show_content and file_info.get("preview_reason"):
        lines.append(f"正文暂不可预览：{file_info['preview_reason']}。")
    role = _string_or_none(file_info.get("role"))
    if role:
        role_label = {
            "initial_geometry": "初始结构",
            "optimized_geometry": "优化后的结构",
            "verified_hessian": "已验证 Hessian",
        }.get(role, role.replace("_", " "))
        lines.append(f"来源：{role_label}。")
    source = _string_or_none(file_info.get("source"))
    if source and detail == "full":
        lines.append(f"来源记录：{source}")
    if path:
        lines.append(f"文件：{path}")
    if detail == "full" and file_info.get("sha256"):
        lines.append(f"SHA-256：{file_info['sha256']}")
    return "\n".join(lines)


def _render_record_value(label: str, value: Any) -> str:
    records = value if isinstance(value, list) else [value]
    if not records or not all(isinstance(item, Mapping) for item in records):
        return f"{label}：{_format_scalar(value)}"
    keys: list[str] = []
    for record in records:
        for key in record:
            if str(key) not in keys:
                keys.append(str(key))
    header = " | ".join(keys)
    divider = " | ".join("---" for _ in keys)
    rows = [" | ".join(_format_scalar(record.get(key)) for key in keys) for record in records]
    return f"{label}：\n| {header} |\n| {divider} |\n" + "\n".join(f"| {row} |" for row in rows)


def _render_scalar_table(label: str, value: Any) -> str:
    """Render a scalar or scalar-shaped value as a stable two-column table."""

    if isinstance(value, Mapping):
        rows = [(str(key), _format_scalar(item)) for key, item in value.items()]
    else:
        rows = [("值", _format_scalar(value))]
    body = "\n".join(f"| {key} | {item} |" for key, item in rows)
    return f"{label}：\n| 字段 | 值 |\n| --- | --- |\n{body}"


def _code_fence(text: str) -> tuple[str, str]:
    longest = 0
    for line in text.splitlines():
        match = re.search(r"`+", line)
        if match:
            longest = max(longest, len(match.group(0)))
    fence = "`" * max(3, longest + 1)
    return fence, fence


def render_confirmation(preview: Mapping[str, Any]) -> str:
    """Render the exact pending preview without exposing its internal JSON."""

    request = _mapping(preview.get("request"))
    structure = _mapping(preview.get("structure"))
    parameters = _mapping(preview.get("parameters"))
    resources = _mapping(preview.get("resources"))
    operation = str(preview.get("operation") or "calculation")
    system = _confirmation_system_label(structure, request.get("description"))
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
    if atom_count is not None and not _has_complete_confirmation_identity(structure):
        system = f"{system}（{_format_scalar(atom_count)} 个原子）"
    heading_separator = "  " if _has_complete_confirmation_identity(structure) else ""
    plan_steps = preview.get("plan_steps")
    if isinstance(plan_steps, Sequence) and not isinstance(plan_steps, (str, bytes)) and plan_steps:
        lines = [f"准备对{system}{heading_separator}执行以下完整计算计划："]
        lines.extend(_confirmation_step_line(step) for step in plan_steps)
    else:
        lines = [f"准备对{system}{heading_separator}{task_phrase}，采用 {method}。"]
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
        reason = data.get("reason")
        prefix = ""
        if reason == "ambiguous_subject":
            prefix = "当前有多个可能的结果。"
        elif reason == "ambiguous_property":
            prefix = "当前结果中有多个性质可能匹配。"
        elif reason == "invalid_binding":
            prefix = "所选任务尚未得到所问性质；已保存的其他性质不能替代它。"
        if isinstance(question, str) and question.strip():
            separator = "" if prefix.endswith(("。", "；")) else ""
            return f"{prefix}{separator}{question.strip()}"
        if prefix:
            return f"{prefix}请说明要查询哪个任务或哪一种性质。"
        return "请说明要查询哪个任务或哪一种性质。"
    if status == "unavailable":
        description = data.get("missing_description") or data.get("question")
        suffix = f"（{description}）" if isinstance(description, str) and description else ""
        return f"当前可查询范围内没有找到所问结果{suffix}。这不表示该计算从未进行过。"

    category = data.get("category")
    if category in {
        "ambiguous_molecule",
        "molecule_search_incomplete",
        "molecule_source_unverified",
    }:
        candidates = data.get("candidates")
        details: list[str] = []
        if isinstance(candidates, list):
            for index, candidate in enumerate(candidates[:5], start=1):
                item = _mapping(candidate)
                choice = item.get("choice_id") or f"candidate_{index}"
                title = item.get("Title") or item.get("title") or "未命名结构"
                cid = item.get("CID") or item.get("cid")
                formula = item.get("MolecularFormula") or item.get("formula")
                smiles = item.get("IsomericSMILES") or item.get("isomeric_smiles")
                parts = [f"{choice}: {title}"]
                if cid:
                    parts.append(f"CID {cid}")
                if formula:
                    parts.append(f"分子式 {formula}")
                if smiles:
                    parts.append(f"SMILES {smiles}")
                details.append("；".join(parts))
        suffix = f"候选包括：{'、'.join(details)}。" if details else ""
        if category == "ambiguous_molecule":
            prefix = "已核验出多个不同结构，请回复候选编号、CID 或明确的 SMILES。"
        elif category == "molecule_search_incomplete":
            prefix = "当前检索尚未完整，不能确认唯一结构；请回复候选编号、CID 或明确的 SMILES。"
        else:
            prefix = "部分来源记录无法可靠核验；请明确提供 CID 或 SMILES 后继续。"
        return f"{prefix}{suffix}".strip()
    if category == "molecule_name_not_found":
        raw_query = data.get("raw_query") or data.get("lookup_query") or "该名称"
        return f"来源未识别名称“{raw_query}”，原计算任务已保留。请补充英文名称、CID 或明确 SMILES。"
    if category == "molecule_identity_not_found":
        if data.get("input_kind") == "formula" or data.get("requested_formula"):
            return "没有找到符合当前分子式约束的结构；请提供明确的 CID 或 SMILES。"
        return "没有找到可验证的结构；请补充明确的名称、CID 或 SMILES。"
    if category == "identity_mismatch":
        return "候选结构与用户给出的分子身份约束不一致；请提供明确的 CID 或 SMILES。"

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

    facts = facts_from_result(run, result, registry, structure=structure)
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
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for fact in values:
        task_key = str(fact.get("task_key") or _system_label(fact, None))
        groups.setdefault(task_key, []).append(fact)
        if fact.get("run_status") not in {None, "succeeded"}:
            notice = "该任务尚未全部完成；以下仅是其中已验证步骤的结果。"
            if notice not in lines:
                lines.append(notice)

    for group in groups.values():
        context = _fact_context(group[0], include_task_identity=len(groups) > 1)
        if context:
            lines.append(f"{context}：")
        for fact in group:
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
    partial_facts: Sequence[Mapping[str, Any]] = (),
    repairs: Sequence[Mapping[str, Any]] = (),
    incomplete_targets: Sequence[str] = (),
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
        diagnostics = result.diagnostics if result is not None else {}
        budget_reason = _budget_stop_reason(run.pending_data)
        reason = (
            (run.pending_data.get("reason") if run.pending_data else None)
            or budget_reason
            or (run.pending_data.get("repair_rejected") if run.pending_data else None)
            or diagnostics.get("reason")
        )
        category = (
            (run.pending_data.get("category") if run.pending_data else None)
            or ("budget_exhausted" if budget_reason else None)
            or diagnostics.get("category")
        )
        label = f"（{category}）" if category else ""
        lines = [f"任务未全部完成{label}。"]
        if incomplete_targets:
            lines.append(f"尚未完成：{'、'.join(incomplete_targets)}。")
        if repairs:
            repair_text = [
                text for item in repairs if (text := _repair_record_sentence(item)) is not None
            ]
            if repair_text:
                lines.append(f"已尝试的修复：{'；'.join(repair_text)}。")
        if reason:
            lines.append(f"停止原因：{reason}。")
        facts = list(partial_facts)
        if facts:
            lines.append("以下是仍然有效的已完成结果；它们不代表整个任务成功：")
            lines.append(render_selected_facts(facts))
        return "\n".join(lines)
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
    partial_facts: Sequence[Mapping[str, Any]] = (),
    repairs: Sequence[Mapping[str, Any]] = (),
    incomplete_targets: Sequence[str] = (),
) -> str:
    """Explain that confirmation is idempotent and show the existing result."""

    if result is None:
        if run.status == "failed":
            return "该计算已经结束，无需再次确认。\n" + render_run(
                run,
                result,
                registry,
                structure=structure,
                partial_facts=partial_facts,
                repairs=repairs,
                incomplete_targets=incomplete_targets,
            )
        return "该计算已经结束，无需再次确认。当前没有可呈现的结果。"
    if result.status == "succeeded" and run.status == "succeeded":
        return "该计算已经完成，无需再次确认。\n" + render_result(
            run, result, registry, structure=structure
        )
    return "该计算已经结束，无需再次确认。\n" + render_run(
        run,
        result,
        registry,
        structure=structure,
        partial_facts=partial_facts,
        repairs=repairs,
        incomplete_targets=incomplete_targets,
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
    targets: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render facts selected by structured property targets, never by prose labels."""

    if facts is not None:
        return render_selected_facts(facts)
    if run is None or result is None:
        return "当前可查询范围内没有找到所问结果。这不表示该计算从未进行过。"
    if not targets:
        return "请先明确所需的科学性质；我不会从结果展示文案猜测查询目标。"
    candidates = facts_from_result(run, result, registry)
    selected, covered = select_facts_for_question(targets, candidates)
    if not covered:
        return "本次任务尚未得到所问性质；已保存的其他性质不能替代它。"
    return render_selected_facts(selected)


def fact_matches_question(target: Mapping[str, Any], fact: Mapping[str, Any]) -> bool:
    """Match one structured (subject, property) target to one verified fact."""

    return target.get("subject_ref") == fact.get("subject_ref") and target.get(
        "property"
    ) == fact.get("result_property")


def select_facts_for_question(
    targets: Sequence[Mapping[str, Any]], facts: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], bool]:
    """Match each requested property only within its explicitly selected task."""

    if not targets:
        return [], False
    selected: list[Mapping[str, Any]] = []
    covered: set[tuple[Any, Any]] = set()
    requested: set[tuple[Any, Any]] = set()
    for target in targets:
        identity = (target.get("subject_ref"), target.get("property"))
        if not all(isinstance(value, str) and value for value in identity):
            return [], False
        requested.add(identity)
    for fact in facts:
        identity = (fact.get("subject_ref"), fact.get("result_property"))
        if identity in requested:
            selected.append(fact)
            covered.add(identity)
    return selected, requested <= covered


def facts_from_result(
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
                result_property=(
                    tool.result_properties.get(name, name) if tool is not None else name
                ),
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
                result_property=(
                    tool.result_properties.get(name, name) if tool is not None else name
                ),
                metadata=metadata,
                structure=structure,
            )
        )
    for name, check in result.scientific_checks.items():
        values.append(
            _fact(
                run=run,
                result=result,
                step=step,
                name=name,
                kind="check",
                value={"status": check.status, "reason": check.reason},
                expected_type="scientific_check",
                result_property=(
                    tool.result_properties.get(name, name) if tool is not None else name
                ),
                metadata={
                    **(_metadata(tool, name) if tool is not None else {}),
                    "label": (
                        _metadata(tool, name).get("label", name.replace("_", " "))
                        if tool is not None
                        else name.replace("_", " ")
                    ),
                    "description": (
                        _metadata(tool, name).get(
                            "description", "程序根据已验证计算文件得出的科学目标检查"
                        )
                        if tool is not None
                        else "程序根据已验证计算文件得出的科学目标检查"
                    ),
                },
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
    result_property: str | None,
    metadata: Mapping[str, str],
    structure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = _mapping(step.parameters) if step is not None else {}
    task_key = f"{run.id}:{result.step_id}"
    return {
        "task_key": task_key,
        "subject_ref": task_key,
        "task_description": run.request.description,
        "task_created_at": run.created_at,
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
        "result_property": result_property,
        "metadata": dict(metadata),
    }


def _metadata(tool: Any, name: str) -> dict[str, str]:
    label = name.replace("_", " ")
    description = tool.description if tool is not None else "已验证的计算结果"
    metadata = dict(tool.result_metadata.get(name, {})) if tool is not None else {}
    metadata.setdefault("label", label)
    metadata.setdefault("description", description)
    return metadata


def _fact_context(fact: Mapping[str, Any], *, include_task_identity: bool = False) -> str:
    system = _string_or_none(fact.get("system"))
    tool = fact.get("step_tool")
    operation = (
        "优化"
        if tool == "optimize_geometry"
        else "单点"
        if tool == "single_point"
        else "距离测量"
        if tool == "geometry_distance"
        else "频率计算"
        if tool == "frequency"
        else "初始结构生成"
        if tool == "generate_geometry"
        else "分子解析"
        if tool == "resolve_molecule"
        else ""
    )
    method = _method_label(fact.get("method_profile"))
    environment = _environment_label(fact.get("environment"))
    details = [
        value
        for value in (
            method if method != "未指定方法" else None,
            environment if environment != "未指定环境" else None,
        )
        if value
    ]
    if include_task_identity:
        description = _string_or_none(fact.get("task_description"))
        if description:
            compact = " ".join(description.split())
            if len(compact) > 100:
                compact = compact[:97].rstrip() + "..."
            details.append(f"任务：{compact}")
        created_at = _string_or_none(fact.get("task_created_at"))
        if created_at:
            details.append(f"创建于 {created_at[:16]}")
    qualifier = f"（{'；'.join(details)}）" if details else ""
    if system and operation:
        return f"{system}{operation}结果{qualifier}"
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
                name = item.get("check") or item.get("field") or item.get("port")
                if name:
                    label = str(name).replace("_", " ")
                    if label not in labels:
                        labels.append(label)
    return f"结果目标：{'、'.join(labels)}。" if labels else ""


def _confirmation_step_line(value: Any) -> str:
    item = _mapping(value)
    index = item.get("index")
    ordinal = _format_scalar(index) if type(index) is int else "?"
    tool = str(item.get("tool") or "未知工具")
    tool_labels = {
        "resolve_molecule": "解析分子",
        "generate_geometry": "生成初始结构",
        "optimize_geometry": "几何优化",
        "frequency": "频率计算",
        "single_point": "独立单点计算",
        "geometry_distance": "原子间距离测量",
    }
    operation_names = {
        "SP": "单点计算",
        "Opt": "几何优化",
        "Freq": "频率计算",
    }
    operations = item.get("operations")
    operation_labels = (
        [
            operation_names.get(str(operation), str(operation))
            for operation in operations
            if isinstance(operations, list)
        ]
        if isinstance(operations, list)
        else []
    )
    label = tool_labels.get(tool, tool)
    if operation_labels and not all(operation == label for operation in operation_labels):
        label = f"{label}（{'、'.join(operation_labels)}）"

    parameters = _mapping(item.get("parameters"))
    details: list[str] = []
    if parameters.get("query"):
        details.append(f"分子查询“{parameters['query']}”")
    method_profile = parameters.get("method_profile")
    if method_profile is not None:
        details.append(f"方法 {_method_label(method_profile)}")
    environment = parameters.get("environment")
    if environment is not None:
        details.append(_environment_label(environment))
    if parameters.get("charge") is not None:
        details.append(f"电荷 {_format_scalar(parameters['charge'])}")
    if parameters.get("multiplicity") is not None:
        details.append(f"多重度 {_format_scalar(parameters['multiplicity'])}")
    if parameters.get("atom_i") is not None and parameters.get("atom_j") is not None:
        details.append(
            f"第 {_format_scalar(parameters['atom_i'])}、"
            f"{_format_scalar(parameters['atom_j'])} 号原子"
            "（XYZ 从 1 编号）"
        )
    for name, label_text in (("geom_maxiter", "几何迭代上限"), ("scf_maxiter", "SCF 迭代上限")):
        if parameters.get(name) is not None:
            details.append(f"{label_text} {_format_scalar(parameters[name])}")

    inputs = item.get("inputs")
    if isinstance(inputs, list):
        for raw_input in inputs:
            bound = _mapping(raw_input)
            name = str(bound.get("name") or "输入")
            if bound.get("source_step"):
                details.append(f"{name}来自{bound['source_step']}.{bound.get('port') or '输出'}")
            elif bound.get("history_geometry") is True:
                details.append(f"{name}使用已验证的历史优化结构")
            elif bound.get("artifact_role"):
                details.append(f"{name}使用{_artifact_role_label(str(bound['artifact_role']))}")

    checks = item.get("goal_checks")
    if isinstance(checks, list):
        for raw_check in checks:
            check = _mapping(raw_check)
            check_name = _check_label(str(check.get("check") or "科学检查"))
            status = str(check.get("required_status") or "passed")
            details.append(
                f"须先满足{check.get('source_step', '前置步骤')}的{check_name}（{status}）"
            )

    targets = item.get("requested_results")
    if isinstance(targets, list) and targets:
        names = [
            str(_mapping(target).get("label") or _mapping(target).get("name") or "结果")
            for target in targets
        ]
        details.append(f"目标：{'、'.join(names)}")
    suffix = f"；{'；'.join(details)}" if details else ""
    return f"{ordinal}. {label}{suffix}。"


def _artifact_role_label(value: str) -> str:
    return {
        "initial_geometry": "初始结构",
        "input_geometry": "提供的结构",
        "optimized_geometry": "已优化结构",
        "restart_candidate": "受限修复候选结构",
    }.get(value, value.replace("_", " "))


def _check_label(value: str) -> str:
    return {
        "frequency_complete": "频率完整性检查",
        "local_minimum_supported": "局部极小值检查",
    }.get(value, value.replace("_", " "))


def _fact_sentence(fact: Mapping[str, Any]) -> str:
    metadata = _mapping(fact.get("metadata"))
    label = str(metadata.get("label") or fact.get("name") or "结果")
    if fact.get("kind") == "port":
        return f"{label}已生成并通过校验，可作为后续计算的结构输入。"
    if fact.get("kind") == "check":
        value = _mapping(fact.get("value"))
        status = value.get("status")
        status_label = {"passed": "通过", "not_met": "未满足", "unverified": "未能验证"}.get(
            str(status), "未知"
        )
        reason = _string_or_none(value.get("reason"))
        suffix = f"；{reason}" if reason else ""
        return f"{label}{status_label}{suffix}。"
    if fact.get("expected_type") == "angstrom" and isinstance(fact.get("value"), Mapping):
        distance = _mapping(fact.get("value"))
        raw = distance.get("value")
        if type(raw) in {int, float} and math.isfinite(float(raw)):
            indices = distance.get("atom_indices")
            symbols = distance.get("atom_symbols")
            if (
                isinstance(indices, list)
                and len(indices) == 2
                and isinstance(symbols, list)
                and len(symbols) == 2
            ):
                pair = (
                    f"第 {_format_scalar(indices[0])} 号 {symbols[0]} 与第 "
                    f"{_format_scalar(indices[1])} 号 {symbols[1]} 原子"
                )
                return f"{label}为 **{_format_scalar(raw)} Å**（{pair}；来源：已验证几何）。"
    value, unit = _display_value(fact.get("value"), fact.get("expected_type"))
    return f"{label}为 **{value}{unit}**。"


def _repair_record_sentence(record: Mapping[str, Any]) -> str | None:
    action_labels = {
        "restart_optimization": "从已校验的候选结构重启几何优化",
        "increase_scf_maxiter": "提高 SCF 迭代上限",
    }
    action = record.get("action")
    label = action_labels.get(str(action))
    if label is None:
        return None
    patch = _mapping(record.get("parameter_patch"))
    changes = [
        f"{_PARAMETER_LABELS.get(str(name), str(name))}调整为 {_format_scalar(value)}"
        for name, value in patch.items()
    ]
    attempt = record.get("failed_attempt")
    suffix = f"（第 {_format_scalar(attempt)} 次失败后）" if attempt is not None else ""
    detail = f"：{'、'.join(changes)}" if changes else ""
    return f"{label}{detail}{suffix}"


def _budget_stop_reason(pending_data: Mapping[str, Any]) -> str | None:
    labels = {
        "max_attempts_per_science_step": "该计算步骤已达到最大尝试次数",
        "max_extra_orca_executions": "已达到额外 ORCA 执行预算",
        "max_plan_revisions": "已达到最大计划修订次数",
    }
    return labels.get(str(pending_data.get("budget_exhausted")))


def _display_value(value: Any, expected_type: str | None) -> tuple[str, str]:
    if expected_type == "frequency" and isinstance(value, Mapping):
        modes = value.get("modes")
        if not isinstance(modes, list):
            return "不可用", ""
        rendered: list[str] = []
        for mode in modes:
            item = _mapping(mode)
            index, number = item.get("index"), item.get("value")
            if type(index) is not int or type(number) not in {int, float}:
                continue
            if not math.isfinite(float(number)):
                continue
            rendered.append(f"{index}: {_format_scalar(number)}")
        text = ", ".join(rendered)
        if value.get("scaling_factor") is not None:
            factor = _format_scalar(value.get("scaling_factor"))
            applied = "已应用" if value.get("scaling_applied") else "未应用"
            text += f"；缩放因子 {factor}（{applied}）"
        return text or "不可用", " cm⁻¹"
    if isinstance(value, Mapping) and "value" in value:
        token = value.get("token")
        raw = token if isinstance(token, str) and token.strip() else value.get("value")
        unit = value.get("unit") or _display_unit(expected_type)
        if unit == "angstrom":
            unit = "Å"
        return _format_scalar(raw), f" {unit}" if unit else ""
    unit = f" {_display_unit(expected_type)}" if _display_unit(expected_type) else ""
    return _format_scalar(value), unit


def _display_unit(expected_type: str | None) -> str:
    if expected_type in {None, "integer", "text", "boolean", "molecular_geometry"}:
        return ""
    return "Å" if expected_type == "angstrom" else str(expected_type)


def _render_failed_result(result: Result) -> str:
    diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    category = diagnostics.get("category") or "unknown_failure"
    reason = diagnostics.get("reason") or "没有得到已验证的科学结果"
    return f"本次计算未成功完成（{category}）：{reason}。"


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


def _confirmation_system_label(structure: Mapping[str, Any], description: Any) -> str:
    """Show the verified molecule identity in a confirmation heading when available."""

    title = _string_or_none(structure.get("title"))
    formula = _string_or_none(structure.get("formula"))
    smiles = _string_or_none(structure.get("isomeric_smiles")) or _string_or_none(
        structure.get("canonical_smiles")
    )
    if not any((title, formula, smiles)):
        return _system_label(structure, description)

    if title is None and formula == "H2O":
        name = "水分子（H₂O）"
        formula_text = None
    else:
        name = title or "名称未提供"
        formula_text = _pretty_formula(formula) if formula else "分子式未提供"
    smiles_text = smiles or "未提供"
    parts = [name]
    if formula_text:
        parts.append(formula_text)
    parts.append(f"(SMILES:{smiles_text})")
    return "  ".join(parts)


def _has_complete_confirmation_identity(structure: Mapping[str, Any]) -> bool:
    """Return whether the heading can carry title, formula, and SMILES."""

    title = _string_or_none(structure.get("title"))
    formula = _string_or_none(structure.get("formula"))
    smiles = _string_or_none(structure.get("isomeric_smiles")) or _string_or_none(
        structure.get("canonical_smiles")
    )
    return bool(title and formula and smiles)


def _pretty_formula(value: str) -> str:
    return value.translate(_SUBSCRIPT_DIGITS)


def _operation_label(
    operation: str, parameters: Mapping[str, Any], structure: Mapping[str, Any]
) -> str:
    if operation == "Opt":
        return "几何优化"
    if operation == "SP":
        return "单点计算"
    if operation == "Freq":
        return "频率计算"
    return "计算任务"


def _method_label(value: Any) -> str:
    if not value:
        return "未指定方法"
    text = str(value)
    try:
        from bg6022.orca.profiles import get_profile

        profile = get_profile(text)
    except ValueError:
        return text
    return profile.display_name or profile.name


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
    "AnswerOutput",
    "AnswerSection",
    "compose_answer",
    "context_answer",
    "explain_result",
    "fact_matches_question",
    "render_already_finished",
    "render_answer_output",
    "validate_result_answer",
    "render_clarification",
    "render_confirmation",
    "facts_from_result",
    "render_result",
    "render_run",
    "render_selected_facts",
    "select_facts_for_question",
]
