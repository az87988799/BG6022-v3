"""Private, read-only Tool for literal evidence from registered ORCA stdout."""

from __future__ import annotations

import unicodedata
from threading import Event

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from bg6022.models import Tool
from bg6022.session import (
    ArtifactReadError,
    check_read_deadline,
    read_registered_artifact_bytes,
)

MAX_QUERIES = 3
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_SECONDS = 5.0
MAX_SNIPPETS = 3
MAX_LINES = 120
MAX_TEXT_BYTES = 8192
MAX_LINE_BYTES = 4096


class OutputQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence: StrictStr = Field(min_length=1, max_length=240)
    search_terms: list[StrictStr] = Field(min_length=1, max_length=4)

    @field_validator("evidence")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or any(unicodedata.category(c).startswith("C") for c in value):
            raise ValueError("query evidence must be nonempty plain text")
        return value

    @field_validator("search_terms")
    @classmethod
    def literal_terms(cls, values: list[str]) -> list[str]:
        for term in values:
            if not 1 <= len(term) <= 80 or not term.strip():
                raise ValueError("search terms must contain 1–80 nonblank characters")
            if any(unicodedata.category(c).startswith("C") for c in term):
                raise ValueError("control characters are not search terms")
        return list(dict.fromkeys(values))


class InspectOrcaOutputParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    queries: list[OutputQuerySpec] = Field(min_length=1, max_length=MAX_QUERIES)


def _lines(payload: bytes, cancel: Event, deadline: float):
    offset = 0
    number = 0
    while offset < len(payload):
        check_read_deadline(cancel, deadline)
        end = payload.find(b"\n", offset)
        if end < 0:
            end = len(payload)
        number += 1
        raw = payload[offset : min(end, offset + MAX_LINE_BYTES)].rstrip(b"\r")
        yield number, raw.decode("utf-8", errors="replace"), end - offset > MAX_LINE_BYTES
        offset = end + 1


def visible_text(value: str) -> str:
    return "".join(
        f"\\u{ord(c):04x}" if unicodedata.category(c).startswith("C") and c != "\n" else c
        for c in value
    )


def search_output_bytes(payload, queries, *, deadline, cancel, limits):
    """Literal candidates, never a final-state parser; bounded storage on both passes."""
    terms = [[" ".join(t.casefold().split()) for t in q.search_terms] for q in queries]
    term_hits = [[[] for _ in words] for words in terms]
    long_lines = False
    for number, line, clipped in _lines(payload, cancel, deadline):
        long_lines |= clipped
        normalized = " ".join(line.casefold().split())
        for index, words in enumerate(terms):
            for term_index, word in enumerate(words):
                hits = term_hits[index][term_index]
                if word in normalized and len(hits) < 5:
                    hits.append(number)
    candidates = [sorted({n for hits in groups for n in hits}) for groups in term_hits]
    windows = []
    for hits in candidates:
        merged = []
        for hit in hits:
            start, end = max(1, hit - 3), hit + 18
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        windows.append(merged)
    evidence = []
    for index, ranges in enumerate(windows):
        snippets = []
        truncated = long_lines or any(len(hits) >= 5 for hits in term_hits[index])
        for start, end in ranges:
            if limits["snippets"] <= 0 or limits["lines"] <= 0 or limits["bytes"] <= 0:
                truncated = True
                break
            text_lines = []
            actual_end = start
            for number, line, clipped in _lines(payload, cancel, deadline):
                if number < start:
                    continue
                if number > end:
                    break
                rendered = visible_text(line)
                available = limits["bytes"] - (1 if text_lines else 0)
                if limits["lines"] <= 0 or available <= 0:
                    truncated = True
                    break
                encoded = rendered.encode("utf-8")
                if len(encoded) > available:
                    rendered = encoded[:available].decode("utf-8", errors="ignore")
                    truncated = True
                truncated |= clipped
                limits["bytes"] -= len(rendered.encode("utf-8")) + (1 if text_lines else 0)
                limits["lines"] -= 1
                text_lines.append(rendered)
                actual_end = number
            if text_lines:
                snippets.append(
                    {"start_line": start, "end_line": actual_end, "text": "\n".join(text_lines)}
                )
                limits["snippets"] -= 1
        evidence.append(
            {
                "query_index": index,
                "lookup_status": "found"
                if snippets
                else ("unavailable" if ranges else "not_found"),
                "snippets": snippets,
                "truncated": truncated,
                "ambiguous": len(ranges) > 1,
            }
        )
    return evidence


def validate_output_evidence(evidence, query_count, limits):
    """Check the private Tool's return shape and actual use against trusted limits."""
    if not isinstance(evidence, list) or len(evidence) != query_count:
        raise ValueError("invalid query evidence")
    usage = {"snippets": 0, "lines": 0, "bytes": 0}
    for index, item in enumerate(evidence):
        if (
            not isinstance(item, dict)
            or item.get("query_index") != index
            or item.get("lookup_status") not in {"found", "not_found", "unavailable"}
            or type(item.get("truncated")) is not bool
            or type(item.get("ambiguous")) is not bool
            or not isinstance(item.get("snippets"), list)
        ):
            raise ValueError("invalid query evidence identity or shape")
        if bool(item["snippets"]) != (item["lookup_status"] == "found"):
            raise ValueError("query status does not match snippets")
        for snippet in item["snippets"]:
            if (
                not isinstance(snippet, dict)
                or not isinstance(snippet.get("text"), str)
                or type(snippet.get("start_line")) is not int
                or type(snippet.get("end_line")) is not int
                or not 1 <= snippet["start_line"] <= snippet["end_line"]
            ):
                raise ValueError("invalid snippet shape")
            lines = snippet["text"].count("\n") + 1
            if lines != snippet["end_line"] - snippet["start_line"] + 1:
                raise ValueError("snippet line range mismatch")
            usage["snippets"] += 1
            usage["lines"] += lines
            usage["bytes"] += len(snippet["text"].encode("utf-8"))
    if any(usage[key] > limits[key] for key in usage):
        raise ValueError("query Tool exceeded the remaining excerpt budget")
    return usage


def make_orca_output_tool(*, remaining_file_bytes, deadline, remaining_excerpt_limits):
    def execute(step, context):
        parameters = InspectOrcaOutputParameters.model_validate(step.parameters, strict=True)
        bytes_read = 0
        try:
            payload = read_registered_artifact_bytes(
                context.data_root,
                context.run,
                context.frozen_inputs["source"],
                max_bytes=remaining_file_bytes,
                deadline=deadline,
                cancel=context.cancel,
            )
            bytes_read = len(payload)
            evidence = search_output_bytes(
                payload,
                parameters.queries,
                deadline=deadline,
                cancel=context.cancel,
                limits=remaining_excerpt_limits,
            )
        except ArtifactReadError as error:
            return context.make_result(
                "cancelled" if error.category == "cancelled" else "failed",
                diagnostics={"category": error.category, "bytes_read": bytes_read},
            )
        return context.make_result(
            "succeeded",
            values={"evidence": evidence},
            diagnostics={"query_kind": "raw_output", "bytes_read": bytes_read},
        )

    return Tool(
        name="inspect_orca_output",
        description="Read bounded registered ORCA stdout excerpts.",
        operations=[],
        parameter_model=InspectOrcaOutputParameters.__name__,
        parameter_schema=InspectOrcaOutputParameters.model_json_schema(),
        parameter_type=InspectOrcaOutputParameters,
        input_ports={"source": "orca_output"},
        output_ports={},
        results={"evidence": "record_list"},
        result_properties={"evidence": "orca_output"},
        requires_compute_permission=False,
        execution_budget="none",
        execute_function=execute,
    )
