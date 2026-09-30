"""Private, read-only Tool for literal evidence from registered ORCA stdout."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
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
MAX_CANDIDATES = 24

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_SEPARATOR = re.compile(r"^\s*[-=*]{3,}\s*$")
_ORBIT_ROW = re.compile(rf"^\s*(\d+)\s+({_NUMBER})\s+({_NUMBER})(?:\s+({_NUMBER}))?\s*$")
FRONTIER_HINTS = {"lumo_energy", "homo_energy", "homo_lumo_gap", "frontier_orbitals"}


def _decimal(token):
    if not isinstance(token, str) or len(token) > 80:
        return None
    try:
        value = Decimal(token.replace("D", "E").replace("d", "e"))
        return value if value.is_finite() and abs(value.adjusted()) <= 1000 else None
    except InvalidOperation:
        return None


def _heading(line, previous):
    stripped = line.strip().strip("*").strip()
    if re.match(r"NO\s+OCC\b", stripped, re.I):
        return None
    if stripped.upper() in {"DIPOLE MOMENT", "ORBITAL ENERGIES"}:
        return stripped.upper()
    if (
        _SEPARATOR.match(previous)
        and re.fullmatch(r"[A-Za-z][A-Za-z ()/–-]{2,90}", stripped)
        and stripped not in {"X", "Y", "Z"}
    ):
        return stripped
    return None


def _consume_value(candidate, number, line):
    """Local adapters consume the same scanned, hash-verified byte snapshot."""
    heading = candidate["heading"].upper()
    if heading == "DIPOLE MOMENT":
        match = re.search(rf"Magnitude\s*\(Debye\)\s*:\s*({_NUMBER})", line, re.I)
        if match and (value := _decimal(match[1])) is not None and value >= 0:
            if len(candidate.setdefault("dipoles", [])) < 2:
                candidate["dipoles"].append((match[1], number))
            candidate["priority"] = 10
        if re.search(r"Type of density|Method\s*:", line, re.I):
            if len(candidate.setdefault("definition", [])) < 2:
                candidate["definition"].append(line.strip()[:160])
    if heading == "ORBITAL ENERGIES":
        if re.search(r"\bNO\s+OCC\s+E\(Eh\)", line, re.I):
            candidate["orbital_header"] = (number, line.strip()[:160])
            candidate["orbital_ev"] = bool(re.search(r"E\(eV\)", line, re.I))
        match = _ORBIT_ROW.match(line)
        if match and candidate.get("orbital_header"):
            index, occ = int(match[1]), _decimal(match[2])
            token = match[4] if candidate["orbital_ev"] else match[3]
            value = _decimal(token) if token else None
            last_index = candidate.get("last_index")
            last_value = candidate.get("last_value")
            if (
                occ not in {Decimal(0), Decimal(2)}
                or value is None
                or (last_index is None and index not in {0, 1})
                or (last_index is not None and index != last_index + 1)
                or (last_value is not None and value is not None and value < last_value)
                or (candidate.get("lumo") and occ == 2)
            ):
                candidate["orbital_invalid"] = True
            candidate["last_index"], candidate["last_value"] = index, value
            if occ == 2:
                candidate["homo"] = (token, number)
            elif occ == 0 and not candidate.get("lumo"):
                candidate["lumo"] = (token, number)
            if candidate.get("homo") and candidate.get("lumo"):
                candidate["priority"] = 10
    match = re.search(
        rf"(Number of (?:Electrons|Alpha Electrons|Beta Electrons|Correlated Electrons))"
        rf"\s*(?:\.{{2,}}|:)\s*({_NUMBER})\s*$",
        line,
        re.I,
    )
    if match and (value := _decimal(match[2])) is not None and value >= 0 and value == int(value):
        if len(candidate.setdefault("electron_fields", [])) < 4:
            candidate["electron_fields"].append((match[1], match[2], number))
        candidate["priority"] = max(candidate["priority"], 8)


def _observations(candidate, hint, *, spin=False):
    if hint == "dipole_moment":
        values = candidate.get("dipoles", [])
        if len(values) != 1:
            return []
        token, line = values[0]
        return [
            {
                "property_hint": hint,
                "view_kind": "observed_value",
                "token": token,
                "unit": "Debye",
                "source_lines": [line],
                "section": "DIPOLE MOMENT",
                "definition": "; ".join(candidate.get("definition", [])),
            }
        ]
    if hint in FRONTIER_HINTS:
        if (
            spin
            or candidate.get("orbital_invalid")
            or not candidate.get("orbital_header")
            or not candidate.get("homo")
            or not candidate.get("lumo")
        ):
            return []
        homo, hline = candidate["homo"]
        lumo, lline = candidate["lumo"]
        unit = "eV" if candidate["orbital_ev"] else "Eh"
        # Bound operands above, but do not round their subtraction to Python's
        # process-global 28-digit Decimal default.
        with localcontext() as context:
            context.prec = 2100
            gap = str(_decimal(lumo) - _decimal(homo))
        if _decimal(gap) is None:
            return []
        base = {
            "unit": unit,
            "section": "ORBITAL ENERGIES",
            "header_line": candidate["orbital_header"][0],
            "header": candidate["orbital_header"][1],
        }
        result = [
            {
                **base,
                "property_hint": "homo_energy",
                "view_kind": "observed_value",
                "token": homo,
                "source_lines": [hline],
            },
            {
                **base,
                "property_hint": "lumo_energy",
                "view_kind": "observed_value",
                "token": lumo,
                "source_lines": [lline],
            },
            {
                **base,
                "property_hint": "homo_lumo_gap",
                "view_kind": "derived_value",
                "token": gap,
                "source_lines": [hline, lline],
                "operands": {"HOMO": homo, "LUMO": lumo},
                "formula": "LUMO - HOMO",
            },
        ]
        return (
            result
            if hint == "frontier_orbitals"
            else [v for v in result if v["property_hint"] == hint]
        )
    if hint and hint.startswith("orca_printed_"):
        definition = {
            "orca_printed_electron_count": "number of electrons",
            "orca_printed_alpha_electrons": "number of alpha electrons",
            "orca_printed_beta_electrons": "number of beta electrons",
            "orca_printed_correlated_electrons": "number of correlated electrons",
        }.get(hint)
        fields = [v for v in candidate.get("electron_fields", []) if v[0].casefold() == definition]
        if len(fields) == 1:
            label, token, line = fields[0]
            return [
                {
                    "property_hint": hint,
                    "view_kind": "observed_value",
                    "token": token,
                    "unit": "electrons",
                    "source_lines": [line],
                    "section": label,
                    "definition": label,
                }
            ]
    return []


class OutputQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence: StrictStr = Field(min_length=1, max_length=240)
    search_terms: list[StrictStr] = Field(min_length=1, max_length=4)
    property_hint: StrictStr | None = Field(default=None, max_length=80)
    question_key: StrictStr | None = Field(default=None, max_length=64)

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

    @field_validator("property_hint", "question_key")
    @classmethod
    def plain_hint(cls, value):
        if value is not None and (
            not value.strip() or any(unicodedata.category(c).startswith("C") for c in value)
        ):
            raise ValueError("query hints must be nonblank plain text")
        return value


def dump_query_spec(query):
    """Keep legacy query snapshots unchanged when optional goal hints are absent."""
    return OutputQuerySpec.model_validate(query, strict=True).model_dump(
        mode="json", exclude_none=True
    )


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


def _stage_binding(candidate, context, anchors):
    locations = (context or {}).get("source_locations", {})
    positions = [
        locations.get(k)
        for k in ("final_energy_section", "final_scf", "final_energy", "normal_termination")
    ]
    if (
        not context
        or not context.get("verified_stage")
        or not all(type(n) is int for n in positions)
    ):
        return "source_only"
    section, scf, energy, termination = positions
    if not (
        section <= scf < energy < termination
        and "FINAL ENERGY EVALUATION" in anchors.get(section, "").upper()
        and "SCF" in anchors.get(scf, "").upper()
        and "CONVERG" in anchors.get(scf, "").upper()
        and "FINAL SINGLE POINT ENERGY" in anchors.get(energy, "")
        and "ORCA TERMINATED NORMALLY" in anchors.get(termination, "")
    ):
        return "source_only"
    # Explicit final SCF anchors can bind a preceding orbital table. A later
    # property job still needs its own density/geometry evidence.
    return (
        "selected_stage"
        if scf <= candidate["start_line"] <= candidate["end_line"] < energy
        else "source_only"
    )


def search_output_bytes(payload, queries, *, deadline, cancel, limits, stage_context=None):
    """Scan blocks and numeric rows with bounded candidate storage and fair excerpts."""
    terms = [[" ".join(t.casefold().split()) for t in q.search_terms] for q in queries]
    pools, active, saturated = (
        [[] for _ in queries],
        [None for _ in queries],
        [False for _ in queries],
    )
    limitations, anchors = set(), {}
    wanted = {
        v for v in (stage_context or {}).get("source_locations", {}).values() if type(v) is int
    }
    previous, heading, heading_line, number = "", "", 1, 0
    spin = (stage_context or {}).get("multiplicity", 1) != 1
    for number, line, clipped in _lines(payload, cancel, deadline):
        if clipped:
            limitations.add("line_limit")
        if "\ufffd" in line:
            limitations.add("invalid_encoding")
        if number in wanted:
            anchors[number] = line
        normalized = " ".join(line.casefold().split())
        spin |= any(
            s in normalized
            for s in ("spin up orbitals", "spin down orbitals", "alpha orbitals", "beta orbitals")
        )
        new_heading = _heading(line, previous)
        if new_heading:
            for i, current in enumerate(active):
                if current is not None and current["structural"]:
                    current["end_line"], active[i] = number - 1, None
            heading, heading_line = new_heading, number
        for i, query in enumerate(queries):
            current = active[i]
            if current is not None and not current["structural"] and number > current["end_line"]:
                active[i] = current = None
            forced = bool(
                new_heading
                and (
                    (query.property_hint == "dipole_moment" and heading == "DIPOLE MOMENT")
                    or (query.property_hint in FRONTIER_HINTS and heading == "ORBITAL ENERGIES")
                )
            )
            hit = forced or any(word in normalized for word in terms[i])
            if hit and current is None:
                structural = heading.upper() in {"DIPOLE MOMENT", "ORBITAL ENERGIES"}
                current = {
                    "start_line": heading_line if structural else max(1, number - 3),
                    "end_line": number + 18,
                    "heading": heading if structural else "",
                    "structural": structural,
                    "priority": 3 if structural else 1,
                    "clipped": False,
                }
                if len(pools[i]) < MAX_CANDIDATES:
                    pools[i].append(current)
                else:
                    saturated[i] = True
                active[i] = current
            if current is not None:
                current["end_line"] = (
                    number if current["structural"] else number + 18 if hit else current["end_line"]
                )
                current["clipped"] |= clipped or "\ufffd" in line
                _consume_value(current, number, line)
                if current not in pools[i]:
                    weakest = min(pools[i], key=lambda c: c["priority"])
                    if current["priority"] > weakest["priority"]:
                        pools[i].remove(weakest)
                        pools[i].append(current)
        previous = line
    evidence, selections = [], []
    for i, (query, pool) in enumerate(zip(queries, pools, strict=True)):
        for candidate in pool:
            candidate["end_line"] = min(candidate["end_line"], number)
        local = sorted(pool, key=lambda c: (-c["priority"], c["start_line"]))
        if local:
            local = [c for c in local if c["priority"] == local[0]["priority"]]
            bound = [
                c for c in local if _stage_binding(c, stage_context, anchors) == "selected_stage"
            ]
            if len(bound) == 1:
                local = bound
        observations = (
            _observations(local[0], query.property_hint, spin=spin) if len(local) == 1 else []
        )
        binding = (
            _stage_binding(local[0], stage_context, anchors) if len(local) == 1 else "source_only"
        )
        local_limits = sorted(limitations | ({"candidate_limit"} if saturated[i] else set()))
        if query.property_hint and binding != "selected_stage":
            local_limits.append("stage_unbound")
        for value in observations:
            value.update(
                binding_status=binding,
                limitations=local_limits,
                question_key=query.question_key,
                required_scope_complete=(
                    binding == "selected_stage"
                    and not saturated[i]
                    and not local[0]["clipped"]
                    and "invalid_encoding" not in local_limits
                ),
            )
        status = (
            "no_match"
            if not local
            else "multiple"
            if len(local) > 1
            else "text_only"
            if query.property_hint and not observations
            else "unique"
        )
        evidence.append(
            {
                "query_index": i,
                "lookup_status": "not_found",
                "snippets": [],
                "search_status": "limited" if set(local_limits) - {"stage_unbound"} else "complete",
                "candidate_status": status,
                "binding_status": binding,
                "limitations": local_limits,
                "excerpt_complete": False,
                "observations": observations,
                "ambiguous": len(local) > 1,
                "truncated": False,
                "required_scope_complete": False,
            }
        )
        selections.append(local)
    # One opportunity for every question before the next excerpt of any question.
    for round_index in range(MAX_SNIPPETS):
        for i, local in enumerate(selections):
            if round_index >= len(local):
                continue
            item, candidate = evidence[i], local[round_index]
            if min(limits.values()) <= 0:
                item["limitations"].append("excerpt_budget")
                continue
            start, end = candidate["start_line"], candidate["end_line"]
            if candidate.get("homo") and candidate.get("lumo"):
                start, end = max(start, candidate["homo"][1] - 1), candidate["lumo"][1] + 1
            waiting = max(
                1, sum(bool(c) and not evidence[j]["snippets"] for j, c in enumerate(selections))
            )
            line_budget, byte_budget = (
                max(1, limits["lines"] // waiting),
                max(1, limits["bytes"] // waiting),
            )
            text_lines, used, actual_end = [], 0, start - 1
            cut = candidate["clipped"]
            for n, text, clipped in _lines(payload, cancel, deadline):
                if n < start:
                    continue
                if n > end:
                    break
                rendered = visible_text(text)
                cost = len(rendered.encode()) + bool(text_lines)
                if len(text_lines) >= line_budget or used + cost > byte_budget:
                    cut = True
                    break
                text_lines.append(rendered)
                used, actual_end = used + cost, n
                cut |= clipped
            if text_lines:
                item["snippets"].append(
                    {"start_line": start, "end_line": actual_end, "text": "\n".join(text_lines)}
                )
                limits["snippets"] -= 1
                limits["lines"] -= len(text_lines)
                limits["bytes"] -= used
            if cut:
                item["limitations"].append("excerpt_budget")
    for i, item in enumerate(evidence):
        local = selections[i]
        item["lookup_status"] = (
            "found" if item["snippets"] else "unavailable" if local else "not_found"
        )
        item["excerpt_complete"] = (
            bool(local)
            and len(item["snippets"]) == len(local)
            and "excerpt_budget" not in item["limitations"]
        )
        item["truncated"] = bool(local) and not item["excerpt_complete"]
        item["required_scope_complete"] = (
            item["excerpt_complete"]
            if queries[i].property_hint == "raw_excerpt"
            else (
                bool(item["observations"])
                and all(o["required_scope_complete"] for o in item["observations"])
                if queries[i].property_hint
                else item["excerpt_complete"] and not item["ambiguous"]
            )
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
            or item.get("search_status") not in {"complete", "limited", "unavailable"}
            or item.get("candidate_status") not in {"no_match", "multiple", "text_only", "unique"}
            or item.get("binding_status") not in {"source_only", "selected_stage", "unavailable"}
            or type(item.get("excerpt_complete")) is not bool
            or type(item.get("required_scope_complete")) is not bool
            or not isinstance(item.get("limitations"), list)
            or len(item["limitations"]) > 8
            or any(not isinstance(v, str) or len(v) > 80 for v in item["limitations"])
            or not isinstance(item.get("observations"), list)
            or len(item["observations"]) > 3
        ):
            raise ValueError("invalid query evidence identity or shape")
        if bool(item["snippets"]) != (item["lookup_status"] == "found"):
            raise ValueError("query status does not match snippets")
        if item["ambiguous"] != (item["candidate_status"] == "multiple"):
            raise ValueError("candidate status does not match ambiguity")
        for observation in item["observations"]:
            if (
                not isinstance(observation, dict)
                or observation.get("view_kind") not in {"observed_value", "derived_value"}
                or not isinstance(observation.get("property_hint"), str)
                or observation.get("unit") not in {"Debye", "eV", "Eh", "electrons"}
                or _decimal(observation.get("token")) is None
                or observation.get("binding_status") != item["binding_status"]
                or type(observation.get("required_scope_complete")) is not bool
                or not isinstance(observation.get("source_lines"), list)
                or not 1 <= len(observation["source_lines"]) <= 2
                or any(type(n) is not int or n < 1 for n in observation["source_lines"])
            ):
                raise ValueError("invalid readonly observation")
            if observation["required_scope_complete"] and (
                item["binding_status"] != "selected_stage"
                or item["candidate_status"] != "unique"
                or set(item["limitations"]) - {"line_limit"}
            ):
                raise ValueError("unbound or limited observation cannot complete its numeric goal")
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


def make_orca_output_tool(
    *, remaining_file_bytes, deadline, remaining_excerpt_limits, stage_context=None
):
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
                stage_context=stage_context,
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
