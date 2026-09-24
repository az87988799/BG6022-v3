"""Shared, deterministic helpers for user text that enters either intake path."""

from __future__ import annotations

import re

from bg6022.tools.molecule import parse_xyz_bytes


def extract_single_inline_xyz(message: str) -> tuple[str, int] | None:
    """Find one complete XYZ block and retain its exact original text."""

    lines = message.splitlines(keepends=True)
    matches: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        count_text = line.strip()
        if not count_text.isdecimal():
            continue
        count = int(count_text)
        if count <= 0 or count > 10000 or index + count + 2 > len(lines):
            continue
        block = "".join(lines[index : index + count + 2])
        try:
            parsed = parse_xyz_bytes(block.encode("utf-8"))
        except (TypeError, ValueError):
            continue
        matches.append((block, parsed.atom_count))
        if len(matches) > 1:
            return None
    return matches[0] if matches else None


def mentions_computation(message: str) -> bool:
    return (
        re.search(
            r"(?i)(?:\b(?:opt|sp|freq)\b|geometry\s+optimization|optimiz|"
            r"single[ -]?point|frequency|frequencies|几何优化|优化|单点|频率|计算)",
            message,
        )
        is not None
    )


__all__ = ["extract_single_inline_xyz", "mentions_computation"]
