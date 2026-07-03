"""Deterministic report-quality gate: the reader must never meet an uncited,
unclear, or template-breaking report.

Zero-LLM checks over the synthesized markdown; violations trigger one repair
pass (in the orchestrator) and are disclosed as warnings if they survive it.
Checks are deliberately conservative — false positives would churn synthesis
tokens — so only substantive prose is held to the citation bar.
"""

from __future__ import annotations

import re

from .report import CITATION_RE

REQUIRED_SECTIONS = ("TL;DR", "Method Landscape", "Decision Guide", "Caveats")

# Prose shorter than this can be a transition line; longer without a citation
# is a factual paragraph making uncited claims.
MIN_UNCITED_PARAGRAPH_CHARS = 300

_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")


def _sections(body: str) -> dict[str, str]:
    """Map '## Heading' → section text (up to the next heading)."""
    sections: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(body))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[start:end]
    return sections


def _is_prose(paragraph: str) -> bool:
    stripped = paragraph.lstrip()
    return bool(stripped) and not stripped.startswith(("#", "|", ">", "-", "*", "```"))


def report_violations(body: str, known_ids: set[str]) -> list[str]:
    """All the ways the synthesized body breaks the report contract."""
    violations: list[str] = []
    sections = _sections(body)

    for required in REQUIRED_SECTIONS:
        if not any(required.casefold() in title.casefold() for title in sections):
            violations.append(f"required section '## {required}' is missing")

    for bad in sorted({c for c in CITATION_RE.findall(body) if c not in known_ids}):
        violations.append(f"cites unknown finding id {bad}")

    for title, text in sections.items():
        rows = [
            line
            for line in text.splitlines()
            if _TABLE_ROW_RE.match(line.strip()) and not set(line) <= {"|", "-", " ", ":"}
        ]
        if "landscape" in title.casefold() and len(rows) > 1:
            for row in rows[1:]:  # skip the header row
                if not CITATION_RE.search(row):
                    method = row.strip("| ").split("|")[0].strip()
                    violations.append(f"Method Landscape row '{method}' cites no finding ids")
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if (
                _is_prose(paragraph)
                and len(paragraph) >= MIN_UNCITED_PARAGRAPH_CHARS
                and not CITATION_RE.search(paragraph)
            ):
                violations.append(
                    f"uncited factual paragraph in '## {title}' " f"(starts: “{paragraph[:60]}…”)"
                )
    return violations


def repair_instruction(violations: list[str]) -> str:
    return (
        "Your report draft violates the report contract:\n"
        + "\n".join(f"- {v}" for v in violations)
        + "\n\nRewrite the COMPLETE report now, fixing every violation. Keep all "
        "valid content and citations; add citations only to finding ids you were "
        "given (never invent ids); include every required section."
    )
