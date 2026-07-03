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
_FENCE_RE = re.compile(r"^```.*?(?:^```\s*$|\Z)", re.MULTILINE | re.DOTALL)
# Markdown block markers that exempt a paragraph from the prose-citation bar:
# headings, tables, quotes, and true list items (marker + space). A paragraph
# opening with bold ('**Claim** …') is still prose.
_NON_PROSE_RE = re.compile(r"^(#|\||>|[-*+]\s|\d+\.\s)")


def _sections(body: str) -> list[tuple[str, str]]:
    """(title, text) per '## Heading', preserving duplicates; text before the
    first heading is included under the empty title."""
    matches = list(_HEADING_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()] if matches else body
    if preamble.strip():
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((match.group(1).strip(), body[start:end]))
    return sections


def _is_prose(paragraph: str) -> bool:
    stripped = paragraph.lstrip()
    return bool(stripped) and not _NON_PROSE_RE.match(stripped)


def report_violations(body: str, known_ids: set[str]) -> list[str]:
    """All the ways the synthesized body breaks the report contract."""
    # Fenced code blocks are opaque: a '## ' inside one must not satisfy a
    # required-section check, and fence bodies are not uncited prose.
    body = _FENCE_RE.sub("", body)
    violations: list[str] = []
    sections = _sections(body)
    titles = [title for title, _ in sections]

    for required in REQUIRED_SECTIONS:
        if not any(required.casefold() in title.casefold() for title in titles):
            violations.append(f"required section '## {required}' is missing")

    for bad in sorted({c for c in CITATION_RE.findall(body) if c not in known_ids}):
        violations.append(f"cites unknown finding id {bad}")

    for title, text in sections:
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
                where = f"'## {title}'" if title else "the report preamble"
                violations.append(
                    f"uncited factual paragraph in {where} (starts: “{paragraph[:60]}…”)"
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
