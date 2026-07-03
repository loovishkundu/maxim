"""arXiv search via the public Atom API. No key exists — always available."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from ..llm import ClientTool, SourceDoc, ToolOutcome
from .base import get_text

_SPEC = {
    "name": "arxiv_search",
    "description": (
        "Search arXiv preprints. Returns title, date, and abstract. Abstract "
        "text may be quoted directly; cite the arXiv URL shown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search phrase"}},
        "required": ["query"],
    },
}

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def _entry_text(entry: ET.Element, tag: str) -> str:
    node = entry.find(f"atom:{tag}", _ATOM)
    return " ".join((node.text or "").split()) if node is not None else ""


async def _search(tool_input: dict[str, Any]) -> ToolOutcome:
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return ToolOutcome(content="error: empty query", error=True)
    raw = await get_text(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": f"all:{query}",
            "max_results": 8,
            "sortBy": "relevance",
        },
    )
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return ToolOutcome(content=f"arXiv returned unparseable XML: {exc}", error=True)
    entries = root.findall("atom:entry", _ATOM)
    if not entries:
        return ToolOutcome(content=f"no arXiv results for {query!r}")

    lines: list[str] = []
    sources: list[SourceDoc] = []
    for entry in entries:
        title = _entry_text(entry, "title") or "(untitled)"
        url = _entry_text(entry, "id")
        published = _entry_text(entry, "published")[:10]
        summary = _entry_text(entry, "summary")
        lines.append(f"- {title} ({published})\n  {url}")
        if summary:
            lines.append(f"  abstract: {summary}")
        if url:
            sources.append(SourceDoc(url=url, text=f"{title}\n{summary}".strip()))
    return ToolOutcome(content="\n".join(lines), sources=sources)


def tool() -> ClientTool:
    return ClientTool(spec=_SPEC, handler=_search, max_uses=6)
