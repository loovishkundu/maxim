"""Semantic Scholar paper search. Keyless; SEMANTIC_SCHOLAR_API_KEY raises limits."""

from __future__ import annotations

import os
from typing import Any

from ..llm import ClientTool, SourceDoc, ToolOutcome
from .base import get_json

_SPEC = {
    "name": "paper_search",
    "description": (
        "Search peer-reviewed papers and preprints (Semantic Scholar). Returns "
        "title, venue, year, citation count, and abstract. Abstract text may be "
        "quoted directly; cite the paper URL shown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search phrase"}},
        "required": ["query"],
    },
}

_FIELDS = "title,abstract,url,year,venue,citationCount,externalIds"


async def _search(tool_input: dict[str, Any]) -> ToolOutcome:
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return ToolOutcome(content="error: empty query", error=True)
    headers: dict[str, str] = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    data = await get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": 8, "fields": _FIELDS},
        headers=headers,
    )
    papers = data.get("data") or []
    if not papers:
        return ToolOutcome(content=f"no papers found for {query!r}")

    lines: list[str] = []
    sources: list[SourceDoc] = []
    for paper in papers:
        title = paper.get("title") or "(untitled)"
        url = paper.get("url") or (
            f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
        )
        year = paper.get("year") or "?"
        venue = paper.get("venue") or "preprint"
        citations = paper.get("citationCount") or 0
        abstract = paper.get("abstract") or ""
        lines.append(f"- {title} ({venue} {year}, {citations} citations)\n  {url}")
        if abstract:
            lines.append(f"  abstract: {abstract}")
            sources.append(SourceDoc(url=url, text=f"{title}\n{abstract}"))
    return ToolOutcome(content="\n".join(lines), sources=sources)


def tool() -> ClientTool:
    return ClientTool(spec=_SPEC, handler=_search, max_uses=6)
