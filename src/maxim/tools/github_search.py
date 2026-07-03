"""GitHub issue/discussion search. Works keyless; GITHUB_TOKEN raises limits."""

from __future__ import annotations

import os
from typing import Any

from ..llm import ClientTool, SourceDoc, ToolOutcome
from ..schemas import EngagementStats
from .base import get_json, truncate

_SPEC = {
    "name": "github_search",
    "description": (
        "Search GitHub issues and discussions for real-world reports about a "
        "method or library (bugs, benchmarks, production experience). Returned "
        "issue bodies may be quoted directly; cite the issue URL shown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "search terms; GitHub qualifiers like repo:owner/name allowed",
            }
        },
        "required": ["query"],
    },
}


async def _search(tool_input: dict[str, Any]) -> ToolOutcome:
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return ToolOutcome(content="error: empty query", error=True)
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = await get_json(
        "https://api.github.com/search/issues",
        params={"q": query, "per_page": 8},
        headers=headers,
    )
    items = data.get("items") or []
    if not items:
        return ToolOutcome(content=f"no GitHub issues/discussions match {query!r}")

    lines: list[str] = []
    sources: list[SourceDoc] = []
    engagement: dict[str, EngagementStats] = {}
    for item in items:
        url = str(item.get("html_url") or "")
        title = item.get("title") or "(untitled)"
        comments = int(item.get("comments") or 0)
        reactions = int((item.get("reactions") or {}).get("total_count") or 0)
        state = item.get("state") or "?"
        body = truncate(str(item.get("body") or ""))
        lines.append(
            f"- {title} [{state}] — {comments} comments, {reactions} reactions\n  {url}"
            + (f"\n  body: {body}" if body else "")
        )
        if url:
            sources.append(SourceDoc(url=url, text=f"{title}\n{item.get('body') or ''}".strip()))
            engagement[url] = EngagementStats(
                source="github", comments=comments, reactions=reactions
            )
    return ToolOutcome(content="\n".join(lines), sources=sources, engagement=engagement)


def tool() -> ClientTool:
    return ClientTool(spec=_SPEC, handler=_search, max_uses=6)
