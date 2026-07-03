"""Hacker News search via the public Algolia API. No key needed."""

from __future__ import annotations

from typing import Any

from ..llm import ClientTool, SourceDoc, ToolOutcome
from ..schemas import EngagementStats
from .base import get_json, truncate

_SPEC = {
    "name": "hn_search",
    "description": (
        "Search Hacker News threads. Returns top stories with points and comment "
        "counts — the primary way to find practitioner discussion and sentiment. "
        "Story text (when present) may be quoted directly; cite the thread URL shown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search phrase"}},
        "required": ["query"],
    },
}


async def _search(tool_input: dict[str, Any]) -> ToolOutcome:
    query = str(tool_input.get("query", "")).strip()
    if not query:
        return ToolOutcome(content="error: empty query", error=True)
    data = await get_json(
        "https://hn.algolia.com/api/v1/search",
        params={"query": query, "tags": "story", "hitsPerPage": 8},
    )
    hits = data.get("hits") or []
    if not hits:
        return ToolOutcome(content=f"no Hacker News results for {query!r}")

    lines: list[str] = []
    sources: list[SourceDoc] = []
    engagement: dict[str, EngagementStats] = {}
    for hit in hits:
        item_url = f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = hit.get("title") or "(untitled)"
        points = int(hit.get("points") or 0)
        comments = int(hit.get("num_comments") or 0)
        created = str(hit.get("created_at") or "")[:10]
        entry = (
            f"- {title} — {points} points, {comments} comments ({created})\n  thread: {item_url}"
        )
        if hit.get("url"):
            entry += f"\n  links to: {hit['url']}"
        story_text = hit.get("story_text") or ""
        if story_text:
            entry += f"\n  text: {truncate(story_text, 500)}"
        lines.append(entry)

        sources.append(SourceDoc(url=item_url, text=f"{title}\n{story_text}".strip()))
        stats = EngagementStats(source="hn", points=points, comments=comments)
        engagement[item_url] = stats
        if hit.get("url"):
            engagement[str(hit["url"])] = stats
    return ToolOutcome(content="\n".join(lines), sources=sources, engagement=engagement)


def tool() -> ClientTool:
    return ClientTool(spec=_SPEC, handler=_search, max_uses=6)
