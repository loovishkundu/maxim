"""Perspective-specific client-side tools, all graceful without API keys.

Technical perspectives get paper search (Semantic Scholar + arXiv); the
community researcher gets thread search (HN Algolia + GitHub issues), whose
structured engagement metadata feeds the mechanical sentiment floors.
"""

from __future__ import annotations

from ..llm import ClientTool
from . import arxiv_api, github_search, hn_algolia, semantic_scholar


def client_tools_for(perspective: str) -> list[ClientTool]:
    if perspective == "community":
        return [hn_algolia.tool(), github_search.tool()]
    if perspective in ("ai_agentic", "classical_ml", "data_science", "statistics"):
        return [semantic_scholar.tool(), arxiv_api.tool()]
    return []
