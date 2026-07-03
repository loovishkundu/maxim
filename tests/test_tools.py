"""Client-side tools against canned HTTP responses (httpx.MockTransport).

Covers parsing, SourceCache-bound sources, engagement extraction, keyless
operation, and error degradation — no network.
"""

import json

import httpx
import pytest

import maxim.tools.base as tools_base
from maxim.tools import arxiv_api, client_tools_for, github_search, hn_algolia, semantic_scholar

HN_RESPONSE = {
    "hits": [
        {
            "objectID": "101",
            "title": "STL saved our telemetry pipeline",
            "url": "https://blog.example.com/stl",
            "points": 412,
            "num_comments": 220,
            "created_at": "2026-05-01T12:00:00Z",
            "story_text": "We replaced Prophet with STL and cut false positives.",
        },
        {
            "objectID": "102",
            "title": "Ask HN: anomaly detection?",
            "url": None,
            "points": 3,
            "num_comments": 1,
            "created_at": "2026-04-01T12:00:00Z",
        },
    ]
}

GITHUB_RESPONSE = {
    "items": [
        {
            "title": "STL blows up on NaN gaps",
            "html_url": "https://github.com/org/lib/issues/42",
            "state": "open",
            "comments": 12,
            "reactions": {"total_count": 7},
            "body": "In production we hit a crash when sensor gaps produce NaNs.",
        }
    ]
}

S2_RESPONSE = {
    "data": [
        {
            "paperId": "abc",
            "title": "Robust STL for streaming telemetry",
            "abstract": "We propose RobustSTL, which outperforms baselines on telemetry.",
            "url": "https://www.semanticscholar.org/paper/abc",
            "year": 2025,
            "venue": "KDD",
            "citationCount": 31,
        }
    ]
}

ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <title>Anomaly Detection with  Seasonal
      Decomposition</title>
    <summary>We study STL variants for vehicle telemetry.</summary>
    <published>2026-01-15T00:00:00Z</published>
  </entry>
</feed>"""


@pytest.fixture
def http(monkeypatch):
    """Route all tool HTTP through a canned-response transport."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        host = request.url.host
        if host == "hn.algolia.com":
            return httpx.Response(200, json=HN_RESPONSE)
        if host == "api.github.com":
            return httpx.Response(200, json=GITHUB_RESPONSE)
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json=S2_RESPONSE)
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_RESPONSE)
        return httpx.Response(500, text="unexpected host")

    monkeypatch.setattr(
        tools_base,
        "_make_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return recorded


async def test_hn_search_parses_engagement_and_sources(http):
    outcome = await hn_algolia.tool().handler({"query": "STL anomaly"})
    assert not outcome.error
    assert "412 points, 220 comments" in outcome.content
    thread_url = "https://news.ycombinator.com/item?id=101"
    assert thread_url in outcome.content
    # Story text is cached under the thread URL so quotes verify mechanically.
    by_url = {s.url: s for s in outcome.sources}
    assert "cut false positives" in by_url[thread_url].text
    # Engagement is keyed by thread URL and by the linked article URL.
    assert outcome.engagement[thread_url].points == 412
    assert outcome.engagement["https://blog.example.com/stl"].comments == 220
    assert outcome.engagement[thread_url].source == "hn"


async def test_github_search_parses_reactions(http, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    outcome = await github_search.tool().handler({"query": "STL NaN"})
    url = "https://github.com/org/lib/issues/42"
    assert url in outcome.content
    assert outcome.engagement[url].reactions == 7
    assert outcome.engagement[url].comments == 12
    assert "crash when sensor gaps" in {s.url: s for s in outcome.sources}[url].text
    # Keyless: no Authorization header was sent.
    github_requests = [r for r in http if r.url.host == "api.github.com"]
    assert "authorization" not in {k.lower() for k in github_requests[0].headers}


async def test_github_token_attached_when_present(http, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh_test")
    await github_search.tool().handler({"query": "x"})
    request = [r for r in http if r.url.host == "api.github.com"][0]
    assert request.headers["authorization"] == "Bearer gh_test"


async def test_semantic_scholar_returns_quotable_abstract(http):
    outcome = await semantic_scholar.tool().handler({"query": "robust STL"})
    assert "Robust STL for streaming telemetry" in outcome.content
    assert "KDD 2025, 31 citations" in outcome.content
    source = outcome.sources[0]
    assert "outperforms baselines" in source.text
    assert source.url == "https://www.semanticscholar.org/paper/abc"


async def test_arxiv_parses_atom_and_normalizes_whitespace(http):
    outcome = await arxiv_api.tool().handler({"query": "seasonal decomposition"})
    assert "Anomaly Detection with Seasonal Decomposition" in outcome.content
    assert "2026-01-15" in outcome.content
    assert outcome.sources[0].url == "http://arxiv.org/abs/2601.00001v1"
    assert "STL variants" in outcome.sources[0].text


async def test_empty_query_is_an_error_without_http(http):
    outcome = await hn_algolia.tool().handler({"query": "  "})
    assert outcome.error
    assert not http  # never hit the network


async def test_http_failure_propagates_for_llm_layer_to_catch(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"message": "rate limited"})

    monkeypatch.setattr(
        tools_base,
        "_make_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await github_search.tool().handler({"query": "x"})


def test_perspective_tool_assignment():
    community = {t.name for t in client_tools_for("community")}
    technical = {t.name for t in client_tools_for("statistics")}
    assert community == {"hn_search", "github_search"}
    assert technical == {"paper_search", "arxiv_search"}
    assert client_tools_for("nonsense") == []


def test_specs_are_valid_tool_definitions():
    for perspective in ("community", "statistics"):
        for tool in client_tools_for(perspective):
            assert set(tool.spec) == {"name", "description", "input_schema"}
            json.dumps(tool.spec)  # serializable
            assert tool.max_uses > 0
