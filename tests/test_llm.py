from types import SimpleNamespace as NS

from maxim.llm import CitedQuote, SourceDoc, harvest_blocks, strip_dangling_tool_use


def _fetch_result(url: str, text: str) -> NS:
    return NS(
        type="web_fetch_tool_result",
        content=NS(
            type="web_fetch_result",
            url=url,
            retrieved_at="2026-07-03T00:00:00Z",
            content=NS(source=NS(data=text)),
        ),
    )


class TestHarvest:
    def test_harvests_fetched_page(self):
        cache: dict[str, SourceDoc] = {}
        quotes: list[CitedQuote] = []
        harvest_blocks([_fetch_result("https://a.com", "page text")], cache, quotes)
        assert cache["https://a.com"].text == "page text"

    def test_skips_fetch_error_blocks(self):
        cache: dict[str, SourceDoc] = {}
        harvest_blocks(
            [NS(type="web_fetch_tool_result", content=NS(type="web_fetch_tool_result_error"))],
            cache,
            [],
        )
        assert not cache

    def test_harvests_citations_from_text_blocks(self):
        quotes: list[CitedQuote] = []
        block = NS(
            type="text",
            text="answer",
            citations=[NS(cited_text="a cited sentence", url="https://b.com", title="B")],
        )
        harvest_blocks([block], {}, quotes)
        assert quotes[0].text == "a cited sentence"
        assert quotes[0].url == "https://b.com"

    def test_tolerates_weird_shapes(self):
        harvest_blocks([NS(type="web_fetch_tool_result", content=None), NS(type="text")], {}, [])


class TestStripDangling:
    def test_strips_unresolved_trailing_tool_use(self):
        content = [
            NS(type="text", text="searching..."),
            NS(type="server_tool_use", id="srvtoolu_1", name="web_fetch"),
        ]
        out = strip_dangling_tool_use(content)
        assert [b.type for b in out] == ["text"]

    def test_keeps_resolved_tool_use(self):
        content = [
            NS(type="server_tool_use", id="srvtoolu_1", name="web_search"),
            NS(type="web_search_tool_result", tool_use_id="srvtoolu_1", content=[]),
            NS(type="text", text="done"),
        ]
        assert strip_dangling_tool_use(content) == content

    def test_strips_multiple_trailing_unresolved(self):
        content = [
            NS(type="text", text="ok"),
            NS(type="server_tool_use", id="a", name="web_search"),
            NS(type="server_tool_use", id="b", name="web_fetch"),
        ]
        out = strip_dangling_tool_use(content)
        assert len(out) == 1

    def test_empty_content(self):
        assert strip_dangling_tool_use([]) == []
