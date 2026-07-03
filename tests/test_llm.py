from types import SimpleNamespace as NS

from maxim.config import Settings
from maxim.llm import (
    LLM,
    CitedQuote,
    ClientTool,
    SourceDoc,
    ToolOutcome,
    harvest_blocks,
    strip_dangling_tool_use,
    thinking_kwargs,
)
from maxim.usage import UsageLedger


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


def _usage():
    return NS(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        server_tool_use=None,
    )


def _message(content, stop_reason):
    return NS(content=content, stop_reason=stop_reason, usage=_usage())


class FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self._responses.pop(0))


def _tool_use(tid, name, tool_input):
    return NS(type="tool_use", id=tid, name=name, input=tool_input)


def _llm(responses):
    return LLM(
        settings=Settings(),
        ledger=UsageLedger(budget_usd=10.0),
        client=FakeAnthropicClient(responses),
    )


def _client_tool(handler, max_uses=5, name="hn_search"):
    return ClientTool(
        spec={"name": name, "description": "d", "input_schema": {"type": "object"}},
        handler=handler,
        max_uses=max_uses,
    )


class TestClientToolLoop:
    async def test_tool_round_trip(self):
        async def handler(tool_input):
            assert tool_input == {"query": "stl"}
            return ToolOutcome(
                content="TOOL RESULT TEXT",
                sources=[SourceDoc(url="https://x.test/1", text="cached body")],
                engagement={"https://x.test/1": {"points": 42}},
            )

        llm = _llm(
            [
                _message([_tool_use("tu1", "hn_search", {"query": "stl"})], "tool_use"),
                _message([NS(type="text", text="done", citations=None)], "end_turn"),
            ]
        )
        result = await llm.run_agentic(
            stage="researcher:community",
            system="s",
            messages=[{"role": "user", "content": "go"}],
            tools=[_client_tool(handler)],
            model="claude-opus-4-8",
            effort="medium",
            max_tokens=1000,
            max_continuations=6,
        )

        # The tool result was fed back on the same conversation.
        second_call = llm.client.calls[1]
        tool_result_msg = second_call["messages"][-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tu1"
        assert tool_result_msg["content"][0]["content"] == "TOOL RESULT TEXT"
        # Only the API spec (not the ClientTool wrapper) went to the API.
        assert second_call["tools"] == [_client_tool(handler).spec]
        # Sources and engagement were harvested for verification.
        assert result.source_cache["https://x.test/1"].text == "cached body"
        assert result.engagement["https://x.test/1"] == {"points": 42}
        assert result.continuations == 1
        assert not result.truncated

    async def test_tool_budget_enforced_mechanically(self):
        calls = []

        async def handler(tool_input):
            calls.append(tool_input)
            return ToolOutcome(content="ok")

        llm = _llm(
            [
                _message([_tool_use("tu1", "hn_search", {"query": "a"})], "tool_use"),
                _message([_tool_use("tu2", "hn_search", {"query": "b"})], "tool_use"),
                _message([NS(type="text", text="done", citations=None)], "end_turn"),
            ]
        )
        await llm.run_agentic(
            stage="s",
            system="s",
            messages=[{"role": "user", "content": "go"}],
            tools=[_client_tool(handler, max_uses=1)],
            model="claude-opus-4-8",
            effort="medium",
            max_tokens=1000,
            max_continuations=6,
        )
        assert len(calls) == 1  # second call blocked by the cap, not executed
        third_call = llm.client.calls[2]
        blocked = third_call["messages"][-1]["content"][0]
        assert blocked["is_error"] is True
        assert "budget exhausted" in blocked["content"]

    async def test_tool_exception_degrades_to_error_result(self):
        async def handler(tool_input):
            raise RuntimeError("api down")

        llm = _llm(
            [
                _message([_tool_use("tu1", "hn_search", {"query": "a"})], "tool_use"),
                _message([NS(type="text", text="done", citations=None)], "end_turn"),
            ]
        )
        result = await llm.run_agentic(
            stage="s",
            system="s",
            messages=[{"role": "user", "content": "go"}],
            tools=[_client_tool(handler)],
            model="claude-opus-4-8",
            effort="medium",
            max_tokens=1000,
            max_continuations=6,
        )
        error_result = llm.client.calls[1]["messages"][-1]["content"][0]
        assert error_result["is_error"] is True
        assert "api down" in error_result["content"]
        assert result.final_stop_reason == "end_turn"

    async def test_tool_use_at_continuation_cap_strips_dangling(self):
        async def handler(tool_input):
            return ToolOutcome(content="ok")

        llm = _llm(
            [
                _message(
                    [
                        NS(type="text", text="thinking", citations=None),
                        _tool_use("tu1", "hn_search", {"query": "a"}),
                    ],
                    "tool_use",
                ),
            ]
        )
        result = await llm.run_agentic(
            stage="s",
            system="s",
            messages=[{"role": "user", "content": "go"}],
            tools=[_client_tool(handler)],
            model="claude-opus-4-8",
            effort="medium",
            max_tokens=1000,
            max_continuations=0,
        )
        # Cap hit mid-tool-call: transcript must stay replayable (no dangling
        # tool_use) and the gather is marked truncated.
        assert result.truncated
        last = result.messages[-1]
        assert last["role"] == "assistant"
        assert [getattr(b, "type", None) for b in last["content"]] == ["text"]


class TestThinkingKwargs:
    def test_adaptive_models_get_thinking_and_effort(self):
        kwargs = thinking_kwargs("claude-opus-4-8", "low")
        assert kwargs == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}

    def test_haiku_gets_neither(self):
        # haiku-4-5 400s on adaptive thinking AND on output_config.effort;
        # the critic batch calls run on it, so this must stay empty.
        assert thinking_kwargs("claude-haiku-4-5", "low") == {}

    def test_fable_and_sonnet5_supported(self):
        assert thinking_kwargs("claude-fable-5", "high")
        assert thinking_kwargs("claude-sonnet-5", "high")


class TestDocumentHarvest:
    def test_base64_pdf_source_not_harvested_as_text(self):
        # A base64 PDF's data is a str; caching it as page text would fail
        # every genuine quote against base64 noise. It must be skipped so the
        # evidence degrades to verification_skipped instead.
        cache: dict[str, SourceDoc] = {}
        block = _fetch_result("https://a.com/paper.pdf", "unused")
        block.content.content = NS(source=NS(type="base64", data="JVBERi0xLjQK..."))
        harvest_blocks([block], cache, [])
        assert "https://a.com/paper.pdf" not in cache

    def test_plain_text_source_still_harvested(self):
        cache: dict[str, SourceDoc] = {}
        block = _fetch_result("https://a.com", "page text")
        harvest_blocks([block], cache, [])
        assert cache["https://a.com"].text == "page text"

    async def test_no_handler_execution_when_budget_already_spent(self):
        calls = []

        async def handler(tool_input):
            calls.append(tool_input)
            return ToolOutcome(content="ok")

        llm = _llm(
            [
                _message(
                    [_tool_use("tu1", "hn_search", {"query": "a"})],
                    "tool_use",
                ),
            ]
        )
        result = await llm.run_agentic(
            stage="s",
            system="s",
            messages=[{"role": "user", "content": "go"}],
            tools=[_client_tool(handler)],
            model="claude-opus-4-8",
            effort="medium",
            max_tokens=1000,
            max_continuations=0,
        )
        # The result would be discarded, so the tool must never execute.
        assert calls == []
        assert result.truncated
