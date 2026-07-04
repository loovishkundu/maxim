"""Thin wrapper around one shared AsyncAnthropic client.

All Anthropic API knowledge lives here: structured parsing with retry, the
agentic streaming loop with pause_turn continuation, prompt-cache-friendly
system blocks, and harvesting of fetched page text / server citations into the
SourceCache that grounds verification.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .resilience import (
    CircuitOpenError,
    TransientServerError,
    api_resilient,
    reraise_if_transient_status,
)
from .usage import UsageLedger

T = TypeVar("T", bound=BaseModel)

THINKING = {"type": "adaptive"}

# Models on the adaptive-thinking / effort API surface. Haiku 4.5 supports
# neither adaptive thinking nor output_config.effort — sending either 400s —
# so calls to models outside this list go out with both omitted.
_ADAPTIVE_MODEL_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-fable",
    "claude-mythos",
)


def thinking_kwargs(model: str, effort: str) -> dict[str, Any]:
    """Per-model thinking/effort request params (empty for e.g. haiku)."""
    if model.startswith(_ADAPTIVE_MODEL_PREFIXES):
        return {"thinking": THINKING, "output_config": {"effort": effort}}
    return {}


_TOOL_RESULT_TYPES = {"web_search_tool_result", "web_fetch_tool_result", "tool_result"}
_TOOL_USE_TYPES = {"server_tool_use", "tool_use"}


# One resilience policy (retry + circuit breaker) per request shape. The
# breaker is per decorated function, so a failing stage type opens its own
# breaker without blocking the others.
@api_resilient()
async def _parse_request(client: anthropic.AsyncAnthropic, **kwargs: Any) -> Any:
    try:
        return await client.messages.parse(**kwargs)
    except anthropic.APIStatusError as exc:
        reraise_if_transient_status(exc)
        raise


@api_resilient()
async def _stream_request(client: anthropic.AsyncAnthropic, **kwargs: Any) -> Any:
    try:
        async with client.messages.stream(**kwargs) as stream:
            return await stream.get_final_message()
    except anthropic.APIStatusError as exc:
        reraise_if_transient_status(exc)
        raise


@api_resilient()
async def _stream_text_request(
    client: anthropic.AsyncAnthropic,
    on_text: Callable[[str], None] | None,
    emitted: dict[str, int],
    **kwargs: Any,
) -> Any:
    """Streamed text with exactly-once on_text delivery across retries.

    `emitted` persists across retry attempts (same dict object), so a retry
    replays only the suffix the callback has not seen yet.
    """
    try:
        async with client.messages.stream(**kwargs) as stream:
            if on_text is not None:
                seen = 0
                async for chunk in stream.text_stream:
                    seen += len(chunk)
                    if seen > emitted["chars"]:
                        on_text(chunk[-(seen - emitted["chars"]) :])
                        emitted["chars"] = seen
            return await stream.get_final_message()
    except anthropic.APIStatusError as exc:
        reraise_if_transient_status(exc)
        raise


class LLMError(RuntimeError):
    """A call failed after all retries."""


@dataclass
class SourceDoc:
    url: str
    text: str
    retrieved_at: str | None = None


@dataclass
class CitedQuote:
    text: str
    url: str | None
    title: str | None


@dataclass
class ToolOutcome:
    """What a client-side tool hands back.

    `sources` land in the SourceCache so quotes taken from tool results stay
    mechanically verifiable; `engagement` maps url → stats for the community
    researcher's mechanical floors.
    """

    content: str
    sources: list[SourceDoc] = field(default_factory=list)
    engagement: dict[str, Any] = field(default_factory=dict)
    error: bool = False


@dataclass
class ClientTool:
    """A locally-executed tool: API spec + async handler + hard use cap."""

    spec: dict[str, Any]  # {"name", "description", "input_schema"}
    handler: Callable[[dict[str, Any]], Any]  # async (input) -> ToolOutcome
    max_uses: int = 5

    @property
    def name(self) -> str:
        return str(self.spec["name"])


@dataclass
class AgenticResult:
    messages: list[dict[str, Any]]
    final_stop_reason: str | None
    source_cache: dict[str, SourceDoc]
    cited_quotes: list[CitedQuote]
    continuations: int
    truncated: bool  # gather ended early: continuation cap, max_tokens, or budget
    queries: list[str] = field(default_factory=list)  # web_search queries issued
    engagement: dict[str, Any] = field(default_factory=dict)  # url → stats from tools


@dataclass
class StreamResult:
    text: str
    stop_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


def _system_blocks(system: str) -> list[dict[str, Any]]:
    """System prompt as a cacheable block; the text must be byte-stable per role."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _document_text(doc: Any) -> str | None:
    source = getattr(doc, "source", None)
    if getattr(source, "type", None) == "base64":
        # PDF payload (Base64PDFSource.data is also a str) — harvesting it as
        # page text would fail every genuine quote against base64 noise.
        # Returning None leaves the evidence verification_skipped instead.
        return None
    data = getattr(source, "data", None)
    if isinstance(data, str) and data.strip():
        return data
    content = getattr(source, "content", None)
    if isinstance(content, list):
        joined = "\n".join(getattr(b, "text", "") or "" for b in content)
        return joined if joined.strip() else None
    return None


def harvest_blocks(
    content: list[Any],
    source_cache: dict[str, SourceDoc],
    cited_quotes: list[CitedQuote],
    queries: list[str] | None = None,
) -> None:
    """Pull fetched page text, server citations, and search queries out of
    response content blocks.

    Deliberately defensive about shapes: a missed harvest degrades a finding to
    verification_skipped, it must never crash the run.
    """
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use" and queries is not None:
            if getattr(block, "name", None) == "web_search":
                tool_input = getattr(block, "input", None)
                query = tool_input.get("query") if isinstance(tool_input, dict) else None
                if isinstance(query, str) and query.strip():
                    queries.append(query.strip())
        elif btype == "web_fetch_tool_result":
            inner = getattr(block, "content", None)
            if getattr(inner, "type", None) != "web_fetch_result":
                continue  # error block
            url = getattr(inner, "url", None)
            text = _document_text(getattr(inner, "content", None))
            if url and text:
                source_cache[url] = SourceDoc(
                    url=url,
                    text=text,
                    retrieved_at=getattr(inner, "retrieved_at", None),
                )
        elif btype == "text":
            for cit in getattr(block, "citations", None) or []:
                cited = getattr(cit, "cited_text", None)
                if cited:
                    cited_quotes.append(
                        CitedQuote(
                            text=cited,
                            url=getattr(cit, "url", None),
                            title=getattr(cit, "title", None),
                        )
                    )


def strip_dangling_tool_use(content: list[Any]) -> list[Any]:
    """Drop trailing tool_use blocks that never received a result.

    A turn that ended on pause_turn (continuation cap) or max_tokens can end
    with an unresolved (server_)tool_use block; replaying that as a prefix for
    a follow-up call is an invalid conversation shape. Removing whole
    unresolved blocks from the tail is the documented-safe cleanup.
    """
    resolved: set[str] = set()
    for block in content:
        if getattr(block, "type", None) in _TOOL_RESULT_TYPES:
            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id:
                resolved.add(tool_use_id)
    out = list(content)
    while out:
        last = out[-1]
        if (
            getattr(last, "type", None) in _TOOL_USE_TYPES
            and getattr(last, "id", None) not in resolved
        ):
            out.pop()
            continue
        break
    return out


@dataclass
class LLM:
    settings: Settings
    ledger: UsageLedger
    client: anthropic.AsyncAnthropic = field(
        default_factory=lambda: anthropic.AsyncAnthropic(max_retries=4)
    )

    async def close(self) -> None:
        await self.client.close()

    # ------------------------------------------------------------------ parse

    async def parse(
        self,
        *,
        stage: str,
        system: str,
        messages: list[dict[str, Any]],
        output_format: type[T],
        model: str,
        effort: str,
        max_tokens: int = 16_000,
    ) -> T:
        """Structured parse with validation-retry. Used by every stage."""
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for _attempt in range(self.settings.max_parse_retries + 1):
            try:
                response = await _parse_request(
                    self.client,
                    model=model,
                    max_tokens=max_tokens,
                    system=_system_blocks(system),
                    messages=attempt_messages,
                    output_format=output_format,
                    **thinking_kwargs(model, effort),
                )
            except (
                anthropic.APIConnectionError,
                anthropic.APIStatusError,
                httpx.TransportError,
                TransientServerError,
                CircuitOpenError,
            ) as exc:
                # The SDK and the resilience layer already retried transient
                # failures; whatever reaches here is fatal for this call.
                raise LLMError(f"{stage}: API call failed: {exc}") from exc
            except ValidationError as exc:
                # messages.parse validates eagerly and raises the raw pydantic
                # error itself — route it through the same validation-retry
                # budget instead of letting it crash the stage.
                last_error = exc
                continue
            self.ledger.record(stage, model, response.usage)
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "refusal":
                raise LLMError(f"{stage}: model refused the request")
            if stop_reason == "max_tokens":
                # Retrying would truncate again and re-bill the whole prefix.
                raise LLMError(
                    f"{stage}: structured output truncated at max_tokens={max_tokens} — "
                    "raise the token budget for this stage"
                )
            parsed = getattr(response, "parsed_output", None)
            if parsed is not None:
                return parsed
            # Fall back to manual validation of the text payload, then retry
            # with the validation error appended.
            text = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"),
                "",
            )
            try:
                return output_format.model_validate_json(text)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": text or "(empty)"},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed schema validation with this "
                            f"error:\n{exc}\n\nEmit the corrected output now, matching "
                            "the schema exactly."
                        ),
                    },
                ]
        raise LLMError(f"{stage}: structured output failed validation: {last_error}")

    # ---------------------------------------------------------------- agentic

    async def run_agentic(
        self,
        *,
        stage: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any] | ClientTool],
        model: str,
        effort: str,
        max_tokens: int,
        max_continuations: int,
        on_progress: Callable[[str], None] | None = None,
    ) -> AgenticResult:
        """Streamed agentic turn with server and client tools.

        Resumes on pause_turn; executes ClientTool calls locally and feeds the
        results (and their sources) back. Tool rounds and pause_turn resumes
        share one continuation budget so total turns stay bounded. Returns the
        full transcript (for a follow-up parse call on the same conversation)
        plus everything harvested for verification. The transcript is always
        left in a replayable state: on an early stop the trailing unresolved
        tool_use blocks are stripped.
        """
        api_tools: list[dict[str, Any]] = []
        handlers: dict[str, ClientTool] = {}
        for tool in tools:
            if isinstance(tool, ClientTool):
                api_tools.append(tool.spec)
                handlers[tool.name] = tool
            else:
                api_tools.append(tool)
        uses: dict[str, int] = dict.fromkeys(handlers, 0)

        transcript = list(messages)
        source_cache: dict[str, SourceDoc] = {}
        cited_quotes: list[CitedQuote] = []
        queries: list[str] = []
        engagement: dict[str, Any] = {}
        continuations = 0
        stop_reason: str | None = None
        truncated = False
        # The _20260209 web tools run code execution in a server-side
        # container. A continuation with pending code-execution tool uses
        # (pause_turn resume, or the tool_result follow-up when Claude mixed
        # client tools into the turn) MUST pass the container id back, or the
        # API rejects it: "container_id is required when there are pending
        # tool uses generated by code execution with tools."
        container_id: str | None = None

        while True:
            request_kwargs: dict[str, Any] = {}
            if container_id is not None:
                request_kwargs["container"] = container_id
            try:
                final = await _stream_request(
                    self.client,
                    model=model,
                    max_tokens=max_tokens,
                    system=_system_blocks(system),
                    messages=transcript,
                    tools=api_tools,
                    **thinking_kwargs(model, effort),
                    **request_kwargs,
                )
            except (
                anthropic.APIConnectionError,
                anthropic.APIStatusError,
                httpx.TransportError,
                TransientServerError,
                CircuitOpenError,
            ) as exc:
                raise LLMError(f"{stage}: API call failed: {exc}") from exc
            self.ledger.record(stage, model, final.usage)
            new_container_id = getattr(getattr(final, "container", None), "id", None)
            if new_container_id:
                container_id = new_container_id
            harvest_blocks(final.content, source_cache, cited_quotes, queries)
            if on_progress is not None:
                searches, _fetches = self.ledger.stage_counts(stage)
                on_progress(f"{len(source_cache)} pages cached · {searches} searches")
            stop_reason = final.stop_reason
            if stop_reason == "refusal":
                raise LLMError(f"{stage}: model refused the request")
            if (
                stop_reason == "pause_turn"
                and continuations < max_continuations
                and not self.ledger.over_budget
            ):
                # Pass content blocks back exactly as received (thinking included).
                transcript = transcript + [{"role": "assistant", "content": final.content}]
                continuations += 1
                continue
            if (
                stop_reason == "tool_use"
                and handlers
                and continuations < max_continuations
                and not self.ledger.over_budget
            ):
                # Budget checked BEFORE running handlers: executing tools whose
                # results are guaranteed to be discarded wastes real HTTP calls.
                results = await self._run_client_tools(
                    final.content, handlers, uses, source_cache, engagement
                )
                if results:
                    transcript = transcript + [
                        {"role": "assistant", "content": final.content},
                        {"role": "user", "content": results},
                    ]
                    continuations += 1
                    continue
                # Degenerate tool_use with no blocks: fall through to the
                # early-stop cleanup rather than leaving unresolved tool_use.
            # Terminal: either a clean end_turn, or an early stop (continuation
            # cap / budget while paused / max_tokens / unresolvable tool_use).
            # Early stops can leave a dangling tool_use that would poison the
            # follow-up parse call.
            truncated = stop_reason in ("pause_turn", "max_tokens", "tool_use")
            content = strip_dangling_tool_use(final.content) if truncated else final.content
            if content:
                transcript = transcript + [{"role": "assistant", "content": content}]
            break

        return AgenticResult(
            messages=transcript,
            final_stop_reason=stop_reason,
            source_cache=source_cache,
            cited_quotes=cited_quotes,
            continuations=continuations,
            truncated=truncated,
            queries=queries,
            engagement=engagement,
        )

    async def _run_client_tools(
        self,
        content: list[Any],
        handlers: dict[str, ClientTool],
        uses: dict[str, int],
        source_cache: dict[str, SourceDoc],
        engagement: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute client tool_use blocks; never raises — a broken tool
        degrades to an is_error result the model can route around."""
        results: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool = handlers.get(getattr(block, "name", ""))
            entry: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id}
            if tool is None:
                entry.update(content=f"unknown tool {block.name!r}", is_error=True)
            elif uses[tool.name] >= tool.max_uses:
                entry.update(
                    content="tool use budget exhausted — work with what you have",
                    is_error=True,
                )
            else:
                uses[tool.name] += 1
                try:
                    outcome: ToolOutcome = await tool.handler(dict(block.input or {}))
                except Exception as exc:  # degrade, never crash the researcher
                    outcome = ToolOutcome(content=f"tool failed: {exc}", error=True)
                for doc in outcome.sources:
                    source_cache[doc.url] = doc
                engagement.update(outcome.engagement)
                entry["content"] = outcome.content
                if outcome.error:
                    entry["is_error"] = True
            results.append(entry)
        return results

    # ----------------------------------------------------------------- stream

    async def stream_text(
        self,
        *,
        stage: str,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        effort: str,
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
    ) -> StreamResult:
        """Plain streamed generation returning the concatenated text + stop reason."""
        try:
            final = await _stream_text_request(
                self.client,
                on_text,
                {"chars": 0},
                model=model,
                max_tokens=max_tokens,
                system=_system_blocks(system),
                messages=messages,
                **thinking_kwargs(model, effort),
            )
        except (
            anthropic.APIConnectionError,
            anthropic.APIStatusError,
            httpx.TransportError,
            TransientServerError,
            CircuitOpenError,
        ) as exc:
            raise LLMError(f"{stage}: API call failed: {exc}") from exc
        self.ledger.record(stage, model, final.usage)
        if final.stop_reason == "refusal":
            raise LLMError(f"{stage}: model refused the request")
        text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
        return StreamResult(text=text, stop_reason=final.stop_reason)


def dump_for_prompt(model: BaseModel | list[BaseModel]) -> str:
    """Compact JSON for embedding pydantic payloads in user messages."""
    if isinstance(model, list):
        return json.dumps([m.model_dump() for m in model], ensure_ascii=False)
    return json.dumps(model.model_dump(), ensure_ascii=False)
