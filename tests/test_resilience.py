"""The API resilience layer: retry transient errors, never 4xx, break fast."""

import anthropic
import httpx
import pytest

import maxim.llm as llm_module
from maxim.config import Settings
from maxim.llm import LLM, LLMError
from maxim.resilience import CircuitOpenError, api_resilient
from maxim.usage import UsageLedger

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_REQUEST)


def bad_request() -> anthropic.BadRequestError:
    return anthropic.BadRequestError(
        "container_id is required",
        response=httpx.Response(400, request=_REQUEST),
        body=None,
    )


def flaky(failures, exc_factory):
    calls = {"n": 0}

    @api_resilient(delay=0.001)
    async def call():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc_factory()
        return "ok"

    return call, calls


async def test_transient_errors_are_retried():
    call, calls = flaky(2, connection_error)
    assert await call() == "ok"
    assert calls["n"] == 3  # two failures + success


async def test_client_errors_are_never_retried():
    call, calls = flaky(5, bad_request)
    with pytest.raises(anthropic.BadRequestError):
        await call()
    assert calls["n"] == 1  # a 400 is a bug, not a blip


async def test_exhausted_retries_reraise_the_transient_error():
    call, calls = flaky(10, connection_error)
    with pytest.raises(anthropic.APIConnectionError):
        await call()
    assert calls["n"] == 3  # max_attempts


async def test_circuit_opens_after_sustained_failures():
    call, calls = flaky(100, connection_error)
    for _ in range(2):  # 2 calls x 3 attempts = 6 failures = threshold
        with pytest.raises(anthropic.APIConnectionError):
            await call()
    before = calls["n"]
    with pytest.raises(CircuitOpenError):
        await call()
    assert calls["n"] == before  # breaker open: no real call was made


async def test_llm_layer_converts_circuit_open_to_llmerror(monkeypatch):
    async def open_circuit(client, **kwargs):
        raise CircuitOpenError("circuit open")

    monkeypatch.setattr(llm_module, "_parse_request", open_circuit)
    llm = LLM(settings=Settings(), ledger=UsageLedger(budget_usd=10.0), client=object())
    with pytest.raises(LLMError, match="API call failed"):
        await llm.parse(
            stage="planner",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            output_format=None.__class__,
            model="claude-opus-4-8",
            effort="low",
        )


def mid_stream_drop() -> httpx.ReadError:
    # What a connection reset during SSE body iteration actually raises —
    # the SDK's exception wrapping covers only the initial send().
    return httpx.ReadError("Connection reset by peer", request=_REQUEST)


def overloaded_status() -> anthropic.APIStatusError:
    # Mid-stream `error` SSE events arrive on a live 200 response, so the SDK
    # raises the generic base class, not InternalServerError.
    return anthropic.APIStatusError(
        "overloaded",
        response=httpx.Response(200, request=_REQUEST),
        body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )


async def test_mid_stream_transport_errors_are_retried():
    call, calls = flaky(2, mid_stream_drop)
    assert await call() == "ok"
    assert calls["n"] == 3


async def test_reclassified_overload_is_retried():
    from maxim.resilience import reraise_if_transient_status

    calls = {"n": 0}

    @api_resilient(delay=0.001)
    async def call():
        calls["n"] += 1
        if calls["n"] <= 2:
            try:
                raise overloaded_status()
            except anthropic.APIStatusError as exc:
                reraise_if_transient_status(exc)
                raise
        return "ok"

    assert await call() == "ok"
    assert calls["n"] == 3


def test_typed_4xx_is_never_reclassified():
    from maxim.resilience import reraise_if_transient_status

    # BadRequestError is a typed subclass — reclassification must not touch it.
    reraise_if_transient_status(bad_request())  # no raise


async def test_persistent_transport_error_becomes_llmerror(monkeypatch):
    async def dead_stream(client, on_text, emitted, **kwargs):
        raise mid_stream_drop()

    monkeypatch.setattr(llm_module, "_stream_text_request", dead_stream)
    llm = LLM(settings=Settings(), ledger=UsageLedger(budget_usd=10.0), client=object())
    with pytest.raises(LLMError, match="API call failed"):
        await llm.stream_text(
            stage="synthesizer",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            model="claude-sonnet-5",
            effort="high",
            max_tokens=100,
        )


async def test_parse_validation_error_uses_the_retry_budget(monkeypatch):
    # messages.parse validates eagerly and can raise raw pydantic
    # ValidationError — it must consume the validation-retry budget and end
    # as LLMError, never crash the stage.
    from pydantic import BaseModel, TypeAdapter

    class Expected(BaseModel):
        value: int

    def validation_error():
        try:
            TypeAdapter(Expected).validate_json('{"wrong": true}')
        except Exception as exc:
            return exc

    calls = {"n": 0}

    async def raising_parse(client, **kwargs):
        calls["n"] += 1
        raise validation_error()

    monkeypatch.setattr(llm_module, "_parse_request", raising_parse)
    llm = LLM(settings=Settings(), ledger=UsageLedger(budget_usd=10.0), client=object())
    with pytest.raises(LLMError, match="failed validation"):
        await llm.parse(
            stage="canonicalizer",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            output_format=Expected,
            model="claude-opus-4-8",
            effort="low",
        )
    assert calls["n"] == Settings().max_parse_retries + 1
