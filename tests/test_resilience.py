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
