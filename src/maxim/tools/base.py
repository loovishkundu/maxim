"""Shared HTTP plumbing for the client-side research tools.

Every tool degrades gracefully: no API key required for any of them (keys
only raise rate limits), and failures surface as is_error tool results the
model can route around — never as researcher crashes.

Transient failures (dropped connections, timeouts, 429, 5xx) retry with
backoff before giving up; other 4xx are terminal and propagate immediately.
No circuit breaker here: the four tools share this layer, and one dead host
must not block the others — a persistently failing tool already degrades to
capped is_error results.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..resilience import transient_resilient

TIMEOUT_S = 15.0

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class TransientHTTPError(Exception):
    """A retryable HTTP failure: worth another attempt after backoff."""


def _make_client() -> httpx.AsyncClient:
    # Separated for tests, which swap in a MockTransport-backed client.
    return httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True)


@transient_resilient(
    (httpx.TransportError, TransientHTTPError),
    max_attempts=3,
    delay=0.25,
    max_delay=5.0,
    failure_threshold=None,
)
async def _request(
    url: str,
    params: dict[str, Any] | None,
    headers: dict[str, str] | None,
) -> httpx.Response:
    async with _make_client() as client:
        response = await client.get(url, params=params, headers=headers)
    if response.status_code in _RETRYABLE_STATUSES:
        raise TransientHTTPError(f"{response.status_code} from {response.url.host}")
    response.raise_for_status()  # remaining 4xx are terminal, not retried
    return response


async def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    return (await _request(url, params, headers)).json()


async def get_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    return (await _request(url, params, headers)).text


def truncate(text: str, limit: int = 1500) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
