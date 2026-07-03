"""Shared HTTP plumbing for the client-side research tools.

Every tool degrades gracefully: no API key required for any of them (keys
only raise rate limits), and failures surface as is_error tool results the
model can route around — never as researcher crashes.
"""

from __future__ import annotations

from typing import Any

import httpx

TIMEOUT_S = 15.0


def _make_client() -> httpx.AsyncClient:
    # Separated for tests, which swap in a MockTransport-backed client.
    return httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True)


async def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    async with _make_client() as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    async with _make_client() as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text


def truncate(text: str, limit: int = 1500) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
