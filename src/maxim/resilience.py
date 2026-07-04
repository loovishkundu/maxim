"""API-call resilience via pyresilience: retry transient failures, never 4xx.

The anthropic SDK already retries transport blips internally (max_retries);
this layer sits above it for sustained trouble — connection drops that
outlive the SDK's retries, 429s, 5xx/529 overloads — with exponential
backoff + jitter. A circuit breaker per call type fails remaining calls
fast once the API is clearly down, so a broken run dies in seconds instead
of grinding through every researcher's full timeout.

Client errors (4xx) are protocol or configuration bugs: retrying cannot fix
them, so they are never retried and they never trip the breaker.
"""

from __future__ import annotations

import anthropic
import httpx
from pyresilience import CircuitBreakerConfig, CircuitOpenError, RetryConfig, resilient

__all__ = [
    "TRANSIENT_API_ERRORS",
    "CircuitOpenError",
    "TransientServerError",
    "api_resilient",
    "reraise_if_transient_status",
    "transient_resilient",
]


class TransientServerError(Exception):
    """A server-side failure surfaced with a shape retry_on can't see.

    Mid-stream `error` SSE events (e.g. overloaded_error) arrive on a live
    200 response, so the SDK raises the generic APIStatusError base class —
    not InternalServerError. Reclassified here so the retry layer treats
    them like the 5xx they really are."""


# Worth retrying above the SDK's own retry layer. RateLimitError (429) and
# InternalServerError (5xx, incl. 529 overloaded) subclass APIStatusError;
# APIConnectionError covers failures around request send. httpx.TransportError
# covers drops DURING stream body iteration, which the SDK does not wrap
# (its exception mapping surrounds only the initial send).
TRANSIENT_API_ERRORS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    httpx.TransportError,
    TransientServerError,
)

# API error-body types that are transient regardless of the carrier status.
_TRANSIENT_ERROR_TYPES = frozenset({"overloaded_error", "api_error"})


def reraise_if_transient_status(exc: anthropic.APIStatusError) -> None:
    """Reclassify a generic APIStatusError whose body says 'overloaded'.

    Only exact base-class instances are reclassified — typed subclasses
    (BadRequestError, RateLimitError, ...) already carry their meaning."""
    if type(exc) is anthropic.APIStatusError and getattr(exc, "type", None) in (
        _TRANSIENT_ERROR_TYPES
    ):
        raise TransientServerError(str(exc)) from exc


def transient_resilient(
    retry_on: tuple[type[BaseException], ...],
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 20.0,
    failure_threshold: int | None = 6,
    recovery_timeout: float = 30.0,
):
    """Decorator factory: retry the given transient errors; optional breaker.

    A factory (not a shared decorator) so every decorated function owns its
    own breaker state — parse calls tripping must not open the gather
    breaker, and tests can build isolated instances with tiny delays. Pass
    failure_threshold=None to skip the breaker (used for the external tool
    APIs, where one dead host must not block the others sharing a policy).
    """
    breaker = (
        CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            error_types=retry_on,
        )
        if failure_threshold is not None
        else None
    )
    return resilient(
        retry=RetryConfig(
            max_attempts=max_attempts,
            delay=delay,
            backoff_factor=2.0,
            max_delay=max_delay,
            jitter=True,
            retry_on=retry_on,
        ),
        circuit_breaker=breaker,
    )


def api_resilient(
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    failure_threshold: int = 6,
    recovery_timeout: float = 30.0,
):
    """Resilience policy for Anthropic API calls."""
    return transient_resilient(
        TRANSIENT_API_ERRORS,
        max_attempts=max_attempts,
        delay=delay,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
