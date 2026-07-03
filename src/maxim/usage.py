"""Token/cost accounting across the run, and the budget gate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING,
    WEB_SEARCH_PER_1K_USD,
)
from .schemas import RunUsage, StageUsage


@dataclass
class _StageAccumulator:
    stage: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    web_fetches: int = 0

    def cost_usd(self) -> float:
        pricing = PRICING.get(self.model)
        if pricing is None:
            return 0.0
        in_price = pricing.input_per_mtok / 1_000_000
        out_price = pricing.output_per_mtok / 1_000_000
        cost = (
            self.input_tokens * in_price
            + self.output_tokens * out_price
            + self.cache_read_tokens * in_price * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * in_price * CACHE_WRITE_MULTIPLIER
            + self.web_searches * (WEB_SEARCH_PER_1K_USD / 1000)
        )
        return cost


@dataclass
class UsageLedger:
    budget_usd: float
    started_at: float = field(default_factory=time.monotonic)
    unknown_models: set[str] = field(default_factory=set)
    _stages: dict[tuple[str, str], _StageAccumulator] = field(default_factory=dict)

    def reset_clock(self) -> None:
        """Restart wall-clock measurement (call after the confirmation gate)."""
        self.started_at = time.monotonic()

    def record(self, stage: str, model: str, usage: Any) -> None:
        """Record a response's usage object (anthropic Usage or compatible)."""
        if model not in PRICING:
            # Cost estimates (and therefore the budget gate) are blind to this
            # model's spend; surfaced as a run warning by the orchestrator.
            self.unknown_models.add(model)
        key = (stage, model)
        acc = self._stages.get(key)
        if acc is None:
            acc = _StageAccumulator(stage=stage, model=model)
            self._stages[key] = acc
        acc.calls += 1
        acc.input_tokens += getattr(usage, "input_tokens", 0) or 0
        acc.output_tokens += getattr(usage, "output_tokens", 0) or 0
        acc.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        acc.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        server = getattr(usage, "server_tool_use", None)
        if server is not None:
            acc.web_searches += getattr(server, "web_search_requests", 0) or 0
            acc.web_fetches += getattr(server, "web_fetch_requests", 0) or 0

    @property
    def cost_usd(self) -> float:
        return sum(acc.cost_usd() for acc in self._stages.values())

    @property
    def over_budget(self) -> bool:
        return self.cost_usd >= self.budget_usd

    def stage_counts(self, stage: str) -> tuple[int, int]:
        """(web_searches, web_fetches) accumulated for a stage."""
        searches = sum(a.web_searches for (s, _), a in self._stages.items() if s == stage)
        fetches = sum(a.web_fetches for (s, _), a in self._stages.items() if s == stage)
        return searches, fetches

    def to_run_usage(self) -> RunUsage:
        stages = [
            StageUsage(
                stage=acc.stage,
                model=acc.model,
                calls=acc.calls,
                input_tokens=acc.input_tokens,
                output_tokens=acc.output_tokens,
                cache_read_tokens=acc.cache_read_tokens,
                cache_write_tokens=acc.cache_write_tokens,
                web_searches=acc.web_searches,
                web_fetches=acc.web_fetches,
                cost_usd=round(acc.cost_usd(), 4),
            )
            for acc in self._stages.values()
        ]
        return RunUsage(
            stages=stages,
            total_cost_usd=round(self.cost_usd, 4),
            wall_seconds=round(time.monotonic() - self.started_at, 1),
        )
