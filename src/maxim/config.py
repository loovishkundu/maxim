"""Run configuration: models, effort levels, depth presets, pricing estimates.

Every numeric budget in the pipeline lives here so behavior is tunable in one
place. Pricing figures are ESTIMATES for the cost display — they are not billing
truth and will drift; treat the printed cost as an order-of-magnitude guide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


# Estimates (USD per million tokens). Update as published pricing changes.
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-8": ModelPricing(5.00, 25.00),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25
WEB_SEARCH_PER_1K_USD = 10.0


@dataclass(frozen=True)
class LoopPolicy:
    """Pass-gate thresholds and loop caps for one depth.

    Per-depth values, not constants: niche perspectives can be legitimately
    thin, and hard gates would burn budget on futile retries. Ratios are over
    the number of drafted findings in an iteration.
    """

    min_findings: int
    max_evidence_retries: int
    max_revalidates: int
    max_replans: int
    retry_weak_ratio: float = 0.2  # weak+unsupported share above this → RETRY
    replan_unsupported_ratio: float = 0.5  # unsupported share above this → REPLAN
    revalidate_mechanical_ratio: float = 0.3  # mechanical failures at/below this → RE-VALIDATE
    replan_coverage_gaps: int = 2  # gaps at/above this → REPLAN


@dataclass(frozen=True)
class DepthPreset:
    """Per-depth budget knobs. All hard caps enforced in code, not prompts."""

    researcher_effort: str
    web_search_max_uses: int
    web_fetch_max_uses: int
    max_continuations: int
    researcher_timeout_s: float
    gather_max_tokens: int
    synthesis_max_tokens: int
    loop: LoopPolicy


DEPTHS: dict[str, DepthPreset] = {
    "quick": DepthPreset(
        researcher_effort="low",
        web_search_max_uses=5,
        web_fetch_max_uses=4,
        max_continuations=3,
        researcher_timeout_s=300.0,
        gather_max_tokens=8_000,
        synthesis_max_tokens=16_000,
        loop=LoopPolicy(min_findings=2, max_evidence_retries=1, max_revalidates=1, max_replans=0),
    ),
    "standard": DepthPreset(
        researcher_effort="medium",
        web_search_max_uses=10,
        web_fetch_max_uses=7,
        max_continuations=6,
        researcher_timeout_s=600.0,
        gather_max_tokens=12_000,
        synthesis_max_tokens=24_000,
        loop=LoopPolicy(min_findings=3, max_evidence_retries=2, max_revalidates=2, max_replans=1),
    ),
    "deep": DepthPreset(
        researcher_effort="high",
        web_search_max_uses=20,
        web_fetch_max_uses=14,
        max_continuations=8,
        researcher_timeout_s=900.0,
        gather_max_tokens=20_000,
        synthesis_max_tokens=32_000,
        loop=LoopPolicy(min_findings=4, max_evidence_retries=2, max_revalidates=2, max_replans=1),
    ),
}


@dataclass
class Settings:
    depth: str = "standard"
    perspectives: list[str] | None = None  # None = whatever the planner scopes in
    budget_usd: float = 10.0
    max_concurrency: int = 3
    out_dir: Path = field(default_factory=lambda: Path("./maxim-reports"))
    out: Path | None = None  # explicit report file path; overrides out_dir naming
    assume_yes: bool = False
    quiet: bool = False
    json_output: bool = False

    planner_model: str = "claude-opus-4-8"
    researcher_model: str = "claude-opus-4-8"
    critic_model: str = "claude-opus-4-8"
    synthesizer_model: str = "claude-opus-4-8"

    planner_effort: str = "medium"
    critic_effort: str = "low"
    synthesizer_effort: str = "high"

    max_parse_retries: int = 2
    web_fetch_max_content_tokens: int = 25_000

    @property
    def preset(self) -> DepthPreset:
        return DEPTHS[self.depth]
