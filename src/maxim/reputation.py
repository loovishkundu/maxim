"""Source reputation tiers and recency scoring.

Deterministic, zero-LLM. Tiers are a *score*, not a filter — only the small
SEO-farm blocklist hard-blocks (wired into the server search tools). Everything
else degrades confidence rather than discarding evidence.

Tiers:
- A: peer-reviewed venues, arXiv, standards bodies.
- B: engineering blogs of known companies / research labs, official docs,
  conference talk pages.
- C: reputable general articles, individual blogs, unknown-but-plausible pages.
- D: forums and community threads (great for sentiment, weak for technical
  claims on their own).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from urllib.parse import urlparse

from .schemas import EvidenceKind

Tier = str  # "A" | "B" | "C" | "D"

# Hard-blocked at the search-tool level: content farms that pollute results.
BLOCKED_DOMAINS: list[str] = [
    "pinterest.com",
    "slideshare.net",
    "scribd.com",
    "coursehero.com",
    "chegg.com",
    "w3schools.blog",
]

_TIER_A = {
    "arxiv.org",
    "aclanthology.org",
    "acm.org",
    "dl.acm.org",
    "doi.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "jmlr.org",
    "nature.com",
    "neurips.cc",
    "nist.gov",
    "openreview.net",
    "proceedings.mlr.press",
    "sciencedirect.com",
    "semanticscholar.org",
    "springer.com",
    "usenix.org",
    "vldb.org",
}

_TIER_B = {
    "anthropic.com",
    "aws.amazon.com",
    "azure.microsoft.com",
    "blog.cloudflare.com",
    "cloud.google.com",
    "databricks.com",
    "deepmind.google",
    "developer.nvidia.com",
    "discord.com",
    "dropbox.tech",
    "duckdb.org",
    "elastic.co",
    "eng.uber.com",
    "engineering.atspotify.com",
    "engineering.fb.com",
    "github.blog",
    "grafana.com",
    "huggingface.co",
    "instagram-engineering.com",
    "kafka.apache.org",
    "martinfowler.com",
    "microsoft.com",
    "netflixtechblog.com",
    "openai.com",
    "pytorch.org",
    "redis.io",
    "research.google",
    "scikit-learn.org",
    "shopify.engineering",
    "slack.engineering",
    "stripe.com",
    "tensorflow.org",
    "postgresql.org",
    "python.org",
}

_TIER_D = {
    "dev.to",
    "hackernoon.com",
    "news.ycombinator.com",
    "quora.com",
    "reddit.com",
    "old.reddit.com",
    "stackexchange.com",
    "stackoverflow.com",
}

_ENG_BLOG_HOST = re.compile(r"^(eng|engineering|tech|research)\.")

# Method-knowledge decay per perspective (months). Stats fundamentals age
# slowly; agentic-AI churns monthly.
HALF_LIVES_MONTHS: dict[str, float] = {
    "ai_agentic": 12.0,
    "classical_ml": 36.0,
    "data_science": 36.0,
    "statistics": 120.0,
    "community": 12.0,
}


def _host(url: str) -> str:
    host = urlparse(url).netloc.casefold()
    return host.removeprefix("www.")


def _in(host: str, registry: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in registry)


def classify_source(url: str, kind: EvidenceKind) -> Tier:
    """Domain registry first, evidence-kind heuristics for unknowns."""
    host = _host(url)
    if not host:
        return "D"
    if _in(host, _TIER_D):
        return "D"
    if _in(host, _TIER_A):
        return "A"
    if _in(host, _TIER_B):
        return "B"
    if _in(host, {"github.com"}):
        # Repos/issues: community signal, not authority.
        return "D" if kind in ("anecdote", "production_report") else "C"
    if kind == "paper":
        return "A"
    if kind in ("docs", "talk"):
        return "B"
    if kind == "blog" and _ENG_BLOG_HOST.match(host):
        return "B"
    if kind in ("anecdote",):
        return "D"
    return "C"


_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%B %Y", "%b %Y")


def parse_published(raw: str | None) -> dt.date | None:
    if not raw:
        return None
    text = raw.strip()
    if "T" in text:
        # ISO-8601 timestamps ("2026-01-15T00:00:00Z") must not fall through
        # to the bare-year fallback, which would snap them to July 1.
        try:
            return dt.date.fromisoformat(text[:10])
        except ValueError:
            pass
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        return dt.date(int(match.group()), 7, 1)  # mid-year for bare years
    return None


def effective_half_life(perspective: str, horizon_months: int) -> float:
    """Blend the perspective's decay rate with the planner's topic horizon.

    Geometric mean keeps both signals: a fast-moving topic shortens even the
    statistics half-life; a stable topic stretches the AI one. Clamped to a
    sane range.
    """
    base = HALF_LIVES_MONTHS.get(perspective, 24.0)
    blended = math.sqrt(base * max(horizon_months, 1))
    return max(6.0, min(blended, 120.0))


def recency_score(
    published: dt.date | None,
    perspective: str,
    horizon_months: int,
    today: dt.date | None = None,
) -> float | None:
    """0.5 ** (age / half_life); None when the date is unknown."""
    if published is None:
        return None
    today = today or dt.date.today()
    age_months = max((today - published).days / 30.44, 0.0)
    half_life = effective_half_life(perspective, horizon_months)
    return round(0.5 ** (age_months / half_life), 3)


def stamp_evidence(evidence, perspective: str, horizon_months: int):
    """Return a copy of an Evidence with tier and recency_score stamped."""
    return evidence.model_copy(
        update={
            "tier": classify_source(evidence.source_url, evidence.kind),
            "recency_score": recency_score(
                parse_published(evidence.published), perspective, horizon_months
            ),
        }
    )


def tier_mix(tiers: list[Tier]) -> dict[Tier, float]:
    """Fraction of evidence per tier, e.g. {"A": 0.4, "B": 0.3, ...}."""
    if not tiers:
        return {}
    total = len(tiers)
    return {t: round(tiers.count(t) / total, 2) for t in ("A", "B", "C", "D") if t in tiers}


def tier_badge(mix: dict[Tier, float]) -> str:
    if not mix:
        return "no sources"
    return " ".join(f"{t}:{int(round(f * 100))}%" for t, f in mix.items())
