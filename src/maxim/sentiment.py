"""Community-sentiment rigor: mechanical, zero-LLM.

HN/Reddit/GitHub are noisy, so community findings clear extra bars:

- Engagement floors — evidence whose thread demonstrably has no traction
  (from tool metadata) does not count toward corroboration. Evidence without
  metadata (plain web fetches) is not penalized; the floors exist to filter
  known drive-by threads, not to punish unknown ones.
- Corroboration — a sentiment claim needs ≥2 distinct qualifying threads,
  else the sentiment is stripped and the finding survives only as a
  single-anecdote observation.
- Honesty — per method, fewer than 3 independent threads renders the pulse
  as insufficient_data, never a fabricated verdict. Sample sizes are always
  stamped so the synthesizer can hedge.
"""

from __future__ import annotations

from .schemas import EngagementStats, Finding, MethodPulse, Sentiment

# Floors per source (PLAN §6). Below → the thread doesn't corroborate.
_HN_MIN_POINTS = 10
_HN_MIN_COMMENTS = 5
_GITHUB_MIN_REACTIONS = 3
_GITHUB_MIN_COMMENTS = 5  # proxy for maintainer engagement, invisible to search
_REDDIT_MIN_POINTS = 20

MIN_CORROBORATION_THREADS = 2
MIN_PULSE_THREADS = 3


def meets_floor(stats: EngagementStats | None) -> bool:
    """False only when metadata is present AND below its source's floor."""
    if stats is None:
        return True
    if stats.source == "hn":
        return (stats.points or 0) >= _HN_MIN_POINTS or (stats.comments or 0) >= _HN_MIN_COMMENTS
    if stats.source == "github":
        return (stats.reactions or 0) >= _GITHUB_MIN_REACTIONS or (
            stats.comments or 0
        ) >= _GITHUB_MIN_COMMENTS
    if stats.source == "reddit":
        return (stats.points or 0) >= _REDDIT_MIN_POINTS
    return True


def _qualifying_threads(finding: Finding) -> set[str]:
    """Distinct source URLs that can corroborate sentiment: not mechanically
    failed, and not below a known engagement floor."""
    return {
        ev.source_url
        for ev in finding.evidence
        if ev.status != "failed" and meets_floor(ev.engagement)
    }


def apply_sentiment_rigor(findings: list[Finding]) -> list[Finding]:
    """Stamp sample sizes; strip sentiment lacking corroboration."""
    out: list[Finding] = []
    for finding in findings:
        if finding.perspective != "community":
            out.append(finding)
            continue
        sample = len(_qualifying_threads(finding))
        update: dict = {"sentiment_sample_size": sample}
        if finding.sentiment is not None and sample < MIN_CORROBORATION_THREADS:
            update["sentiment"] = None
            update["caveats"] = finding.caveats + [
                f"sentiment demoted: only {sample} qualifying thread(s) — "
                "kept as a single-anecdote observation"
            ]
        out.append(finding.model_copy(update=update))
    return out


def _majority_sentiment(sentiments: list[str]) -> Sentiment:
    counts = {s: sentiments.count(s) for s in set(sentiments)}
    best = max(counts.values())
    winners = [s for s, n in counts.items() if n == best]
    if len(winners) == 1:
        return winners[0]  # type: ignore[return-value]
    return "mixed"


def build_pulse(findings: list[Finding]) -> list[MethodPulse]:
    """Aggregate community findings into per-method pulses."""
    community = [f for f in findings if f.perspective == "community"]
    by_method: dict[str, list[Finding]] = {}
    for finding in community:
        by_method.setdefault(finding.method_name, []).append(finding)

    pulses: list[MethodPulse] = []
    for method, group in by_method.items():
        threads: dict[str, str] = {}  # url → display line
        for finding in group:
            for ev in finding.evidence:
                if ev.status != "failed" and meets_floor(ev.engagement):
                    threads.setdefault(ev.source_url, f"{ev.source_title} — {ev.source_url}")
        sentiments = [f.sentiment for f in group if f.sentiment is not None]
        if len(threads) < MIN_PULSE_THREADS or not sentiments:
            sentiment: Sentiment = "insufficient_data"
        else:
            sentiment = _majority_sentiment(sentiments)
        how: list[str] = []
        seen = set()
        for finding in group:
            for item in finding.how_people_test_it:
                if item.casefold() not in seen:
                    seen.add(item.casefold())
                    how.append(item)
        pulses.append(
            MethodPulse(
                method=method,
                sentiment=sentiment,
                sample_size=len(threads),
                notable_threads=sorted(threads.values())[:4],
                how_people_test_it=how,
            )
        )
    return sorted(pulses, key=lambda p: p.method.casefold())
