"""Deterministic loop routing for the researcher state machine.

The decision to retry / re-validate / replan is a pure function of critic
verdicts and mechanical checks — no LLM judgment. Priority when several
triggers fire: REPLAN (structural failure needs a new angle, more evidence for
a bad plan is waste) > RETRY (better evidence for flagged claims) >
RE-VALIDATE (mechanical repair only). A trigger whose cap is exhausted falls
through to the next priority rather than blocking it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .config import LoopPolicy

Action = Literal["accept", "replan", "retry", "revalidate"]


@dataclass(frozen=True)
class LoopState:
    """How many of each loop this researcher has already spent."""

    evidence_retries: int = 0
    revalidates: int = 0
    replans: int = 0

    def spend(self, action: Action) -> LoopState:
        if action == "retry":
            return LoopState(self.evidence_retries + 1, self.revalidates, self.replans)
        if action == "revalidate":
            return LoopState(self.evidence_retries, self.revalidates + 1, self.replans)
        if action == "replan":
            return LoopState(self.evidence_retries, self.revalidates, self.replans + 1)
        return self


@dataclass(frozen=True)
class IterationOutcome:
    """What one draft→verify→critique pass produced, as counts.

    weak: findings kept but only partially_supported (a fix_hint could rescue).
    unsupported: critic-rejected (unsupported / contradicted / source_unreliable).
    mechanical_failed: rejected before the critic — quote provably absent from
    its fetched source.
    """

    drafted: int
    validated: int
    weak: int
    unsupported: int
    mechanical_failed: int
    coverage_gaps: int


@dataclass(frozen=True)
class Decision:
    action: Action
    reasons: list[str] = field(default_factory=list)
    # True when a repair trigger fired but its cap was spent: the dossier is
    # accepted with known pending work, which callers must surface (gap note,
    # confidence cap) rather than passing off as healthy.
    degraded: bool = False


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def decide(outcome: IterationOutcome, state: LoopState, policy: LoopPolicy) -> Decision:
    """Route the next loop iteration from one pass's outcome. Deterministic."""
    reasons: list[str] = []

    unsupported_ratio = _ratio(outcome.unsupported, outcome.drafted)
    weak_ratio = _ratio(outcome.weak + outcome.unsupported, outcome.drafted)
    mechanical_ratio = _ratio(outcome.mechanical_failed, outcome.drafted)

    structural = []
    if outcome.validated < policy.min_findings:
        structural.append(
            f"only {outcome.validated} validated findings (need {policy.min_findings})"
        )
    if unsupported_ratio > policy.replan_unsupported_ratio:
        structural.append(f"{unsupported_ratio:.0%} of drafted claims unsupported")
    if outcome.coverage_gaps >= policy.replan_coverage_gaps:
        structural.append(f"{outcome.coverage_gaps} coverage gaps against the brief")

    if structural:
        if state.replans < policy.max_replans:
            return Decision("replan", structural)
        reasons.append("structural failure but replan cap spent: " + "; ".join(structural))

    if weak_ratio > policy.retry_weak_ratio and (outcome.weak + outcome.unsupported) > 0:
        if state.evidence_retries < policy.max_evidence_retries:
            reasons.append(
                f"{weak_ratio:.0%} of drafted claims weak/unsupported — retry with fix hints"
            )
            return Decision("retry", reasons)
        reasons.append("weak evidence but retry cap spent")

    if outcome.mechanical_failed > 0 and mechanical_ratio <= policy.revalidate_mechanical_ratio:
        if state.revalidates < policy.max_revalidates:
            reasons.append(
                f"{outcome.mechanical_failed} quote(s) failed mechanical verification — "
                "targeted re-fetch"
            )
            return Decision("revalidate", reasons)
        reasons.append("mechanical failures but re-validate cap spent")

    degraded = bool(reasons)
    reasons.append("accepting dossier" + (" (loop caps exhausted)" if degraded else ""))
    return Decision("accept", reasons, degraded=degraded)
