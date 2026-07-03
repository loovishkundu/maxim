"""The grounding critic: fresh-context, claim-vs-evidence-only judgment.

The critic never sees the researcher's conversation — only each claim, its
quotes, and an excerpt of the actual cached source text around the quote. That
isolation is deliberate: a critic reading the researcher's narrative gets
seduced by it.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM, SourceDoc, dump_for_prompt
from .prompts import CRITIC_SYSTEM
from .schemas import (
    Confidence,
    CritiqueResult,
    Finding,
    RejectedFinding,
    ResearchBrief,
    Verdict,
)
from .verification import excerpt_around

_UNAVAILABLE = "[source text unavailable — not mechanically verified]"


def build_payload(
    brief: ResearchBrief,
    findings: list[Finding],
    source_cache: dict[str, SourceDoc],
) -> str:
    lines: list[str] = [
        "Research brief sub-questions (for coverage_gaps):",
        dump_for_prompt(brief),
        "",
        "Findings to judge:",
    ]
    for finding in findings:
        lines.append(f"\n### {finding.id}")
        lines.append(f"CLAIM: {finding.claim}")
        lines.append(f"METHOD: {finding.method_name}")
        for i, ev in enumerate(finding.evidence, 1):
            lines.append(f'EVIDENCE {i} ({ev.kind}, {ev.source_url}): "{ev.quote}"')
            doc = source_cache.get(ev.source_url)
            excerpt = excerpt_around(ev.quote, doc.text) if doc else None
            if ev.status == "verified" and excerpt:
                lines.append(f"SOURCE CONTEXT: …{excerpt}…")
            elif excerpt:
                lines.append(f"SOURCE CONTEXT (quote NOT found verbatim here): …{excerpt}…")
            else:
                lines.append(f"SOURCE CONTEXT: {_UNAVAILABLE}")
    return "\n".join(lines)


async def critique(
    *,
    stage: str,
    brief: ResearchBrief,
    findings: list[Finding],
    source_cache: dict[str, SourceDoc],
    settings: Settings,
    llm: LLM,
) -> CritiqueResult:
    payload = build_payload(brief, findings, source_cache)
    return await llm.parse(
        stage=stage,
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": payload}],
        output_format=CritiqueResult,
        model=settings.critic_model,
        effort=settings.critic_effort,
    )


def _stamp_confidence(finding: Finding, verdict: Verdict) -> Confidence:
    """Deterministic rubric — the model never sets confidence.

    "high" demands the full chain: critic-supported + mechanically verified
    quote + a reputable (tier A/B) source. Unstamped tiers (None) count as
    unknown, not reputable.
    """
    any_verified = any(ev.status == "verified" for ev in finding.evidence)
    strong_source = any(
        ev.status == "verified" and ev.tier in ("A", "B") for ev in finding.evidence
    )
    if verdict == "supported":
        if strong_source:
            return "high"
        return "medium" if any_verified else "low"
    if verdict == "partially_supported":
        return "medium" if any_verified else "low"
    return "low"


def _normalize_id(finding_id: str) -> str:
    """Tolerate critic id drift like 'f-AI1' or bare 'ai1'."""
    norm = finding_id.strip().casefold()
    return norm.removeprefix("f-")


def apply_critique(
    findings: list[Finding],
    result: CritiqueResult,
) -> tuple[list[Finding], list[RejectedFinding]]:
    """Split findings into validated (confidence stamped) and rejected."""
    verdict_by_id: dict[str, tuple[Verdict, str | None]] = {
        _normalize_id(v.finding_id): (v.verdict, v.fix_hint) for v in result.verdicts
    }
    validated: list[Finding] = []
    rejected: list[RejectedFinding] = []
    for finding in findings:
        entry = verdict_by_id.get(_normalize_id(finding.id))
        unreviewed = entry is None
        verdict, fix_hint = entry or ("partially_supported", None)
        caveats = list(finding.caveats)
        if verdict in ("supported", "partially_supported"):
            if verdict == "partially_supported" and fix_hint:
                caveats.append(f"critic: {fix_hint}")
            if unreviewed:
                # Fail toward caution: the critic never judged this finding,
                # so it keeps its mechanical verification but no LLM approval.
                caveats.append("critic returned no verdict — claim is unreviewed")
            validated.append(
                finding.model_copy(
                    update={
                        "verdict": verdict,
                        "confidence": (
                            "low" if unreviewed else _stamp_confidence(finding, verdict)
                        ),
                        "caveats": caveats,
                    }
                )
            )
        else:
            reason = f"critic verdict: {verdict}"
            if fix_hint:
                reason += f" — {fix_hint}"
            rejected.append(
                RejectedFinding(
                    finding=finding.model_copy(update={"verdict": verdict}),
                    reason=reason,
                )
            )
    return validated, rejected
