"""Stage 4 — synthesis: validated dossiers → the final report body.

Generated as streamed prose (structure enforced by the system prompt template),
constrained by the citation contract: the synthesizer may only cite finding
ids; report.py resolves them and flags anything unknown.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .config import Settings
from .llm import LLM, StreamResult
from .prompts import SYNTHESIZER_SYSTEM
from .schemas import PERSPECTIVE_LABELS, MethodPulse, ResearchDossier, ResearchPlan


def _findings_payload(dossier: ResearchDossier) -> list[dict]:
    return [
        {
            "id": f.id,
            "claim": f.claim,
            "method": f.method_name,
            "confidence": f.confidence,
            "verdict": f.verdict,
            "caveats": f.caveats,
            "evidence": [
                {
                    "quote": ev.quote,
                    "url": ev.source_url,
                    "title": ev.source_title,
                    "kind": ev.kind,
                    "published": ev.published,
                    "verified": ev.status == "verified",
                }
                for ev in f.evidence
            ],
        }
        for f in dossier.findings
    ]


def build_synthesis_input(
    plan: ResearchPlan,
    dossiers: list[ResearchDossier],
    canonical_methods: list[str] | None = None,
    pulse: list[MethodPulse] | None = None,
) -> str:
    parts: list[str] = [
        f"Topic: {plan.topic}",
        f"Domain: {plan.domain}",
        f"Planner rationale: {plan.rationale}",
        f"Assumptions made: {json.dumps(plan.assumptions)}",
    ]
    if canonical_methods:
        parts.append(
            "Canonical method names (use EXACTLY these spellings in the Method "
            f"Landscape table): {json.dumps(canonical_methods, ensure_ascii=False)}"
        )
    if pulse:
        parts.append(
            "Community pulse (MECHANICAL aggregates — sentiment and sample sizes "
            "are computed, not guessed; render insufficient_data as '–' and hedge "
            "by sample size):\n"
            + json.dumps([p.model_dump() for p in pulse], ensure_ascii=False, indent=1)
        )
    if plan.out_of_scope:
        parts.append(
            "Out-of-scope perspectives: "
            + "; ".join(f"{o.perspective}: {o.reason}" for o in plan.out_of_scope)
        )
    for dossier in dossiers:
        label = PERSPECTIVE_LABELS[dossier.perspective]
        parts.append(f"\n===== {label} dossier =====")
        if not dossier.ok:
            parts.append(f"THIS AGENT FAILED ({dossier.failure}) — no findings. Say so honestly.")
            continue
        parts.append(f"Summary: {dossier.summary}")
        if dossier.gaps:
            parts.append(f"Gaps (unanswered): {json.dumps(dossier.gaps)}")
        if dossier.rejected:
            parts.append(
                f"{len(dossier.rejected)} finding(s) were REJECTED by verification/critique "
                "— do not use them; mention the rejection count in Caveats."
            )
        parts.append(
            "Validated findings (cite by id):\n"
            + json.dumps(_findings_payload(dossier), ensure_ascii=False, indent=1)
        )
    parts.append("\nWrite the full report now, following the template.")
    return "\n".join(parts)


async def synthesize(
    plan: ResearchPlan,
    dossiers: list[ResearchDossier],
    settings: Settings,
    llm: LLM,
    canonical_methods: list[str] | None = None,
    pulse: list[MethodPulse] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> StreamResult:
    return await llm.stream_text(
        stage="synthesizer",
        system=SYNTHESIZER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": build_synthesis_input(plan, dossiers, canonical_methods, pulse),
            }
        ],
        model=settings.synthesizer_model,
        effort=settings.synthesizer_effort,
        max_tokens=settings.preset.synthesis_max_tokens,
        on_text=on_text,
    )
