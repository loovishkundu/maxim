"""Stage 1 — query understanding: topic → ResearchPlan; per-task replanning."""

from __future__ import annotations

import json

from .config import Settings
from .llm import LLM, LLMError, dump_for_prompt
from .prompts import PLANNER_SYSTEM, REPLANNER_SYSTEM
from .schemas import PERSPECTIVES, Finding, RejectedFinding, ResearchBrief, ResearchPlan


async def make_plan(topic: str, settings: Settings, llm: LLM) -> ResearchPlan:
    scope_note = ""
    if settings.perspectives:
        allowed = ", ".join(settings.perspectives)
        scope_note = (
            f"\n\nThe user restricted this run to these perspectives only: {allowed}. "
            "Produce briefs only for those; list the rest in out_of_scope with reason "
            '"excluded by user".'
        )
    user = (
        f"Topic to research:\n\n{topic}\n\n"
        f"Canonical perspectives: {', '.join(PERSPECTIVES)}.{scope_note}"
    )
    plan = await llm.parse(
        stage="planner",
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=ResearchPlan,
        model=settings.planner_model,
        effort=settings.planner_effort,
    )
    if settings.perspectives:
        plan.briefs = [b for b in plan.briefs if b.perspective in settings.perspectives]
    # Dedupe: the planner is an LLM and may emit two briefs for one perspective,
    # which would collide finding ids (F-<prefix><n>) and merge ledger stages.
    seen: set[str] = set()
    unique = []
    for brief in plan.briefs:
        if brief.perspective not in seen:
            seen.add(brief.perspective)
            unique.append(brief)
    plan.briefs = unique
    if not plan.briefs:
        raise LLMError("planner produced no in-scope research briefs")
    return plan


async def replan_task(
    brief: ResearchBrief,
    plan: ResearchPlan,
    *,
    reasons: list[str],
    queries_tried: list[str],
    validated: list[Finding],
    rejected: list[RejectedFinding],
    settings: Settings,
    llm: LLM,
) -> ResearchBrief:
    """Produce a revised brief for one perspective after a structural failure.

    Seeded with what the failed pass already tried (queries — don't repeat),
    what it validated (locked in — don't re-cover), and what got rejected and
    why (steer toward groundable sources).
    """
    user = "\n".join(
        [
            f"Topic: {plan.topic}",
            f"Domain: {plan.domain}",
            f"Original brief (JSON):\n{dump_for_prompt(brief)}",
            "",
            f"Structural failure: {'; '.join(reasons)}",
            f"Queries already tried (do not repeat): {json.dumps(queries_tried)}",
            "Already-validated claims (locked in, do not re-cover): "
            + json.dumps([f.claim for f in validated]),
            "Rejected claims and why: "
            + json.dumps([f"{r.finding.claim} — {r.reason}" for r in rejected]),
            "",
            "Write the revised brief now.",
        ]
    )
    revised = await llm.parse(
        stage=f"replanner:{brief.perspective}",
        system=REPLANNER_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=ResearchBrief,
        model=settings.planner_model,
        effort=settings.planner_effort,
    )
    # The perspective is not the model's to change: ids, prompts, and report
    # sections are all keyed on it.
    if revised.perspective != brief.perspective:
        revised = revised.model_copy(update={"perspective": brief.perspective})
    return revised
