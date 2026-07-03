"""Stage 1 — query understanding: topic → ResearchPlan."""

from __future__ import annotations

from .config import Settings
from .llm import LLM, LLMError
from .prompts import PLANNER_SYSTEM
from .schemas import PERSPECTIVES, ResearchPlan


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
