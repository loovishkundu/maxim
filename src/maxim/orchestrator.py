"""Pipeline glue: plan → confirm → two-wave fan-out → synthesize → report.

Wave 1 runs the technical perspectives in parallel; their method names are
then canonicalized (one cheap call) and the community researcher runs as
wave 2, seeded with the union of methods wave 1 ACTUALLY found — sentiment
about methods nobody surfaced is noise.

`run_pipeline` is the single programmatic seam (no global state, no printing —
progress goes through a callback), which is what keeps the future Claude skill
and MCP wrapper cheap.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .config import Settings
from .llm import LLM, LLMError
from .methods import apply_canonical_names, canonical_names, canonicalize_methods
from .planner import make_plan
from .report import assemble_report, fallback_report
from .researcher import run_researcher, stub_dossier
from .schemas import (
    SCHEMA_VERSION,
    MethodPulse,
    ResearchBrief,
    ResearchDossier,
    ResearchPlan,
    RunResult,
)
from .sentiment import build_pulse
from .synthesizer import synthesize
from .usage import UsageLedger

ProgressFn = Callable[[str, str], None]
ConfirmFn = Callable[[ResearchPlan], bool]

# Researchers self-terminate at a soft deadline and salvage partial findings;
# the hard wait_for backstop only fires if one hangs past the grace window.
TIMEOUT_GRACE_S = 60.0


class PlanRejected(Exception):
    """The user declined the plan at the confirmation gate."""


async def run_pipeline(
    topic: str,
    settings: Settings,
    progress: ProgressFn,
    confirm: ConfirmFn,
    on_synthesis_text: Callable[[str], None] | None = None,
) -> RunResult:
    ledger = UsageLedger(budget_usd=settings.budget_usd)
    llm = LLM(settings=settings, ledger=ledger)
    warnings: list[str] = []
    try:
        progress("planner", "classifying topic and scoping perspectives…")
        plan = await make_plan(topic, settings, llm)
        progress("planner", f"{len(plan.briefs)} perspectives in scope")

        if not confirm(plan):
            raise PlanRejected
        # Wall-clock should measure the pipeline, not how long the user stared
        # at the confirmation prompt.
        ledger.reset_clock()

        semaphore = asyncio.Semaphore(settings.max_concurrency)
        timeout = settings.preset.researcher_timeout_s

        async def guarded(brief: ResearchBrief) -> ResearchDossier:
            async with semaphore:
                label = f"researcher:{brief.perspective}"
                if ledger.over_budget:
                    warnings.append(f"{brief.perspective}: skipped — budget exhausted")
                    return stub_dossier(brief, "skipped: budget exhausted")
                try:
                    dossier = await asyncio.wait_for(
                        run_researcher(
                            brief,
                            plan,
                            settings,
                            llm,
                            progress=lambda msg, _l=label: progress(_l, msg),
                            deadline=time.monotonic() + timeout,
                        ),
                        timeout=timeout + TIMEOUT_GRACE_S,
                    )
                    progress(
                        label,
                        f"done — {len(dossier.findings)} validated, "
                        f"{len(dossier.rejected)} rejected",
                    )
                    return dossier
                except TimeoutError:
                    # The soft deadline should have salvaged partial results;
                    # reaching this backstop means a call hung outright.
                    warnings.append(
                        f"{brief.perspective}: hung past {timeout:.0f}s + grace and was "
                        "cancelled — the cancelled call's usage is not counted; real "
                        "spend may exceed the estimate"
                    )
                    progress(label, "timed out")
                    searches, fetches = ledger.stage_counts(label)
                    return stub_dossier(brief, f"timed out after {timeout:.0f}s", searches, fetches)
                except Exception as exc:  # degrade, never die: synthesis still runs
                    warnings.append(f"{brief.perspective}: failed — {exc}")
                    progress(label, f"failed: {exc}")
                    searches, fetches = ledger.stage_counts(label)
                    return stub_dossier(brief, str(exc), searches, fetches)

        wave1_briefs = [b for b in plan.briefs if b.perspective != "community"]
        community_brief = next((b for b in plan.briefs if b.perspective == "community"), None)

        dossiers = list(await asyncio.gather(*(guarded(b) for b in wave1_briefs)))

        # Canonicalize method names across wave 1 so the landscape table and
        # the community wave speak one vocabulary.
        methods_union = [m for d in dossiers for m in d.methods_identified]
        mapping: dict[str, str] = {}
        if methods_union:
            progress("canonicalizer", f"normalizing {len(set(methods_union))} method names…")
            mapping = await canonicalize_methods(methods_union, settings, llm)
            dossiers = [apply_canonical_names(d, mapping) for d in dossiers]
        canonical = canonical_names(mapping)

        pulse: list[MethodPulse] = []
        if community_brief is not None:
            if canonical:
                # Seed wave 2 with what wave 1 actually found, not planner
                # guesses; the planner's own seeds stay as a fallback when
                # wave 1 surfaced nothing.
                community_brief = community_brief.model_copy(
                    update={"must_cover_methods": canonical}
                )
            community_dossier = apply_canonical_names(await guarded(community_brief), mapping)
            dossiers.append(community_dossier)
            pulse = build_pulse(community_dossier.findings)

        if ledger.unknown_models:
            models = ", ".join(sorted(ledger.unknown_models))
            warnings.append(
                f"no pricing configured for model(s) {models} — cost estimates and the "
                "budget gate cannot see that spend"
            )

        any_findings = any(d.findings for d in dossiers)
        synthesis_failed = False
        fallback_reason: str | None = None
        if not any_findings:
            fallback_reason = "no validated findings from any researcher"
        elif ledger.over_budget:
            fallback_reason = f"budget (${settings.budget_usd:.2f}) exhausted before synthesis"

        if fallback_reason is None:
            progress("synthesizer", "writing report…")
            try:
                synthesis = await synthesize(
                    plan,
                    dossiers,
                    settings,
                    llm,
                    canonical_methods=canonical,
                    pulse=pulse,
                    on_text=on_synthesis_text,
                )
            except LLMError as exc:
                # The research is already paid for — degrade to the raw dump
                # rather than losing the run.
                synthesis_failed = True
                fallback_reason = f"synthesis failed: {exc}"
            else:
                if synthesis.truncated:
                    synthesis_failed = True
                    warnings.append(
                        "synthesis hit its token cap — the report body is truncated; "
                        "re-run with a deeper preset or fewer perspectives"
                    )
                report_md = assemble_report(
                    plan,
                    dossiers,
                    synthesis.text,
                    ledger.to_run_usage(),
                    warnings,
                    settings.depth,
                )

        if fallback_reason is not None:
            warnings.append(fallback_reason)
            report_md = fallback_report(
                plan,
                dossiers,
                fallback_reason,
                ledger.to_run_usage(),
                warnings,
                settings.depth,
            )

        partial = (
            ledger.over_budget
            or not any_findings
            or synthesis_failed
            or any(not d.ok for d in dossiers)
        )
        return RunResult(
            schema_version=SCHEMA_VERSION,
            topic=topic,
            plan=plan,
            dossiers=dossiers,
            canonical_methods=canonical,
            pulse=pulse,
            report_markdown=report_md,
            usage=ledger.to_run_usage(),
            partial=partial,
            warnings=warnings,
        )
    finally:
        await llm.close()
