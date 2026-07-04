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
from .quality import report_violations
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
from .synthesizer import repair_synthesis, synthesize
from .usage import UsageLedger

ProgressFn = Callable[[str, str], None]
ConfirmFn = Callable[[ResearchPlan], bool]

# Researchers self-terminate at a soft deadline and salvage partial findings;
# the hard wait_for backstop only fires if one hangs past the grace window.
TIMEOUT_GRACE_S = 60.0
# Total tries per researcher: a hard failure with nothing to salvage gets one
# fresh-conversation retry before it costs the report a whole section.
RESEARCHER_ATTEMPTS = 2


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
                # One fresh retry for hard failures: a researcher that dies in
                # its first gather must not cost a whole report section when a
                # second attempt would have worked. Mid-loop failures salvage
                # the checkpoint instead — validated findings beat a re-run.
                failure_note = "all attempts failed"
                for attempt in range(1, RESEARCHER_ATTEMPTS + 1):
                    checkpoint: dict[str, ResearchDossier] = {}
                    try:
                        dossier = await asyncio.wait_for(
                            run_researcher(
                                brief,
                                plan,
                                settings,
                                llm,
                                progress=lambda msg, _l=label: progress(_l, msg),
                                deadline=time.monotonic() + timeout,
                                checkpoint=checkpoint,
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
                        # The soft deadline should have ended the loop
                        # gracefully; this backstop means a call hung outright.
                        # No retry — the wall clock is spent — but the last
                        # completed pass's snapshot beats losing the run.
                        warnings.append(
                            f"{brief.perspective}: hung past {timeout:.0f}s + grace and "
                            "was cancelled — the cancelled call's usage is not counted; "
                            "real spend may exceed the estimate"
                        )
                        snapshot = checkpoint.get("dossier")
                        if snapshot is not None:
                            progress(
                                label,
                                f"timed out — salvaged {len(snapshot.findings)} findings "
                                "from the last completed pass",
                            )
                            return snapshot
                        progress(label, "timed out")
                        failure_note = f"timed out after {timeout:.0f}s"
                        break
                    except Exception as exc:  # degrade, never die
                        snapshot = checkpoint.get("dossier")
                        if snapshot is not None:
                            warnings.append(
                                f"{brief.perspective}: failed mid-run ({exc}) — salvaged "
                                f"{len(snapshot.findings)} validated findings"
                            )
                            progress(label, "failed — salvaged partial results")
                            return snapshot
                        if attempt < RESEARCHER_ATTEMPTS and not ledger.over_budget:
                            warnings.append(
                                f"{brief.perspective}: attempt {attempt} failed ({exc}) "
                                "— retrying with a fresh conversation"
                            )
                            progress(label, "failed — retrying with a fresh conversation…")
                            continue
                        warnings.append(f"{brief.perspective}: failed — {exc}")
                        progress(label, f"failed: {exc}")
                        failure_note = str(exc)
                        break
                searches, fetches = ledger.stage_counts(label)
                return stub_dossier(brief, failure_note, searches, fetches)

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
        quality_failed = False
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
                body = synthesis.text
                # Rigid quality gate: the reader must not meet uncited or
                # template-breaking output. One repair pass, then disclose.
                known_ids = {f.id for d in dossiers for f in d.findings}
                violations = report_violations(body, known_ids)
                if violations and not synthesis.truncated and not ledger.over_budget:
                    progress(
                        "synthesizer",
                        f"quality gate: {len(violations)} violation(s) — repairing…",
                    )
                    try:
                        repaired = await repair_synthesis(
                            plan,
                            dossiers,
                            settings,
                            llm,
                            draft_text=body,
                            violations=violations,
                            canonical_methods=canonical,
                            pulse=pulse,
                        )
                    except LLMError as exc:
                        warnings.append(f"report repair pass failed: {exc}")
                    else:
                        repaired_violations = report_violations(repaired.text, known_ids)
                        # A repair is accepted only when it is complete (not
                        # truncated) and strictly improves: either clean, or
                        # fewer violations without introducing new kinds.
                        improved = not repaired_violations or (
                            len(repaired_violations) < len(violations)
                            and set(repaired_violations) <= set(violations)
                        )
                        if not repaired.truncated and improved:
                            body = repaired.text
                            violations = repaired_violations
                if violations:
                    quality_failed = True
                    warnings.append("report quality gate: " + "; ".join(violations))
                report_md = assemble_report(
                    plan,
                    dossiers,
                    body,
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
                pulse=pulse,
            )

        partial = (
            ledger.over_budget
            or not any_findings
            or synthesis_failed
            or quality_failed
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
