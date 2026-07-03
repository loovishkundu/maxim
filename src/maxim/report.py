"""Report assembly: citation resolution, sources, appendices, metadata footer."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from .schemas import (
    PERSPECTIVE_LABELS,
    Finding,
    ResearchDossier,
    ResearchPlan,
    RunUsage,
)

CITATION_RE = re.compile(r"\[(F-[a-z]{2}\d+)\]")


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug[:max_len].rstrip("-") or "topic"


def report_path(out_dir: Path, topic: str) -> Path:
    date = dt.date.today().isoformat()
    base = out_dir / f"{slugify(topic)}-{date}.md"
    if not base.exists():
        return base
    n = 2
    while (candidate := out_dir / f"{slugify(topic)}-{date}-{n}.md").exists():
        n += 1
    return candidate


def _verification_mark(finding: Finding) -> str:
    if any(ev.status == "verified" for ev in finding.evidence):
        return "verified"
    return "unverified"


def _sources_section(dossiers: list[ResearchDossier], cited_ids: list[str]) -> str:
    lines = ["## Sources"]
    cited_set = set(cited_ids)
    for dossier in dossiers:
        if not dossier.findings:
            continue
        lines.append(f"\n### {PERSPECTIVE_LABELS[dossier.perspective]}")
        for f in dossier.findings:
            cited_note = "" if f.id in cited_set else " (not cited in report body)"
            lines.append(f"- **[{f.id}]** {f.claim}{cited_note}")
            for ev in f.evidence:
                published = f", {ev.published}" if ev.published else ""
                status = "✓" if ev.status == "verified" else f"({ev.status})"
                lines.append(
                    f"  - {ev.source_title} — <{ev.source_url}> ({ev.kind}{published}) {status}"
                )
    return "\n".join(lines)


def _rejected_appendix(dossiers: list[ResearchDossier]) -> str | None:
    rows: list[str] = []
    for dossier in dossiers:
        for rej in dossier.rejected:
            rows.append(
                f"- ({PERSPECTIVE_LABELS[dossier.perspective]}) “{rej.finding.claim}” — "
                f"{rej.reason}"
            )
    if not rows:
        return None
    return (
        "## Appendix: Rejected Claims\n"
        "These claims were drafted during research but did not survive grounding; "
        "they are listed for transparency and should not be relied on.\n\n" + "\n".join(rows)
    )


def _footer(
    plan: ResearchPlan,
    dossiers: list[ResearchDossier],
    usage: RunUsage,
    warnings: list[str],
    depth: str,
) -> str:
    lines = [
        "---",
        "## Run Metadata",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')} · depth: {depth}",
        f"- Estimated cost: ${usage.total_cost_usd:.2f} · wall time: {usage.wall_seconds:.0f}s",
        f"- Recency horizon: {plan.recency_horizon_months} months",
    ]
    for dossier in dossiers:
        status = "ok" if dossier.ok else f"FAILED ({dossier.failure})"
        lines.append(
            f"- {PERSPECTIVE_LABELS[dossier.perspective]}: {len(dossier.findings)} validated, "
            f"{len(dossier.rejected)} rejected, {dossier.web_searches} searches, "
            f"{dossier.web_fetches} fetches — {status}"
        )
    lines.append("- Stage usage:")
    for stage in usage.stages:
        lines.append(
            f"  - {stage.stage} [{stage.model}]: {stage.calls} calls, "
            f"{stage.input_tokens:,} in / {stage.output_tokens:,} out "
            f"(cache r/w {stage.cache_read_tokens:,}/{stage.cache_write_tokens:,}) "
            f"≈ ${stage.cost_usd:.2f}"
        )
    if warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {w}" for w in warnings)
    lines.append("- Costs are estimates from configured pricing, not billing truth.")
    return "\n".join(lines)


def assemble_report(
    plan: ResearchPlan,
    dossiers: list[ResearchDossier],
    synthesis_md: str,
    usage: RunUsage,
    warnings: list[str],
    depth: str,
) -> str:
    """Resolve finding-id citations, append sources/appendices/footer."""
    known = {f.id for d in dossiers for f in d.findings}
    cited_ids = CITATION_RE.findall(synthesis_md)
    unknown = sorted({c for c in cited_ids if c not in known})
    body = synthesis_md
    for bad in unknown:
        body = body.replace(f"[{bad}]", f"[{bad} — unresolved citation]")
        warnings.append(f"synthesizer cited unknown finding id {bad}")

    sections = [body, _sources_section(dossiers, cited_ids)]
    rejected = _rejected_appendix(dossiers)
    if rejected:
        sections.append(rejected)
    sections.append(_footer(plan, dossiers, usage, warnings, depth))
    return "\n\n".join(sections) + "\n"


def fallback_report(
    plan: ResearchPlan,
    dossiers: list[ResearchDossier],
    reason: str,
    usage: RunUsage,
    warnings: list[str],
    depth: str,
) -> str:
    """Zero-LLM raw dump used when synthesis cannot run (budget, no findings,
    or a synthesis-stage failure). States the actual reason and keeps the
    run-metadata footer so failure causes reach the report file, not just
    stderr."""
    lines = [
        f"# {plan.topic} (raw findings — {reason})",
        "",
        f"Domain: {plan.domain}",
        "",
    ]
    for dossier in dossiers:
        lines.append(f"## {PERSPECTIVE_LABELS[dossier.perspective]}")
        if not dossier.ok:
            lines.append(f"_Agent failed: {dossier.failure}_")
            continue
        lines.append(dossier.summary)
        for f in dossier.findings:
            lines.append(
                f"- **{f.method_name}** ({f.confidence}, {_verification_mark(f)}): " f"{f.claim}"
            )
            for ev in f.evidence:
                lines.append(f"  - {ev.source_title} — <{ev.source_url}>")
        lines.append("")
    lines.append(_footer(plan, dossiers, usage, warnings, depth))
    return "\n".join(lines)
