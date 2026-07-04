"""CLI: argument parsing, progress rendering, confirmation gate, exit codes.

Contract for wrappers (Claude Code skill, scripts):
- stdout carries the result only: the report path, or the RunResult JSON with --json
- all progress/human chatter goes to stderr
- exit codes: 0 ok · 2 partial (a researcher failed / budget or token caps hit /
  synthesis degraded) · 3 hard failure (no result) · 4 plan declined by the user
- --quiet implies --yes (a suppressed confirmation prompt would otherwise hang)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from .config import DEPTHS, Settings
from .llm import LLMError
from .orchestrator import PlanRejected, run_pipeline
from .report import report_path
from .schemas import PERSPECTIVE_LABELS, PERSPECTIVES, ResearchPlan

EXIT_OK = 0
EXIT_PARTIAL = 2
EXIT_FAILURE = 3
EXIT_CANCELLED = 4

_COST_HINTS = {"quick": "$2–3", "standard": "$5–8", "deep": "$15–25"}
# Default budget ceilings leave headroom above the cost hints so a default run
# never degrades mid-flight; an explicit --budget-usd always wins.
_DEFAULT_BUDGETS = {"quick": 5.0, "standard": 12.0, "deep": 35.0}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maxim",
        description=(
            "Multi-perspective research assistant: plans, researches in parallel, "
            "verifies claims against sources, and writes one cited report."
        ),
    )
    parser.add_argument("topic", help="topic / concept / field to research")
    parser.add_argument(
        "--depth",
        choices=sorted(DEPTHS),
        default="standard",
        help="budget preset (default: standard)",
    )
    parser.add_argument(
        "--perspectives",
        help=f"comma-separated subset of: {','.join(PERSPECTIVES)}",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="cost ceiling; research/synthesis stop when the estimate passes it "
        "(default: scaled to depth — quick $5, standard $12, deep $35)",
    )
    parser.add_argument("--out", type=Path, help="report file path (default: ./maxim-reports/)")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="parallel researcher agents (default: 3)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="bypass the server-side page-fetch cache (slower; use for "
        "fast-moving topics where today's version of a page matters)",
    )
    parser.add_argument("--json", action="store_true", help="emit RunResult JSON on stdout")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output (implies --yes)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="skip plan confirmation")
    return parser


def _parse_perspectives(raw: str | None, parser: argparse.ArgumentParser) -> list[str] | None:
    if raw is None:
        return None
    wanted = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in wanted if p not in PERSPECTIVES]
    if bad:
        parser.error(f"unknown perspective(s): {', '.join(bad)}")
    return wanted


def _render_plan(plan: ResearchPlan, settings: Settings, console: Console) -> None:
    lines = [
        f"[bold]Domain:[/bold] {plan.domain}",
        f"[bold]Rationale:[/bold] {plan.rationale}",
        f"[bold]Recency horizon:[/bold] {plan.recency_horizon_months} months",
        "",
        "[bold]Perspectives in scope:[/bold]",
    ]
    for brief in plan.briefs:
        lines.append(f"  • {PERSPECTIVE_LABELS[brief.perspective]}: {brief.objective}")
    for oos in plan.out_of_scope:
        lines.append(
            f"  ◦ [dim]{PERSPECTIVE_LABELS[oos.perspective]} — out of scope: {oos.reason}[/dim]"
        )
    if plan.assumptions:
        lines.append("")
        lines.append("[bold]Assumptions:[/bold]")
        lines.extend(f"  • {a}" for a in plan.assumptions)
    lines.append("")
    lines.append(
        f"[bold]Estimated cost:[/bold] ~{_COST_HINTS[settings.depth]} "
        f"(cap: ${settings.budget_usd:.2f})"
    )
    console.print(Panel("\n".join(lines), title="Research plan", border_style="cyan"))


def _make_confirm(settings: Settings, console: Console):
    def confirm(plan: ResearchPlan) -> bool:
        if not settings.quiet:
            _render_plan(plan, settings, console)
        if settings.assume_yes or not sys.stdin.isatty():
            return True
        answer = console.input("Proceed with this plan? [Y/n] ").strip().casefold()
        return answer in ("", "y", "yes")

    return confirm


async def _run(settings: Settings, topic: str, console: Console) -> int:
    def progress(label: str, message: str) -> None:
        if not settings.quiet:
            console.print(f"[dim]{label:>24}[/dim]  {message}")

    result = await run_pipeline(
        topic, settings, progress=progress, confirm=_make_confirm(settings, console)
    )

    path = settings.out or report_path(settings.out_dir, topic)
    write_error: str | None = None
    dumped_to_stdout = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.report_markdown, encoding="utf-8")
    except OSError as exc:
        # The run is already paid for — never discard the result over a write
        # failure. Salvage to a temp file; failing even that, dump the
        # markdown to stdout so it exists SOMEWHERE.
        write_error = f"could not write report to {path}: {exc}"
        console.print(f"[bold red]error:[/bold red] {write_error}")
        try:
            fallback = Path(tempfile.mkdtemp(prefix="maxim-")) / path.name
            fallback.write_text(result.report_markdown, encoding="utf-8")
            console.print(f"[yellow]report salvaged to:[/yellow] {fallback}")
            path = fallback
            write_error = None
        except OSError:
            if not settings.json_output:
                print(result.report_markdown)
                dumped_to_stdout = True

    if not settings.quiet:
        console.print()
        if write_error is None:
            console.print(f"[bold green]Report:[/bold green] {path}")
        console.print(
            f"[dim]cost ≈ ${result.usage.total_cost_usd:.2f} · "
            f"{result.usage.wall_seconds:.0f}s · "
            f"{sum(len(d.findings) for d in result.dossiers)} validated findings · "
            f"{sum(len(d.rejected) for d in result.dossiers)} rejected[/dim]"
        )
        for warning in result.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")

    if settings.json_output:
        print(result.model_dump_json())
    elif write_error is None:
        print(path)

    if write_error is not None:
        # The result still reached the user (JSON payload or stdout dump):
        # that is a partial outcome, not a hard "no result" failure.
        if settings.json_output or dumped_to_stdout:
            return EXIT_PARTIAL
        return EXIT_FAILURE
    return EXIT_PARTIAL if result.partial else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_dotenv()

    budget = args.budget_usd if args.budget_usd is not None else _DEFAULT_BUDGETS[args.depth]
    if budget <= 0:
        parser.error("--budget-usd must be positive")

    settings = Settings(
        depth=args.depth,
        perspectives=_parse_perspectives(args.perspectives, parser),
        budget_usd=budget,
        max_concurrency=max(1, args.max_concurrency),
        out=args.out,
        # A quiet console swallows the confirmation prompt but input() would
        # still block on a question the user cannot see — quiet implies yes.
        assume_yes=args.yes or args.quiet,
        quiet=args.quiet,
        json_output=args.json,
        web_fetch_use_cache=not args.fresh,
    )

    # Not constructed with quiet=: errors must always reach stderr; progress and
    # summaries are gated on settings.quiet at each call site instead.
    console = Console(stderr=True)

    if not settings.quiet and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        console.print(
            "[yellow]warning:[/yellow] ANTHROPIC_API_KEY is not set (checked env and .env). "
            "The SDK may still find credentials from an `ant auth login` profile."
        )

    try:
        return asyncio.run(_run(settings, args.topic, console))
    except PlanRejected:
        console.print("Cancelled — no research was run.")
        return EXIT_CANCELLED
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 130
    except LLMError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        return EXIT_FAILURE
    except Exception as exc:  # keep the exit-code contract even for surprises
        console.print(f"[bold red]unexpected error:[/bold red] {type(exc).__name__}: {exc}")
        return EXIT_FAILURE
