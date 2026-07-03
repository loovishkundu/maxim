"""Researcher loop state machine, driven end-to-end by a scripted LLM.

Each scenario scripts the draft dossiers and critique results per iteration
and asserts on the real loop: routing, repair messages, id continuity,
supersession of weak findings, rejected-list hygiene, and graceful stops
(deadline / budget) that salvage partial results.
"""

import time

from conftest import GOOD_QUOTE, PAGE_TEXT, SOURCE_URL, make_brief, make_plan

from maxim.config import Settings
from maxim.llm import AgenticResult, SourceDoc
from maxim.researcher import run_researcher
from maxim.schemas import (
    ClaimVerdict,
    CritiqueResult,
    DraftDossier,
    DraftEvidence,
    DraftFinding,
)
from maxim.usage import UsageLedger

REPAIR_URL = "https://eng.example.com/stl-follow-up"
REPAIR_PAGE = (
    "Follow-up benchmarks on 12 vehicle fleets confirmed STL held a 90%+ "
    "detection rate across seasons and sensor types."
)
REPAIR_QUOTE = "confirmed STL held a 90%+ detection rate across seasons"


def df(claim, method, quote=GOOD_QUOTE, url=SOURCE_URL) -> DraftFinding:
    return DraftFinding(
        claim=claim,
        method_name=method,
        evidence=[
            DraftEvidence(
                quote=quote,
                source_url=url,
                source_title="Example post",
                published="2026-01",
                kind="blog",
            )
        ],
        caveats=[],
    )


def dd(findings, gaps=None) -> DraftDossier:
    return DraftDossier(
        summary="scripted summary",
        findings=findings,
        methods_identified=[f.method_name for f in findings],
        gaps=gaps or [],
    )


def cr(verdicts, gaps=None) -> CritiqueResult:
    return CritiqueResult(
        verdicts=[
            ClaimVerdict(finding_id=fid, verdict=verdict, fix_hint=hint)
            for fid, verdict, hint in verdicts
        ],
        coverage_gaps=gaps or [],
    )


class ScriptedLLM:
    """Pops scripted drafts/critiques per call; records every agentic turn."""

    def __init__(self, drafts, critiques, budget_usd=10.0):
        self.drafts = list(drafts)
        self.critiques = list(critiques)
        self.ledger = UsageLedger(budget_usd=budget_usd)
        self.agentic_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def parse(self, *, stage, system, messages, output_format, model, effort, **_):
        self.parse_calls.append({"messages": messages, "output_format": output_format})
        if output_format is DraftDossier:
            return self.drafts.pop(0)
        if output_format is CritiqueResult:
            return self.critiques.pop(0)
        raise AssertionError(f"unexpected parse: {output_format}")

    async def run_agentic(self, *, messages, tools, **_):
        self.agentic_calls.append({"messages": messages, "tools": tools})
        return AgenticResult(
            messages=messages + [{"role": "assistant", "content": "gathered"}],
            final_stop_reason="end_turn",
            source_cache={
                SOURCE_URL: SourceDoc(url=SOURCE_URL, text=PAGE_TEXT),
                REPAIR_URL: SourceDoc(url=REPAIR_URL, text=REPAIR_PAGE),
            },
            cited_quotes=[],
            continuations=0,
            truncated=False,
        )


async def _run(llm, deadline=None):
    return await run_researcher(
        make_brief("ai_agentic"),
        make_plan(),
        Settings(depth="standard"),
        llm,
        progress=lambda _msg: None,
        deadline=deadline,
    )


SUPPORTED_4 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 5)])


async def test_healthy_first_pass_accepts():
    llm = ScriptedLLM(
        drafts=[SUPPORTED_4],
        critiques=[cr([(f"F-ai{i}", "supported", None) for i in range(1, 5)])],
    )
    dossier = await _run(llm)
    assert dossier.iterations == 1
    assert dossier.loop_actions == []
    assert [f.id for f in dossier.findings] == ["F-ai1", "F-ai2", "F-ai3", "F-ai4"]
    # eng.* blog is tier B: supported + verified + reputable ⇒ high stays high
    assert all(f.confidence == "high" for f in dossier.findings)
    assert len(llm.agentic_calls) == 1  # gather only, no repair turns


async def test_retry_loop_repairs_weak_findings():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", "find the benchmark numbers"),
            ("F-ai5", "partially_supported", "quote is about a different metric"),
        ]
    )
    draft2 = dd(
        [
            df("claim 4 repaired", "m4", quote=REPAIR_QUOTE, url=REPAIR_URL),
            df("claim 5 repaired", "m5", quote=REPAIR_QUOTE, url=REPAIR_URL),
        ]
    )
    critique2 = cr([("F-ai6", "supported", None), ("F-ai7", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1, draft2], critiques=[critique1, critique2])

    dossier = await _run(llm)

    assert dossier.iterations == 2
    assert dossier.loop_actions == ["retry"]
    # weak m4/m5 superseded by their repaired versions; ids never reused
    assert [f.id for f in dossier.findings] == ["F-ai1", "F-ai2", "F-ai3", "F-ai6", "F-ai7"]
    assert all(f.verdict == "supported" for f in dossier.findings)
    assert dossier.rejected == []

    # The retry turn happened on the same conversation with the critic's hints.
    assert len(llm.agentic_calls) == 2
    retry_msg = llm.agentic_calls[1]["messages"][-1]["content"]
    assert "find the benchmark numbers" in retry_msg
    assert "claim 4" in retry_msg
    # And the repair draft locks the already-validated claims.
    repair_draft_msg = llm.parse_calls[-2]["messages"][-1]["content"]
    assert "do not repeat" in repair_draft_msg.casefold()
    assert "claim 1" in repair_draft_msg


async def test_unrepaired_weak_finding_is_kept():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    # Repair pass only rescues m4; m5 comes back with nothing.
    draft2 = dd([df("claim 4 repaired", "m4", quote=REPAIR_QUOTE, url=REPAIR_URL)])
    critique2 = cr([("F-ai6", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1, draft2], critiques=[critique1, critique2])

    dossier = await _run(llm)

    by_id = {f.id: f for f in dossier.findings}
    assert "F-ai6" in by_id  # repaired m4
    assert "F-ai5" in by_id  # unrepaired weak m5 survives, still flagged
    assert "F-ai4" not in by_id  # superseded weak m4 dropped
    assert by_id["F-ai5"].verdict == "partially_supported"


async def test_revalidate_refetches_without_searching():
    draft1 = dd(
        [df(f"claim {i}", f"m{i}") for i in range(1, 4)]
        + [df("broken-quote claim", "m4", quote="this text is in no fetched page at all")]
    )
    critique1 = cr([(f"F-ai{i}", "supported", None) for i in range(1, 4)])
    # Repair pass re-submits the same claim with the real quote.
    draft2 = dd([df("broken-quote claim", "m4")])
    critique2 = cr([("F-ai5", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1, draft2], critiques=[critique1, critique2])

    dossier = await _run(llm)

    assert dossier.loop_actions == ["revalidate"]
    assert [f.id for f in dossier.findings] == ["F-ai1", "F-ai2", "F-ai3", "F-ai5"]
    # The mechanically-rejected version was repaired — it must leave the
    # rejected list rather than shaming a now-validated claim in the appendix.
    assert dossier.rejected == []

    # The repair turn carries the broken quote + URL and uses fetch-only tools.
    revalidate_msg = llm.agentic_calls[1]["messages"][-1]["content"]
    assert "this text is in no fetched page at all" in revalidate_msg
    assert SOURCE_URL in revalidate_msg
    tool_types = {t["type"] for t in llm.agentic_calls[1]["tools"]}
    assert tool_types == {"web_fetch_20260209"}


async def test_deadline_stops_loop_and_salvages_findings():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1])

    # Deadline already passed: the loop wants a retry but must stop instead.
    dossier = await _run(llm, deadline=time.monotonic())

    assert dossier.iterations == 1
    assert dossier.loop_actions == []
    assert len(dossier.findings) == 5  # salvaged, not a stub
    assert any("deadline" in g for g in dossier.gaps)
    # Work was pending, so nothing keeps "high" confidence.
    assert all(f.confidence in ("medium", "low") for f in dossier.findings)
    assert len(llm.agentic_calls) == 1


async def test_budget_exhaustion_stops_loop():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1], budget_usd=0.0)

    dossier = await _run(llm)

    assert dossier.budget_exhausted
    assert dossier.iterations == 1
    assert len(dossier.findings) == 5
    assert any("budget" in g for g in dossier.gaps)


async def test_structural_failure_recorded_when_replan_unavailable():
    draft1 = dd([df("claim 1", "m1"), df("claim 2", "m2")])
    critique1 = cr([("F-ai1", "supported", None), ("F-ai2", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1])

    dossier = await _run(llm)

    assert dossier.iterations == 1
    assert len(dossier.findings) == 2
    assert any("structural weakness" in g for g in dossier.gaps)


async def test_retry_cap_exhausts_then_accepts():
    weak_critique = lambda ids: cr(  # noqa: E731
        [(f"F-ai{i}", "partially_supported", "hint") for i in ids]
    )
    # Every pass drafts 4 findings that all come back weak → retry, retry, stop.
    drafts = [
        dd([df(f"claim {i}", f"m{i}") for i in range(1, 5)]),
        dd([df(f"claim r1-{i}", f"m{i}") for i in range(1, 5)]),
        dd([df(f"claim r2-{i}", f"m{i}") for i in range(1, 5)]),
    ]
    critiques = [
        weak_critique(range(1, 5)),
        weak_critique(range(5, 9)),
        weak_critique(range(9, 13)),
    ]
    llm = ScriptedLLM(drafts=drafts, critiques=critiques)

    dossier = await _run(llm)

    assert dossier.loop_actions == ["retry", "retry"]
    assert dossier.iterations == 3
    # Weak repairs supersede by method name each round: 4 findings remain.
    assert len(dossier.findings) == 4
