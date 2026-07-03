"""Researcher loop state machine, driven end-to-end by a scripted LLM.

Each scenario scripts the draft dossiers and critique results per iteration
and asserts on the real loop: routing, repair messages, id continuity,
supersession of weak findings, rejected-list hygiene, and graceful stops
(deadline / budget) that salvage partial results.
"""

import time

from conftest import GOOD_QUOTE, PAGE_TEXT, SOURCE_URL, make_brief, make_plan

from maxim.config import Settings
from maxim.llm import AgenticResult, LLMError, SourceDoc
from maxim.researcher import run_researcher
from maxim.schemas import (
    ClaimVerdict,
    CoverageResult,
    CritiqueResult,
    DraftDossier,
    DraftEvidence,
    DraftFinding,
    EngagementStats,
    ResearchBrief,
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

    def __init__(
        self,
        drafts,
        critiques,
        budget_usd=10.0,
        briefs=None,
        replan_raises=False,
        draft_error_on=None,
        truncate_calls=(),
    ):
        self.drafts = list(drafts)
        self.critiques = list(critiques)
        self.briefs = list(briefs or [])
        self.replan_raises = replan_raises
        self.draft_error_on = draft_error_on  # 1-based draft-parse call number
        self.truncate_calls = set(truncate_calls)  # 1-based agentic call numbers
        self.draft_calls = 0
        self.ledger = UsageLedger(budget_usd=budget_usd)
        self.agentic_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self._last_critique: CritiqueResult | None = None

    async def parse(self, *, stage, system, messages, output_format, model, effort, **_):
        self.parse_calls.append(
            {"stage": stage, "messages": messages, "output_format": output_format}
        )
        if output_format is DraftDossier:
            self.draft_calls += 1
            if self.draft_calls == self.draft_error_on:
                raise LLMError("researcher: structured output failed validation")
            return self.drafts.pop(0)
        if output_format is CritiqueResult:
            self._last_critique = self.critiques.pop(0)
            return self._last_critique
        if output_format is CoverageResult:
            # The scripted CritiqueResult carries the intended gaps; the real
            # critique() sources them from this separate coverage pass.
            gaps = self._last_critique.coverage_gaps if self._last_critique else []
            return CoverageResult(coverage_gaps=gaps)
        if output_format is ResearchBrief:
            if self.replan_raises:
                raise LLMError("replanner: API call failed: 529")
            return self.briefs.pop(0)
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
            truncated=len(self.agentic_calls) in self.truncate_calls,
            queries=[f"scripted query {len(self.agentic_calls)}"],
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
    draft_calls = [c for c in llm.parse_calls if c["output_format"] is DraftDossier]
    repair_draft_msg = draft_calls[-1]["messages"][-1]["content"]
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


def _revised_brief() -> ResearchBrief:
    return ResearchBrief(
        perspective="ai_agentic",
        objective="Attack the topic via benchmark write-ups instead",
        sub_questions=["Which benchmarks exist?"],
        seed_queries=["telemetry anomaly benchmark 2026"],
        must_cover_methods=[],
        avoid=[],
    )


async def test_replan_restarts_in_fresh_conversation():
    # Pass 1: only 1 validated finding (< min 3) → structural → REPLAN.
    draft1 = dd([df("claim 1", "m1")])
    critique1 = cr([("F-ai1", "supported", None)])
    draft2 = dd([df(f"replanned claim {i}", f"n{i}") for i in range(1, 5)])
    critique2 = cr([(f"F-ai{i}", "supported", None) for i in range(2, 6)])
    llm = ScriptedLLM(
        drafts=[draft1, draft2],
        critiques=[critique1, critique2],
        briefs=[_revised_brief()],
    )

    dossier = await _run(llm)

    assert dossier.loop_actions == ["replan"]
    assert dossier.iterations == 2
    # Pass-1 validated finding is kept; replanned findings continue the ids.
    assert [f.id for f in dossier.findings] == ["F-ai1", "F-ai2", "F-ai3", "F-ai4", "F-ai5"]

    # The replan gather is a FRESH conversation (one user message, no prior
    # transcript) built on the revised brief, carrying the tried queries.
    replan_gather = llm.agentic_calls[1]["messages"]
    assert len(replan_gather) == 1
    content = replan_gather[0]["content"]
    assert "benchmark write-ups" in content
    assert "scripted query 1" in content  # do-not-repeat list
    assert "claim 1" in content  # locked-in claim list

    # replan_task itself was seeded with the failure reasons.
    replan_parse = next(c for c in llm.parse_calls if c["output_format"] is ResearchBrief)
    assert "1 validated" in replan_parse["messages"][0]["content"]


async def test_replan_failure_keeps_partial_results():
    draft1 = dd([df("claim 1", "m1")])
    critique1 = cr([("F-ai1", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1], replan_raises=True)

    dossier = await _run(llm)

    assert dossier.ok
    assert dossier.iterations == 1
    assert len(dossier.findings) == 1
    assert any("replan failed" in g for g in dossier.gaps)
    # Repair work was pending, so confidence is capped.
    assert dossier.findings[0].confidence == "medium"


async def test_quick_depth_records_structural_weakness_without_replan():
    # quick's LoopPolicy has max_replans=0: the weakness is recorded, no
    # replanner call is made.
    draft1 = dd([df("claim 1", "m1")])
    critique1 = cr([("F-ai1", "supported", None)])
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1])

    dossier = await run_researcher(
        make_brief("ai_agentic"),
        make_plan(),
        Settings(depth="quick"),
        llm,
        progress=lambda _msg: None,
    )

    assert dossier.iterations == 1
    assert len(dossier.findings) == 1
    assert all(c["output_format"] is not ResearchBrief for c in llm.parse_calls)
    assert dossier.loop_actions == []
    # The weakness is not silently passed off as healthy.
    assert any("replan cap spent" in g for g in dossier.gaps)
    assert dossier.findings[0].confidence == "medium"


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


async def test_coverage_judged_against_frozen_claims_on_repair_passes():
    # Pass 1 covers everything; pass 2 repairs two weak claims. The coverage
    # judge must still see the frozen pass-1 claims, else answered
    # sub-questions read as phantom gaps and trigger a spurious REPLAN.
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
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

    await _run(llm)

    coverage_payloads = [
        c["messages"][0]["content"] for c in llm.parse_calls if c["output_format"] is CoverageResult
    ]
    assert len(coverage_payloads) == 2
    # Pass-2 coverage sees frozen claims AND the repaired survivors.
    assert "claim 1" in coverage_payloads[1]
    assert "claim 4 repaired" in coverage_payloads[1]


async def test_later_rejection_evicts_stale_weak_twin():
    # Pass 1 validates "claim 4" weakly; the retry resubmits the same claim
    # with new evidence and the critic (after opus arbitration) says
    # contradicted. The newer verdict must win: the weak twin leaves the
    # findings and the contradiction is recorded, never silently erased.
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    draft2 = dd(
        [
            df("claim 4", "m4", quote=REPAIR_QUOTE, url=REPAIR_URL),
            df("claim 6", "m6", quote=REPAIR_QUOTE, url=REPAIR_URL),
        ]
    )
    critique2_batch = cr([("F-ai6", "contradicted", None), ("F-ai7", "supported", None)])
    critique2_arbitration = cr([("F-ai6", "contradicted", None)])
    draft3 = dd([])
    llm = ScriptedLLM(
        drafts=[draft1, draft2, draft3],
        critiques=[critique1, critique2_batch, critique2_arbitration],
    )

    dossier = await _run(llm)

    claims = [f.claim for f in dossier.findings]
    assert "claim 4" not in claims  # stale weak twin evicted by the newer verdict
    assert "claim 6" in claims
    assert any(
        r.finding.claim == "claim 4" and "contradicted" in r.reason for r in dossier.rejected
    )


async def test_duplicate_rejections_collapse_to_one_entry():
    broken = "this text is in no fetched page at all"
    draft1 = dd(
        [df(f"claim {i}", f"m{i}") for i in range(1, 4)]
        + [df("broken-quote claim", "m4", quote=broken)]
    )
    critique1 = cr([(f"F-ai{i}", "supported", None) for i in range(1, 4)])
    # The repair pass re-submits the same broken claim with the same bad quote.
    draft2 = dd([df("broken-quote claim", "m4", quote=broken)])
    critique2 = cr([])
    llm = ScriptedLLM(drafts=[draft1, draft2], critiques=[critique1, critique2])

    dossier = await _run(llm)

    matching = [r for r in dossier.rejected if r.finding.claim == "broken-quote claim"]
    assert len(matching) == 1  # not one entry per failed pass


async def test_repair_gather_truncation_is_disclosed():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    draft2 = dd([df("claim 4 repaired", "m4", quote=REPAIR_QUOTE, url=REPAIR_URL)])
    critique2 = cr([("F-ai6", "supported", None)])
    llm = ScriptedLLM(
        drafts=[draft1, draft2],
        critiques=[critique1, critique2],
        truncate_calls={2},  # the retry gather hits a cap
    )

    dossier = await _run(llm)

    assert any("stopped early" in g for g in dossier.gaps)


async def test_mid_loop_draft_failure_salvages_validated_findings():
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
        ]
    )
    llm = ScriptedLLM(drafts=[draft1], critiques=[critique1], draft_error_on=2)

    dossier = await _run(llm)

    assert dossier.ok
    assert len(dossier.findings) == 5  # pass-1 results survive the pass-2 crash
    assert any("draft extraction failed" in g for g in dossier.gaps)
    # Work was pending: confidence capped.
    assert all(f.confidence in ("medium", "low") for f in dossier.findings)


async def test_checkpoint_snapshot_written_each_pass():
    # If the hard-timeout backstop cancels the coroutine mid-pass, the
    # orchestrator salvages this snapshot — it must hold the last completed
    # pass with capped confidence, not a stub.
    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 6)])
    critique1 = cr(
        [(f"F-ai{i}", "supported", None) for i in range(1, 4)]
        + [
            ("F-ai4", "partially_supported", None),
            ("F-ai5", "partially_supported", None),
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
    checkpoint: dict = {}

    await run_researcher(
        make_brief("ai_agentic"),
        make_plan(),
        Settings(depth="standard"),
        llm,
        progress=lambda _msg: None,
        checkpoint=checkpoint,
    )

    snapshot = checkpoint["dossier"]
    assert len(snapshot.findings) == 5  # last completed pass
    assert all(f.confidence in ("medium", "low") for f in snapshot.findings)
    assert any("hard timeout" in g for g in snapshot.gaps)


TIER_C_URL = "https://randomblog.dev/post"


async def test_tier_collapse_routes_to_replan():
    # All findings validate but only from tier-C sources — PLAN §5's tier
    # collapse trigger must fire a REPLAN toward reputable sources.
    class TierCLLM(ScriptedLLM):
        async def run_agentic(self, *, messages, tools, **kwargs):
            result = await super().run_agentic(messages=messages, tools=tools, **kwargs)
            result.source_cache[TIER_C_URL] = SourceDoc(url=TIER_C_URL, text=PAGE_TEXT)
            return result

    draft1 = dd([df(f"claim {i}", f"m{i}", url=TIER_C_URL) for i in range(1, 5)])
    critique1 = cr([(f"F-ai{i}", "supported", None) for i in range(1, 5)])
    draft2 = dd([df("claim 5", "m5", quote=REPAIR_QUOTE, url=REPAIR_URL)])
    critique2 = cr([("F-ai5", "supported", None)])
    llm = TierCLLM(
        drafts=[draft1, draft2],
        critiques=[critique1, critique2],
        briefs=[_revised_brief()],
    )

    dossier = await _run(llm)

    assert "replan" in dossier.loop_actions
    replan_parse = next(c for c in llm.parse_calls if c["output_format"] is ResearchBrief)
    assert "tier collapse" in replan_parse["messages"][0]["content"]


async def test_engagement_stamped_through_run_researcher():
    # The tools→sentiment join: engagement harvested during gather must land
    # on the enriched Evidence, tolerating URL spelling drift.
    stats = EngagementStats(source="hn", points=50, comments=20, thread_id="hn:1")

    drifted_url = SOURCE_URL.replace("eng.example.com", "ENG.Example.com") + "/"

    class EngagementLLM(ScriptedLLM):
        async def run_agentic(self, *, messages, tools, **kwargs):
            result = await super().run_agentic(messages=messages, tools=tools, **kwargs)
            # Tool registered the article under host-case + trailing-slash drift.
            result.engagement = {drifted_url: stats}
            return result

    draft1 = dd([df(f"claim {i}", f"m{i}") for i in range(1, 5)])
    critique1 = cr([(f"F-ai{i}", "supported", None) for i in range(1, 5)])
    llm = EngagementLLM(drafts=[draft1], critiques=[critique1])

    dossier = await _run(llm)

    assert all(f.evidence[0].engagement == stats for f in dossier.findings)
