"""End-to-end pipeline test with a fake LLM — no network.

Exercises: planning, fan-out, the mechanical fabrication gate, critique
application, synthesis citation resolution, budget gate, and researcher-failure
degradation.
"""

from conftest import GOOD_QUOTE, PAGE_TEXT, SOURCE_URL, make_draft_dossier, make_plan

import maxim.orchestrator as orchestrator
from maxim.config import Settings
from maxim.llm import AgenticResult, LLMError, SourceDoc, StreamResult
from maxim.schemas import (
    CanonicalMethods,
    ClaimVerdict,
    CoverageResult,
    CritiqueResult,
    DraftDossier,
    MethodGroup,
    ResearchPlan,
)
from maxim.usage import UsageLedger

GOOD_SYNTH = """# Report

## TL;DR
- STL is solid [F-ai1]

## Method Landscape
| Method | Perspective | Maturity | Community Signal | Best When | Findings |
|---|---|---|---|---|---|
| STL decomposition | Statistics | mature | – | seasonal data | [F-st1] |

## Decision Guide
Start with STL [F-st1].

## Caveats
Coverage gaps remain [F-ai1].
"""


class FakeLLM:
    def __init__(
        self,
        settings,
        ledger,
        fail_perspectives=frozenset(),
        synth_stop_reason="end_turn",
        synth_raises=False,
        synth_texts=None,
        perspectives=("ai_agentic", "statistics"),
    ):
        self.settings = settings
        self.ledger = ledger
        self.fail_perspectives = fail_perspectives
        self.synth_stop_reasons = list(
            synth_stop_reason if isinstance(synth_stop_reason, list) else [synth_stop_reason]
        )
        self.synth_raises = synth_raises
        self.synth_texts = list(synth_texts) if synth_texts else [GOOD_SYNTH]
        self.synth_calls = 0
        self.perspectives = perspectives
        self.draft_calls: dict[str, int] = {}
        self.agentic_stages: list[str] = []
        self.agentic_messages: dict[str, str] = {}

    async def close(self):
        pass

    async def parse(
        self, *, stage, system, messages, output_format, model, effort, max_tokens=16_000
    ):
        if output_format is ResearchPlan:
            return make_plan(self.perspectives)
        if output_format is CanonicalMethods:
            return CanonicalMethods(
                groups=[
                    MethodGroup(
                        canonical="STL decomposition", variants=["STL decomposition", "STL"]
                    ),
                    MethodGroup(canonical="Prophet", variants=["Prophet"]),
                ]
            )
        if output_format is DraftDossier:
            # First pass drafts 3 good findings + 1 fabricated quote. The loop
            # then RE-VALIDATEs; the repair pass legitimately drops the broken
            # claim (empty repair dossier) and the loop accepts.
            self.draft_calls[stage] = self.draft_calls.get(stage, 0) + 1
            if self.draft_calls[stage] == 1:
                return make_draft_dossier()
            return DraftDossier(summary="", findings=[], methods_identified=[], gaps=[])
        if output_format is CritiqueResult:
            # The fabricated finding (drafted 4th) never reaches the critic;
            # survivors keep ids 1-3 in every perspective.
            return CritiqueResult(
                verdicts=[
                    ClaimVerdict(finding_id=f"F-{p}{n}", verdict="supported", fix_hint=None)
                    for p in ("ai", "st", "cm")
                    for n in (1, 2, 3)
                ],
                coverage_gaps=[],
            )
        if output_format is CoverageResult:
            return CoverageResult(coverage_gaps=["maturity question unanswered"])
        raise AssertionError(f"unexpected output_format {output_format}")

    async def run_agentic(
        self,
        *,
        stage,
        system,
        messages,
        tools,
        model,
        effort,
        max_tokens,
        max_continuations,
        on_progress=None,
    ):
        perspective = stage.split(":", 1)[1]
        self.agentic_stages.append(stage)
        self.agentic_messages.setdefault(stage, messages[0]["content"])
        if perspective in self.fail_perspectives:
            raise RuntimeError("boom: simulated researcher crash")
        return AgenticResult(
            messages=messages + [{"role": "assistant", "content": "searched"}],
            final_stop_reason="end_turn",
            source_cache={SOURCE_URL: SourceDoc(url=SOURCE_URL, text=PAGE_TEXT)},
            cited_quotes=[],
            continuations=1,
            truncated=False,
        )

    async def stream_text(
        self, *, stage, system, messages, model, effort, max_tokens, on_text=None
    ):
        if self.synth_raises:
            raise LLMError("synthesizer: API call failed: 529 overloaded")
        self.synth_calls += 1
        text = self.synth_texts.pop(0) if len(self.synth_texts) > 1 else self.synth_texts[0]
        if len(self.synth_stop_reasons) > 1:
            stop_reason = self.synth_stop_reasons.pop(0)
        else:
            stop_reason = self.synth_stop_reasons[0]
        return StreamResult(text=text, stop_reason=stop_reason)


def _settings(**kwargs) -> Settings:
    return Settings(assume_yes=True, quiet=True, **kwargs)


def _install(monkeypatch, **fake_kwargs):
    holder: dict[str, FakeLLM] = {}

    def factory(settings, ledger):
        holder["llm"] = FakeLLM(settings, ledger, **fake_kwargs)
        return holder["llm"]

    monkeypatch.setattr(orchestrator, "LLM", factory)
    return holder


async def test_happy_path(monkeypatch):
    _install(monkeypatch)
    events: list[tuple[str, str]] = []
    result = await orchestrator.run_pipeline(
        "anomaly detection for vehicle telemetry",
        _settings(),
        progress=lambda label, msg: events.append((label, msg)),
        confirm=lambda plan: True,
    )

    assert result.schema_version == "1"
    assert len(result.dossiers) == 2
    for dossier in result.dossiers:
        assert dossier.ok
        # good findings validated, fabricated finding mechanically rejected
        assert len(dossier.findings) == 3
        assert all(f.verdict == "supported" for f in dossier.findings)
        assert all(f.confidence == "high" for f in dossier.findings)
        assert all(ev.status == "verified" for f in dossier.findings for ev in f.evidence)
        assert len(dossier.rejected) == 1
        assert "mechanical" in dossier.rejected[0].reason
        assert "maturity question unanswered" in dossier.gaps
        # The fabricated quote triggered one RE-VALIDATE repair pass; the
        # repair legitimately dropped the claim and the loop accepted.
        assert dossier.iterations == 2
        assert dossier.loop_actions == ["revalidate"]

    assert "[F-ai1]" in result.report_markdown
    assert "## Sources" in result.report_markdown
    assert "## Appendix: Rejected Claims" in result.report_markdown
    assert not result.partial
    assert any(label == "synthesizer" for label, _ in events)


async def test_good_quote_survives_verification():
    # guards the fixture: the "good" quote really is in the page text
    from maxim.verification import match_quote

    assert match_quote(GOOD_QUOTE, PAGE_TEXT) == 1.0


async def test_researcher_failure_degrades(monkeypatch):
    _install(monkeypatch, fail_perspectives=frozenset({"statistics"}))
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    by_perspective = {d.perspective: d for d in result.dossiers}
    assert not by_perspective["statistics"].ok
    assert by_perspective["ai_agentic"].ok
    assert result.partial
    assert any("statistics" in w for w in result.warnings)
    # synthesis still ran on the surviving dossier
    assert "[F-ai1]" in result.report_markdown


async def test_budget_gate_skips_research(monkeypatch):
    _install(monkeypatch)
    # Force over-budget from the start: record cost onto the ledger the pipeline
    # creates by patching UsageLedger to start over budget.
    monkeypatch.setattr(orchestrator, "UsageLedger", lambda budget_usd: _over_budget_ledger())
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(budget_usd=0.0),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert result.partial
    assert all(not d.ok for d in result.dossiers)
    assert "budget exhausted" in result.report_markdown


def _over_budget_ledger() -> UsageLedger:
    class FakeUsageObj:
        input_tokens = 10_000_000
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        server_tool_use = None

    ledger = UsageLedger(budget_usd=0.0)
    ledger.record("preexisting", "claude-opus-4-8", FakeUsageObj())
    return ledger


async def test_truncated_synthesis_marks_partial(monkeypatch):
    _install(monkeypatch, synth_stop_reason="max_tokens")
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert result.partial
    assert any("token cap" in w for w in result.warnings)
    # The (truncated) synthesized body is still shipped, with sources appended.
    assert "[F-ai1]" in result.report_markdown


async def test_synthesis_failure_degrades_to_fallback(monkeypatch):
    _install(monkeypatch, synth_raises=True)
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert result.partial
    assert any("synthesis failed" in w for w in result.warnings)
    # Research is not lost: the fallback dump carries the validated findings.
    assert "STL decomposition" in result.report_markdown
    assert "synthesis failed" in result.report_markdown


async def test_plan_rejected(monkeypatch):
    _install(monkeypatch)
    try:
        await orchestrator.run_pipeline(
            "topic",
            _settings(),
            progress=lambda *_: None,
            confirm=lambda plan: False,
        )
    except orchestrator.PlanRejected:
        pass
    else:
        raise AssertionError("expected PlanRejected")


async def test_two_wave_community_seeded_with_canonical_methods(monkeypatch):
    holder = _install(monkeypatch, perspectives=("ai_agentic", "statistics", "community"))
    events: list[tuple[str, str]] = []
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda label, msg: events.append((label, msg)),
        confirm=lambda plan: True,
    )
    fake = holder["llm"]

    # Wave order: community's first gather comes after both wave-1 gathers.
    first_gather_index = {stage: i for i, stage in reversed(list(enumerate(fake.agentic_stages)))}
    assert first_gather_index["researcher:community"] > first_gather_index["researcher:ai_agentic"]
    assert first_gather_index["researcher:community"] > first_gather_index["researcher:statistics"]

    # The community brief was reseeded with the canonicalized wave-1 methods.
    community_msg = fake.agentic_messages["researcher:community"]
    assert "STL decomposition" in community_msg
    assert "Prophet" in community_msg

    assert result.canonical_methods == ["STL decomposition", "Prophet"]
    assert any(label == "canonicalizer" for label, _ in events)
    by_perspective = {d.perspective: d for d in result.dossiers}
    assert set(by_perspective) == {"ai_agentic", "statistics", "community"}

    # Pulse is mechanical: with no model-claimed sentiment and few threads,
    # honesty demands insufficient_data — never an invented verdict.
    assert result.pulse
    assert all(p.sentiment == "insufficient_data" for p in result.pulse)


BAD_SYNTH = "# Report\n\nSTL is solid [F-ai1] and stats agree [F-st1]. Bogus [F-xx7]."


async def test_quality_gate_repairs_bad_draft(monkeypatch):
    holder = _install(monkeypatch, synth_texts=[BAD_SYNTH, GOOD_SYNTH])
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert holder["llm"].synth_calls == 2  # draft + one repair pass
    assert not result.partial
    assert "## TL;DR" in result.report_markdown
    assert "[F-xx7" not in result.report_markdown


async def test_quality_gate_discloses_unrepaired_violations(monkeypatch):
    holder = _install(monkeypatch, synth_texts=[BAD_SYNTH, BAD_SYNTH])
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert holder["llm"].synth_calls == 2  # exactly one repair attempt, no loop
    assert result.partial
    assert any("report quality gate" in w for w in result.warnings)
    # The bad citation is still annotated for the reader, never silently kept.
    assert "[F-xx7 — unresolved citation]" in result.report_markdown


async def test_hard_timeout_salvages_checkpoint_snapshot(monkeypatch):
    import asyncio
    from dataclasses import replace

    import maxim.config as config
    import maxim.researcher as researcher_mod

    # Shrink the clocks: soft deadline effectively disabled (margin 0 with a
    # 0.3s budget the fake iteration finishes well inside), and the repair
    # gather hangs so the wait_for backstop fires.
    monkeypatch.setitem(
        config.DEPTHS, "standard", replace(config.DEPTHS["standard"], researcher_timeout_s=0.3)
    )
    monkeypatch.setattr(orchestrator, "TIMEOUT_GRACE_S", 0.05)
    monkeypatch.setattr(researcher_mod, "DEADLINE_MARGIN_S", 0.0)

    holder = _install(monkeypatch)
    fake = None

    original_run_agentic = FakeLLM.run_agentic

    async def hanging_run_agentic(self, *, stage, **kwargs):
        if stage == "researcher:statistics" and any(
            s == "researcher:statistics" for s in self.agentic_stages
        ):
            await asyncio.sleep(30)  # second gather (the repair turn) hangs
        return await original_run_agentic(self, stage=stage, **kwargs)

    monkeypatch.setattr(FakeLLM, "run_agentic", hanging_run_agentic)

    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    fake = holder["llm"]
    assert fake is not None

    by_perspective = {d.perspective: d for d in result.dossiers}
    salvaged = by_perspective["statistics"]
    # Not a stub: the last completed pass survived the hard cancel.
    assert salvaged.ok
    assert len(salvaged.findings) == 3
    assert all(f.confidence in ("medium", "low") for f in salvaged.findings)
    assert any("hard timeout" in g for g in salvaged.gaps)
    assert any("hung past" in w for w in result.warnings)


async def test_repair_that_introduces_new_violations_is_rejected(monkeypatch):
    # The repair has fewer violations by count but invents a NEW bad citation
    # — count-only comparison would accept it; the subset rule must not.
    repair_with_new_bug = GOOD_SYNTH + "\nAlso bogus [F-yy8].\n"
    _install(monkeypatch, synth_texts=[BAD_SYNTH, repair_with_new_bug])
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert result.partial
    assert "[F-yy8" not in result.report_markdown  # bad repair rejected
    assert "[F-xx7 — unresolved citation]" in result.report_markdown  # original kept


async def test_truncated_repair_never_replaces_complete_draft(monkeypatch):
    # The repair fixes every violation but hit max_tokens — a truncated body
    # must not silently replace the complete draft.
    _install(
        monkeypatch,
        synth_texts=[BAD_SYNTH, GOOD_SYNTH],
        synth_stop_reason=["end_turn", "max_tokens"],
    )
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert result.partial
    assert "[F-xx7 — unresolved citation]" in result.report_markdown  # original kept


async def test_fallback_report_carries_pulse_and_tiers(monkeypatch):
    _install(
        monkeypatch,
        perspectives=("ai_agentic", "statistics", "community"),
        synth_raises=True,
    )
    result = await orchestrator.run_pipeline(
        "topic",
        _settings(),
        progress=lambda *_: None,
        confirm=lambda plan: True,
    )
    assert "Community Pulse (mechanical aggregates)" in result.report_markdown
    assert "[tier B]" in result.report_markdown
