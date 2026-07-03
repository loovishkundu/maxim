import re

from conftest import make_brief, make_finding

from maxim.config import Settings
from maxim.critic import apply_critique, build_payload, critique
from maxim.llm import SourceDoc
from maxim.schemas import ClaimVerdict, CoverageResult, CritiqueResult


def test_apply_critique_splits_and_stamps():
    findings = [
        make_finding("F-ai1", status="verified"),
        make_finding("F-ai2", status="skipped"),
        make_finding("F-ai3", status="verified"),
    ]
    result = CritiqueResult(
        verdicts=[
            ClaimVerdict(finding_id="F-ai1", verdict="supported", fix_hint=None),
            ClaimVerdict(
                finding_id="F-ai2", verdict="partially_supported", fix_hint="find a benchmark"
            ),
            ClaimVerdict(finding_id="F-ai3", verdict="contradicted", fix_hint=None),
        ],
        coverage_gaps=[],
    )
    validated, rejected = apply_critique(findings, result)

    assert [f.id for f in validated] == ["F-ai1", "F-ai2"]
    assert validated[0].confidence == "high"  # supported + verified evidence
    assert validated[1].confidence == "low"  # partial + no verified evidence
    assert any("find a benchmark" in c for c in validated[1].caveats)
    assert [r.finding.id for r in rejected] == ["F-ai3"]
    assert "contradicted" in rejected[0].reason


def test_missing_verdict_kept_but_demoted_to_low():
    findings = [make_finding("F-ai1", status="verified")]
    validated, rejected = apply_critique(findings, CritiqueResult(verdicts=[], coverage_gaps=[]))
    assert len(validated) == 1
    assert validated[0].verdict == "partially_supported"
    # Unreviewed by the critic: confidence is forced to low regardless of
    # mechanical verification.
    assert validated[0].confidence == "low"
    assert any("unreviewed" in c for c in validated[0].caveats)
    assert not rejected


def test_supported_but_weak_tier_caps_at_medium():
    finding = make_finding("F-ai1", status="verified")
    weak = finding.model_copy(
        update={"evidence": [finding.evidence[0].model_copy(update={"tier": "D"})]}
    )
    result = CritiqueResult(
        verdicts=[ClaimVerdict(finding_id="F-ai1", verdict="supported", fix_hint=None)],
        coverage_gaps=[],
    )
    validated, _ = apply_critique([weak], result)
    # Verified quote + supported claim, but only a forum source: not "high".
    assert validated[0].confidence == "medium"


def test_supported_unstamped_tier_caps_at_medium():
    finding = make_finding("F-ai1", status="verified")
    unstamped = finding.model_copy(
        update={"evidence": [finding.evidence[0].model_copy(update={"tier": None})]}
    )
    result = CritiqueResult(
        verdicts=[ClaimVerdict(finding_id="F-ai1", verdict="supported", fix_hint=None)],
        coverage_gaps=[],
    )
    validated, _ = apply_critique([unstamped], result)
    assert validated[0].confidence == "medium"


def test_verdict_id_drift_is_tolerated():
    findings = [make_finding("F-ai1", status="verified")]
    result = CritiqueResult(
        verdicts=[ClaimVerdict(finding_id="AI1", verdict="supported", fix_hint=None)],
        coverage_gaps=[],
    )
    validated, rejected = apply_critique(findings, result)
    assert validated[0].verdict == "supported"
    assert validated[0].confidence == "high"
    assert not any("unreviewed" in c for c in validated[0].caveats)


def test_payload_marks_unverified_context():
    finding = make_finding("F-ai1", status="failed")
    payload = build_payload(
        make_brief(),
        [finding],
        {finding.evidence[0].source_url: SourceDoc(url="u", text="unrelated text entirely")},
    )
    assert "F-ai1" in payload
    assert "source text unavailable" in payload or "NOT found verbatim" in payload


class RecordingLLM:
    """Records every parse call; scripts verdicts per model tier."""

    def __init__(self, batch_verdict="supported", arbitration_verdict="supported", gaps=None):
        self.batch_verdict = batch_verdict
        self.arbitration_verdict = arbitration_verdict
        self.gaps = gaps or []
        self.calls: list[dict] = []

    async def parse(self, *, stage, system, messages, output_format, model, effort, **_):
        content = messages[0]["content"]
        self.calls.append(
            {"model": model, "effort": effort, "format": output_format, "content": content}
        )
        if output_format is CoverageResult:
            return CoverageResult(coverage_gaps=self.gaps)
        ids = re.findall(r"### (F-\w+)", content)
        verdict = (
            self.arbitration_verdict if "arbitrating reviewer" in content else self.batch_verdict
        )
        return CritiqueResult(
            verdicts=[ClaimVerdict(finding_id=i, verdict=verdict, fix_hint=None) for i in ids],
            coverage_gaps=[],
        )


def _many_findings(n):
    return [make_finding(f"F-ai{i}", status="verified") for i in range(1, n + 1)]


async def test_critique_batches_and_runs_coverage_once():
    llm = RecordingLLM(gaps=["gap 1"])
    result = await critique(
        stage="critic:ai_agentic",
        brief=make_brief(),
        findings=_many_findings(20),
        source_cache={},
        settings=Settings(),
        llm=llm,
    )
    batch_calls = [c for c in llm.calls if c["format"] is CritiqueResult]
    coverage_calls = [c for c in llm.calls if c["format"] is CoverageResult]
    assert len(batch_calls) == 3  # 8 + 8 + 4
    assert len(coverage_calls) == 1
    assert all(c["model"] == "claude-haiku-4-5" for c in batch_calls)
    assert len(result.verdicts) == 20
    assert result.coverage_gaps == ["gap 1"]
    # Batch payloads must not double-report coverage.
    assert all("separate pass" in c["content"] for c in batch_calls)


async def test_contradicted_verdict_escalates_and_arbitration_wins():
    llm = RecordingLLM(batch_verdict="contradicted", arbitration_verdict="supported")
    result = await critique(
        stage="critic:ai_agentic",
        brief=make_brief(),
        findings=_many_findings(2),
        source_cache={},
        settings=Settings(),
        llm=llm,
    )
    escalations = [c for c in llm.calls if "arbitrating reviewer" in c["content"]]
    assert len(escalations) == 2  # one per contradicted finding, one-by-one
    assert all(c["model"] == "claude-opus-4-8" for c in escalations)
    assert all(c["effort"] == "low" for c in escalations)
    assert {v.verdict for v in result.verdicts} == {"supported"}


async def test_unsupported_on_verified_evidence_escalates():
    # The judge rejecting a mechanically verified quote is a split signal.
    llm = RecordingLLM(batch_verdict="unsupported", arbitration_verdict="unsupported")
    await critique(
        stage="critic:ai_agentic",
        brief=make_brief(),
        findings=[make_finding("F-ai1", status="verified")],
        source_cache={},
        settings=Settings(),
        llm=llm,
    )
    assert any("arbitrating reviewer" in c["content"] for c in llm.calls)


async def test_unsupported_on_skipped_evidence_does_not_escalate():
    llm = RecordingLLM(batch_verdict="unsupported")
    await critique(
        stage="critic:ai_agentic",
        brief=make_brief(),
        findings=[make_finding("F-ai1", status="skipped")],
        source_cache={},
        settings=Settings(),
        llm=llm,
    )
    assert not any("arbitrating reviewer" in c["content"] for c in llm.calls)
    # No opus involved anywhere: everything stayed on the batch model.
    assert all(c["model"] == "claude-haiku-4-5" for c in llm.calls)
