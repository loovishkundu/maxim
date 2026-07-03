from conftest import make_brief, make_finding

from maxim.critic import apply_critique, build_payload
from maxim.llm import SourceDoc
from maxim.schemas import ClaimVerdict, CritiqueResult


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
