from conftest import make_dossier, make_plan

from maxim.report import (
    assemble_report,
    fallback_report,
    report_path,
    slugify,
)
from maxim.schemas import RejectedFinding, RunUsage, StageUsage


def _usage() -> RunUsage:
    return RunUsage(
        stages=[
            StageUsage(
                stage="planner",
                model="claude-opus-4-8",
                calls=1,
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=0,
                cache_write_tokens=800,
                web_searches=0,
                web_fetches=0,
                cost_usd=0.02,
            )
        ],
        total_cost_usd=0.02,
        wall_seconds=12.5,
    )


def test_slugify():
    assert slugify("Vector Search for Multi-Tenant SaaS!") == "vector-search-for-multi-tenant-saas"
    assert slugify("///") == "topic"


def test_report_path_collision(tmp_path):
    first = report_path(tmp_path, "my topic")
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("x")
    second = report_path(tmp_path, "my topic")
    assert second != first
    assert second.name.endswith("-2.md")


def test_assemble_resolves_known_and_flags_unknown():
    plan = make_plan()
    dossiers = [make_dossier("ai_agentic", "F-ai1"), make_dossier("statistics", "F-st1")]
    synthesis = "# T\n\nSTL works well [F-ai1]. Bogus claim [F-zz9]."
    warnings: list[str] = []
    report = assemble_report(plan, dossiers, synthesis, _usage(), warnings, "standard")

    assert "[F-ai1]" in report
    assert "[F-zz9 — unresolved citation]" in report
    assert any("F-zz9" in w for w in warnings)
    assert "## Sources" in report
    assert "https://eng.example.com/anomaly-detection" in report
    assert "## Run Metadata" in report
    assert "$0.02" in report


def test_assemble_includes_rejected_appendix():
    plan = make_plan()
    dossier = make_dossier()
    dossier.rejected.append(
        RejectedFinding(finding=make_finding_rejected(), reason="critic verdict: unsupported")
    )
    report = assemble_report(plan, [dossier], "body [F-ai1]", _usage(), [], "quick")
    assert "## Appendix: Rejected Claims" in report
    assert "critic verdict: unsupported" in report


def make_finding_rejected():
    from conftest import make_finding

    f = make_finding("F-ai2")
    return f.model_copy(update={"verdict": "unsupported"})


def test_fallback_report_states_reason_and_keeps_metadata():
    plan = make_plan()
    report = fallback_report(
        plan,
        [make_dossier()],
        reason="all researchers timed out",
        usage=_usage(),
        warnings=["statistics: timed out after 600s"],
        depth="standard",
    )
    assert "all researchers timed out" in report
    assert "budget" not in report.split("\n")[0]  # no false budget claim in the title
    assert "STL decomposition" in report
    assert "https://eng.example.com/anomaly-detection" in report
    assert "## Run Metadata" in report
    assert "statistics: timed out after 600s" in report
