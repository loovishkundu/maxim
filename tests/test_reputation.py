import datetime as dt

from maxim.reputation import (
    BLOCKED_DOMAINS,
    classify_source,
    effective_half_life,
    parse_published,
    recency_score,
    stamp_evidence,
    tier_badge,
    tier_mix,
)
from maxim.schemas import DraftEvidence
from maxim.verification import verify_evidence


class TestClassifySource:
    def test_registry_tier_a(self):
        assert classify_source("https://arxiv.org/abs/2401.1", "paper") == "A"
        assert classify_source("https://www.arxiv.org/abs/2401.1", "paper") == "A"
        assert classify_source("https://proceedings.mlr.press/v202/x.html", "paper") == "A"

    def test_registry_tier_a_subdomain(self):
        assert classify_source("https://ieeexplore.ieee.org/document/1", "paper") == "A"
        assert classify_source("https://link.springer.com/article/1", "article") == "A"

    def test_registry_tier_b(self):
        assert classify_source("https://netflixtechblog.com/x", "blog") == "B"
        assert classify_source("https://eng.uber.com/x", "blog") == "B"

    def test_registry_tier_d_forums(self):
        assert classify_source("https://news.ycombinator.com/item?id=1", "anecdote") == "D"
        assert classify_source("https://www.reddit.com/r/mlops/x", "anecdote") == "D"
        assert classify_source("https://stackoverflow.com/questions/1", "docs") == "D"

    def test_kind_fallback_paper_is_a(self):
        assert classify_source("https://unknown-journal.org/p/12", "paper") == "A"

    def test_kind_fallback_docs_and_talk_are_b(self):
        assert classify_source("https://some-project.io/docs/x", "docs") == "B"
        assert classify_source("https://someconf.com/talks/42", "talk") == "B"

    def test_eng_subdomain_blog_is_b(self):
        assert classify_source("https://eng.example-corp.com/post", "blog") == "B"
        assert classify_source("https://engineering.acme.dev/post", "blog") == "B"

    def test_unknown_blog_is_c(self):
        assert classify_source("https://randomblog.dev/post", "blog") == "C"
        assert classify_source("https://mysite.com/article", "article") == "C"

    def test_unknown_anecdote_is_d(self):
        assert classify_source("https://someforum.io/thread/9", "anecdote") == "D"

    def test_github_split_by_kind(self):
        assert classify_source("https://github.com/org/repo/issues/1", "anecdote") == "D"
        assert classify_source("https://github.com/org/repo", "docs") == "C"

    def test_garbage_url(self):
        assert classify_source("not a url", "blog") == "D"


class TestParsePublished:
    def test_iso_formats(self):
        assert parse_published("2026-01-15") == dt.date(2026, 1, 15)
        assert parse_published("2026-01") == dt.date(2026, 1, 1)

    def test_human_formats(self):
        assert parse_published("March 5, 2025") == dt.date(2025, 3, 5)
        assert parse_published("Mar 2025") == dt.date(2025, 3, 1)

    def test_bare_year_maps_to_midyear(self):
        assert parse_published("published in 2023") == dt.date(2023, 7, 1)

    def test_unparseable(self):
        assert parse_published(None) is None
        assert parse_published("last Tuesday") is None
        assert parse_published("") is None


class TestRecency:
    TODAY = dt.date(2026, 7, 3)

    def test_half_life_exact(self):
        # effective half-life for ai_agentic with a 12-month horizon is
        # sqrt(12*12) = 12 months; a 12-month-old source scores ~0.5.
        published = dt.date(2025, 7, 3)
        score = recency_score(published, "ai_agentic", 12, today=self.TODAY)
        assert abs(score - 0.5) < 0.02

    def test_stats_ages_slowly(self):
        published = dt.date(2016, 7, 3)  # 10 years old
        stats = recency_score(published, "statistics", 120, today=self.TODAY)
        ai = recency_score(published, "ai_agentic", 12, today=self.TODAY)
        assert stats > 0.4
        assert ai < 0.01

    def test_horizon_modulates(self):
        # Fast-moving topic (12mo horizon) shortens the statistics half-life.
        assert effective_half_life("statistics", 12) < effective_half_life("statistics", 120)

    def test_clamped(self):
        assert effective_half_life("statistics", 480) <= 120.0
        assert effective_half_life("ai_agentic", 1) >= 6.0

    def test_unknown_date_scores_none(self):
        assert recency_score(None, "ai_agentic", 12) is None

    def test_future_date_scores_one(self):
        future = dt.date(2026, 8, 1)
        assert recency_score(future, "ai_agentic", 12, today=self.TODAY) == 1.0


class TestStampAndMix:
    def test_stamp_evidence(self):
        draft = DraftEvidence(
            quote="q" * 40,
            source_url="https://arxiv.org/abs/2401.1",
            source_title="Paper",
            published="2026-01-15",
            kind="paper",
        )
        ev = stamp_evidence(verify_evidence(draft, {}, []), "classical_ml", 24)
        assert ev.tier == "A"
        assert ev.recency_score is not None
        assert 0 < ev.recency_score <= 1

    def test_tier_mix_and_badge(self):
        mix = tier_mix(["A", "A", "B", "D"])
        assert mix == {"A": 0.5, "B": 0.25, "D": 0.25}
        assert tier_badge(mix) == "A:50% B:25% D:25%"

    def test_empty_mix(self):
        assert tier_mix([]) == {}
        assert tier_badge({}) == "no sources"


def test_blocked_domains_wired_into_tools():
    from maxim.config import Settings
    from maxim.researcher import _web_tools

    for tool in _web_tools(Settings()):
        assert tool["blocked_domains"] == BLOCKED_DOMAINS


class TestReviewRegressions:
    def test_github_lookalike_domains_not_matched(self):
        # host.startswith("github.com") used to match unrelated registrable
        # domains; suffix matching also has to catch real subdomains.
        assert classify_source("https://gist.github.com/u/1", "docs") == "C"
        assert classify_source("https://github.community/t/1", "anecdote") == "D"  # kind fallback
        assert classify_source("https://github.comfoo.io/x", "blog") == "C"  # not github

    def test_iso_timestamp_parses_exact_date(self):
        assert parse_published("2026-01-15T00:00:00Z") == dt.date(2026, 1, 15)
        assert parse_published("2026-01-15T09:30:00+02:00") == dt.date(2026, 1, 15)
