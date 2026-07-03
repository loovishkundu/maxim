"""Mechanical sentiment rigor: floors, corroboration, pulse honesty."""

from conftest import make_finding

from maxim.schemas import EngagementStats, Evidence
from maxim.sentiment import apply_sentiment_rigor, build_pulse, meets_floor


def ev(url, status="verified", engagement=None, title="Thread") -> Evidence:
    return Evidence(
        quote="a long enough verbatim quote about the method here",
        source_url=url,
        source_title=title,
        published=None,
        kind="anecdote",
        status=status,
        match_ratio=1.0 if status == "verified" else None,
        engagement=engagement,
    )


def community_finding(fid, evidence, sentiment="positive", method="STL", how=()) -> object:
    base = make_finding(fid, perspective="community")
    return base.model_copy(
        update={
            "evidence": evidence,
            "sentiment": sentiment,
            "method_name": method,
            "how_people_test_it": list(how),
        }
    )


class TestFloors:
    def test_unknown_engagement_passes(self):
        assert meets_floor(None)

    def test_hn_floor(self):
        assert meets_floor(EngagementStats(source="hn", points=10, comments=0))
        assert meets_floor(EngagementStats(source="hn", points=0, comments=5))
        assert not meets_floor(EngagementStats(source="hn", points=9, comments=4))

    def test_github_floor(self):
        assert meets_floor(EngagementStats(source="github", reactions=3))
        assert meets_floor(EngagementStats(source="github", reactions=0, comments=5))
        assert not meets_floor(EngagementStats(source="github", reactions=2, comments=1))

    def test_reddit_floor(self):
        assert meets_floor(EngagementStats(source="reddit", points=20))
        assert not meets_floor(EngagementStats(source="reddit", points=19))


HOT = EngagementStats(source="hn", points=100, comments=50)
COLD = EngagementStats(source="hn", points=1, comments=0)


class TestCorroboration:
    def test_two_qualifying_threads_keep_sentiment(self):
        finding = community_finding(
            "F-cm1", [ev("https://a.test/1", engagement=HOT), ev("https://b.test/2")]
        )
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment == "positive"
        assert out.sentiment_sample_size == 2

    def test_single_thread_demotes_sentiment(self):
        finding = community_finding("F-cm1", [ev("https://a.test/1", engagement=HOT)])
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment is None
        assert out.sentiment_sample_size == 1
        assert any("single-anecdote" in c for c in out.caveats)

    def test_below_floor_thread_does_not_corroborate(self):
        finding = community_finding(
            "F-cm1",
            [ev("https://a.test/1", engagement=HOT), ev("https://b.test/2", engagement=COLD)],
        )
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment is None  # cold thread doesn't count: 1 < 2

    def test_failed_evidence_does_not_corroborate(self):
        finding = community_finding(
            "F-cm1",
            [ev("https://a.test/1", engagement=HOT), ev("https://b.test/2", status="failed")],
        )
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment is None

    def test_non_community_findings_untouched(self):
        finding = make_finding("F-ai1")
        (out,) = apply_sentiment_rigor([finding])
        assert out is finding


class TestPulse:
    def test_insufficient_data_below_three_threads(self):
        findings = apply_sentiment_rigor(
            [
                community_finding(
                    "F-cm1", [ev("https://a.test/1", engagement=HOT), ev("https://b.test/2")]
                )
            ]
        )
        (pulse,) = build_pulse(findings)
        assert pulse.sentiment == "insufficient_data"
        assert pulse.sample_size == 2

    def test_majority_sentiment_across_findings(self):
        findings = apply_sentiment_rigor(
            [
                community_finding("F-cm1", [ev("https://a.test/1"), ev("https://b.test/2")]),
                community_finding("F-cm2", [ev("https://c.test/3"), ev("https://d.test/4")]),
                community_finding(
                    "F-cm3",
                    [ev("https://e.test/5"), ev("https://f.test/6")],
                    sentiment="negative",
                ),
            ]
        )
        (pulse,) = build_pulse(findings)
        assert pulse.sentiment == "positive"  # 2 positive vs 1 negative
        assert pulse.sample_size == 6

    def test_tie_renders_mixed(self):
        findings = apply_sentiment_rigor(
            [
                community_finding("F-cm1", [ev("https://a.test/1"), ev("https://b.test/2")]),
                community_finding(
                    "F-cm2",
                    [ev("https://c.test/3"), ev("https://d.test/4")],
                    sentiment="negative",
                ),
            ]
        )
        (pulse,) = build_pulse(findings)
        assert pulse.sentiment == "mixed"

    def test_how_people_test_it_merged_unique(self):
        findings = [
            community_finding(
                "F-cm1",
                [ev("https://a.test/1"), ev("https://b.test/2")],
                how=["injected-anomaly benchmarks", "shadow deployments"],
            ),
            community_finding(
                "F-cm2",
                [ev("https://c.test/3"), ev("https://d.test/4")],
                how=["Injected-anomaly benchmarks", "A/B against Prophet"],
            ),
        ]
        (pulse,) = build_pulse(apply_sentiment_rigor(findings))
        assert pulse.how_people_test_it == [
            "injected-anomaly benchmarks",
            "shadow deployments",
            "A/B against Prophet",
        ]

    def test_methods_grouped_separately(self):
        findings = [
            community_finding("F-cm1", [ev("https://a.test/1")], method="STL"),
            community_finding("F-cm2", [ev("https://b.test/2")], method="Prophet"),
        ]
        pulses = build_pulse(apply_sentiment_rigor(findings))
        assert [p.method for p in pulses] == ["Prophet", "STL"]

    def test_no_community_findings_no_pulse(self):
        assert build_pulse([make_finding("F-ai1")]) == []


class TestThreadIdentity:
    def test_hn_item_and_article_urls_count_as_one_thread(self):
        # Same HN discussion registered under two URLs — corroboration must
        # not be satisfied by one thread talking to itself.
        stats = EngagementStats(
            source="hn",
            points=100,
            comments=50,
            thread_id="https://news.ycombinator.com/item?id=101",
        )
        finding = community_finding(
            "F-cm1",
            [
                ev("https://news.ycombinator.com/item?id=101", engagement=stats),
                ev("https://blog.example.com/stl", engagement=stats),
            ],
        )
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment_sample_size == 1
        assert out.sentiment is None  # demoted: one thread, not two

    def test_url_spelling_drift_counts_as_one_thread(self):
        finding = community_finding(
            "F-cm1",
            [ev("https://a.test/thread/"), ev("HTTPS://A.TEST/thread#comment-3")],
        )
        (out,) = apply_sentiment_rigor([finding])
        assert out.sentiment_sample_size == 1

    def test_pulse_threads_deduped_by_identity(self):
        stats = EngagementStats(source="hn", points=100, thread_id="hn:101")
        findings = apply_sentiment_rigor(
            [
                community_finding(
                    "F-cm1",
                    [
                        ev("https://a.test/1", engagement=stats),
                        ev("https://b.test/2", engagement=stats),
                    ],
                )
            ]
        )
        (pulse,) = build_pulse(findings)
        assert pulse.sample_size == 1


def test_normalize_url():
    from maxim.sentiment import normalize_url

    assert normalize_url("HTTPS://Example.COM/Path/") == "https://example.com/Path"
    assert normalize_url("https://a.test/x#frag") == "https://a.test/x"
    assert normalize_url("https://a.test/x?q=1") == "https://a.test/x?q=1"
