from maxim.llm import CitedQuote, SourceDoc
from maxim.schemas import DraftEvidence
from maxim.verification import excerpt_around, match_quote, normalize, verify_evidence

PAGE = """
Gradient boosting remains the strongest baseline for tabular data in production.
In our benchmarks, XGBoost outperformed the neural approaches on 7 of 9 datasets,
while requiring an order of magnitude less tuning effort. “Deep learning is not a
silver bullet for tabular problems,” the authors conclude.
"""


def _evidence(quote: str, url: str = "https://example.com/post") -> DraftEvidence:
    return DraftEvidence(
        quote=quote, source_url=url, source_title="Example", published=None, kind="blog"
    )


class TestNormalize:
    def test_collapses_whitespace_and_case(self):
        assert normalize("  Hello\n\tWORLD  ") == "hello world"

    def test_maps_curly_punctuation(self):
        assert normalize("“smart” — quotes’") == '"smart" - quotes\''


class TestMatchQuote:
    def test_exact_substring(self):
        assert match_quote("XGBoost outperformed the neural approaches", PAGE) == 1.0

    def test_whitespace_and_case_insensitive(self):
        assert match_quote("xgboost OUTPERFORMED   the neural approaches", PAGE) == 1.0

    def test_curly_vs_straight_quotes(self):
        quote = '"Deep learning is not a silver bullet for tabular problems,"'
        assert match_quote(quote, PAGE) == 1.0

    def test_minor_typo_still_matches(self):
        quote = "XGBoost outperformd the neural approaches on 7 of 9 datasets"
        assert match_quote(quote, PAGE) >= 0.85

    def test_paraphrase_fails(self):
        quote = "Boosted trees beat deep nets in most of the evaluated benchmark suites"
        assert match_quote(quote, PAGE) < 0.85

    def test_fabricated_fails(self):
        quote = "LightGBM achieved a 40% latency reduction over all competitors"
        assert match_quote(quote, PAGE) < 0.85

    def test_stitched_quote_fails(self):
        # A "quote" spliced from fragments of adjacent sentences must not pass:
        # matched characters are spread over a span much wider than the quote.
        page = (
            "Gradient boosting models such as XGBoost are widely deployed. In several "
            "published benchmarks they outperformed neural network approaches on most "
            "tabular datasets, while the tuning effort required was an order of "
            "magnitude lower than for deep models."
        )
        stitched = (
            "Gradient boosting outperformed neural network approaches on most tabular "
            "datasets with lower tuning effort"
        )
        assert match_quote(stitched, page) < 0.85

    def test_empty(self):
        assert match_quote("", PAGE) == 0.0
        assert match_quote("something", "") == 0.0


class TestVerifyEvidence:
    CACHE = {"https://example.com/post": SourceDoc(url="https://example.com/post", text=PAGE)}

    def test_verified_from_cache(self):
        ev = verify_evidence(
            _evidence("XGBoost outperformed the neural approaches"), self.CACHE, []
        )
        assert ev.status == "verified"
        assert ev.match_ratio == 1.0

    def test_failed_when_quote_absent(self):
        ev = verify_evidence(_evidence("a completely invented sentence about SVMs"), self.CACHE, [])
        assert ev.status == "failed"

    def test_skipped_when_url_not_cached(self):
        ev = verify_evidence(_evidence("anything", url="https://not-fetched.com/x"), self.CACHE, [])
        assert ev.status == "skipped"
        assert ev.match_ratio is None

    def test_cited_quote_rescues_uncached_url(self):
        cited = [
            CitedQuote(
                text="XGBoost outperformed the neural approaches",
                url="https://not-fetched.com/x",
                title=None,
            )
        ]
        ev = verify_evidence(
            _evidence(
                "XGBoost outperformed the neural approaches", url="https://not-fetched.com/x"
            ),
            self.CACHE,
            cited,
        )
        assert ev.status == "verified"

    def test_cited_quote_from_other_url_does_not_rescue(self):
        # A citation from a DIFFERENT page must not verify a misattributed quote.
        cited = [
            CitedQuote(
                text="XGBoost outperformed the neural approaches",
                url="https://other-page.com/y",
                title=None,
            )
        ]
        ev = verify_evidence(
            _evidence(
                "XGBoost outperformed the neural approaches", url="https://not-fetched.com/x"
            ),
            self.CACHE,
            cited,
        )
        assert ev.status == "skipped"

    def test_cited_quote_never_overrides_mechanical_failed(self):
        # Source text IS cached and the quote is provably absent: final verdict,
        # no citation rescue.
        cited = [
            CitedQuote(
                text="a completely invented sentence about SVMs",
                url="https://example.com/post",
                title=None,
            )
        ]
        ev = verify_evidence(
            _evidence("a completely invented sentence about SVMs"), self.CACHE, cited
        )
        assert ev.status == "failed"

    def test_short_generic_quote_not_rescued(self):
        cited = [CitedQuote(text="state of the art results", url="https://u.com/a", title=None)]
        ev = verify_evidence(_evidence("state of the art", url="https://u.com/a"), {}, cited)
        assert ev.status == "skipped"  # too short for citation rescue


class TestExcerpt:
    def test_excerpt_contains_quote(self):
        out = excerpt_around("XGBoost outperformed the neural approaches", PAGE, radius=40)
        assert out is not None
        assert "xgboost outperformed" in out

    def test_no_locatable_anchor(self):
        assert excerpt_around("zzz qqq totally absent phrase", PAGE) is None
