"""Mechanical verification: is each quoted evidence actually in its source?

This layer is deterministic and costs zero LLM tokens. It runs BEFORE the LLM
critic, because an LLM comparing a claim against a model-written quote cannot
detect a fabricated quote — a substring match against the actually-fetched page
text can.

Statuses:
- verified: quote found in the cached source text (or matches a server-side
  citation from the SAME url, which the API guarantees came from real content)
  at >= threshold.
- failed: we HAVE the source text and the quote is not in it. Never rescued —
  a quote provably absent from its claimed source is misattributed at best.
- skipped: no cached text for that URL (paywall, JS-rendered, video, never
  fetched) — not the model's fault, but confidence gets capped downstream.
"""

from __future__ import annotations

import difflib
import re

from .llm import CitedQuote, SourceDoc
from .schemas import DraftEvidence, Evidence, VerificationStatus

MATCH_THRESHOLD = 0.85
# Below this (normalized chars), a quote is too generic for the citation-rescue
# path — short common phrases appear inside unrelated citations by accident.
MIN_RESCUE_QUOTE_LEN = 30

_WS = re.compile(r"\s+")
_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
        "…": "...",
    }
)


def normalize(text: str) -> str:
    return _WS.sub(" ", text.translate(_PUNCT_MAP)).casefold().strip()


def _anchors(qn: str) -> list[str]:
    """Short substrings of the quote used to locate candidate windows."""
    anchors: list[str] = []
    if len(qn) >= 24:
        anchors.append(qn[:24])
        mid = len(qn) // 2
        anchors.append(qn[mid : mid + 24])
    words = sorted(qn.split(), key=len, reverse=True)
    if words and len(words[0]) >= 6:
        anchors.append(words[0])
    return anchors or [qn]


def _window_score(qn: str, window: str) -> float:
    """Score for the quote appearing verbatim-ish within the window.

    Matched characters are normalized by the SPAN they occupy in the window,
    not just the quote length: a "quote" stitched together from fragments
    scattered across several sentences matches many characters but over a span
    much wider than the quote itself, and scores low. A genuine quote with a
    typo or two occupies a span ≈ its own length and scores high.
    """
    if not qn:
        return 0.0
    sm = difflib.SequenceMatcher(None, qn, window, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return 0.0
    matched = sum(b.size for b in blocks)
    span = (blocks[-1].b + blocks[-1].size) - blocks[0].b
    return matched / max(len(qn), span)


def _best_match(quote_n: str, text_n: str) -> tuple[float, int]:
    """(best score, best position in normalized text; -1 if nothing located)."""
    if not quote_n or not text_n:
        return 0.0, -1
    pos = text_n.find(quote_n)
    if pos != -1:
        return 1.0, pos

    qlen = len(quote_n)
    best = 0.0
    best_pos = -1
    seen_positions: set[int] = set()
    for anchor in _anchors(quote_n):
        start = 0
        hits = 0
        while hits < 20:
            pos = text_n.find(anchor, start)
            if pos == -1:
                break
            hits += 1
            start = pos + 1
            window_start = max(0, pos - qlen)
            if window_start in seen_positions:
                continue
            seen_positions.add(window_start)
            window = text_n[window_start : pos + qlen + 64]
            score = _window_score(quote_n, window)
            if score > best:
                best, best_pos = score, window_start
            if best >= 0.999:
                return best, best_pos
    if best >= MATCH_THRESHOLD:
        return best, best_pos

    # Coarse fallback scan for quotes whose anchors were all slightly wrong.
    # The window extends by `step` so every possible quote offset is fully
    # covered by at least one window.
    step = max(qlen // 2, 24)
    limit = 2000
    for n, i in enumerate(range(0, len(text_n), step)):
        if n >= limit:
            break
        window = text_n[i : i + qlen + step]
        sm = difflib.SequenceMatcher(None, quote_n, window, autojunk=False)
        if sm.real_quick_ratio() < MATCH_THRESHOLD or sm.quick_ratio() < MATCH_THRESHOLD:
            continue
        score = _window_score(quote_n, window)
        if score > best:
            best, best_pos = score, i
        if best >= 0.999:
            break
    return best, best_pos


def match_quote(quote: str, text: str) -> float:
    """Best-effort score in [0, 1] for `quote` appearing verbatim in `text`."""
    score, _ = _best_match(normalize(quote), normalize(text))
    return score


def excerpt_around(quote: str, text: str, radius: int = 500) -> str | None:
    """Slice of the (normalized) source text around the quote's best match.

    Fed to the LLM critic so it judges the quote in context. Uses the location
    where the quote actually matched best — not merely the first occurrence of
    an anchor word, which on long pages can sit far from the real quote.
    """
    qn = normalize(quote)
    tn = normalize(text)
    score, pos = _best_match(qn, tn)
    if pos == -1 or score < 0.3:
        return None
    start = max(0, pos - radius)
    end = min(len(tn), pos + len(qn) + radius)
    return tn[start:end]


def verify_evidence(
    draft: DraftEvidence,
    source_cache: dict[str, SourceDoc],
    cited_quotes: list[CitedQuote],
) -> Evidence:
    status: VerificationStatus
    ratio: float | None

    doc = source_cache.get(draft.source_url)
    if doc is not None:
        ratio = round(match_quote(draft.quote, doc.text), 3)
        status = "verified" if ratio >= MATCH_THRESHOLD else "failed"
    else:
        ratio = None
        status = "skipped"

    if status == "skipped" and cited_quotes:
        # Server-side citations are pre-verified ground truth for pages we
        # never got local text for — but only from the SAME url (otherwise we
        # would stamp "verified" onto a misattributed source), and only for
        # quotes long enough not to match generic citation text by accident.
        # A mechanical "failed" (source text present, quote absent) is final.
        quote_n = normalize(draft.quote)
        if len(quote_n) >= MIN_RESCUE_QUOTE_LEN:
            for cited in cited_quotes:
                if cited.url != draft.source_url:
                    continue
                cited_ratio = match_quote(draft.quote, cited.text)
                if cited_ratio >= MATCH_THRESHOLD:
                    status = "verified"
                    ratio = round(cited_ratio, 3)
                    break

    return Evidence(
        quote=draft.quote,
        source_url=draft.source_url,
        source_title=draft.source_title,
        published=draft.published,
        kind=draft.kind,
        status=status,
        match_ratio=ratio,
    )
