"""Critic calibration: known-good and known-bad finding/quote pairs the live
critic must classify correctly — guards against judge drift when prompts or
models change.

Excluded from the default run (costs real API money). Run explicitly:

    uv run pytest -m calibration
"""

import os

import pytest
from conftest import PAGE_TEXT, SOURCE_URL, make_brief, make_finding
from dotenv import load_dotenv

from maxim.config import Settings
from maxim.critic import apply_critique, critique
from maxim.llm import LLM, SourceDoc
from maxim.usage import UsageLedger

load_dotenv()

pytestmark = [
    pytest.mark.calibration,
    pytest.mark.skipif(
        not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        reason="calibration needs live API credentials",
    ),
]

CACHE = {SOURCE_URL: SourceDoc(url=SOURCE_URL, text=PAGE_TEXT)}


async def _judge(finding):
    llm = LLM(settings=Settings(), ledger=UsageLedger(budget_usd=2.0))
    try:
        result = await critique(
            stage="critic:calibration",
            brief=make_brief(),
            findings=[finding],
            source_cache=CACHE,
            settings=Settings(),
            llm=llm,
        )
    finally:
        await llm.close()
    return apply_critique([finding], result)


async def test_grounded_claim_is_supported():
    # The quote directly carries the claim: the critic must say supported.
    validated, rejected = await _judge(make_finding("F-ai1", status="verified"))
    assert not rejected
    assert validated[0].verdict == "supported"
    assert validated[0].confidence == "high"  # verified + tier B + supported


async def test_claim_contradicted_by_its_own_quote_is_rejected():
    # The quote says STL CAUGHT 92% of anomalies; the claim asserts the
    # opposite. A drifting judge that rubber-stamps this must fail CI.
    finding = make_finding("F-ai1", status="verified").model_copy(
        update={
            "claim": (
                "STL missed most injected anomalies on vehicle telemetry and "
                "produced a high false-positive rate."
            )
        }
    )
    validated, rejected = await _judge(finding)
    assert rejected, "contradicted claim was not rejected"
    assert rejected[0].finding.verdict in ("contradicted", "unsupported")


async def test_plausible_claim_without_source_text_is_not_supported():
    # Evidence whose source text is unavailable must make the critic MORE
    # skeptical, not less — plausibility alone is not support.
    finding = make_finding("F-ai1", status="skipped").model_copy(
        update={
            "claim": (
                "STL is the industry-standard anomaly detector across all "
                "vehicle telemetry pipelines worldwide."
            )
        }
    )
    llm = LLM(settings=Settings(), ledger=UsageLedger(budget_usd=2.0))
    try:
        result = await critique(
            stage="critic:calibration",
            brief=make_brief(),
            findings=[finding],
            source_cache={},  # nothing fetched: no context excerpt available
            settings=Settings(),
            llm=llm,
        )
    finally:
        await llm.close()
    validated, rejected = apply_critique([finding], result)
    if validated:
        assert validated[0].verdict != "supported"
        assert validated[0].confidence == "low"
