from dataclasses import dataclass

from maxim.usage import UsageLedger


@dataclass
class FakeServerToolUse:
    web_search_requests: int = 0
    web_fetch_requests: int = 0


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    server_tool_use: FakeServerToolUse | None = None


def test_cost_math_opus():
    ledger = UsageLedger(budget_usd=100)
    ledger.record(
        "researcher:ai_agentic",
        "claude-opus-4-8",
        FakeUsage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=100_000,
            server_tool_use=FakeServerToolUse(web_search_requests=10),
        ),
    )
    # 1M in * $5 + 100k out * $25/M + 1M cache read * $0.5/M + 100k write * $6.25/M + 10 searches
    expected = 5.0 + 2.5 + 0.5 + 0.625 + 0.1
    assert abs(ledger.cost_usd - expected) < 1e-9


def test_unknown_model_costs_zero():
    ledger = UsageLedger(budget_usd=1)
    ledger.record("x", "some-unknown-model", FakeUsage(input_tokens=10_000_000))
    assert ledger.cost_usd == 0.0


def test_budget_gate():
    ledger = UsageLedger(budget_usd=0.01)
    assert not ledger.over_budget
    ledger.record("x", "claude-opus-4-8", FakeUsage(input_tokens=10_000))
    assert ledger.over_budget  # 10k * $5/M = $0.05 >= $0.01


def test_stage_counts_and_run_usage():
    ledger = UsageLedger(budget_usd=10)
    ledger.record(
        "researcher:statistics",
        "claude-opus-4-8",
        FakeUsage(input_tokens=100, server_tool_use=FakeServerToolUse(3, 2)),
    )
    ledger.record(
        "researcher:statistics",
        "claude-opus-4-8",
        FakeUsage(input_tokens=50, server_tool_use=FakeServerToolUse(1, 1)),
    )
    assert ledger.stage_counts("researcher:statistics") == (4, 3)
    run_usage = ledger.to_run_usage()
    assert len(run_usage.stages) == 1
    stage = run_usage.stages[0]
    assert stage.calls == 2
    assert stage.input_tokens == 150
    assert run_usage.total_cost_usd == round(ledger.cost_usd, 4)


def test_usage_with_missing_fields():
    class Bare:
        input_tokens = 5

    ledger = UsageLedger(budget_usd=10)
    ledger.record("x", "claude-opus-4-8", Bare())  # no output/cache/server fields
    assert ledger.cost_usd > 0
