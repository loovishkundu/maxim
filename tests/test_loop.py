"""The loop router is the safety-critical core of M2: these tests pin the
routing table (trigger → action), the priority order, and cap fall-through."""

from maxim.config import DEPTHS, LoopPolicy
from maxim.loop import Decision, IterationOutcome, LoopState, decide

POLICY = LoopPolicy(min_findings=3, max_evidence_retries=2, max_revalidates=2, max_replans=1)


def outcome(**overrides) -> IterationOutcome:
    healthy = dict(
        drafted=8, validated=7, weak=1, unsupported=0, mechanical_failed=0, coverage_gaps=0
    )
    healthy.update(overrides)
    return IterationOutcome(**healthy)


class TestAccept:
    def test_healthy_dossier_accepts(self):
        decision = decide(outcome(), LoopState(), POLICY)
        assert decision.action == "accept"

    def test_exactly_min_findings_accepts(self):
        decision = decide(outcome(validated=3, drafted=4, weak=0), LoopState(), POLICY)
        assert decision.action == "accept"


class TestReplan:
    def test_too_few_findings_triggers_replan(self):
        decision = decide(outcome(validated=1, drafted=2, weak=0), LoopState(), POLICY)
        assert decision.action == "replan"
        assert any("1 validated" in r for r in decision.reasons)

    def test_majority_unsupported_triggers_replan(self):
        decision = decide(
            outcome(drafted=8, validated=3, weak=0, unsupported=5), LoopState(), POLICY
        )
        assert decision.action == "replan"

    def test_coverage_gaps_trigger_replan(self):
        decision = decide(outcome(coverage_gaps=2), LoopState(), POLICY)
        assert decision.action == "replan"

    def test_one_gap_does_not_replan(self):
        decision = decide(outcome(coverage_gaps=1), LoopState(), POLICY)
        assert decision.action == "accept"

    def test_replan_cap_falls_through(self):
        # Structural failure with the replan spent: falls to retry (weak claims
        # exist), never loops on replan.
        decision = decide(
            outcome(drafted=8, validated=2, weak=3, unsupported=3),
            LoopState(replans=1),
            POLICY,
        )
        assert decision.action == "retry"
        assert any("replan cap spent" in r for r in decision.reasons)


class TestRetry:
    def test_weak_claims_trigger_retry(self):
        decision = decide(
            outcome(drafted=10, validated=7, weak=3, unsupported=0), LoopState(), POLICY
        )
        assert decision.action == "retry"

    def test_weak_share_at_threshold_accepts(self):
        # 2/10 == the 0.2 threshold: not strictly above → no retry.
        decision = decide(
            outcome(drafted=10, validated=8, weak=2, unsupported=0), LoopState(), POLICY
        )
        assert decision.action == "accept"

    def test_retry_cap_falls_through_to_accept(self):
        decision = decide(
            outcome(drafted=10, validated=7, weak=3, unsupported=0),
            LoopState(evidence_retries=2),
            POLICY,
        )
        assert decision.action == "accept"
        assert any("caps exhausted" in r for r in decision.reasons)


class TestRevalidate:
    def test_mechanical_failures_trigger_revalidate(self):
        decision = decide(
            outcome(drafted=10, validated=8, weak=0, mechanical_failed=2), LoopState(), POLICY
        )
        assert decision.action == "revalidate"

    def test_widespread_mechanical_failure_is_not_revalidate(self):
        # 4/10 broken quotes exceeds the 30% ceiling — targeted repair will not
        # save this; it accepts (or replans via other triggers), never grinds.
        decision = decide(
            outcome(drafted=10, validated=6, weak=0, mechanical_failed=4), LoopState(), POLICY
        )
        assert decision.action == "accept"

    def test_revalidate_cap_falls_through(self):
        decision = decide(
            outcome(drafted=10, validated=8, weak=0, mechanical_failed=2),
            LoopState(revalidates=2),
            POLICY,
        )
        assert decision.action == "accept"


class TestPriority:
    def test_replan_beats_retry_and_revalidate(self):
        decision = decide(
            outcome(
                drafted=10,
                validated=2,
                weak=3,
                unsupported=3,
                mechanical_failed=2,
                coverage_gaps=3,
            ),
            LoopState(),
            POLICY,
        )
        assert decision.action == "replan"

    def test_retry_beats_revalidate(self):
        decision = decide(
            outcome(drafted=10, validated=6, weak=3, unsupported=0, mechanical_failed=1),
            LoopState(),
            POLICY,
        )
        assert decision.action == "retry"


class TestState:
    def test_spend_increments_the_right_counter(self):
        state = LoopState().spend("retry").spend("retry").spend("replan")
        assert state == LoopState(evidence_retries=2, revalidates=0, replans=1)

    def test_spend_accept_is_identity(self):
        assert LoopState().spend("accept") == LoopState()

    def test_zero_drafted_replans_not_divides(self):
        decision = decide(
            IterationOutcome(
                drafted=0,
                validated=0,
                weak=0,
                unsupported=0,
                mechanical_failed=0,
                coverage_gaps=0,
            ),
            LoopState(),
            POLICY,
        )
        assert decision.action == "replan"  # no findings at all is structural


def test_depth_presets_carry_loop_policies():
    assert DEPTHS["quick"].loop.max_replans == 0  # quick never replans
    assert DEPTHS["standard"].loop.max_replans == 1
    assert DEPTHS["deep"].loop.min_findings > DEPTHS["quick"].loop.min_findings


def test_decision_is_a_value_object():
    assert Decision("accept").action == "accept"
    assert Decision("accept").reasons == []
