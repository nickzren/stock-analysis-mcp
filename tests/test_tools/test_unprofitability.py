"""Tests for the shared unprofitability classifier."""

from stock_analysis.tools.analyze.signals import (
    UnprofitabilityResult,
    classify_unprofitability,
)


class TestClassifyUnprofitability:
    """Tests covering each individual trigger and the per-call-site composite predicates."""

    def test_no_inputs_returns_no_triggers(self) -> None:
        result = classify_unprofitability()
        assert result.triggers == frozenset()
        assert result.any is False
        assert result.by_margin is False
        assert result.by_eps is False
        assert result.by_signal is False
        assert result.by_pe_not_meaningful is False
        assert result.by_no_pe_with_negative_fcf is False

    def test_negative_margin_fires_by_margin(self) -> None:
        result = classify_unprofitability(net_margin=-0.05)
        assert result.by_margin is True
        assert result.any is True

    def test_zero_margin_does_not_fire(self) -> None:
        result = classify_unprofitability(net_margin=0.0)
        assert result.by_margin is False

    def test_positive_margin_does_not_fire(self) -> None:
        result = classify_unprofitability(net_margin=0.10)
        assert result.by_margin is False

    def test_zero_eps_fires_by_eps(self) -> None:
        """trailing_eps <= 0 is the threshold (zero counts as nonpositive)."""
        result = classify_unprofitability(trailing_eps=0.0)
        assert result.by_eps is True

    def test_negative_eps_fires_by_eps(self) -> None:
        result = classify_unprofitability(trailing_eps=-1.50)
        assert result.by_eps is True

    def test_positive_eps_does_not_fire(self) -> None:
        result = classify_unprofitability(trailing_eps=2.0)
        assert result.by_eps is False

    def test_signaled_unprofitable_fires_by_signal(self) -> None:
        result = classify_unprofitability(signaled_unprofitable=True)
        assert result.by_signal is True

    def test_pe_not_meaningful_fires_only_when_set(self) -> None:
        assert classify_unprofitability(pe_not_meaningful=True).by_pe_not_meaningful is True
        assert classify_unprofitability(pe_not_meaningful=False).by_pe_not_meaningful is False

    def test_no_pe_with_negative_fcf_requires_both_conditions(self) -> None:
        # pe_trailing None AND has_negative_fcf_signal True -> fires
        result = classify_unprofitability(pe_trailing=None, has_negative_fcf_signal=True)
        assert result.by_no_pe_with_negative_fcf is True
        # pe_trailing present -> does not fire even if FCF negative
        result = classify_unprofitability(pe_trailing=15.0, has_negative_fcf_signal=True)
        assert result.by_no_pe_with_negative_fcf is False
        # pe_trailing None but no negative FCF -> does not fire
        result = classify_unprofitability(pe_trailing=None, has_negative_fcf_signal=False)
        assert result.by_no_pe_with_negative_fcf is False

    def test_multiple_triggers_compose(self) -> None:
        result = classify_unprofitability(
            net_margin=-0.05,
            trailing_eps=-1.0,
            signaled_unprofitable=True,
        )
        assert result.by_margin is True
        assert result.by_eps is True
        assert result.by_signal is True
        assert len(result.triggers) == 3


class TestOrchestratorSemantics:
    """Orchestrator uses by_margin | by_eps (no signals, no PE inference)."""

    @staticmethod
    def _orchestrator_predicate(result: UnprofitabilityResult) -> bool:
        return result.by_margin or result.by_eps

    def test_orchestrator_ignores_signaled_alone(self) -> None:
        """Drift: orchestrator does NOT flag based on signal alone."""
        result = classify_unprofitability(signaled_unprofitable=True)
        assert self._orchestrator_predicate(result) is False

    def test_orchestrator_ignores_pe_not_meaningful_alone(self) -> None:
        result = classify_unprofitability(pe_not_meaningful=True)
        assert self._orchestrator_predicate(result) is False

    def test_orchestrator_fires_on_margin(self) -> None:
        result = classify_unprofitability(net_margin=-0.01)
        assert self._orchestrator_predicate(result) is True

    def test_orchestrator_fires_on_eps(self) -> None:
        result = classify_unprofitability(trailing_eps=-0.5)
        assert self._orchestrator_predicate(result) is True


class TestActionZonesSemantics:
    """Action zones uses by_margin | by_eps | by_signal | by_no_pe_with_negative_fcf."""

    @staticmethod
    def _action_zones_predicate(result: UnprofitabilityResult) -> bool:
        return (
            result.by_margin
            or result.by_eps
            or result.by_signal
            or result.by_no_pe_with_negative_fcf
        )

    def test_action_zones_fires_on_signal_alone(self) -> None:
        result = classify_unprofitability(signaled_unprofitable=True)
        assert self._action_zones_predicate(result) is True

    def test_action_zones_fires_on_no_pe_with_negative_fcf(self) -> None:
        """Drift: action_zones infers unprofitability when there's no P/E and FCF is negative."""
        result = classify_unprofitability(pe_trailing=None, has_negative_fcf_signal=True)
        assert self._action_zones_predicate(result) is True

    def test_action_zones_ignores_pe_not_meaningful_alone(self) -> None:
        """Drift: action_zones does NOT look at pe_not_meaningful."""
        result = classify_unprofitability(pe_not_meaningful=True)
        assert self._action_zones_predicate(result) is False

    def test_action_zones_skip_when_only_pe_present(self) -> None:
        """No triggers when only a valid PE is provided."""
        result = classify_unprofitability(pe_trailing=15.0)
        assert self._action_zones_predicate(result) is False


class TestVerdictBusinessQualitySemantics:
    """Verdict (business_quality) uses by_margin | by_eps | by_signal | by_pe_not_meaningful."""

    @staticmethod
    def _verdict_predicate(result: UnprofitabilityResult) -> bool:
        return (
            result.by_margin
            or result.by_eps
            or result.by_signal
            or result.by_pe_not_meaningful
        )

    def test_verdict_fires_on_pe_not_meaningful_alone(self) -> None:
        """Drift: verdict flags when PE is not meaningful even without margin/eps data."""
        result = classify_unprofitability(pe_not_meaningful=True)
        assert self._verdict_predicate(result) is True

    def test_verdict_ignores_no_pe_with_negative_fcf(self) -> None:
        """Drift: verdict does NOT use the no-PE + negative-FCF inference."""
        result = classify_unprofitability(pe_trailing=None, has_negative_fcf_signal=True)
        assert self._verdict_predicate(result) is False

    def test_verdict_fires_on_signal(self) -> None:
        result = classify_unprofitability(signaled_unprofitable=True)
        assert self._verdict_predicate(result) is True

    def test_verdict_fires_on_margin(self) -> None:
        result = classify_unprofitability(net_margin=-0.02)
        assert self._verdict_predicate(result) is True


class TestKnownDriftCases:
    """Cases where the three call sites historically disagree.

    These tests document the drift so future alignment is an intentional decision.
    """

    def test_signal_only_orchestrator_disagrees(self) -> None:
        """Only the signal fires. orchestrator says no, action_zones/verdict say yes."""
        result = classify_unprofitability(signaled_unprofitable=True)
        assert (result.by_margin or result.by_eps) is False  # orchestrator
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_no_pe_with_negative_fcf
        ) is True  # action_zones
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_pe_not_meaningful
        ) is True  # verdict

    def test_pe_not_meaningful_only_verdict_unique(self) -> None:
        """Only pe_not_meaningful fires. Only verdict flags."""
        result = classify_unprofitability(pe_not_meaningful=True)
        assert (result.by_margin or result.by_eps) is False
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_no_pe_with_negative_fcf
        ) is False
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_pe_not_meaningful
        ) is True

    def test_no_pe_negative_fcf_only_action_zones_unique(self) -> None:
        """Only no-PE + negative-FCF fires. Only action_zones flags."""
        result = classify_unprofitability(
            pe_trailing=None,
            has_negative_fcf_signal=True,
        )
        assert (result.by_margin or result.by_eps) is False
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_no_pe_with_negative_fcf
        ) is True
        assert (
            result.by_margin or result.by_eps or result.by_signal
            or result.by_pe_not_meaningful
        ) is False
