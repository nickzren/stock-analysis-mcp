"""Regression tests for decision_context trigger emission.

These tests pin down the contract that:
  - Growth triggers fire for the signal names actually emitted by signals.py
    (`high_revenue_growth` / `negative_revenue_growth`), not the previously
    broken aliases (`high_growth` / `declining_growth`).
"""

from stock_analysis.tools.analyze.decision_context import _build_fundamental_conditions


class TestGrowthTriggers:
    """Growth-decelerates / growth-accelerates emit on the right signal names."""

    def test_high_revenue_growth_emits_growth_decelerates(self) -> None:
        bullish_list = ["high_revenue_growth"]
        bearish_list: list[str] = []

        bullish, bearish = _build_fundamental_conditions(
            is_unprofitable=False,
            net_margin=0.15,
            fcf=1_000_000.0,
            revenue_yoy=0.25,
            next_earnings_date=None,
            bullish_list=bullish_list,
            bearish_list=bearish_list,
        )

        conditions = [c.get("condition") for c in bearish]
        assert "growth_decelerates" in conditions

    def test_negative_revenue_growth_emits_growth_accelerates(self) -> None:
        bullish_list: list[str] = []
        bearish_list = ["negative_revenue_growth"]

        bullish, bearish = _build_fundamental_conditions(
            is_unprofitable=False,
            net_margin=0.10,
            fcf=500_000.0,
            revenue_yoy=-0.05,
            next_earnings_date=None,
            bullish_list=bullish_list,
            bearish_list=bearish_list,
        )

        conditions = [c.get("condition") for c in bullish]
        assert "growth_accelerates" in conditions

    def test_stale_signal_names_do_not_fire(self) -> None:
        """Previously-buggy aliases must not produce growth triggers."""
        bullish_list = ["high_growth"]
        bearish_list = ["declining_growth"]

        bullish, bearish = _build_fundamental_conditions(
            is_unprofitable=False,
            net_margin=0.10,
            fcf=500_000.0,
            revenue_yoy=0.20,
            next_earnings_date=None,
            bullish_list=bullish_list,
            bearish_list=bearish_list,
        )

        all_conditions = [c.get("condition") for c in (bullish + bearish)]
        assert "growth_accelerates" not in all_conditions
        assert "growth_decelerates" not in all_conditions

    def test_no_growth_signal_emits_no_growth_trigger(self) -> None:
        """Without matching growth signals, no growth triggers should fire."""
        bullish, bearish = _build_fundamental_conditions(
            is_unprofitable=False,
            net_margin=0.10,
            fcf=500_000.0,
            revenue_yoy=0.10,
            next_earnings_date=None,
            bullish_list=[],
            bearish_list=[],
        )

        all_conditions = [c.get("condition") for c in (bullish + bearish)]
        assert "growth_accelerates" not in all_conditions
        assert "growth_decelerates" not in all_conditions
