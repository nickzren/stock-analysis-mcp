"""Tests for the shared hard-gate helpers."""

from stock_analysis.tools.analyze.gates import (
    EARNINGS_BLACKOUT_DAYS,
    check_data_quality,
    check_earnings_blackout,
    check_liquidity,
    falling_knife_assessment,
    is_falling_knife_technicals,
)


class TestEarningsBlackout:
    def test_fires_within_window(self) -> None:
        fired, blocker = check_earnings_blackout({"earnings": {"days_until": 3}})
        assert fired is True
        assert blocker == {
            "id": "earnings_blackout",
            "reason": f"earnings in 3 days (<= {EARNINGS_BLACKOUT_DAYS})",
        }

    def test_does_not_fire_outside_window(self) -> None:
        fired, blocker = check_earnings_blackout({"earnings": {"days_until": 6}})
        assert fired is False
        assert blocker is None

    def test_missing_days_until_does_not_fire(self) -> None:
        fired, blocker = check_earnings_blackout({"earnings": {}})
        assert fired is False
        assert blocker is None


class TestLiquidity:
    def test_weak_liquidity_blocker(self) -> None:
        weak, missing, blockers = check_liquidity({"liquidity": {"avg_dollar_volume": 500_000}})
        assert weak is True and missing is False
        assert blockers[0]["id"] == "weak_liquidity"

    def test_missing_liquidity_blocker(self) -> None:
        weak, missing, blockers = check_liquidity({"liquidity": {}})
        assert weak is False and missing is True
        assert blockers[0]["id"] == "liquidity_missing"

    def test_healthy_liquidity_no_blockers(self) -> None:
        weak, missing, blockers = check_liquidity({"liquidity": {"avg_dollar_volume": 50_000_000}})
        assert (weak, missing, blockers) == (False, False, [])


class TestDataQuality:
    def test_critical_tool_failure_fires(self) -> None:
        fired, blocker = check_data_quality(
            {"tool_failures": [{"tool": "technicals", "error": "boom"}]},
            critical_tools=frozenset({"technicals"}),
        )
        assert fired is True
        assert blocker is not None and blocker["id"] == "data_quality_critical"

    def test_clean_inputs_do_not_fire(self) -> None:
        fired, blocker = check_data_quality({"fundamentals_status": "available"})
        assert fired is False and blocker is None


class TestFallingKnife:
    def test_scores_match_dip_rubric(self) -> None:
        score, reasons = falling_knife_assessment(
            death_cross=True,
            below_sma200=True,
            return_3m=-0.35,
            position_in_range=0.05,
            sma_200_slope=-0.001,
            days_since_52w_high=200,
        )
        assert score == 8
        assert reasons == [
            "death_cross_active",
            "below_sma200",
            "severe_3m_decline",
            "near_52w_low",
            "sma200_downtrend",
            "stale_52w_high",
        ]

    def test_none_inputs_score_zero(self) -> None:
        score, reasons = falling_knife_assessment(
            death_cross=None,
            below_sma200=None,
            return_3m=None,
            position_in_range=None,
            sma_200_slope=None,
            days_since_52w_high=None,
        )
        assert score == 0 and reasons == []

    def test_from_technicals_payload(self) -> None:
        technicals_data = {
            "moving_averages": {
                "sma_200_slope_pct_per_day": -0.002,
                "rules": {
                    "death_cross": {"triggered": True},
                    "above_sma200": {"triggered": False},
                },
            },
            "price_position": {"position_in_range": 0.05, "days_since_52w_high": 200},
            "returns": {"return_3m": -0.35},
        }
        is_knife, reasons = is_falling_knife_technicals(technicals_data)
        assert is_knife is True
        assert "death_cross_active" in reasons


class TestFallingKnifeDeclinePrecondition:
    """Structure-lag artifacts alone must not qualify a chart as a knife."""

    def test_recovery_chart_scores_zero(self) -> None:
        # HOOD 2026-07-06 regression: +66% 3m, above SMA200, death cross
        # persistence + barely-negative slope + stale 52w high scored 4.
        score, reasons = falling_knife_assessment(
            death_cross=True,
            below_sma200=False,
            return_3m=0.66,
            position_in_range=0.56,
            sma_200_slope=-0.000514,
            days_since_52w_high=186,
        )
        assert (score, reasons) == (0, [])

    def test_shallow_decline_above_sma200_scores_zero(self) -> None:
        # -1% over 3m above SMA200 is not a knife either (Codex tightening).
        score, reasons = falling_knife_assessment(
            death_cross=True,
            below_sma200=False,
            return_3m=-0.01,
            position_in_range=0.56,
            sma_200_slope=-0.001,
            days_since_52w_high=200,
        )
        assert (score, reasons) == (0, [])

    def test_below_sma200_still_scores_despite_positive_returns(self) -> None:
        score, reasons = falling_knife_assessment(
            death_cross=True,
            below_sma200=True,
            return_3m=0.05,
            position_in_range=0.3,
            sma_200_slope=-0.001,
            days_since_52w_high=200,
        )
        assert score == 5  # 2 + 1 + 1 + 1
        assert "death_cross_active" in reasons

    def test_significant_decline_above_sma200_still_scores(self) -> None:
        # Early violent break: above SMA200 but down >20% in 3m qualifies.
        score, _ = falling_knife_assessment(
            death_cross=True,
            below_sma200=False,
            return_3m=-0.25,
            position_in_range=0.3,
            sma_200_slope=-0.001,
            days_since_52w_high=200,
        )
        assert score >= 4
