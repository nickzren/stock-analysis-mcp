"""Tests for swing setup detection (dict-fixture driven, no dataframes)."""

from typing import Any

from stock_analysis.tools.trade_setup.setups import detect_setup


def make_technicals(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "current_price": 100.0,
        "moving_averages": {
            "sma_20": 99.0,
            "sma_50": 95.0,
            "sma_200": 90.0,
            "sma_200_slope_pct_per_day": 0.001,
            "rules": {
                "above_sma20": {"triggered": True},
                "above_sma50": {"triggered": True},
                "above_sma200": {"triggered": True},
                "golden_cross": {"triggered": True},
                "death_cross": {"triggered": False},
            },
        },
        "rsi": {"value": 45.0, "bullish_divergence": False},
        "atr": {"value": 2.0, "value_pct": 0.02},
        "price_position": {"position_in_range": 0.6, "days_since_52w_high": 30},
        "returns": {"return_3m": 0.05, "return_1w_zscore": 0.0},
        "volume": {"ratio": 1.0},
    }
    base.update(overrides)
    return base


def make_features(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "last_close": 100.0,
        "prior_close": 99.5,
        "prior_day_high": 101.0,
        "prior_day_low": 98.5,
        "high_20d_prior": 105.0,
        "low_20d_prior": 95.0,
        "swing_low": 97.0,
        "last_volume_ratio": 1.0,
        "bandwidth_pctile_6m": 0.5,
        "trigger_state": {
            "close_above_prior_day_high": False,
            "close_above_20d_high": False,
        },
    }
    base.update(overrides)
    return base


class TestPullback:
    def test_detects_pullback_in_uptrend(self) -> None:
        # 100 is ~4.8% off the 105 20d high, RSI 45, uptrend rules all true.
        setup = detect_setup(make_technicals(), make_features(), actionable_price=100.0)
        assert setup is not None
        assert setup["type"] == "pullback_in_uptrend"
        assert setup["trigger_price"] == 101.0  # prior-day high
        assert setup["trigger_satisfied"] is False
        assert setup["stop_price"] == 97.0  # swing low
        assert setup["stop_basis"] == "swing_low"
        assert setup["target_primary"] == {"price": 105.0, "basis": "prior_20d_high"}
        assert setup["stop_price"] < setup["trigger_price"]

    def test_no_pullback_in_downtrend(self) -> None:
        t = make_technicals()
        t["moving_averages"]["rules"]["above_sma200"] = {"triggered": False}
        assert detect_setup(t, make_features(), actionable_price=100.0) is None

    def test_no_pullback_when_rsi_out_of_band(self) -> None:
        t = make_technicals(rsi={"value": 60.0, "bullish_divergence": False})
        assert detect_setup(t, make_features(), actionable_price=100.0) is None

    def test_no_pullback_when_too_extended_off_high(self) -> None:
        f = make_features(high_20d_prior=115.0)  # 13% off high — beyond 8% band
        assert detect_setup(make_technicals(), f, actionable_price=100.0) is None

    def test_atr_stop_fallback_when_no_swing_low(self) -> None:
        f = make_features(swing_low=None)
        setup = detect_setup(make_technicals(), f, actionable_price=100.0)
        assert setup is not None
        assert setup["stop_basis"] == "atr"
        assert setup["stop_price"] == 97.0  # 101 - 2.0 * 2.0


class TestBreakout:
    def _breakout_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # Not a pullback: only 1.9% off the 20d high; compressed bandwidth.
        t = make_technicals(rsi={"value": 60.0, "bullish_divergence": False})
        f = make_features(high_20d_prior=102.0, bandwidth_pctile_6m=0.10,
                          last_volume_ratio=2.0)
        return t, f

    def test_detects_breakout_qualifier(self) -> None:
        t, f = self._breakout_inputs()
        setup = detect_setup(t, f, actionable_price=100.0)
        assert setup is not None
        assert setup["type"] == "breakout"
        assert setup["trigger_price"] == 102.0
        assert setup["trigger_satisfied"] is False
        assert setup["stop_basis"] == "below_breakout_level"
        assert setup["stop_price"] == round(102.0 * 0.99, 2)

    def test_trigger_satisfied_needs_volume(self) -> None:
        t, f = self._breakout_inputs()
        f["last_volume_ratio"] = 1.0
        f["trigger_state"]["close_above_20d_high"] = True
        setup = detect_setup(t, f, actionable_price=102.5)
        assert setup is not None
        assert setup["trigger_satisfied"] is False

    def test_no_breakout_without_compression(self) -> None:
        t, f = self._breakout_inputs()
        f["bandwidth_pctile_6m"] = 0.60
        assert detect_setup(t, f, actionable_price=100.0) is None


class TestMeanReversion:
    def _meanrev_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        t = make_technicals(rsi={"value": 25.0, "bullish_divergence": True})
        t["moving_averages"]["rules"]["golden_cross"] = {"triggered": False}
        t["moving_averages"]["rules"]["above_sma50"] = {"triggered": False}
        t["returns"]["return_3m"] = -0.10
        # 100 vs 105 high is in pullback band, but RSI 25 fails pullback RSI gate.
        f = make_features(high_20d_prior=120.0)  # too far off high for pullback
        return t, f

    def test_detects_oversold_bounce(self) -> None:
        t, f = self._meanrev_inputs()
        setup = detect_setup(t, f, actionable_price=100.0)
        assert setup is not None
        assert setup["type"] == "oversold_mean_reversion"
        assert setup["trigger_price"] == 101.0
        assert setup["stop_basis"] == "atr"
        assert setup["stop_price"] == 96.0  # 101 - 2.5 * 2.0
        # SMA20 (99) is below entry (101) -> falls back to 1R target
        assert setup["target_primary"] is None

    def test_sma20_target_when_above_entry(self) -> None:
        t, f = self._meanrev_inputs()
        t["moving_averages"]["sma_20"] = 108.0
        setup = detect_setup(t, f, actionable_price=100.0)
        assert setup is not None
        assert setup["target_primary"] == {"price": 108.0, "basis": "sma_20"}

    def test_no_meanrev_below_support(self) -> None:
        t, f = self._meanrev_inputs()
        t["moving_averages"]["sma_200"] = 120.0  # price 100 < 0.9 * 120
        assert detect_setup(t, f, actionable_price=100.0) is None


def test_exact_3pct_off_high_boundary_favors_pullback_over_breakout() -> None:
    # off_high == (100 - 97) / 100 == 0.03 exactly: satisfies both
    # PULLBACK_MIN_OFF_HIGH (>=0.03) and BREAKOUT_MAX_BELOW_HIGH (<=0.03).
    # SETUP_PRIORITY checks pullback before breakout, so pullback wins.
    f = make_features(high_20d_prior=100.0, bandwidth_pctile_6m=0.10, last_volume_ratio=2.0)
    setup = detect_setup(make_technicals(), f, actionable_price=97.0)
    assert setup is not None
    assert setup["type"] == "pullback_in_uptrend"


def test_no_setup_on_boring_chart() -> None:
    t = make_technicals(rsi={"value": 60.0, "bullish_divergence": False})
    f = make_features(high_20d_prior=120.0)  # not near high, not oversold
    assert detect_setup(t, f, actionable_price=100.0) is None


def test_returns_none_when_atr_missing_and_no_structural_stop() -> None:
    t = make_technicals(atr={"value": None, "value_pct": None})
    f = make_features(swing_low=None)
    assert detect_setup(t, f, actionable_price=100.0) is None


class TestBreakoutStructuralTargets:
    def _inputs(self, **feat_over: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        t = make_technicals(rsi={"value": 60.0, "bullish_divergence": False})
        t["price_position"]["week_52_high"] = 150.0
        f = make_features(high_20d_prior=102.0, low_20d_prior=95.0,
                          bandwidth_pctile_6m=0.10, last_volume_ratio=2.0)
        f.update(feat_over)
        return t, f

    def test_measured_move_target_when_nearest(self) -> None:
        t, f = self._inputs()
        setup = detect_setup(t, f, actionable_price=100.0)
        assert setup is not None and setup["type"] == "breakout"
        # measured move = 102 + (102 - 95) = 109 < 52w high 150
        assert setup["target_primary"] == {"price": 109.0, "basis": "measured_move"}

    def test_week52_high_wins_when_nearer(self) -> None:
        t, f = self._inputs()
        t["price_position"]["week_52_high"] = 105.0
        setup = detect_setup(t, f, actionable_price=100.0)
        assert setup["target_primary"] == {"price": 105.0, "basis": "week_52_high"}

    def test_run_past_anchor_filters_low_targets(self) -> None:
        # Trigger satisfied at 112: measured move 109 is BELOW the anchor,
        # so the 52w high is the only valid structural target.
        t, f = self._inputs()
        setup = detect_setup(t, f, actionable_price=112.0)
        assert setup is not None
        assert setup["trigger_satisfied"] is True
        assert setup["target_primary"] == {"price": 150.0, "basis": "week_52_high"}

    def test_ath_break_falls_back_to_none(self) -> None:
        # Anchor above every candidate -> None (plan builder's 1R fallback).
        t, f = self._inputs()
        t["price_position"]["week_52_high"] = 102.0
        setup = detect_setup(t, f, actionable_price=115.0)
        assert setup is not None
        assert setup["target_primary"] is None

    def test_candidate_equal_to_anchor_is_filtered(self) -> None:
        # Strictly-above means a candidate AT the anchor is excluded:
        # measured move = 102 + (102 - 95) = 109 == anchor when the run-past
        # actionable price is exactly 109.
        t, f = self._inputs()
        setup = detect_setup(t, f, actionable_price=109.0)
        assert setup is not None
        assert setup["target_primary"] == {"price": 150.0, "basis": "week_52_high"}
