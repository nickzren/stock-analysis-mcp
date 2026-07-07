"""Screen must agree with the full card's detectors: no false negatives."""

import math
from typing import Any

import pandas as pd

from stock_analysis.tools.trade_setup.setups import detect_setup
from stock_analysis.utils.scan_screen import build_screen_payload, screen_symbol
from stock_analysis.utils.swing_features import compute_setup_features


def make_df(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    volumes = volumes or [1_000_000.0] * n
    dates = pd.date_range("2025-06-02", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": list(dates), "open": closes, "high": highs,
                         "low": lows, "close": closes, "volume": volumes})


def uptrend_breakout_df():
    # 260 bars: long gentle rise -> SMA200 defined, golden structure, tight tail
    closes = [80.0 + i * 0.12 for i in range(260)]
    volumes = [1_000_000.0] * 259 + [2_500_000.0]
    return make_df(closes, volumes=volumes)


def zscore_meltdown_df():
    # Long steady uptrend, then a sharp one-week drop that leaves RSI above 30
    # but the 1-week return z-score deeply negative. Above 0.9*SMA200.
    closes = [80.0 + i * 0.12 for i in range(255)] + [108.8, 108.4, 108.0, 107.6, 107.5]
    return make_df(closes)


class TestPayloadCompleteness:
    def test_every_detector_input_present(self) -> None:
        payload = build_screen_payload(uptrend_breakout_df())
        assert payload is not None
        ma = payload["moving_averages"]
        for key in ("sma_20", "sma_50", "sma_200", "sma_200_slope_pct_per_day"):
            assert key in ma
        for rule in ("above_sma20", "above_sma50", "above_sma200",
                     "golden_cross", "death_cross"):
            assert "triggered" in ma["rules"][rule]
        assert "value" in payload["rsi"]
        assert payload["rsi"]["bullish_divergence"] is False  # documented approx
        assert "value" in payload["atr"]
        assert "return_3m" in payload["returns"]
        assert "return_1w_zscore" in payload["returns"]  # AMENDMENT
        for key in ("week_52_high", "week_52_low", "position_in_range",
                    "days_since_52w_high"):
            assert key in payload["price_position"]
        assert "ratio" in payload["volume"]
        assert "current_price" in payload

    def test_short_frame_returns_none(self) -> None:
        assert build_screen_payload(make_df([100.0] * 10)) is None
        assert build_screen_payload(None) is None


class TestConsistencyBattery:
    """Screen promote/terminate must equal the shared detectors' verdict."""

    def _card_side(self, df) -> dict[str, Any] | None:
        payload = build_screen_payload(df)
        features = compute_setup_features(df)
        assert payload is not None and features is not None
        return detect_setup(payload, features, float(payload["current_price"]))

    def test_breakout_promotes(self) -> None:
        df = uptrend_breakout_df()
        screen = screen_symbol(df)
        assert screen["promote"] is True
        assert screen["setup_type"] == self._card_side(df)["type"]
        assert screen["trigger_price"] == self._card_side(df)["trigger_price"]

    def test_zscore_only_meanrev_promotes(self) -> None:
        # AMENDMENT named regression: RSI >= 30 but zscore < -2 must promote.
        df = zscore_meltdown_df()
        payload = build_screen_payload(df)
        assert payload["rsi"]["value"] >= 30.0
        assert payload["returns"]["return_1w_zscore"] < -2.0
        screen = screen_symbol(df)
        assert screen["promote"] is True
        assert screen["setup_type"] == "oversold_mean_reversion"

    def test_boring_chart_terminates_no_setup(self) -> None:
        # Gentle uptrend, then a mild drift off the highs, both with realistic
        # day-to-day noise (a noiseless straight line makes RSI/z-score read
        # any single-direction drift as an extreme, which is not "boring").
        level = 100.0
        closes = []
        for i in range(260):
            level += 0.02 if i < 230 else -0.03
            noise = 0.35 * math.sin(i * 2.1) + 0.2 * math.sin(i * 0.9 + 1.3)
            closes.append(level + noise)
        df = make_df(closes)
        screen = screen_symbol(df)
        assert screen["promote"] is False
        assert screen["action_hint"] == "no_setup"
        assert self._card_side(df) is None

    def test_knife_terminates_avoid(self) -> None:
        closes = [150.0 - i * 0.25 for i in range(260)]  # long decline, below SMA200
        df = make_df(closes)
        screen = screen_symbol(df)
        assert screen["promote"] is False
        assert screen["action_hint"] == "avoid"
        assert "falling_knife" in screen["blocker_ids"]

    def test_thin_liquidity_terminates_avoid(self) -> None:
        df = uptrend_breakout_df()
        df["volume"] = 3000.0  # ~ $300k/day dollar volume
        screen = screen_symbol(df)
        assert screen["promote"] is False
        assert "weak_liquidity" in screen["blocker_ids"]

    def test_missing_volume_is_liquidity_missing_blocker(self) -> None:
        df = uptrend_breakout_df()
        df["volume"] = float("nan")
        screen = screen_symbol(df)
        assert screen["promote"] is False
        assert "liquidity_missing" in screen["blocker_ids"]  # AMENDMENT shape rule

    def test_insufficient_data_is_wait_for_data(self) -> None:
        screen = screen_symbol(make_df([100.0] * 10))
        assert screen["promote"] is False
        assert screen["action_hint"] == "wait_for_data"
        assert "data_quality_critical" in screen["blocker_ids"]
