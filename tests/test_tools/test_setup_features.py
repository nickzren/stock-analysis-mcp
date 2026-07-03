"""Tests for setup-level technical feature extraction."""

import pandas as pd

from stock_analysis.tools.trade_setup.features import compute_setup_features


def make_df(closes: list[float], highs: list[float] | None = None,
            lows: list[float] | None = None, volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    volumes = volumes or [1_000_000.0] * n
    dates = pd.date_range("2026-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_returns_none_for_short_frames() -> None:
    assert compute_setup_features(make_df([100.0] * 20)) is None


def test_prior_day_and_20d_levels_exclude_current_bar() -> None:
    closes = [100.0] * 29 + [120.0]
    highs = [101.0] * 28 + [110.0, 121.0]  # prior-day high 110, current 121
    df = make_df(closes, highs=highs)
    f = compute_setup_features(df)
    assert f is not None
    assert f["prior_day_high"] == 110.0
    assert f["high_20d_prior"] == 110.0  # excludes the 121 current bar
    assert f["trigger_state"]["close_above_prior_day_high"] is True
    assert f["trigger_state"]["close_above_20d_high"] is True


def test_swing_low_finds_pivot() -> None:
    # V-shape: decline into a pivot low at 90 then recovery
    closes = [100.0] * 15 + [98.0, 95.0, 90.0, 94.0, 97.0] + [99.0] * 10
    lows = [c - 0.5 for c in closes]
    f = compute_setup_features(make_df(closes, lows=lows))
    assert f is not None
    assert f["swing_low"] == 89.5


def test_swing_low_prefers_most_recent_strict_pivot() -> None:
    # Older deeper pivot (low 84.5), recovery, then recent shallower pivot (low 91.5)
    closes = ([100.0] * 8 + [95.0, 90.0, 85.0, 91.0, 96.0]
              + [100.0] * 5 + [96.0, 92.0, 95.0, 98.0] + [99.0] * 8)
    lows = [c - 0.5 for c in closes]
    f = compute_setup_features(make_df(closes, lows=lows))
    assert f is not None
    assert f["swing_low"] == 91.5


def test_volume_ratio_uses_prior_20d_average() -> None:
    volumes = [1_000_000.0] * 29 + [3_000_000.0]
    f = compute_setup_features(make_df([100.0] * 30, volumes=volumes))
    assert f is not None
    assert f["last_volume_ratio"] == 3.0


def test_bandwidth_percentile_is_low_after_contraction() -> None:
    # 150 bars: first 100 volatile (alternating ±5), last 50 flat -> bottom of range
    closes = []
    for i in range(100):
        closes.append(100.0 + (5.0 if i % 2 == 0 else -5.0))
    closes.extend([100.0] * 50)
    f = compute_setup_features(make_df(closes))
    assert f is not None
    assert f["bandwidth_pctile_6m"] is not None
    assert f["bandwidth_pctile_6m"] <= 0.25
