"""Tests for the daily-derived short_term block builders."""

from datetime import date

import pandas as pd

from stock_analysis.utils.swing_features import (
    build_short_term_block,
    measured_move_target,
)


def make_df(closes, opens=None, highs=None, lows=None, volumes=None,
            last_date="2026-03-10"):
    n = len(closes)
    opens = opens or list(closes)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    volumes = volumes or [1_000_000.0] * n
    dates = pd.date_range("2026-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    dates[-1] = last_date
    return pd.DataFrame({"date": dates, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": volumes})


TODAY = date(2026, 3, 10)


def test_returns_none_for_short_frames() -> None:
    assert build_short_term_block(make_df([100.0] * 20), today=TODAY,
                                  session="regular") is None
    assert build_short_term_block(None, today=TODAY, session="closed") is None


class TestLevels:
    def test_prior_windows_exclude_current_bar(self) -> None:
        closes = [100.0] * 29 + [120.0]
        highs = [101.0] * 24 + [104.0, 101.0, 101.0, 101.0, 110.0, 121.0]
        lows = [99.0] * 28 + [95.0, 119.0]
        block = build_short_term_block(make_df(closes, highs=highs, lows=lows),
                                       today=TODAY, session="closed")
        assert block is not None
        levels = block["levels"]
        assert levels["prior_day_high"] == 110.0
        assert levels["prior_day_low"] == 95.0
        assert levels["prior_5d_high"] == 110.0   # max of bars -6:-1
        assert levels["prior_5d_low"] == 95.0
        assert levels["prior_20d_high"] == 110.0
        assert levels["prior_20d_low"] == 95.0

    def test_swing_high_strict_pivot(self) -> None:
        # Older higher pivot (high 113), recovery dip, recent lower pivot (high 106)
        closes = ([100.0] * 8 + [105.0, 110.0, 112.0, 108.0, 104.0]
                  + [100.0] * 5 + [103.0, 105.0, 102.0, 100.0] + [99.0] * 8)
        highs = [c + 1.0 for c in closes]
        block = build_short_term_block(make_df(closes, highs=highs),
                                       today=TODAY, session="closed")
        assert block is not None
        assert block["levels"]["swing_high"] == 106.0  # most recent strict pivot


class TestGap:
    def test_gap_up_unfilled(self) -> None:
        closes = [100.0] * 29 + [104.0]
        opens = [100.0] * 29 + [103.0]
        lows = [99.0] * 29 + [102.5]  # never touched prior close 100
        block = build_short_term_block(make_df(closes, opens=opens, lows=lows),
                                       today=TODAY, session="closed")
        gap = block["gap"]
        assert gap["gap_pct"] == 0.03
        assert gap["direction"] == "up"
        assert gap["filled"] is False
        assert gap["prior_close"] == 100.0

    def test_gap_down_filled(self) -> None:
        closes = [100.0] * 29 + [99.5]
        opens = [100.0] * 29 + [98.0]
        highs = [101.0] * 29 + [100.2]  # traded back through prior close
        block = build_short_term_block(make_df(closes, opens=opens, highs=highs),
                                       today=TODAY, session="closed")
        gap = block["gap"]
        assert gap["direction"] == "down"
        assert gap["filled"] is True

    def test_no_gap_below_epsilon(self) -> None:
        closes = [100.0] * 29 + [100.3]
        opens = [100.0] * 29 + [100.05]  # 0.05% < 0.1% epsilon
        block = build_short_term_block(make_df(closes, opens=opens),
                                       today=TODAY, session="closed")
        gap = block["gap"]
        assert gap["gap_pct"] == 0.0005
        assert gap["direction"] is None
        assert gap["filled"] is None


class TestRvol:
    def test_full_day_basis_off_hours(self) -> None:
        volumes = [1_000_000.0] * 29 + [2_000_000.0]
        block = build_short_term_block(make_df([100.0] * 30, volumes=volumes),
                                       today=TODAY, session="after_hours")
        assert block["rvol"] == {"value": 2.0, "basis": "full_day"}

    def test_partial_day_basis_today_regular(self) -> None:
        volumes = [1_000_000.0] * 29 + [500_000.0]
        block = build_short_term_block(
            make_df([100.0] * 30, volumes=volumes, last_date="2026-03-10"),
            today=TODAY, session="regular")
        assert block["rvol"] == {"value": 0.5, "basis": "partial_day"}

    def test_prior_day_bar_regular_is_full_day(self) -> None:
        block = build_short_term_block(
            make_df([100.0] * 30, last_date="2026-03-09"),
            today=TODAY, session="regular")
        assert block["rvol"]["basis"] == "full_day"


def test_compression_flag() -> None:
    closes = []
    for i in range(100):
        closes.append(100.0 + (5.0 if i % 2 == 0 else -5.0))
    closes.extend([100.0] * 50)
    block = build_short_term_block(make_df(closes), today=TODAY, session="closed")
    assert block["bandwidth_pctile_6m"] is not None
    assert block["compression"] is True


class TestMeasuredMove:
    def test_projects_base_depth(self) -> None:
        assert measured_move_target(102.0, 95.0) == 109.0

    def test_none_safe(self) -> None:
        assert measured_move_target(None, 95.0) is None
        assert measured_move_target(102.0, None) is None
