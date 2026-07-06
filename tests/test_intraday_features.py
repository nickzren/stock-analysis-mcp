"""Tests for the swing-mode intraday block (disclosure semantics)."""

from datetime import datetime

import pandas as pd
import pytz

from stock_analysis.utils.intraday_features import build_intraday_block

ET = pytz.timezone("America/New_York")
NOW = ET.localize(datetime(2026, 3, 10, 11, 30))  # Tuesday, regular hours


def five_min_df():
    # Three bars: typical prices 101, 102, 104 with volumes 100, 200, 100
    return pd.DataFrame({
        "date": ["2026-03-10T11:15:00-0400", "2026-03-10T11:20:00-0400",
                 "2026-03-10T11:25:00-0400"],
        "open": [100.0, 101.0, 103.0],
        "high": [102.0, 103.0, 105.0],
        "low": [100.0, 101.0, 103.0],
        "close": [101.0, 102.0, 104.0],
        "volume": [100.0, 200.0, 100.0],
    })


def hourly_df(closes):
    n = len(closes)
    dates = pd.date_range("2026-02-01 10:00", periods=n, freq="h",
                          tz="America/New_York").strftime("%Y-%m-%dT%H:%M:%S%z")
    return pd.DataFrame({"date": list(dates), "open": closes,
                         "high": [c + 0.5 for c in closes],
                         "low": [c - 0.5 for c in closes],
                         "close": closes, "volume": [10_000.0] * n})


def daily_df():
    return pd.DataFrame({"date": ["2026-03-09", "2026-03-10"],
                         "open": [100.0, 101.0], "high": [101.0, 105.0],
                         "low": [99.0, 100.0], "close": [100.5, 104.0],
                         "volume": [1_000_000.0, 400_000.0]})


TECHNICALS = {
    "current_price": 104.0,
    "moving_averages": {"sma_20": 103.0, "sma_50": 100.0,
                        "price_vs_sma50": 0.04},
}


def block(**over):
    kwargs = {
        "df_5m": five_min_df(),
        "df_1h": hourly_df([100.0 + 0.1 * i for i in range(60)]),
        "daily_df": daily_df(),
        "technicals_payload": TECHNICALS,
        "session": "regular",
        "now": NOW,
    }
    kwargs.update(over)
    return build_intraday_block(**kwargs)


class TestVwap:
    def test_session_vwap_arithmetic(self) -> None:
        b = block()
        # typical prices: (102+100+101)/3=101, (103+101+102)/3=102, (105+103+104)/3=104
        # vwap = (101*100 + 102*200 + 104*100) / 400 = 102.25
        assert b["vwap"]["value"] == 102.25
        assert b["vwap"]["above"] is True  # last close 104 > 102.25
        assert b["vwap"]["price_vs_vwap_pct"] == round((104.0 - 102.25) / 102.25, 4)
        assert b["session_date"] == "2026-03-10"

    def test_missing_5m_nulls_vwap_with_warning(self) -> None:
        b = block(df_5m=None)
        assert b["vwap"] is None
        assert any(w["id"] == "intraday_unavailable" for w in b["warnings"])


class TestTimeAdjustedRvol:
    def test_regular_session_value(self) -> None:
        b = block()
        # cumulative 5m volume 400; avg full-day 1_000_000 (prior daily bar);
        # elapsed 9:30->11:30 = 120min/390 = 0.30769...
        elapsed = 120 / 390
        expected = round(400.0 / (1_000_000.0 * elapsed), 2)
        assert b["rvol_time_adjusted"]["value"] == expected
        assert b["rvol_time_adjusted"]["elapsed_session_pct"] == round(elapsed * 100, 1)

    def test_off_session_is_null_with_warning(self) -> None:
        evening = ET.localize(datetime(2026, 3, 10, 18, 30))
        b = block(session="after_hours", now=evening)
        assert b["rvol_time_adjusted"] is None
        assert any(w["id"] == "off_session" for w in b["warnings"])


class TestHourlyTrend:
    def test_advance(self) -> None:  # rising closes: above rising EMA
        b = block(df_1h=hourly_df([100.0 + 0.5 * i for i in range(60)]))
        assert b["hourly_trend"]["state"] == "advance"

    def test_breakdown(self) -> None:  # falling closes: below falling EMA
        b = block(df_1h=hourly_df([130.0 - 0.5 * i for i in range(60)]))
        assert b["hourly_trend"]["state"] == "breakdown"

    def test_pullback(self) -> None:
        # Long rise then a SHALLOW 3-bar dip: EMA(20) still rising (~122.5 vs
        # ~121.8 five bars ago) while price (122) sits just below it.
        closes = [100.0 + 0.5 * i for i in range(57)] + [124.5, 123.0, 122.0]
        b = block(df_1h=hourly_df(closes))
        assert b["hourly_trend"]["state"] == "pullback"

    def test_range(self) -> None:
        # Long fall then a sharp bounce: price above the EMA while the EMA
        # is still falling.
        closes = [130.0 - 0.5 * i for i in range(57)] + [104.0, 106.0, 108.0]
        b = block(df_1h=hourly_df(closes))
        assert b["hourly_trend"]["state"] == "range"

    def test_missing_hourly_nulls_with_warning(self) -> None:
        b = block(df_1h=None)
        assert b["hourly_trend"] is None
        assert any(w["id"] == "hourly_unavailable" for w in b["warnings"])


class TestAlignment:
    def test_daily_up_plus_hourly_pullback_flags_aligned(self) -> None:
        closes = [100.0 + 0.5 * i for i in range(57)] + [124.5, 123.0, 122.0]
        b = block(df_1h=hourly_df(closes))
        assert b["alignment"]["daily"] == "up"
        assert b["alignment"]["aligned_pullback"] is True

    def test_daily_down(self) -> None:
        t = {"current_price": 90.0,
             "moving_averages": {"sma_20": 92.0, "sma_50": 95.0,
                                 "price_vs_sma50": -0.05}}
        b = block(technicals_payload=t)
        assert b["alignment"]["daily"] == "down"
        assert b["alignment"]["aligned_pullback"] is False


def test_stale_intraday_nulls_dependents() -> None:
    old = pd.DataFrame({"date": ["2026-03-10T09:35:00-0400"], "open": [100.0],
                        "high": [101.0], "low": [99.0], "close": [100.0],
                        "volume": [100.0]})
    b = block(df_5m=old)  # 115 min old > 15-min ceiling
    assert b["freshness"]["stale"] is True
    assert b["vwap"] is None
    assert b["rvol_time_adjusted"] is None
    assert any(w["id"] == "stale_intraday" for w in b["warnings"])
