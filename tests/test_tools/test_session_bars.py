"""WP5: session-indexed bars, price provenance in analyze, staleness (R5.1-R5.7)."""

import importlib
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
import pytz

from stock_analysis.data import yfinance_client
from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.tools.price_history import _build_summary
from stock_analysis.tools.risk_metrics import _calculate_period_return
from stock_analysis.utils.freshness import build_freshness
from stock_analysis.utils.indicators import (
    calculate_returns,
    calculate_sma,
    days_since_extreme,
    invalid_bars_in_window,
    weekly_return_zscore,
)
from stock_analysis.utils.validators import FetchParams

tech_mod = importlib.import_module("stock_analysis.tools.technicals")
summary_mod = importlib.import_module("stock_analysis.tools.stock_summary")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

ET = pytz.timezone("America/New_York")
SATURDAY = ET.localize(datetime(2026, 3, 7, 12, 0))
NAN = float("nan")
R54_CLOSES = pd.Series([10.0, 20.0, NAN, 40.0, 50.0, 60.0, 70.0])


def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return ET.localize(datetime(y, m, d, hh, mm))


class TestR54NeverCompress:
    def test_five_session_return_uses_the_session_five_bars_back(self) -> None:
        assert calculate_returns(R54_CLOSES, 5) == pytest.approx(2.5)
        assert _calculate_period_return(R54_CLOSES, 5) == pytest.approx(2.5)

    def test_sma5_stays_null_and_discloses_the_invalid_bar(self) -> None:
        assert math.isnan(calculate_sma(R54_CLOSES, 5).iloc[-1])
        assert invalid_bars_in_window(R54_CLOSES, window=5) == 1

    def test_an_invalid_endpoint_nulls_the_return(self) -> None:
        latest_invalid = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, NAN])
        assert calculate_returns(latest_invalid, 5) is None
        assert _calculate_period_return(latest_invalid, 5) is None
        assert _calculate_period_return(pd.Series([NAN, 20.0, 30.0]), 5) is None  # short-frame fallback endpoint

    def test_weekly_zscore_needs_the_latest_weekly_return(self) -> None:
        closes = pd.Series([100.0 + (i % 7) for i in range(60)] + [NAN])
        assert weekly_return_zscore(closes) is None
        assert weekly_return_zscore(closes.iloc[:-1]) is not None

    def test_days_since_extreme_counts_sessions(self) -> None:
        assert days_since_extreme(pd.Series([70.0, 60.0, NAN, 40.0]), kind="high") == 3

    def test_price_action_compares_consecutive_sessions(self) -> None:
        got = tech_mod._build_price_action(pd.Series([10.0, 20.0, 30.0, 25.0, 26.0, 27.0, NAN]))
        assert got == {"higher_closes_2d": None, "break_5d_high": None}
        got = tech_mod._build_price_action(pd.Series([10.0, 20.0, NAN, 25.0, 26.0, 27.0, 31.0]))
        assert got == {"higher_closes_2d": True, "break_5d_high": True}

    def test_price_history_total_return_uses_the_period_endpoints(self) -> None:
        df = pd.DataFrame({"date": ["2026-10-07", "2026-10-08", "2026-10-09"], "open": [1.0] * 3,
                           "high": [12.0] * 3, "low": [9.0] * 3, "close": [10.0, 11.0, NAN],
                           "volume": [1.0] * 3})
        summary = _build_summary(df)
        assert (summary["end_date"], summary["end_price"], summary["total_return"]) == ("2026-10-09", None, None)


def pin_frame(gap: int | None = None, *, volume_gap: int | None = None) -> pd.DataFrame:
    n = 40
    close = [round(100 + 3 * math.sin(i / 3) + 0.25 * i, 4) for i in range(n)]
    high = [round(c + 1.0 + 0.3 * math.cos(i), 4) for i, c in enumerate(close)]
    low = [round(c - 1.0 - 0.2 * math.sin(i), 4) for i, c in enumerate(close)]
    vol = [1_000_000.0 + 10_000 * i for i in range(n)]
    dates = pd.bdate_range(end="2026-03-06", periods=n).strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": vol})
    if gap is not None:
        df.loc[gap, ["open", "high", "low", "close", "volume"]] = NAN
    if volume_gap is not None:
        df.loc[volume_gap, "volume"] = NAN
    return df


async def run_technicals(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    async def fake_history(params: FetchParams) -> pd.DataFrame:
        return df

    monkeypatch.setattr(tech_mod, "fetch_history", fake_history)
    return await tech_mod.technicals("PIN", _now=SATURDAY, **kwargs)  # type: ignore[no-any-return]


def windows(**overrides: int) -> dict[str, int]:
    base = {"sma_20": 0, "sma_50": 0, "sma_200": 0, "ema_12": 0, "ema_26": 0, "rsi_14": 0,
            "bollinger_20": 0, "atr_14": 0, "volume_20": 0}
    return {**base, **overrides}


ALL_ONE = windows(**dict.fromkeys(windows(), 1))


class TestR57IndicatorCharacterization:
    """Pinned on the pre-WP5 code: values are preserved, disclosures are added."""

    @pytest.mark.parametrize(
        ("gap", "values", "latest_invalid", "invalid"),
        [
            (None, (107.44, 107.72, 106.35, 76.6, 2.1), False, windows()),
            (10, (107.44, 107.72, 106.35, 78.0, 2.1), False, windows(sma_50=1, sma_200=1)),
            (35, (None, 107.71, 106.3, 72.8, 2.11), False, ALL_ONE),
            (39, (None, 107.12, 105.98, 73.3, 2.08), True, ALL_ONE),
        ],
        ids=["no_gap", "gap_mid_series", "gap_in_window", "gap_latest_bar"],
    )
    async def test_values_and_disclosures(
        self, monkeypatch: pytest.MonkeyPatch, gap: int | None, values: tuple[Any, ...],
        latest_invalid: bool, invalid: dict[str, int],
    ) -> None:
        r = await run_technicals(monkeypatch, pin_frame(gap))

        ma = r["moving_averages"]
        assert (ma["sma_20"], ma["ema_12"], ma["ema_26"], r["rsi"]["value"], r["atr"]["value"]) == values
        assert r["bar_validity"]["latest_bar_invalid"] is latest_invalid
        assert r["bar_validity"]["invalid_bars_in_window"] == invalid


    async def test_atr_window_counts_a_missing_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pin_frame()
        df.loc[39, "high"] = NAN
        r = await run_technicals(monkeypatch, df)
        assert r["bar_validity"]["latest_bar_invalid"] is False
        assert r["bar_validity"]["invalid_bars_in_window"] == windows(atr_14=1)


class TestR55LatestBarInvalid:
    async def test_ema_rsi_atr_keep_their_values_and_the_bar_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        r = await run_technicals(monkeypatch, pin_frame(39))

        assert r["current_price"] is None
        assert (r["moving_averages"]["ema_12"], r["rsi"]["value"], r["atr"]["value"]) == (107.12, 73.3, 2.08)
        assert r["bar_validity"]["latest_bar_invalid"] is True
        assert r["session_close"]["session_date"] == "2026-03-05"  # the latest valid completed session

    def test_freshness_still_reads_the_raw_latest_bar(self) -> None:
        now = at(2026, 3, 10, 11)
        intraday = pd.DataFrame({"date": ["2026-03-10T10:55:00-0400", "2026-03-10T11:00:00-0400"],
                                 "close": [101.0, NAN]})
        daily = pd.DataFrame({"date": ["2026-03-09"], "close": [100.0]})
        f = build_freshness(intraday_df=intraday, daily_df=daily, session="regular", now=now)
        assert (f["basis"], f["reason_code"]) == ("unverifiable", "nonfinite_close")

    async def test_analyze_warns_when_the_latest_daily_bar_is_invalid(self, analyze: Any) -> None:
        result = await analyze(technicals={"bar_validity": {"latest_bar_invalid": True}})
        assert "latest_daily_bar_invalid" in result["data_quality"]["warnings"]
        assert "latest_daily_bar_invalid" in project_analyze_result(result, "decision")["data_quality"]["warnings"]


class TestR56MissingVolume:
    async def test_price_metrics_survive_and_volume_metrics_are_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        r = await run_technicals(monkeypatch, pin_frame(volume_gap=39))

        assert r["current_price"] == 111.01
        assert (r["moving_averages"]["sma_20"], r["rsi"]["value"], r["atr"]["value"]) == (107.44, 76.6, 2.1)
        assert r["volume"]["current"] is None
        assert r["volume"]["ratio"] is None
        assert r["obv"] == {"current": None, "sma_20": None, "trend": None}
        validity = r["bar_validity"]
        assert (validity["latest_bar_invalid"], validity["latest_volume_invalid"]) == (False, True)
        assert validity["invalid_bars_in_window"] == windows(volume_20=1)


class TestTechnicalsPriceBlocks:
    async def test_session_close_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pin_frame()
        df.attrs["fetched_at"] = "2026-03-07T17:00:00Z"
        r = await run_technicals(monkeypatch, df)
        assert r["session_close"] == {"close": 111.0105, "session_date": "2026-03-06",
                                      "fetched_at": "2026-03-07T17:00:00Z", "cache_hit": False}
        assert "intraday_quote" not in r

    async def test_intraday_quote_is_fetched_only_in_the_regular_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        intervals: list[str] = []
        bar = pd.DataFrame({"date": ["2026-03-10T10:55:00-0400"], "open": [1.0], "high": [1.0], "low": [1.0],
                            "close": [101.8], "volume": [1.0]})

        async def fake_history(params: FetchParams) -> pd.DataFrame:
            intervals.append(params.interval)
            return bar if params.interval == "5m" else pin_frame()

        monkeypatch.setattr(tech_mod, "fetch_history", fake_history)

        r = await tech_mod.technicals("PIN", _now=at(2026, 3, 10, 11), intraday_quote=True)
        assert intervals == ["1d", "5m"]
        assert r["intraday_quote"]["close"] == 101.8
        assert r["intraday_quote"]["bar_time"] == "2026-03-10T14:55:00+00:00"

        intervals.clear()
        r = await tech_mod.technicals("PIN", _now=at(2026, 3, 10, 17), intraday_quote=True)
        assert intervals == ["1d"]
        assert r["intraday_quote"] is None


class TestFetchedAtAndCacheHit:
    async def test_history_records_when_it_was_fetched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [10]},
                             index=pd.DatetimeIndex(["2026-10-08"], tz="America/New_York"))

        class ChartTicker:
            def __init__(self, symbol: str) -> None:
                pass

            def history(self, **kwargs: Any) -> pd.DataFrame:
                return frame.copy()

        monkeypatch.setattr(yfinance_client.yf, "Ticker", ChartTicker)
        df = await yfinance_client.fetch_history(FetchParams("AAPL", "1y", "1d", True))
        fetched = datetime.fromisoformat(df.attrs["fetched_at"].replace("Z", "+00:00"))
        assert abs((datetime.now(UTC) - fetched).total_seconds()) < 60

    async def test_info_cache_reports_hits_and_the_original_fetch_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        async def fake_fetch(symbol: str) -> tuple[dict[str, Any], Any, bool]:
            calls.append(symbol)
            return {"regularMarketPrice": 100.0}, None, False

        monkeypatch.setattr(yfinance_client, "_fetch_info_with_singleflight", fake_fetch)
        yfinance_client.info_cache.clear()
        try:
            info, fetched_at, hit = await yfinance_client.fetch_info_with_cache_state("aapl")
            again, fetched_again, hit_again = await yfinance_client.fetch_info_with_cache_state("AAPL")
            assert (info, hit) == ({"regularMarketPrice": 100.0}, False)
            assert (again, fetched_again, hit_again) == (info, fetched_at, True)
            assert await yfinance_client.fetch_info("AAPL") == info
            assert calls == ["aapl"]
        finally:
            yfinance_client.info_cache.clear()

    async def test_stock_summary_reports_quote_time_and_cache_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake(symbol: str) -> tuple[dict[str, Any], str, bool]:
            return ({"regularMarketPrice": 100.0, "regularMarketTime": 1791554400, "quoteType": "EQUITY",
                     "currency": "USD", "shortName": "Apple"}, "2026-10-09T15:00:00Z", True)

        monkeypatch.setattr(summary_mod, "fetch_info_with_cache_state", fake)
        r = await summary_mod.stock_summary("AAPL")
        assert (r["quote_time"], r["fetched_at"], r["cache_hit"]) == (
            "2026-10-09T14:00:00+00:00", "2026-10-09T15:00:00Z", True)


FRIDAY_PRE_OPEN = at(2026, 10, 9, 8)


@pytest.fixture
def analyze(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def run(*, technicals: dict[str, Any] | None = None, summary: dict[str, Any] | None = None,
                  now: datetime = FRIDAY_PRE_OPEN, account_size: float | None = 10_000) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        async def fake_summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Acme", "currency": "USD", "quote_type": "EQUITY",
                    "market_cap": 1e10, "current_price": 100.0, "quote_time": "2026-10-09T11:00:00+00:00",
                    "fetched_at": "2026-10-09T11:00:05Z", "cache_hit": True, **(summary or {})}

        async def fake_technicals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            seen["technicals_kwargs"] = kwargs
            return {"symbol": symbol, "current_price": 80.0,
                    "session_close": {"close": 80.0, "session_date": "2026-10-08",
                                      "fetched_at": "2026-10-09T12:00:00Z", "cache_hit": False},
                    **(technicals or {})}

        async def fake_risk(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "atr": {"value": 2.0, "as_pct_of_price": 0.025}, "beta": {"value": 1.0}}

        async def fake_fundamentals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "valuation": {"pe_trailing": 15.0, "trailing_eps": 6.0},
                    "profitability": {"net_margin": 0.2},
                    "analyst_coverage": {"price_target_mean": 100.0, "price_target_median": 96.0,
                                         "upside_to_mean_target": 0.0, "upside_to_median_target": -0.04}}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", fake_summary)
        monkeypatch.setattr(orchestrator, "technicals", fake_technicals)
        monkeypatch.setattr(orchestrator, "risk_metrics", fake_risk)
        monkeypatch.setattr(orchestrator, "fundamentals_snapshot", fake_fundamentals)
        for name in ("events_calendar", "stock_news", "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        result: dict[str, Any] = await orchestrator.analyze_stock("ACME", account_size=account_size, _now=now)
        result["_seen"] = seen
        return result

    return run


class TestR51AnalyzePrice:
    async def test_the_bar_beats_cached_info_everywhere_it_drives_a_number(self, analyze: Any) -> None:
        result = await analyze()

        assert result["summary"]["current_price"] == 80.0
        prov = result["summary"]["price_provenance"]
        assert (prov["price_source"], prov["price_as_of"], prov["as_of_kind"]) == (
            "daily_close", "2026-10-08", "session_close")
        assert result["action_zones"]["position_sizing_range"]["shares_range"]["at_price"] == 80.0
        assert result["action_zones"]["stop_calculation"]["stop_price"] < 80.0
        assert result["analyst_coverage"]["upside_to_mean_target"] == 0.25
        assert result["analyst_coverage"]["upside_to_median_target"] == 0.2
        assert result["watchlist_snapshot"]["price"] == 80.0

    async def test_source_and_time_appear_in_every_tier(self, analyze: Any) -> None:
        result = await analyze()
        for detail in ("decision", "standard", "full"):
            prov = project_analyze_result(result, detail)["summary"]["price_provenance"]
            assert (prov["price_source"], prov["price_as_of"], prov["as_of_kind"]) == (
                "daily_close", "2026-10-08", "session_close"), detail
            assert prov["staleness"]["unit"] == "sessions", detail

    async def test_a_fresh_intraday_bar_drives_the_regular_session(self, analyze: Any) -> None:
        now = at(2026, 10, 9, 11)
        bar_time = (now.astimezone(UTC) - timedelta(minutes=5)).isoformat()
        result = await analyze(now=now, technicals={"intraday_quote": {
            "close": 82.0, "bar_time": bar_time, "fetched_at": None, "cache_hit": False}})

        assert result["summary"]["current_price"] == 82.0
        assert result["summary"]["price_provenance"]["price_source"] == "intraday_quote"
        assert result["_seen"]["technicals_kwargs"] == {"_now": now, "intraday_quote": True}

    async def test_info_is_the_labelled_fallback(self, analyze: Any) -> None:
        result = await analyze(technicals={"session_close": None})

        assert result["summary"]["current_price"] == 100.0
        prov = result["summary"]["price_provenance"]
        assert (prov["price_source"], prov["fallback_reason"], prov["cache_hit"]) == (
            "info_quote_fallback", "no_valid_daily_close", True)
        assert prov["price_as_of"] == "2026-10-09T11:00:00+00:00"


class TestComponentFreshness:
    def test_r5_2_equivalent_instants_give_identical_ages(self) -> None:
        now = datetime(2026, 10, 9, 18, tzinfo=UTC)
        et, _ = orchestrator._component_freshness_entry({"as_of": "2026-10-09T10:00:00-04:00"}, now, 168)
        utc, _ = orchestrator._component_freshness_entry({"as_of": "2026-10-09T14:00:00Z"}, now, 168)
        assert et["age_hours"] == utc["age_hours"] == 4.0

    @pytest.mark.parametrize(
        ("now", "behind"),
        [(at(2026, 10, 12, 8), 0), (at(2026, 10, 13, 8), 1)],
        ids=["monday_pre_open_current", "tuesday_pre_open_stale"],
    )
    def test_r5_3_daily_components_count_sessions(self, now: datetime, behind: int) -> None:
        entry, stale = orchestrator._session_freshness_entry(
            {"as_of": "2026-10-12T12:00:00Z", "last_bar_date": "2026-10-09"}, now)
        assert (entry["last_bar_date"], entry["sessions_behind"], entry["stale"]) == (
            "2026-10-09", behind, bool(behind))
        assert stale == (behind or None)

    async def test_analyze_warns_in_sessions(self, analyze: Any) -> None:
        prov = {"price": {"source": "yfinance", "as_of": "2026-10-09T12:00:00Z", "last_bar_date": "2026-10-07"}}
        result = await analyze(technicals={"data_provenance": prov})

        dq = result["data_quality"]
        assert dq["component_freshness"]["price"]["sessions_behind"] == 1
        assert {"price_stale_1_sessions", "risk_stale_1_sessions"} <= set(dq["staleness_warnings"])
