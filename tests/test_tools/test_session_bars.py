"""WP5: session-indexed bars, price provenance in analyze, staleness (R5.1-R5.7)."""

import importlib
import inspect
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
import pytz
import yfinance.scrapers.history as history_module
import yfinance.utils as yfinance_utils
from yfinance.scrapers.history import PriceHistory

from stock_analysis.data import yfinance_client
from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.tools.price_history import _build_summary
from stock_analysis.tools.risk_metrics import _calculate_period_return
from stock_analysis.utils.freshness import build_freshness
from stock_analysis.utils.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_sma_slope,
    days_since_extreme,
    invalid_bars_in_window,
    session_returns,
    weekly_return_zscore,
)
from stock_analysis.utils.validators import FetchParams

tech_mod = importlib.import_module("stock_analysis.tools.technicals")
summary_mod = importlib.import_module("stock_analysis.tools.stock_summary")
risk_mod = importlib.import_module("stock_analysis.tools.risk_metrics")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
price_history_mod = importlib.import_module("stock_analysis.tools.price_history")
data_quality_mod = importlib.import_module("stock_analysis.tools.data_quality")
position_mod = importlib.import_module("stock_analysis.tools.position")

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

    def test_sma5_skips_the_invalid_bar_and_discloses_it(self) -> None:
        """D7 replaced SMA's strict window: the last 5 finite closes, 20..70."""
        assert calculate_sma(R54_CLOSES, 5).iloc[-1] == 48.0
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

    def test_infinite_endpoints_give_no_return(self) -> None:
        inf_latest = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, float("inf")])
        assert calculate_returns(inf_latest, 5) is None
        assert _calculate_period_return(inf_latest, 5) is None
        assert _calculate_period_return(pd.Series([float("inf"), 20.0, 30.0]), 5) is None
        assert weekly_return_zscore(pd.Series([100.0 + (i % 7) for i in range(60)] + [float("inf")])) is None
        assert calculate_sma_slope(pd.Series([100.0] * 20 + [float("inf")])) is None
        assert tech_mod._build_price_action(inf_latest) == {"higher_closes_2d": None, "break_5d_high": None}
        ytd = pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "close": [100.0, float("inf")]})
        assert tech_mod._calculate_ytd_return(ytd) is None

    def test_weekly_zscore_rejects_an_infinite_starting_price(self) -> None:
        closes = [100.0 + (i % 7) for i in range(60)]
        start_inf = pd.Series(closes[:-6] + [float("inf")] + closes[-5:])
        assert session_returns(start_inf, 5).iloc[-1] != session_returns(start_inf, 5).iloc[-1]  # NaN, not -100%
        assert weekly_return_zscore(start_inf) is None
        history_inf = pd.Series(closes[:20] + [float("inf")] + closes[21:])
        history_nan = pd.Series(closes[:20] + [NAN] + closes[21:])
        assert weekly_return_zscore(history_inf) == weekly_return_zscore(history_nan)

    # The drawdown block (not a return series) divides by the infinite peak.
    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
    async def test_risk_return_series_skip_infinite_prices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def run(bad: float) -> dict[str, Any]:
            df = pin_frame(n=60)
            df.loc[30, "close"] = bad

            async def fake_history(params: FetchParams) -> pd.DataFrame:
                return df if params.symbol == "PIN" else pin_frame(n=60)

            monkeypatch.setattr(risk_mod, "fetch_history", fake_history)
            return await risk_mod.risk_metrics("PIN")  # type: ignore[no-any-return]

        with_inf, with_nan = await run(float("inf")), await run(NAN)
        assert with_inf["volatility"] == with_nan["volatility"]
        assert with_inf["volatility"]["annualized"] is not None

    def test_days_since_extreme_counts_sessions(self) -> None:
        assert days_since_extreme(pd.Series([70.0, 60.0, NAN, 40.0]), kind="high") == 3

    def test_price_action_compares_consecutive_sessions(self) -> None:
        got = tech_mod._build_price_action(pd.Series([10.0, 20.0, 30.0, 25.0, 26.0, 27.0, NAN]))
        assert got == {"higher_closes_2d": None, "break_5d_high": None}
        got = tech_mod._build_price_action(pd.Series([10.0, 20.0, NAN, 25.0, 26.0, 27.0, 31.0]))
        assert got == {"higher_closes_2d": True, "break_5d_high": True}

    def test_sma_slope_needs_the_sma_at_both_ends(self) -> None:
        sma = pd.Series([100.0 + i for i in range(30)])
        assert calculate_sma_slope(sma) == pytest.approx((129.0 - 109.0) / 109.0 / 20)
        assert calculate_sma_slope(sma.mask(sma.index == 9)) is None  # invalid 20 sessions back
        assert calculate_sma_slope(sma.mask(sma.index == 29)) is None  # invalid latest

    def test_rsi_divergence_pairs_each_close_with_its_own_session(self) -> None:
        closes = pd.Series([99.0, 101.2, 108.5, 99.3, 100.2, 101.7, 93.7, 100.2, 102.6, 105.9, 91.9, 96.1,
                            91.8, 106.2, 103.9, 90.8, 109.6, 109.3, 103.1, NAN, 93.1, 90.3])
        # An "RSI" that moves exactly with price can never diverge from it.
        assert tech_mod._detect_bullish_rsi_divergence(closes, closes.ffill(), pivot_window=2, lookback=20) is False

    def test_price_history_total_return_uses_the_period_endpoints(self) -> None:
        df = pd.DataFrame({"date": ["2026-10-07", "2026-10-08", "2026-10-09"], "open": [1.0] * 3,
                           "high": [12.0] * 3, "low": [9.0] * 3, "close": [10.0, 11.0, NAN],
                           "volume": [1.0] * 3})
        summary = _build_summary(df)
        assert (summary["end_date"], summary["end_price"], summary["total_return"]) == ("2026-10-09", None, None)


def pin_frame(gap: int | None = None, *, volume_gap: int | None = None, n: int = 40) -> pd.DataFrame:
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
    base = {"sma_20": 0, "sma_50": 0, "sma_200": 0, "ema_12": 0, "ema_26": 0, "macd_12_26_9": 0,
            "rsi_14": 0, "bollinger_20": 0, "atr_14": 0, "volume_20": 0, "obv_20": 0}
    return {**base, **overrides}


ALL_ONE = dict.fromkeys(windows(), 1)


class TestR57IndicatorCharacterization:
    """Pinned on the pre-WP5 code: values are preserved, disclosures are added.
    SMA follows D7 since: a gap inside its window is skipped (107.37 is the mean of
    the last 20 finite closes); an invalid latest bar still has no SMA."""

    @pytest.mark.parametrize(
        ("gap", "values", "latest_invalid", "invalid"),
        [
            (None, (107.44, 107.72, 106.35, 76.6, 2.1), False, windows()),
            (10, (107.44, 107.72, 106.35, 78.0, 2.1), False, windows(sma_50=1, sma_200=1, macd_12_26_9=1)),
            (35, (107.37, 107.71, 106.3, 72.8, 2.11), False, ALL_ONE),
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


class TestReturnEndpoints:
    """A return nulled by an invalid endpoint bar is named, unlike one with too
    little history, so its signals can be marked unknown rather than removed."""

    @pytest.mark.parametrize(
        ("gap", "n", "named"),
        [(None, 70, []), (70 - 64, 70, ["return_3m"]), (69, 70, ["return_1w", "return_1m", "return_3m"]),
         (69 - 21, 70, ["return_1m"]), (0, 40, [])],
        ids=["clean", "3m_start", "latest_bar", "1m_start", "short_history"],
    )
    async def test_invalid_return_endpoints(
        self, monkeypatch: pytest.MonkeyPatch, gap: int | None, n: int, named: list[str]
    ) -> None:
        r = await run_technicals(monkeypatch, pin_frame(gap, n=n))
        assert r["bar_validity"]["invalid_return_endpoints"] == named
        for name in named:
            assert r["returns"][name] is None


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


class TestEmptyLatestRowConsumers:
    """An empty latest session (kept by keepna) is not the freshest bar, and a
    holder's P&L falls back to the latest valid completed-session close."""

    async def test_technicals_and_risk_date_the_last_valid_bar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pin_frame(59, n=60)

        async def fake_history(params: FetchParams) -> pd.DataFrame:
            return df if params.symbol == "PIN" else pin_frame(n=60)

        monkeypatch.setattr(tech_mod, "fetch_history", fake_history)
        monkeypatch.setattr(risk_mod, "fetch_history", fake_history)
        monkeypatch.setattr(price_history_mod, "fetch_history", fake_history)
        tech = await tech_mod.technicals("PIN", _now=SATURDAY)
        risk = await risk_mod.risk_metrics("PIN")
        history = await price_history_mod.price_history("PIN")

        for result in (tech, risk, history):
            assert result["data_provenance"]["price"]["last_bar_date"] == "2026-03-05"

    async def test_data_quality_reports_the_last_bar_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frames = {"PIN": pin_frame(4, n=5), "EMPTY": pin_frame(n=5).assign(open=NAN, high=NAN, low=NAN, close=NAN)}

        async def fake_history(params: FetchParams) -> pd.DataFrame:
            return frames[params.symbol]

        monkeypatch.setattr(data_quality_mod, "fetch_history", fake_history)
        reports = {}
        for symbol in frames:
            report = {"availability": {"price": False}, "staleness": {"price_last_bar": None}, "missing_fields": []}
            await data_quality_mod._check_price_data(symbol, report)
            reports[symbol] = report

        assert reports["PIN"]["staleness"]["price_last_bar"] == "2026-03-05"
        assert reports["PIN"]["availability"]["price"] is True
        assert reports["EMPTY"]["availability"]["price"] is False
        assert reports["EMPTY"]["staleness"]["price_last_bar"] is None

    async def test_position_uses_the_latest_valid_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pin_frame(39)

        async def fake_history(params: FetchParams) -> pd.DataFrame:
            return df

        monkeypatch.setattr(tech_mod, "fetch_history", fake_history)
        result = await position_mod.analyze_position("PIN", cost_basis=100.0, purchase_date="2025-01-02", shares=10)

        assert "error" not in result
        assert result["position"]["current_price"] == pytest.approx(df["close"].iloc[38])
        assert result["data_provenance"]["price"]["last_bar_date"] == "2026-03-05"


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
        assert validity["invalid_bars_in_window"] == windows(volume_20=1, obv_20=1)


    async def test_an_all_missing_volume_window_keeps_the_price_analysis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pin_frame()
        df.loc[20:, "volume"] = NAN
        r = await run_technicals(monkeypatch, df)
        assert r["current_price"] == 111.01
        assert r["volume"] == {"current": None, "avg_20d": None, "ratio": None}
        assert r["bar_validity"]["invalid_bars_in_window"]["volume_20"] == 20


class TestProviderKeepsSessionRows:
    """Through the real pinned yfinance: a chart with an empty session row and a null volume."""

    async def test_empty_sessions_and_missing_volume_survive_the_fetch(self, yf_transport: Any) -> None:
        yf_transport.route("chart", "daily_gaps")

        df = await yfinance_client.fetch_history(FetchParams("EXMP", "1y", "1d", True))

        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        assert df["date"].tolist()[2] == "2026-10-02"
        assert close.tolist()[:2] == [10.0, 20.0] and math.isnan(close.iloc[2])
        assert calculate_returns(close, 5) == pytest.approx(2.5)  # not the compressed 6.0
        assert math.isnan(volume.iloc[-1]) and volume.iloc[0] == 1000.0
        validity = tech_mod._build_bar_validity(close, df["high"], df["low"], volume)
        assert (validity["latest_bar_invalid"], validity["latest_volume_invalid"]) == (False, True)
        assert tech_mod._build_volume(volume)["current"] is None

    async def test_only_daily_frames_keep_empty_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, bool]] = []
        frame = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [10]},
                             index=pd.DatetimeIndex(["2026-10-08 10:00"], tz="America/New_York"))

        class ChartTicker:
            def __init__(self, symbol: str) -> None:
                pass

            def history(self, **kwargs: Any) -> pd.DataFrame:
                seen.append((kwargs["interval"], kwargs["keepna"]))
                return frame.copy()

        monkeypatch.setattr(yfinance_client.yf, "Ticker", ChartTicker)
        for interval in ("1d", "1wk", "5m", "1h"):
            await yfinance_client.fetch_history(FetchParams("AAPL", "1mo", interval, True))
        assert seen == [("1d", True), ("1wk", True), ("5m", False), ("1h", False)]

    async def test_missing_volume_stays_on_its_exchange_session(self, yf_transport: Any) -> None:
        """ASX daily bars are stamped at the Sydney open, the previous day in UTC."""
        yf_transport.route("chart", "sydney_missing_volume")

        df = await yfinance_client.fetch_history(FetchParams("EXMP.AX", "1y", "1d", True))

        assert df["date"].tolist() == ["2026-10-07", "2026-10-08", "2026-10-09"]
        volume = pd.to_numeric(df["volume"], errors="coerce").tolist()
        assert volume[0] == 1000.0 and math.isnan(volume[1]) and volume[2] == 3000.0

    async def test_a_live_duplicate_row_keeps_its_volume(self, yf_transport: Any) -> None:
        """yfinance merges Yahoo's separate live row into the session's row; a null
        volume on the dropped duplicate must not blank the kept one."""
        yf_transport.route("chart", "live_separate_volume")

        df = await yfinance_client.fetch_history(FetchParams("EXMP", "1y", "1d", True))

        assert pd.to_numeric(df["volume"], errors="coerce").iloc[-1] == 2500.0

    async def test_a_row_with_no_prices_has_no_volume(self, yf_transport: Any) -> None:
        """yfinance adds an empty row for a dividend after the last price row and
        zero-fills its volume after the recorder has run."""
        yf_transport.route("chart", "dividend_ahead")

        df = await yfinance_client.fetch_history(FetchParams("EXMP", "1y", "1d", True))

        close, volume = pd.to_numeric(df["close"], errors="coerce"), pd.to_numeric(df["volume"], errors="coerce")
        assert math.isnan(close.iloc[-1]) and math.isnan(volume.iloc[-1])
        assert tech_mod._build_bar_validity(close, df["high"], df["low"], volume)["latest_volume_invalid"] is True

    def test_contract_yfinance_zero_fills_volume_after_dating_sessions(self) -> None:
        """The seam the missing-volume recorder relies on, in the pinned yfinance: the
        quotes are put in exchange time and DST-fixed, their .date becomes the daily
        row date, and only later is volume zero-filled."""
        source = inspect.getsource(PriceHistory.history)
        set_tz = source.index("quotes = utils.set_df_tz(quotes, interval, tz_exchange)")
        fix_dst = source.index("quotes = utils.fix_Yahoo_dst_issue(quotes, interval)")
        dated = source.index("quotes.index = pd.to_datetime(quotes.index.date)")
        zero_filled = source.index("df['Volume'] = df['Volume'].fillna(0)")
        assert set_tz < fix_dst < dated < zero_filled
        assert source.count("fix_Yahoo_dst_issue") == 1
        assert history_module.utils is yfinance_utils
        assert yfinance_utils.fix_Yahoo_dst_issue.__wrapped__ is not None  # type: ignore[attr-defined]


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


class TestAnalyzeBarValidity:
    async def test_gap_counts_and_volume_validity_survive_every_tier(self, analyze: Any) -> None:
        validity = {"latest_bar_invalid": False, "latest_volume_invalid": True,
                    "invalid_bars_in_window": {"sma_20": 1, "rsi_14": 1, "volume_20": 1}}
        result = await analyze(technicals={"bar_validity": validity})

        for detail in ("decision", "standard", "full"):
            dq = project_analyze_result(result, detail)["data_quality"]
            assert dq["bar_validity"] == validity, detail
            assert {"latest_daily_volume_invalid", "invalid_bars_in_indicator_windows"} <= set(dq["warnings"]), detail
            assert "latest_daily_bar_invalid" not in dq["warnings"], detail

    async def test_clean_bars_raise_no_warning(self, analyze: Any) -> None:
        validity = {"latest_bar_invalid": False, "latest_volume_invalid": False,
                    "invalid_bars_in_window": {"sma_20": 0}}
        result = await analyze(technicals={"bar_validity": validity})
        assert not {"latest_daily_bar_invalid", "latest_daily_volume_invalid",
                    "invalid_bars_in_indicator_windows"} & set(result["data_quality"]["warnings"])


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
