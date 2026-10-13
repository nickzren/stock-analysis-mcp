"""D7: SMA and Bollinger skip empty sessions (Nick, 2026-10-12).

Each value is computed from the last N finite closes, all N required, and placed
back on its own session; a session without a finite close has no value. One empty
session must not silence the falling-knife gate for the next ~200 sessions, as it
did once WP5 kept empty rows (main never saw them: yfinance dropped them).
"""

import importlib
import math
from datetime import datetime
from typing import Any

import pandas as pd
import pytest
import pytz

from stock_analysis.utils.indicators import (
    calculate_bollinger_bands,
    calculate_sma,
    invalid_bars_in_window,
)
from stock_analysis.utils.scan_screen import screen_symbol
from stock_analysis.utils.validators import FetchParams

tech_mod = importlib.import_module("stock_analysis.tools.technicals")
risk_mod = importlib.import_module("stock_analysis.tools.risk_metrics")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
position_mod = importlib.import_module("stock_analysis.tools.position")

NAN = float("nan")
SATURDAY = pytz.timezone("America/New_York").localize(datetime(2026, 10, 10, 12, 0))


def values(series: pd.Series) -> list[float | None]:
    return [None if math.isnan(v) else round(v, 6) for v in series]


class TestSma:
    def test_the_window_skips_the_empty_session(self) -> None:
        closes = pd.Series([10.0, 20.0, NAN, 40.0, 50.0, 60.0, 70.0])
        assert values(calculate_sma(closes, 5)) == [None, None, None, None, None, 36.0, 48.0]
        assert invalid_bars_in_window(closes, window=5) == 1  # still disclosed

    def test_multiple_gaps(self) -> None:
        closes = pd.Series([1.0, 2.0, NAN, 4.0, NAN, 6.0, 7.0])
        assert values(calculate_sma(closes, 3)) == [None, None, None, round(7 / 3, 6), None, 4.0, round(17 / 3, 6)]

    def test_insufficient_finite_history_stays_null(self) -> None:
        assert values(calculate_sma(pd.Series([1.0, 2.0, NAN, 4.0, 5.0]), 5)) == [None] * 5

    def test_an_invalid_latest_bar_has_no_value(self) -> None:
        sma = calculate_sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, NAN]), 3)
        assert math.isnan(sma.iloc[-1]) and sma.iloc[-2] == 5.0

    def test_infinite_and_missing_prices_are_skipped(self) -> None:
        closes = pd.Series([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0], dtype="Float64")
        closes[1] = pd.NA
        assert values(calculate_sma(closes, 2)) == [None, None, None, 2.5, 4.5, 5.5]

    def test_the_session_index_is_kept(self) -> None:
        closes = pd.Series([1.0, NAN, 3.0, 5.0], index=[10, 11, 12, 13], name="close")
        sma = calculate_sma(closes, 2)
        assert (list(sma.index), sma.name) == ([10, 11, 12, 13], "close")
        assert values(sma) == [None, None, 2.0, 4.0]


class TestBollinger:
    def test_a_gap_in_the_window_matches_the_bands_without_it(self) -> None:
        closes = [100.0 + (i % 7) for i in range(30)]
        gapped = pd.Series(closes[:25] + [NAN] + closes[25:])
        assert calculate_bollinger_bands(gapped) == calculate_bollinger_bands(pd.Series(closes))
        assert calculate_bollinger_bands(gapped)["middle"] is not None

    def test_an_invalid_latest_bar_has_no_bands(self) -> None:
        bands = calculate_bollinger_bands(pd.Series([100.0 + (i % 7) for i in range(30)] + [NAN]))
        assert bands == {"upper": None, "middle": None, "lower": None, "bandwidth": None, "pct_b": None}


def knife_frame(empty: tuple[int, ...] = ()) -> pd.DataFrame:
    """A year of flat trading, then a 28% slide in ~2 months and a base near the low."""
    closes = []
    for i in range(252):
        if i < 120:
            close = 100.0 + (5.0 if i == 50 else (i % 5) * 0.3)
        elif i < 190:
            close = 100.0 + (i % 3) * 0.2
        elif i < 226:
            close = 100.0 - (i - 189) * (28.0 / 36)
        else:
            close = 72.0 + (i % 2) * 0.1
        closes.append(round(close, 2))
    df = pd.DataFrame({
        "date": pd.bdate_range(end="2026-10-09", periods=252).strftime("%Y-%m-%d").tolist(),
        "open": closes, "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
        "close": closes, "volume": [2_000_000.0] * 252,
    })
    for i in empty:
        df.loc[i, ["open", "high", "low", "close", "volume"]] = NAN
    df.attrs.update({"currency": "USD", "instrument_type": "EQUITY", "symbol": "KNIFE"})
    return df


GAPS = [(), (100,), (60, 100, 180)]
GAP_IDS = ["no_gap", "one_old_gap", "three_gaps"]


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> Any:
    def use(df: pd.DataFrame) -> None:
        async def fake_history(params: FetchParams) -> pd.DataFrame:
            return df if params.symbol == "KNIFE" else knife_frame()

        monkeypatch.setattr(tech_mod, "fetch_history", fake_history)
        monkeypatch.setattr(risk_mod, "fetch_history", fake_history)

    return use


class TestFallingKnifeSurvivesEmptySessions:
    @pytest.mark.parametrize("empty", GAPS, ids=GAP_IDS)
    def test_scan(self, empty: tuple[int, ...]) -> None:
        verdict = screen_symbol(knife_frame(empty), now=SATURDAY, session="closed")
        assert verdict["action_hint"] == "avoid"
        assert "falling_knife" in verdict["blocker_ids"]

    @pytest.mark.parametrize("empty", GAPS, ids=GAP_IDS)
    async def test_analyze(self, monkeypatch: pytest.MonkeyPatch, prices: Any, empty: tuple[int, ...]) -> None:
        prices(knife_frame(empty))

        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Knife", "currency": "USD", "quote_type": "EQUITY",
                    "market_cap": 1e10, "current_price": 72.0}

        async def fundamentals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "valuation": {"pe_trailing": 15.0, "trailing_eps": 6.0},
                    "profitability": {"net_margin": 0.2}}

        async def empty_tool(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        monkeypatch.setattr(orchestrator, "fundamentals_snapshot", fundamentals)
        for name in ("events_calendar", "stock_news", "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty_tool)

        result = await orchestrator.analyze_stock("KNIFE", _now=SATURDAY)

        assert result["dip_assessment"]["dip_classification"]["type"] == "falling_knife"
        assert result["decision_card"]["hard_gates"]["checks"]["falling_knife"] is True

    @pytest.mark.parametrize("empty", GAPS, ids=GAP_IDS)
    async def test_position(self, prices: Any, empty: tuple[int, ...]) -> None:
        prices(knife_frame(empty))

        result = await position_mod.analyze_position("KNIFE", cost_basis=100.0, purchase_date="2025-01-02", shares=10)

        sell = result["sell_signals"]
        assert (sell["broke_sma200"], sell["death_cross"], sell["active_count"]) == (True, True, 3)

    async def test_an_empty_3m_start_bar_leaves_the_signals_unknown(
        self, monkeypatch: pytest.MonkeyPatch, prices: Any
    ) -> None:
        """Codex at 1a9c66a: a null return_3m read as `weak_3m_momentum` removed."""
        prices(knife_frame((251 - 63,)))

        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Knife", "currency": "USD", "quote_type": "EQUITY",
                    "market_cap": 1e10, "current_price": 72.0}

        async def empty_tool(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        for name in ("fundamentals_snapshot", "events_calendar", "stock_news", "ownership_analysis",
                     "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty_tool)

        result = await orchestrator.analyze_stock("KNIFE", _now=SATURDAY)

        assert result["data_quality"]["bar_validity"]["invalid_return_endpoints"] == ["return_3m"]
        snapshot = result["watchlist_snapshot"]
        assert (snapshot["bullish_signals"], snapshot["bearish_signals"]) == (None, None)

    def test_an_empty_latest_session_still_waits_for_data(self) -> None:
        verdict = screen_symbol(knife_frame((251,)), now=SATURDAY, session="closed")
        assert (verdict["action_hint"], verdict["blocker_ids"]) == (
            "wait_for_data", ["data_quality_critical", "freshness_unverifiable@daily", "liquidity_missing"])
