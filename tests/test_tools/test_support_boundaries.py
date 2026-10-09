"""Step 1 WP4 (D5): instrument and currency support boundaries (R4.6-R4.8)."""

import importlib
import inspect
import json
from typing import Any

import pandas as pd
import pytest
from yfinance.base import TickerBase
from yfinance.scrapers.history import PriceHistory

from stock_analysis.data import yfinance_client
from stock_analysis.tools.analyze.action_zones import build_position_sizing_range
from stock_analysis.tools.analyze.decision_card import _compute_hard_gates, _resolve_action_now
from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.scan_screen import screen_symbol
from stock_analysis.utils.validators import FetchParams
from stock_analysis.utils.valuation_basis import classify_support, sizing_unavailable_reason
from tests.test_scan_screen import make_df, uptrend_breakout_df
from tests.test_tools.test_trade_card import blocker_ids, build_card

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

GBP_STOCK = classify_support(quote_type="EQUITY", currency="GBp", symbol="VOD.L", name="Vodafone")
US_STOCK = classify_support(quote_type="EQUITY", currency="USD", symbol="MSFT", name="Microsoft")
ETF = classify_support(quote_type="ETF", currency="USD", symbol="SPY", name="SPDR S&P 500")


class TestTradeCard:
    @pytest.mark.parametrize(
        ("summary", "blocker"),
        [
            ({"currency": "GBp", "quote_type": "EQUITY", "name": "Vodafone"}, "unsupported_currency"),
            ({"currency": "USD", "quote_type": "ETF", "name": "SPDR S&P 500"}, "unsupported_instrument"),
            ({"name": "Unknown"}, "unsupported_currency"),
        ],
        ids=["r4_7_gbp", "r4_8_etf", "unknown_fails_closed"],
    )
    def test_unsupported_is_watch_without_a_plan(self, summary: dict[str, Any], blocker: str) -> None:
        card = build_card(summary_data=summary, actionable_price=101.5)

        assert card["action"] == "watch"
        assert blocker in blocker_ids(card)
        assert card["plan"] is None

    @pytest.mark.parametrize(
        "summary",
        [{"currency": "GBp", "quote_type": "EQUITY", "name": "Vodafone"},
         {"currency": "USD", "quote_type": "ETF", "name": "SPDR S&P 500"}],
        ids=["gbp", "etf"],
    )
    def test_r4_7_no_dollar_liquidity_claim_when_unsupported(self, summary: dict[str, Any]) -> None:
        card = build_card(
            summary_data=summary, risk_data={"liquidity": {"avg_dollar_volume": None}}, actionable_price=101.5
        )

        assert not {"weak_liquidity", "liquidity_missing"} & blocker_ids(card)
        assert card["action"] == "watch"


class TestAnalyzeDecision:
    def test_r4_7_decision_card_caps_new_money_at_watch_and_skips_liquidity(self) -> None:
        gates = _compute_hard_gates(
            events_data={}, data_quality={}, dip_assessment=None,
            fundamentals_summary={}, risk_data={"liquidity": {"avg_dollar_volume": None}}, support=GBP_STOCK,
        )

        action, _ = _resolve_action_now({"mid_term": "buy"}, gates)

        assert action == "watch"
        assert gates["checks"]["unsupported_currency"] is True
        assert gates["checks"]["liquidity_missing"] is False

    def test_holders_still_get_non_new_money_guidance(self) -> None:
        gates = _compute_hard_gates(
            events_data={}, data_quality={}, dip_assessment=None, fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 5e7}}, support=ETF,
        )

        assert _resolve_action_now({"mid_term": "hold_or_reduce"}, gates)[0] == "hold_or_reduce"

    @pytest.mark.parametrize(("support", "sized"), [(GBP_STOCK, False), (ETF, False), (US_STOCK, True)])
    def test_r4_6_r4_7_sizing_only_when_supported(self, support: dict[str, Any], sized: bool) -> None:
        sizing = build_position_sizing_range(
            current_price=100.0, stop_calculation={"stop_distance_pct": 0.05}, regime_classification="medium",
            investor_profile={"account_size": 5000.0}, unavailable_reason=sizing_unavailable_reason(support),
        )

        assert (sizing["dollars_for_account"] is not None) is sized
        assert (sizing["shares_range"] is not None) is sized
        assert (sizing["stop_implied_max"]["dollars_for_account"] is not None) is sized


@pytest.fixture
def analyze(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def run(summary: dict[str, Any], *, bullish: bool = False, account_size: float | None = None) -> dict[str, Any]:
        async def fake_summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "current_price": 100.0, "market_cap": 1e10, **summary}

        async def fundamentals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "valuation": {"pe_trailing": 9.0, "trailing_eps": 11.0},
                    "profitability": {"net_margin": 0.2}, "yield_metrics": {"fcf_yield": 0.09}}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", fake_summary)
        monkeypatch.setattr(orchestrator, "fundamentals_snapshot", fundamentals)
        for name in ("technicals", "risk_metrics", "events_calendar", "stock_news",
                     "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        if bullish:
            real_policy = orchestrator.build_policy_action

            def bullish_policy(**kwargs: Any) -> dict[str, Any]:
                return {**real_policy(**kwargs), "mid_term": "hold_or_add", "long_term": "accumulate"}

            monkeypatch.setattr(orchestrator, "build_policy_action", bullish_policy)
        return await orchestrator.analyze_stock("SPY", account_size=account_size)

    return run


NEW_MONEY = {"buy", "starter", "hold_or_add", "accumulate", "build_position", "starter_position_only",
             "starter_then_add", "speculative_small_position", "speculative_with_thesis"}
GBP_SUMMARY = {"name": "Vodafone", "currency": "GBp", "quote_type": "EQUITY"}
ETF_SUMMARY = {"name": "SPDR S&P 500", "currency": "USD", "quote_type": "ETF"}
UNKNOWN_SUMMARY = {"name": "Unknown"}
US_SUMMARY = {"name": "Example", "currency": "USD", "quote_type": "EQUITY"}


class TestAnalyzeCaps:
    @pytest.mark.parametrize("summary", [GBP_SUMMARY, ETF_SUMMARY, UNKNOWN_SUMMARY], ids=["gbp", "etf", "unknown"])
    async def test_new_money_is_watch_on_every_surface(self, analyze: Any, summary: dict[str, Any]) -> None:
        result = await analyze(summary, bullish=True)

        assert (result["policy_action"]["mid_term"], result["policy_action"]["long_term"]) == ("watch", "watch")
        assert result["decision_card"]["action_now"] == "watch"
        assert not set(result["decision_card"]["horizon_fit"].values()) & NEW_MONEY
        for mode in ("core", "balanced", "speculative"):
            assert result["decision_modes"][mode]["action"] not in NEW_MONEY, mode
            assert result["decision_modes"][mode].get("long_term_view") not in NEW_MONEY, mode
        text = json.dumps(result["executive_summary"]).lower()
        assert "hold or add" not in text and "accumulate" not in text

    async def test_control_a_supported_stock_keeps_its_policy(self, analyze: Any) -> None:
        result = await analyze(US_SUMMARY, bullish=True)

        assert result["policy_action"]["mid_term"] == "hold_or_add"

    @pytest.mark.parametrize(
        ("summary", "reason"),
        [(GBP_SUMMARY, "unsupported_currency"), (ETF_SUMMARY, "unsupported_instrument"),
         (UNKNOWN_SUMMARY, "unsupported_instrument")],
        ids=["gbp", "etf", "unknown"],
    )
    async def test_r4_7_no_share_or_dollar_sizing_anywhere(
        self, analyze: Any, summary: dict[str, Any], reason: str
    ) -> None:
        result = await analyze(summary, account_size=5000.0)

        sizing = result["action_zones"]["position_sizing_range"]
        assert (sizing["dollars_for_account"], sizing["shares_range"], sizing["unavailable_reason"]) == (
            None, None, reason
        )
        card_sizing = result["decision_card"]["sizing"]
        assert (card_sizing["starter_dollars"], card_sizing["whole_shares"]) == (None, None)
        for mode in ("core", "balanced", "speculative"):
            assert "dollars" not in result["decision_modes"][mode]["starter_position"], mode

    async def test_r4_6_a_usd_adr_keeps_its_sizing(self, analyze: Any) -> None:
        result = await analyze({**US_SUMMARY, "financial_currency": "TWD"}, account_size=5000.0)

        assert result["action_zones"]["position_sizing_range"]["dollars_for_account"] is not None
        assert result["decision_card"]["sizing"]["starter_dollars"] is not None

    async def test_r4_8_no_stock_only_text_in_any_tier(self, analyze: Any) -> None:
        result = await analyze(ETF_SUMMARY)

        assert result["dip_assessment"]["status"] == "not_applicable"
        assert result["decision_context"]["fundamentals"]["status"] == "not_applicable"
        for detail in ("decision", "standard", "full"):
            text = json.dumps(project_analyze_result(result, detail))
            for phrase in ("net_margin turns negative", "net_margin >", "guidance cut",
                           "Fundamental data unavailable", "fundamentals_key_inputs_missing"):
                assert phrase not in text, (detail, phrase)


class TestAnalyzeUnsupportedInstrument:
    async def test_r4_8_etf_gets_no_fundamentals_dislocation_or_new_money(self, analyze: Any) -> None:
        result = await analyze({"name": "SPDR S&P 500", "currency": "USD", "quote_type": "ETF"})

        assert result["summary"]["support"]["instrument"]["status"] == "unsupported"
        assert result["data_quality"]["fundamentals_status"] == "not_applicable"
        assert result["fundamentals_summary"]["summary"] is None
        assert result["action_zones"]["valuation_assessment"]["gate"] == "not_applicable"
        assert result["dislocation_framework"]["status"] == "not_applicable"
        assert result["decision_card"]["action_now"] not in {"buy", "starter", "hold_or_add"}
        assert result["policy_action"]["mid_term"] not in {"buy", "hold_or_add", "speculative_small_position"}
        for mode in ("core", "balanced", "speculative"):
            assert result["decision_modes"][mode]["action"] not in {
                "build_position", "starter_position_only", "starter_then_add"
            }

    async def test_support_survives_every_detail_tier(self, analyze: Any) -> None:
        result = await analyze({"name": "Vodafone", "currency": "GBp", "quote_type": "EQUITY"})

        for detail in ("decision", "standard", "full"):
            projected = project_analyze_result(result, detail)
            assert projected["summary"]["support"]["currency"]["status"] == "unsupported", detail
            assert "unsupported_currency" in json.dumps(projected), detail


class TestScan:
    @pytest.mark.parametrize(
        ("currency", "instrument", "blocker"),
        [("GBp", "EQUITY", "unsupported_currency"), ("USD", "ETF", "unsupported_instrument"),
         (None, None, "unsupported_currency")],
        ids=["r4_7_gbp", "etf", "unknown_fails_closed"],
    )
    def test_unsupported_rows_are_flagged_and_not_promoted(
        self, currency: Any, instrument: Any, blocker: str
    ) -> None:
        df = make_df([80.0 + i * 0.12 for i in range(260)], volumes=[1000.0] * 260)  # thin
        df.attrs.update({"currency": currency, "instrument_type": instrument})

        screen = screen_symbol(df)

        assert (screen["promote"], screen["action_hint"]) == (False, "watch")
        assert blocker in screen["blocker_ids"]
        assert not {"weak_liquidity", "liquidity_missing"} & set(screen["blocker_ids"])

    def test_a_supported_us_row_still_promotes(self) -> None:
        assert screen_symbol(uptrend_breakout_df())["promote"] is True


class TestHistoryMetadata:
    async def test_fetch_history_carries_currency_and_instrument_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [10]},
            index=pd.DatetimeIndex(["2026-10-08"], tz="America/New_York"),
        )

        class ChartTicker:
            def __init__(self, symbol: str) -> None:
                self._price_history = type("PH", (), {"_history_metadata": {
                    "currency": "GBp", "instrumentType": "EQUITY"}})()

            def history(self, **kwargs: Any) -> pd.DataFrame:
                return frame.copy()

        monkeypatch.setattr(yfinance_client.yf, "Ticker", ChartTicker)

        df = await yfinance_client.fetch_history(FetchParams("VOD.L", "1y", "1d", True))

        assert (df.attrs["currency"], df.attrs["instrument_type"], df.attrs["symbol"]) == ("GBp", "EQUITY", "VOD.L")

    def test_contract_yfinance_stores_history_metadata_on_the_price_history(self) -> None:
        """The hook _history_metadata reads (no extra request) exists in the pinned yfinance."""
        assert "self._price_history" in inspect.getsource(TickerBase._lazy_load_price_history)
        assert "self._history_metadata" in inspect.getsource(PriceHistory.__init__)
