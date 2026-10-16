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
from stock_analysis.tools.analyze.decision_card import _compute_hard_gates
from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.scan_screen import screen_symbol
from stock_analysis.utils.validators import FetchParams
from stock_analysis.utils.valuation_basis import classify_support, sizing_unavailable_reason
from tests.support import decision_matrix as dm
from tests.support.decision_matrix import decide_on
from tests.test_scan_screen import FRESH, make_df, uptrend_breakout_df
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


class TestOrdinaryAdr:
    def test_an_ordinary_adr_is_a_supported_stock_on_the_trade_card(self) -> None:
        name = "Taiwan Semiconductor Manufacturing Company Limited American Depositary Shares"

        card = build_card(summary_data={"currency": "USD", "quote_type": "EQUITY", "name": name},
                          actionable_price=101.5)

        assert card["action"] == "trade_now" and card["plan"] is not None
        assert card["support"]["instrument"]["status"] == "supported"


class TestAnalyzeDecision:
    def test_r4_7_decision_card_caps_new_money_at_watch_and_skips_liquidity(self) -> None:
        gates = _compute_hard_gates(
            events_data={}, data_quality={}, technicals_data=None,
            fundamentals_summary={}, risk_data={"liquidity": {"avg_dollar_volume": None}}, support=GBP_STOCK,
        )

        decision = decide_on({"mid_term": "buy", "long_term": "accumulate"}, gates["blocking"])

        assert (decision["new_money"]["action"], decision["long_term"]["new_money"]["action"]) == ("watch", "watch")
        assert gates["checks"]["unsupported_currency"] is True
        assert gates["checks"]["liquidity_missing"] is False

    def test_holders_still_get_non_new_money_guidance(self) -> None:
        gates = _compute_hard_gates(
            events_data={}, data_quality={}, technicals_data=None, fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 5e7}}, support=ETF,
        )

        gated = decide_on({"mid_term": "hold_or_reduce"}, gates["blocking"], tilt="bearish")
        ungated = decide_on({"mid_term": "hold_or_reduce"}, tilt="bearish")

        assert gated["new_money"]["action"] == "watch"
        assert gated["if_held"] == ungated["if_held"] == {"action": "reduce", "add_allowed": False,
                                                           "basis": "tilt_bearish"}

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
UNKNOWN_CURRENCY_SUMMARY = {"name": "Example", "quote_type": "EQUITY"}
US_SUMMARY = {"name": "Example", "currency": "USD", "quote_type": "EQUITY"}


class TestAnalyzeCaps:
    @pytest.mark.parametrize("summary", [GBP_SUMMARY, ETF_SUMMARY, UNKNOWN_SUMMARY], ids=["gbp", "etf", "unknown"])
    async def test_new_money_is_watch_on_every_surface(self, analyze: Any, summary: dict[str, Any]) -> None:
        result = await analyze(summary, bullish=True)

        policy = result["diagnostics"]["pre_gate_policy"]
        assert (policy["mid_term"], policy["long_term"]) == ("watch", "watch")
        assert result["decision_card"]["action_now"] == "watch"
        assert "horizon_fit" not in result["decision_card"] and "policy_action" not in result
        for mode in ("core", "balanced", "speculative"):
            assert result["decision_modes"][mode]["action"] not in NEW_MONEY, mode
            assert result["decision_modes"][mode].get("long_term_view") not in NEW_MONEY, mode
        text = json.dumps(result["executive_summary"]).lower()
        assert "hold or add" not in text and "accumulate" not in text

    @pytest.mark.parametrize("summary", [GBP_SUMMARY, UNKNOWN_CURRENCY_SUMMARY], ids=["gbp", "unknown_currency"])
    async def test_the_dip_recommendation_and_entry_actions_are_capped_too(
        self, monkeypatch: pytest.MonkeyPatch, summary: dict[str, Any]
    ) -> None:
        entry = ["consider_small_position", "starter_position", "accumulate", "add_position", "watch_for_reversal"]

        def buy_the_dip(**kwargs: Any) -> dict[str, Any]:
            return {"assessment": {"dip_quality": "good", "recommendation": "buy_the_dip", "rationale": "x"},
                    "entry_timing": {"signals": [{"signal": f"s{i}", "action": action, "rationale": "r"}
                                                 for i, action in enumerate(entry)], "wait_for": []}}

        monkeypatch.setattr(orchestrator, "build_dip_assessment", buy_the_dip)
        fundamentals = {**dm.PROFITABLE, "financial_health": {"debt_to_equity": 0.5}}
        capped = await dm.run("S2_buy_clean", summary=summary, fundamentals=fundamentals)
        control = await dm.run("S2_buy_clean", fundamentals=fundamentals)

        assessment = capped["dip_assessment"]["assessment"]
        assert capped["decision_card"]["new_money"]["action"] == "watch"
        assert "unsupported_currency" in dm.gate_ids(capped)
        assert (assessment["recommendation"], assessment["capped_by"], assessment["support_gates"]) == (
            "watch", "new_money_watch", ["unsupported_currency"])
        assert {s.get("capped_by") for s in capped["dip_assessment"]["entry_timing"]["signals"][:4]} == {
            "new_money_watch"}
        actions = [signal["action"] for signal in capped["dip_assessment"]["entry_timing"]["signals"]]
        assert actions == ["watch"] * 4 + ["watch_for_reversal"]
        assert control["decision_card"]["new_money"]["action"] == "buy"  # the control admits: nothing is capped
        assert control["dip_assessment"]["assessment"]["recommendation"] == "buy_the_dip"
        assert "capped_by" not in control["dip_assessment"]["assessment"]
        assert [signal["action"] for signal in control["dip_assessment"]["entry_timing"]["signals"]] == entry

    async def test_control_a_supported_stock_keeps_its_policy(self, analyze: Any) -> None:
        result = await analyze(US_SUMMARY, bullish=True)

        assert result["diagnostics"]["pre_gate_policy"]["mid_term"] == "hold_or_add"

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

        assert reason in dm.gate_ids(result)
        assert (result["action_zones"]["position_sizing_range"], result["decision_card"]["sizing"],
                result["decision_card"]["entry"]) == (None, None, None)
        for mode in ("core", "balanced", "speculative"):
            assert result["decision_modes"][mode]["starter_position"] is None, mode

    async def test_r4_6_a_usd_adr_keeps_its_sizing(self) -> None:
        result = await dm.run(
            "S2_buy_clean", summary={**dm.EQUITY, "financial_currency": "TWD"}, account_size=5000.0,
            fundamentals={**dm.PROFITABLE, "financial_health": {"debt_to_equity": 0.5}},
        )

        assert (result["summary"]["support"]["currency"]["status"], result["decision_card"]["new_money"]["action"]) == (
            "supported", "buy")
        assert result["action_zones"]["position_sizing_range"]["dollars_for_account"] is not None
        assert result["decision_card"]["sizing"]["starter_dollars"] is not None

    async def test_r4_8_no_stock_only_text_in_any_tier(self, analyze: Any) -> None:
        result = await analyze(ETF_SUMMARY)

        assert result["dip_assessment"]["status"] == "not_applicable"
        assert result["decision_context"]["fundamentals"]["status"] == "not_applicable"
        thesis = result["decision_context"]["thesis_checkpoints"]
        assert (thesis["hold_thesis"], thesis["checkpoints"], thesis["thesis_stop_triggers"]) == (
            "not_applicable", [], None
        )
        check = project_analyze_result(result, "standard")["dislocation_check"]
        assert (check["overall_status"], check["questions"]) == ("not_applicable", [])
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
        assert project_analyze_result(result, "standard")["fundamentals_summary"]["status"] == "not_applicable"
        assert result["action_zones"]["valuation_assessment"]["gate"] == "not_applicable"
        assert result["dislocation_framework"]["status"] == "not_applicable"
        assert result["decision_card"]["action_now"] not in {"buy", "starter", "hold_or_add"}
        assert result["diagnostics"]["pre_gate_policy"]["mid_term"] not in {"buy", "hold_or_add",
                                                                             "speculative_small_position"}
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
        [("GBp", "EQUITY", "unsupported_currency"), ("USD", "ETF", "unsupported_instrument")],
        ids=["r4_7_gbp", "etf"],
    )
    def test_unsupported_rows_are_flagged_and_not_promoted(
        self, currency: Any, instrument: Any, blocker: str
    ) -> None:
        df = make_df([80.0 + i * 0.12 for i in range(260)], volumes=[1000.0] * 260)  # thin
        df.attrs.update({"currency": currency, "instrument_type": instrument})

        screen = screen_symbol(df, now=FRESH, session="closed")

        assert (screen["promote"], screen["action_hint"]) == (False, "watch")
        assert blocker in screen["blocker_ids"]
        assert not {"weak_liquidity", "liquidity_missing"} & set(screen["blocker_ids"])

    def test_unknown_support_is_unevaluated_and_promotes(self) -> None:
        """Unknown support is unverified, not unsupported: stage 1 leaves it, and liquidity, to the card (D-11)."""
        df = make_df([80.0 + i * 0.12 for i in range(260)], volumes=[1000.0] * 260)  # thin
        df.attrs.update({"currency": None, "instrument_type": None})

        screen = screen_symbol(df, now=FRESH, session="closed")

        assert (screen["promote"], screen["action_hint"], screen["blocker_ids"]) == (True, "candidate", [])
        assert {"unsupported_instrument", "unsupported_currency"} <= set(screen["unevaluated"])

    def test_a_supported_us_row_still_promotes(self) -> None:
        assert screen_symbol(uptrend_breakout_df(), now=FRESH, session="closed")["promote"] is True


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


class TestBurnDilutionCurrency:
    """Dilution divides a raise in the statement currency by the market cap in the
    price currency; with different currencies it is not computed."""

    FUND = {"financial_health": {"total_cash": 1e9}, "cash_flow": {"free_cash_flow_ttm": -2e9}}

    @pytest.mark.parametrize(("financial_currency", "computed"), [("USD", True), ("TWD", False), (None, False)])
    def test_dilution_needs_one_currency(self, financial_currency: str | None, computed: bool) -> None:
        summary = {"market_cap": 1e10, "currency": "USD", "financial_currency": financial_currency}

        burn, _ = orchestrator._compute_burn_metrics(self.FUND, summary, is_unprofitable_company=True)

        assert burn is not None
        assert burn["cash_runway_quarters"] == 2.0  # runway is one currency either way
        assert (burn["dilution_analysis"] is not None) is computed
        assert ("dilution_currency_unverified" in burn["warnings"]) is not computed
