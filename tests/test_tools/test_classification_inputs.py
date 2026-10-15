"""WP9: classification-input corrections -- net debt and missing tilt (R9.1-R9.5)."""

import importlib
from typing import Any

import pytest

from stock_analysis.tools.analyze.dislocation_framework import (
    _build_balance_sheet_safety,
    _build_thesis_integrity,
    build_dislocation_framework,
    classify_dislocation_opportunity,
)
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from tests.support.decision_matrix import decide_on

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
fundamentals_module = importlib.import_module("stock_analysis.tools.fundamentals")

NET_CASH = {"net_cash_positive": True, "debt_to_equity": 0.5}
HEALTHY_NET_DEBT = {"net_cash_positive": False, "debt_to_equity": 1.2}
NO_BURN = {"status": "not_applicable"}
SHORT_RUNWAY = {"status": "available", "cash_runway_quarters": 3.0, "liquidity": 5e7}


def fundamentals(health: dict[str, Any], burn: dict[str, Any]) -> dict[str, Any]:
    return {"growth": {"revenue_yoy": 0.12}, "profitability": {"net_margin": 0.2, "fcf_positive": True},
            "health": health, "burn_metrics": burn}


VERDICT = {"tilt": "bullish", "decomposed": {"business_quality": "strong"},
           "horizon_fit": {"mid_term": "ok", "long_term": "ok"}}
ZONES = {"current_zone": "accumulate", "valuation_assessment": {"gate": "attractive", "is_unprofitable": False}}
DEEP_DIP = {"dip_depth": {"from_52w_high": -0.45, "severity": "deep"}}


class TestR91NetDebt:
    @pytest.mark.parametrize("health", [HEALTHY_NET_DEBT, {"net_cash_positive": False, "debt_to_equity": 0.0}],
                             ids=["net_debt", "exactly_zero_net_cash"])
    def test_net_debt_alone_is_watch_with_a_reason(self, health: dict[str, Any]) -> None:
        balance = _build_balance_sheet_safety(health, NO_BURN)
        assert (balance["status"], balance["checks"]["net_cash"]) == ("watch", "fail")
        assert "net_debt" in balance["reasons"]

    def test_the_thesis_is_not_broken_by_net_debt(self) -> None:
        balance = _build_balance_sheet_safety(HEALTHY_NET_DEBT, NO_BURN)
        thesis = _build_thesis_integrity({}, "intact", balance["status"], "ok")
        assert thesis["status"] == "watch"

    def test_dislocation_candidacy_follows_the_corrected_status(self) -> None:
        signals = classify_dislocation_opportunity(VERDICT, ZONES, fundamentals(HEALTHY_NET_DEBT, NO_BURN), DEEP_DIP)
        assert (signals["balance_status"], signals["mismatch_status"]) == ("watch", "unclear")
        assert signals["is_dislocation_candidate"] is False  # watch alone never qualifies

    def test_a_net_cash_company_is_still_safe_and_can_be_a_candidate(self) -> None:
        signals = classify_dislocation_opportunity(VERDICT, ZONES, fundamentals(NET_CASH, NO_BURN), DEEP_DIP)
        assert (signals["balance_status"], signals["is_dislocation_candidate"]) == ("safe", True)

    def test_the_framework_reads_net_debt_as_watch_not_broken(self) -> None:
        framework = build_dislocation_framework(
            VERDICT, ZONES, fundamentals(HEALTHY_NET_DEBT, NO_BURN), DEEP_DIP,
            {"thesis_checkpoints": {}}, {}, decision=decide_on({"mid_term": "hold", "long_term": "hold"}),
            conditions={"buy_only_if": [], "add_only_if": []},
        )
        assert framework["balance_sheet_safety"]["status"] == "watch"
        assert framework["thesis_integrity"]["status"] == "watch"
        assert framework["mismatch_verdict"]["status"] == "unclear"


class TestR92Runway:
    @pytest.mark.parametrize(
        "health",
        [NET_CASH, HEALTHY_NET_DEBT,
         {"net_cash_positive": False, "debt_to_equity": None, "leverage_status": "negative_equity"}],
        ids=["net_cash", "net_debt", "negative_equity"],
    )
    def test_a_failed_runway_is_still_unsafe(self, health: dict[str, Any]) -> None:
        balance = _build_balance_sheet_safety(health, SHORT_RUNWAY)
        assert balance["status"] == "unsafe"
        assert "runway_under_4q" in balance["reasons"]

    def test_negative_equity_is_watch_with_a_reason_never_safe(self) -> None:
        negative = {"net_cash_positive": True, "debt_to_equity": None, "leverage_status": "negative_equity"}
        balance = _build_balance_sheet_safety(negative, NO_BURN)
        assert balance["status"] == "watch"
        assert "negative_equity" in balance["reasons"]

    def test_exactly_zero_net_cash_is_not_net_cash_positive(self) -> None:
        health = fundamentals_module._build_financial_health({"totalCash": 5e9, "totalDebt": 5e9, "debtToEquity": 40.0})
        assert health["rules"]["net_cash_positive"]["triggered"] is False
        balance = _build_balance_sheet_safety({"net_cash_positive": False, "debt_to_equity": 0.4}, NO_BURN)
        assert (balance["status"], balance["reasons"]) == ("watch", ["net_debt"])

    @pytest.mark.parametrize(
        ("health", "burn", "reasons"),
        [(NET_CASH, {"status": "available", "cash_runway_quarters": 6.0}, ["runway_under_8q"]),
         ({"net_cash_positive": True, "debt_to_equity": 2.0}, NO_BURN, ["high_leverage"]),
         ({"net_cash_positive": True, "debt_to_equity": 4.0}, NO_BURN, ["high_leverage"]),
         (NET_CASH, {"status": "available", "cash_runway_quarters": 6.0,
                     "dilution_analysis": {"dilution_if_raised_today": 0.25}}, ["runway_under_8q", "dilution_risk"])],
        ids=["runway_watch", "leverage_watch", "leverage_fail", "dilution"],
    )
    def test_each_reason_is_named(self, health: dict[str, Any], burn: dict[str, Any], reasons: list[str]) -> None:
        balance = _build_balance_sheet_safety(health, burn)
        assert (balance["status"], balance["reasons"]) == ("watch", reasons)

    def test_unknown_net_cash_is_named(self) -> None:
        balance = _build_balance_sheet_safety({"net_cash_positive": None, "debt_to_equity": 0.2}, NO_BURN)
        assert (balance["status"], balance["reasons"]) == ("watch", ["net_cash_unknown"])

    @pytest.mark.parametrize(
        ("health", "burn", "summary"),
        [(HEALTHY_NET_DEBT, NO_BURN, "Balance sheet needs monitoring: net debt."),
         (HEALTHY_NET_DEBT, {"status": "available", "cash_runway_quarters": 10.0}, "Balance sheet needs monitoring: net debt."),
         (NET_CASH, {"status": "available", "cash_runway_quarters": 6.0}, "Balance sheet needs monitoring: runway about 6.0q."),
         ({"net_cash_positive": False, "debt_to_equity": None, "leverage_status": "negative_equity"}, NO_BURN,
          "Balance sheet needs monitoring: net debt, negative equity."),
         (HEALTHY_NET_DEBT, SHORT_RUNWAY,
          "Balance sheet risk is elevated enough to threaten the thesis: runway about 3.0q, net debt.")],
        ids=["net_debt", "net_debt_long_runway", "short_runway", "negative_equity", "failed_runway"],
    )
    def test_the_summary_names_the_actual_causes(self, health: dict[str, Any], burn: dict[str, Any], summary: str) -> None:
        assert _build_balance_sheet_safety(health, burn)["summary"] == summary

    def test_a_safe_balance_sheet_has_no_reasons(self) -> None:
        balance = _build_balance_sheet_safety(NET_CASH, NO_BURN)
        assert (balance["status"], balance["reasons"]) == ("safe", [])


def policy(verdict: dict[str, Any], *, zones: dict[str, Any] | None = None,
           dip: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_policy_action(
        verdict=verdict,
        action_zones=zones or {"current_zone": "hold_neutral",
                               "valuation_assessment": {"gate": "neutral", "is_unprofitable": False}},
        decomposed=verdict.get("decomposed"),
        dip_assessment=dip,
        fundamentals_summary=fundamentals(NET_CASH, NO_BURN),
    )


def with_fit(mid: str, long: str, **tilt: Any) -> dict[str, Any]:
    return {"decomposed": {"business_quality": "strong"}, "horizon_fit": {"mid_term": mid, "long_term": long},
            **tilt}


MISSING_TILT = [{}, {"tilt": None}]


class TestR93MissingTilt:
    @pytest.mark.parametrize("tilt", MISSING_TILT, ids=["absent", "null"])
    @pytest.mark.parametrize(("mid", "long"), [("ok", "ok"), ("caution", "caution")])
    def test_tilt_driven_actions_become_insufficient_data(self, tilt: dict[str, Any], mid: str, long: str) -> None:
        action = policy(with_fit(mid, long, **tilt))

        assert (action["mid_term"], action["long_term"]) == ("insufficient_data", "insufficient_data")
        tilt_reasons = [r for r in action["rationale"] if "tilt=" in r]
        assert tilt_reasons == [f"mid_term={mid}, tilt=missing", f"long_term={long}, tilt=missing"]

    @pytest.mark.parametrize("tilt", MISSING_TILT, ids=["absent", "null"])
    def test_evidence_driven_actions_do_not_need_a_tilt(self, tilt: dict[str, Any]) -> None:
        attractive = {"current_zone": "accumulate",
                      "valuation_assessment": {"gate": "attractive", "is_unprofitable": False}}
        assert policy(with_fit("avoid", "ok", **tilt), zones=attractive)["mid_term"] == "avoid"
        assert policy(with_fit("ok", "ok", **tilt), zones=attractive)["long_term"] == "accumulate"
        assert policy(with_fit("ok", "avoid", **tilt))["long_term"] == "avoid_core_position"
        unprofitable = {"current_zone": "hold_neutral",
                        "valuation_assessment": {"gate": "neutral", "is_unprofitable": True}}
        assert policy(with_fit("ok", "caution", **tilt), zones=unprofitable)["long_term"] == "speculative_only"

    @pytest.mark.parametrize("tilt", MISSING_TILT, ids=["absent", "null"])
    def test_a_dislocation_candidate_keeps_its_evidence_backed_action(self, tilt: dict[str, Any]) -> None:
        action = policy(with_fit("caution", "caution", **tilt), zones=ZONES, dip=DEEP_DIP)
        assert (action["mid_term"], action["long_term"]) == ("speculative_small_position", "small_position_with_stops")


class TestR94ExplicitTilt:
    @pytest.mark.parametrize(
        ("tilt", "mid", "long"),
        [("bearish", "hold_or_reduce", "hold"), ("neutral", "hold", "hold"), ("bullish", "hold_or_add", "hold_or_add")],
    )
    def test_explicit_tilts_are_unchanged(self, tilt: str, mid: str, long: str) -> None:
        action = policy(with_fit("ok", "ok", tilt=tilt))
        assert (action["mid_term"], action["long_term"]) == (mid, long)
        assert f"mid_term=ok, tilt={tilt}, zone=hold_neutral" in action["rationale"]

    def test_explicit_bearish_in_caution_is_unchanged(self) -> None:
        action = policy(with_fit("caution", "caution", tilt="bearish"))
        assert (action["mid_term"], action["long_term"]) == ("wait_for_entry", "wait_or_avoid")


class TestR95FallingKnife:
    @pytest.mark.parametrize(("tilt", "codes"), [(MISSING_TILT[0], ("insufficient_data", "insufficient_data")),
                                                 (MISSING_TILT[1], ("insufficient_data", "insufficient_data")),
                                                 ({"tilt": "bullish"}, ("hold_or_add", "hold_or_add"))],
                             ids=["absent", "null", "bullish"])
    def test_the_falling_knife_is_a_gate_not_a_candidate_rule(self, tilt: dict[str, Any],
                                                               codes: tuple[str, str]) -> None:
        """WP2.2 (plan §1): the candidate keeps its knife-free code; the falling_knife gate makes new money avoid."""
        knife = {"dip_classification": {"type": "falling_knife"}}
        action = policy(with_fit("ok", "ok", **tilt), dip=knife)
        new_money = decide_on(action, [{"id": "falling_knife", "reason": "r"}])["new_money"]

        assert (action["mid_term"], action["long_term"]) == codes == tuple(
            policy(with_fit("ok", "ok", **tilt))[h] for h in ("mid_term", "long_term"))
        assert not [line for line in action["rationale"] if "falling_knife" in line]
        assert (new_money["action"], new_money["decided_by"]) == ("avoid", "falling_knife")


STRONG_TECHNICALS = {
    "current_price": 100.0,
    "moving_averages": {"sma_20": 98.0, "sma_50": 95.0, "sma_200": 90.0,
                        "rules": {"above_sma20": {"triggered": True}, "above_sma50": {"triggered": True},
                                  "above_sma200": {"triggered": True}, "golden_cross": {"triggered": True},
                                  "death_cross": {"triggered": False}}},
    "rsi": {"value": 55.0, "rules": {"oversold": {"triggered": False}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": True}, "bearish_cross": {"triggered": False}}},
    "returns": {"return_1w": 0.02, "return_1w_zscore": 0.4, "return_1m": 0.12, "return_3m": 0.18, "return_1y": 0.4},
    "price_position": {"position_in_range": 0.7, "week_52_low": 70.0, "week_52_high": 120.0, "from_52w_high": -0.17},
}


class TestMissingTiltEndToEnd:
    async def test_a_failed_fundamentals_fetch_is_not_read_as_bearish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "current_price": 100.0, "currency": "USD", "quote_type": "EQUITY"}

        async def technicals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, **STRONG_TECHNICALS}

        async def failed(symbol: str, **kwargs: Any) -> dict[str, Any]:
            raise TimeoutError("fundamentals unavailable")

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        monkeypatch.setattr(orchestrator, "technicals", technicals)
        monkeypatch.setattr(orchestrator, "fundamentals_snapshot", failed)
        for name in ("risk_metrics", "events_calendar", "stock_news", "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        result = await orchestrator.analyze_stock("ACME")

        assert result["verdict"]["tilt"] is None
        assert result["verdict"]["horizon_fit"]["mid_term"] == "ok"
        policy = result["diagnostics"]["pre_gate_policy"]
        assert policy["mid_term"] == "insufficient_data"  # was hold_or_reduce
        assert result["decision_card"]["action_now"] == "insufficient_data"
        balanced = result["decision_modes"]["balanced"]
        assert (balanced["suitability"], balanced["action"]) == ("needs_more_data", "insufficient_data")
        assert not [r for r in policy["rationale"] if "tilt=bearish" in r or "tilt=None" in r]
