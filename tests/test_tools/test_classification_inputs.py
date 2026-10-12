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

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

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
        healthy = {"net_cash_positive": True, "debt_to_equity": 0.5}
        signals = classify_dislocation_opportunity(VERDICT, ZONES, fundamentals(healthy, NO_BURN), DEEP_DIP)
        assert (signals["balance_status"], signals["is_dislocation_candidate"]) == ("safe", True)

    def test_the_framework_reads_net_debt_as_watch_not_broken(self) -> None:
        framework = build_dislocation_framework(
            VERDICT, ZONES, fundamentals(HEALTHY_NET_DEBT, NO_BURN), DEEP_DIP,
            {"thesis_checkpoints": {}}, {}, {"mid_term": "hold", "long_term": "hold"},
        )
        assert framework["balance_sheet_safety"]["status"] == "watch"
        assert framework["thesis_integrity"]["status"] == "watch"
        assert framework["mismatch_verdict"]["status"] == "unclear"


class TestR92Runway:
    @pytest.mark.parametrize(
        "health",
        [{"net_cash_positive": True, "debt_to_equity": 0.5}, HEALTHY_NET_DEBT,
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

    def test_a_safe_balance_sheet_has_no_reasons(self) -> None:
        balance = _build_balance_sheet_safety({"net_cash_positive": True, "debt_to_equity": 0.5}, NO_BURN)
        assert (balance["status"], balance["reasons"]) == ("safe", [])


def policy(verdict: dict[str, Any], *, zones: dict[str, Any] | None = None,
           dip: dict[str, Any] | None = None, health: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_policy_action(
        verdict=verdict,
        action_zones=zones or {"current_zone": "hold_neutral",
                               "valuation_assessment": {"gate": "neutral", "is_unprofitable": False}},
        decomposed=verdict.get("decomposed"),
        dip_assessment=dip,
        fundamentals_summary=fundamentals(health or {"net_cash_positive": True, "debt_to_equity": 0.5}, NO_BURN),
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
    @pytest.mark.parametrize("tilt", [*MISSING_TILT, {"tilt": "bullish"}], ids=["absent", "null", "bullish"])
    def test_the_falling_knife_gate_still_wins(self, tilt: dict[str, Any]) -> None:
        knife = {"dip_classification": {"type": "falling_knife"}}
        action = policy(with_fit("ok", "ok", **tilt), dip=knife)
        assert action["mid_term"] == "wait_for_entry"
        assert "dip_type=falling_knife" in action["rationale"]


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
        assert result["policy_action"]["mid_term"] == "insufficient_data"  # was hold_or_reduce
        assert result["decision_card"]["action_now"] == "insufficient_data"
        assert not [r for r in result["policy_action"]["rationale"] if "tilt=bearish" in r or "tilt=None" in r]
