"""Entry bands and stop disclosures do not override the research decision."""

import math

import pytest

from stock_analysis.tools.analyze.action_zones import build_action_zones
from stock_analysis.tools.analyze.decision_card import _build_entry


@pytest.mark.parametrize(("low", "high", "reason"), [
    (None, 101.0, "missing"), (70.0, None, "missing"),
    (0.0, 101.0, "invalid"), (-1.0, 101.0, "invalid"),
    (70.0, math.inf, "invalid"), (math.nan, 101.0, "invalid"),
    ("70", 101.0, "invalid"), (True, 101.0, "invalid"),
    (115.5, 101.0, "inverted"),
])
def test_unusable_band_loses_both_bounds_but_keeps_reference_price(low, high, reason):
    zones = {"levels": {"strong_buy_below": low, "accumulate_near": high}}
    result = _build_entry(zones, {"current_price": 100.0}, "buy")
    assert result == {
        "current_price": 100.0, "buy_zone_low": None, "buy_zone_high": None,
        "entry_band_unavailable": {"reason": reason},
    }
    assert zones["levels"]["strong_buy_below"] is low
    assert zones["levels"]["accumulate_near"] is high


@pytest.mark.parametrize("action", ["buy", "starter"])
def test_valid_entry_band_keeps_both_bounds(action):
    result = _build_entry({"levels": {"strong_buy_below": 90.0, "accumulate_near": 100.0}},
                          {"current_price": 100.0}, action)
    assert result == {
        "current_price": 100.0, "buy_zone_low": 90.0, "buy_zone_high": 100.0,
        "entry_band_unavailable": None,
    }


def test_blocked_entry_stays_null_even_when_its_band_is_invalid():
    assert _build_entry({"levels": {}}, {"current_price": 100.0}, "wait") is None


def zones(*, price=99.0, atr=2.0, atr_pct=2.0 / 99.0, regime="high"):
    return build_action_zones(
        price,
        {"moving_averages": {"sma_50": 110.0, "sma_200": 101.0},
         "price_position": {"week_52_low": 70.0, "week_52_high": 140.0}},
        {"atr": {"value": atr, "as_pct_of_price": atr_pct}}, {},
        risk_regime={"classification": regime},
    )


def test_stop_warning_compares_selected_distance_before_display_rounding():
    result = zones()
    assert result["levels"]["stop_loss"] == 95.0
    assert result["distance_to_levels"]["stop_loss"] == -0.0404
    assert "stop_too_tight" not in result["zone_warnings"]


def test_actual_stop_rounding_can_still_make_stop_too_tight():
    result = zones(price=100.0, atr=2.001, atr_pct=0.02001)
    assert result["levels"]["stop_loss"] == 96.0
    assert "stop_too_tight" in result["zone_warnings"]


def test_missing_atr_labels_only_the_price_band_as_a_two_percent_fallback():
    result = zones(price=100.0, atr=None, atr_pct=None)
    assert result["levels"]["strong_buy_below"] == 71.4
    assert result["levels"]["reduce_above"] == 137.2
    assert result["levels"]["stop_loss"] is None
    assert result["basis"]["strong_buy_below"] == "52w_low_plus_2pct_fallback"
    assert result["basis"]["reduce_above"] == "52w_high_minus_2pct_fallback"
    assert result["basis"]["stop_loss"] is None
    assert result["stop_calculation"] is None


def test_missing_atr_percentage_does_not_invent_an_atr_basis_for_the_band():
    result = zones(price=100.0, atr=2.0, atr_pct=None)
    assert result["levels"]["stop_loss"] == 96.0
    assert result["basis"]["stop_loss"] == "current_minus_2.0atr"
    assert result["basis"]["strong_buy_below"] == "52w_low_plus_2pct_fallback"
    assert result["basis"]["reduce_above"] == "52w_high_minus_2pct_fallback"


def canonical_sizing():
    return {
        "sizing_status": {"status": "available", "reason": None},
        "sizing": {
            "assumptions": {"entry_price": 100.0, "stop_price": 80.0, "stop_distance_fraction": 0.2,
                            "account_size": 10000.0},
            "limits": {"risk_per_trade_pct": 0.25, "max_position_pct": 10.0},
            "starter": {"allocation_cap_pct": 1.25, "position_dollars": 100.0,
                        "planned_loss_at_stop": 20.0, "whole_shares": 1},
            "full": {"allocation_cap_pct": 1.25, "position_dollars": 100.0,
                     "planned_loss_at_stop": 20.0, "whole_shares": 1},
        },
    }


@pytest.mark.parametrize("action", ["buy", "starter"])
def test_card_aliases_only_read_the_canonical_whole_share_scenarios(action):
    from stock_analysis.tools.analyze.decision_card import adapt_card_sizing

    original = canonical_sizing()
    result = adapt_card_sizing(original, {"action": action, "blocked_by": []})
    sizing = result["sizing"]
    assert result["sizing_status"] == original["sizing_status"]
    assert (sizing["starter_pct"], sizing["starter_dollars"], sizing["whole_shares"], sizing["at_price"]) == (
        1.25, 100.0, 1, 100.0)
    assert sizing["fractional_shares"] is None
    if action == "buy":
        assert sizing["full"] == original["sizing"]["full"]
        assert (sizing["full_pct_cap"], sizing["full_dollars"]) == (1.25, 100.0)
    else:
        assert sizing["full"] is sizing["full_pct_cap"] is sizing["full_dollars"] is None
        assert original["sizing"]["full"] is not None


def test_blocked_sizing_status_takes_precedence_over_missing_inputs():
    from stock_analysis.tools.analyze.decision_card import adapt_card_sizing

    unavailable = {"sizing": None, "sizing_status": {"status": "unavailable", "reason": "stop_price_missing"}}
    assert adapt_card_sizing(unavailable, {"action": "wait", "blocked_by": [{"id": "earnings_blackout"}]}) == {
        "sizing": None,
        "sizing_status": {"status": "not_allowed", "reason": "new_money is wait; blocked by: earnings_blackout"},
    }
    assert adapt_card_sizing(unavailable, {"action": "buy", "blocked_by": []}) == unavailable


def test_zone_aliases_share_the_card_quantity_basis():
    from stock_analysis.tools.analyze.action_zones import build_position_sizing_range

    sizing = canonical_sizing()["sizing"]
    result = build_position_sizing_range(sizing, "medium")
    assert result["dollars_for_account"] == {"min": 100.0, "max": 100.0, "portfolio_assumption": 10000.0}
    assert result["shares_range"] == {"min": 1, "max": 1, "at_price": 100.0}
    assert result["suggested_pct_range"] == [1.25, 1.25]
    assert result["stop_implied_max"]["dollars_for_account"] == 100.0
    assert result["stop_implied_max"]["pct"] == 1.25
    assert result["stop_implied_max"]["stop_distance_pct"] == 20.0


def test_zone_sizing_stays_null_when_the_canonical_full_size_is_absent():
    from stock_analysis.tools.analyze.action_zones import build_position_sizing_range

    assert build_position_sizing_range(None, "medium") is None
    sizing = canonical_sizing()["sizing"]
    sizing["full"] = None
    assert build_position_sizing_range(sizing, "medium") is None


def build_card(*, sizing_result=None):
    from datetime import datetime

    from stock_analysis.tools.analyze.decision_card import build_decision_card

    return build_decision_card(
        summary={"current_price": 100.0}, verdict={}, action_zones={"levels": {}},
        events_data={}, dislocation_framework=None, now=datetime(2026, 10, 19),
        hard_gates={}, decision={"new_money": {"action": "buy", "blocked_by": []},
                                  "long_term": {"new_money": {"action": "buy"}}},
        conditions={"mid_term": {}, "long_term": {}}, data_quality={}, why=[],
        sizing_result=sizing_result,
    )


def test_card_returns_the_precomputed_sizing_and_status_without_rebuilding():
    canonical = canonical_sizing()
    card = build_card(sizing_result=canonical)
    assert card["sizing"] is canonical["sizing"]
    assert card["sizing_status"] is canonical["sizing_status"]


def test_missing_precomputed_size_never_falls_back_to_an_independent_estimate():
    card = build_card()
    assert card["new_money"]["action"] == "buy"
    assert card["sizing"] is None
    assert card["sizing_status"] == {"status": "unavailable", "reason": "sizing_inputs_unavailable"}
