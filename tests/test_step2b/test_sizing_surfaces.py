"""Derived sizes share one budget and keep the planning-stop assumption visible."""

from fractions import Fraction

import pytest

from stock_analysis.tools.analyze.executive_summary import horizon_segments, size_text
from stock_analysis.tools.analyze.investor_profile import build_decision_modes
from stock_analysis.utils.position_sizing import calculate_position_sizing


def _public(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def sizing(**overrides):
    args = {"entry_price": 100.0, "stop_price": 80.0, "risk_per_trade_pct": 0.25,
            "max_position_pct": 10.0, "regime_max_pct": 8.0, "regime_starter_pct": 3.0,
            "preset_starter_pct": 2.5, "account_size": 10_000.0, "risk_origin": "caller",
            "max_position_origin": "profile"}
    return _public(calculate_position_sizing(**(args | overrides)))["sizing"]


def modes(canonical, action="buy"):
    return build_decision_modes(
        summary={"current_price": 100.0},
        verdict={"tilt": "bullish", "decomposed": {"risk": "moderate"},
                 "horizon_fit": {"mid_term": "ok", "long_term": "ok"}},
        action_zones={}, dip_assessment=None, decision_context={}, news_summary={},
        fundamentals_summary={}, events_summary={}, account_size=10_000.0,
        decision={"new_money": {"action": action}, "long_term": {"new_money": {"action": "buy"}}},
        conditions={"mid_term": {"add_only_if": []}, "long_term": {"buy_only_if": []}},
        why=[], canonical_sizing=canonical,
    )


def test_modes_keep_the_caller_budget_and_whole_share_amounts():
    canonical = sizing()
    result = modes(canonical)
    assert result["core"]["action"] == "buy"
    assert result["core"]["starter_position"] is None
    assert result["core"]["max_position"] is None
    for name in ("balanced", "speculative"):
        scenario = result[name]["starter_position"]
        assert scenario["horizon"] == "mid_term"
        assert scenario["pct"] <= canonical["starter"]["allocation_cap_pct"]
        assert (scenario["shares"], scenario["dollars"], scenario["planned_loss_at_stop"]) == (1, 100, 20)
        assert scenario["effective_risk_pct"] == 0.2
    assert result["speculative"]["starter_position"]["pct"] == 1.0
    assert result["speculative"]["capital_at_risk"] is None
    assert result["balanced"]["max_position"]["dollars"] == 100


def test_smaller_canonical_starter_never_increases_in_an_alternative_mode():
    result = modes(sizing(max_position_pct=0.5, max_position_origin="caller"))
    for name in ("balanced", "speculative"):
        scenario = result[name]["starter_position"]
        assert scenario["pct"] == 0.5
        assert scenario["quantity_status"] == "below_one_whole_share"
        assert (scenario["shares"], scenario["dollars"], scenario["planned_loss_at_stop"]) == (0, 0, 0)


def test_modes_cannot_invent_a_size_when_canonical_inputs_are_unavailable():
    for mode in modes(None).values():
        assert mode["starter_position"] is None and mode["max_position"] is None


def test_starter_and_blocked_visibility_is_kept_after_mode_capping():
    for mode in modes(sizing(), "starter").values():
        assert mode["max_position"] is None
    for mode in modes(sizing(), "wait").values():
        assert mode["starter_position"] is None and mode["max_position"] is None


def test_summary_names_the_whole_share_position_conditional_loss_and_limit_origins():
    phrase = size_text("buy", sizing())
    assert "Position: $100 (1% of the account); estimated loss: 0.2% of the account if sold at $80" in phrase
    assert "risk budget: 0.25% (caller)" in phrase
    assert "position cap: 10% (profile)" in phrase
    assert "Full" in phrase and "Starter" in phrase
    assert "None" not in phrase


def test_summary_percentage_only_never_invents_dollars_or_shares():
    phrase = size_text("buy", sizing(account_size=None))
    assert "allocation ceiling: 1.25%" in phrase
    assert "estimated loss ceiling: 0.25%" in phrase
    assert "account size not provided" in phrase
    assert "Position:" not in phrase and "whole shares" not in phrase


def test_summary_mixed_affordability_is_per_scenario():
    phrase = size_text("buy", sizing(account_size=1000.0, entry_price=50.0, stop_price=48.0,
                                     risk_per_trade_pct=1.0))
    assert "Starter: no whole-share position fits this budget" in phrase
    assert "Full — Position: $50 (5% of the account); estimated loss: 0.2%" in phrase


@pytest.mark.parametrize(("action", "reason", "expected"), [
    ("buy", "stop_price_missing", "planning stop"),
    ("starter", "atr_invalid", "ATR"),
])
def test_missing_size_uses_the_status_reason_even_when_research_says_buy(action, reason, expected):
    phrase = size_text(action, None, {"status": "not_allowed" if action == "wait" else "unavailable", "reason": reason})
    assert phrase.startswith("No size estimate:")
    assert expected in phrase
    assert "None" not in phrase


def test_summary_does_not_round_tiny_positive_limits_to_zero():
    phrase = size_text("buy", sizing(account_size=None, risk_per_trade_pct=0.00000001))
    assert "5e-8%" in phrase or "0.00000005%" in phrase
    assert "risk budget: 1e-8%" in phrase or "risk budget: 0.00000001%" in phrase
    assert "ceiling: 0%" not in phrase


def test_holder_and_long_term_segments_never_receive_mid_term_sizes():
    new_money = {"action": "buy", "blocked_by": [], "earliest_reassessment": None}
    card = {"new_money": new_money, "if_held": {"action": "hold"},
            "long_term": {"new_money": new_money}, "sizing": sizing()}
    segments = horizon_segments(card)
    assert "Position:" in segments[0]
    assert "Position:" not in segments[1] and "Position:" not in segments[2]


def test_summary_rounds_non_round_account_prose_without_changing_the_numbers():
    import copy
    import re

    canonical = sizing(account_size=12345.67, risk_per_trade_pct=1.0)
    before = copy.deepcopy(canonical)
    phrase = size_text("buy", canonical)
    assert "Position: $300 (~2.43% of the account); estimated loss: ~0.49%" in phrase
    assert "Position: $600 (~4.86% of the account); estimated loss: ~0.97%" in phrase
    assert not re.search(r"\d\.\d{6}", phrase)
    assert canonical == before


def test_summary_rounds_every_percentage_ceiling_down():
    import re
    from decimal import Decimal

    canonical = sizing(account_size=None, risk_per_trade_pct=0.2599, max_position_pct=1.2999)
    phrase = size_text("buy", canonical)
    displayed = re.findall(r"(?:allocation|loss) ceiling: ~?([\d.e+-]+)%", phrase)
    expected = [canonical[scenario][field] for scenario in ("starter", "full")
                for field in ("allocation_cap_pct", "ceiling_risk_pct")]
    assert len(displayed) == 4
    assert all(Decimal(shown) <= Decimal(str(exact)) for shown, exact in zip(displayed, expected, strict=True))
    assert "allocation ceiling: ~1.29%" in phrase
    assert "estimated loss ceiling: ~0.25%" in phrase
    assert "risk budget: ~0.25%" in phrase and "position cap: ~1.29%" in phrase


def test_tiny_positive_loss_and_subcent_position_prose_never_look_like_zero():
    import re
    from decimal import Decimal

    canonical = sizing(entry_price=0.000123456, stop_price=0.0001, account_size=0.01,
                       risk_per_trade_pct=10.0)
    phrase = size_text("buy", canonical)
    assert not re.search(r"\d\.\d{6}", phrase)
    for shown in re.findall(r"Position: ~?\$([\d.e+-]+)", phrase):
        assert Decimal(shown) > 0
    assert "Position: ~$2.5e-4" in phrase
    tiny = size_text("buy", sizing(account_size=10_000_000.0, entry_price=100.0, stop_price=99.999))
    assert "estimated loss: 2.5e-5%" in tiny
    assert "estimated loss: 0%" not in tiny


def test_blocked_summary_does_not_repeat_the_sizing_status_reason():
    new_money = {"action": "wait", "blocked_by": [{"id": "earnings_blackout"}],
                 "earliest_reassessment": None}
    card = {"new_money": new_money, "if_held": {"action": "hold"}, "long_term": {"new_money": new_money},
            "sizing": None, "sizing_status": {"status": "not_allowed", "reason":
                "new_money is wait; blocked by: earnings_blackout"}}
    assert horizon_segments(card)[0] == "Decision (mid-term, new money): wait; blocked by earnings_blackout."
    assert card["sizing_status"]["reason"] == "new_money is wait; blocked by: earnings_blackout"
