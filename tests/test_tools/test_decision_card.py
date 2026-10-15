"""Tests for the small-account decision card and its hard gates."""

from datetime import datetime

import pytest

from stock_analysis.tools.analyze.decision import NEW_MONEY_ACTIONS
from stock_analysis.tools.analyze.decision_card import (
    _build_sizing,
    _compute_hard_gates,
    build_conditions,
    build_decision_card,
)
from stock_analysis.tools.analyze.gates import GATES
from tests.support.decision_matrix import ET, KNIFE, decide_on


def _empty_policy_action(mid_term: str = "buy") -> dict:
    return {
        "mid_term": mid_term,
        "long_term": "accumulate",
        "rationale": [],
        "conditions_to_upgrade": [],
        "conditions_to_downgrade": [],
    }


def _empty_action_zones(current_price: float = 100.0) -> dict:
    return {
        "current_zone": "accumulate",
        "levels": {
            "strong_buy_below": 90.0,
            "accumulate_near": 100.0,
            "reduce_above": 130.0,
            "stop_loss": 85.0,
        },
        "basis": {"stop_loss": "current_minus_2.0atr"},
        "position_sizing_range": {
            "starter_pct": 3.0,
            "max_pct": 8.0,
            "dollars_for_account": {"min": 150.0, "max": 400.0, "portfolio_assumption": 5000.0},
            "shares_range": {"min": 1, "max": 4, "at_price": current_price},
        },
    }


class TestHardGates:
    """Each blocking condition fires the right gate; clean inputs fire nothing."""

    def test_clean_inputs_yield_no_blocking(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 60, "next_date": "2026-07-15"}},
            data_quality={"fundamentals_status": "available"},
            technicals_data=None,
            fundamentals_summary={
                "valuation": {"valuation_note": None},
                "burn_metrics": {"status": "not_applicable"},
            },
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["any_blocking"] is False
        assert gates["blocking"] == []
        assert all(v is False for v in gates["checks"].values())

    def test_earnings_blackout_fires_at_5_days(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 5}},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["earnings_blackout"] is True
        assert gates["any_blocking"] is True

    def test_earnings_blackout_does_not_fire_at_6_days(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 6}},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["earnings_blackout"] is False

    def test_data_quality_critical_fires(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={"fundamentals_status": "missing"},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["data_quality_critical"] is True
        assert gates["any_blocking"] is True

    def test_falling_knife_fires(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=KNIFE,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["falling_knife"] is True

    def test_missing_runway_only_when_unprofitable_and_burn_unavailable(self) -> None:
        # Unprofitable + burn unavailable → fires
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={
                "valuation": {"valuation_note": "pe_not_meaningful"},
                "burn_metrics": {"status": "unavailable"},
            },
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["missing_runway"] is True

        # Unprofitable but burn AVAILABLE → does not fire
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={
                "valuation": {"valuation_note": "pe_not_meaningful"},
                "burn_metrics": {"status": "available"},
            },
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["missing_runway"] is False

        # Profitable but burn unavailable → does not fire (not applicable)
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={
                "valuation": {"valuation_note": None},
                "burn_metrics": {"status": "unavailable"},
            },
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["missing_runway"] is False

    def test_weak_liquidity_fires_below_threshold(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 500_000}},
        )
        assert gates["checks"]["weak_liquidity"] is True

    def test_weak_liquidity_does_not_fire_above_threshold(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 2_000_000}},
        )
        assert gates["checks"]["weak_liquidity"] is False

    def test_multiple_gates_all_listed(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 3}},
            data_quality={"fundamentals_status": "missing"},
            technicals_data=KNIFE,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 100_000}},
        )
        assert gates["any_blocking"] is True
        gate_ids = {b["id"] for b in gates["blocking"]}
        assert "earnings_blackout" in gate_ids
        assert "data_quality_critical" in gate_ids
        assert "falling_knife" in gate_ids
        assert "weak_liquidity" in gate_ids

    def test_data_quality_critical_fires_on_missing_critical_field(self) -> None:
        """Even if fundamentals_status is OK, missing critical fields (e.g. current_price)
        should still block new buys."""
        gates = _compute_hard_gates(
            events_data={},
            data_quality={
                "fundamentals_status": "available",
                "missing_critical": ["current_price"],
            },
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["data_quality_critical"] is True
        reason = next(b["reason"] for b in gates["blocking"] if b["id"] == "data_quality_critical")
        assert "missing_critical=current_price" in reason

    def test_data_quality_critical_fires_on_critical_tool_failure(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={
                "fundamentals_status": "available",
                "tool_failures": [{"tool": "stock_summary", "error": "timeout"}],
            },
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["data_quality_critical"] is True
        reason = next(b["reason"] for b in gates["blocking"] if b["id"] == "data_quality_critical")
        assert "critical_tool_failed=stock_summary" in reason

    def test_data_quality_does_not_fire_on_noncritical_tool_failure(self) -> None:
        """Non-critical tool failures (e.g., options_signals, news) don't block buys."""
        gates = _compute_hard_gates(
            events_data={},
            data_quality={
                "fundamentals_status": "available",
                "tool_failures": [{"tool": "options_signals", "error": "x"}],
            },
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["data_quality_critical"] is False

    def test_liquidity_missing_fires_when_avg_dollar_volume_none(self) -> None:
        """Missing liquidity data is treated as blocking, distinct from 'too thin'."""
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": None}},
        )
        assert gates["checks"]["liquidity_missing"] is True
        assert gates["checks"]["weak_liquidity"] is False  # mutually exclusive

    def test_liquidity_missing_fires_when_no_liquidity_section(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            technicals_data=None,
            fundamentals_summary={},
            risk_data={},
        )
        assert gates["checks"]["liquidity_missing"] is True


class TestNewMoneyResolution:
    """decide() maps the candidate (D-2) and takes the most restrictive of it and every fired gate (D-5)."""

    @pytest.mark.parametrize(("mid_term", "gates", "action", "decided_by"), [
        ("buy", ["earnings_blackout"], "wait", "earnings_blackout"),
        ("speculative_small_position", ["falling_knife"], "avoid", "falling_knife"),
        ("hold", ["earnings_blackout"], "wait", None),  # a non-admitting candidate: the gate is disclosed, not binding
        ("hold_or_add", [], "starter", None),
        ("hold_or_add", ["earnings_blackout"], "wait", "earnings_blackout"),
        ("avoid", ["earnings_blackout"], "avoid", None),
        ("buy", [], "buy", None),
    ])
    def test_new_money(self, mid_term: str, gates: list[str], action: str, decided_by: str | None) -> None:
        decision = decide_on(_empty_policy_action(mid_term), [{"id": g, "reason": "..."} for g in gates])
        new_money = decision["new_money"]

        assert (new_money["action"], new_money["decided_by"]) == (action, decided_by)
        assert [b["id"] for b in new_money["blocked_by"]] == gates
        assert decision["if_held"]["add_allowed"] is (action in ("buy", "starter"))

    def test_the_vocabulary(self) -> None:
        assert NEW_MONEY_ACTIONS == ("buy", "starter", "wait", "wait_for_data", "insufficient_data", "watch", "avoid")


class TestSizing:
    """Fractional shares emitted alongside whole shares."""

    def test_fractional_computed_from_dollars_and_price(self) -> None:
        sizing = _build_sizing(
            _empty_action_zones(current_price=100.0),
            {"current_price": 100.0},
        )
        assert sizing["whole_shares"] == 1
        # $150 starter / $100 = 1.5 shares
        assert sizing["fractional_shares"] == 1.5
        assert sizing["at_price"] == 100.0

    def test_fractional_none_when_no_price(self) -> None:
        zones = _empty_action_zones(current_price=100.0)
        zones["position_sizing_range"]["shares_range"] = {"at_price": None}
        sizing = _build_sizing(zones, {"current_price": None})
        assert sizing["fractional_shares"] is None

    def test_fractional_none_when_no_dollars(self) -> None:
        zones = _empty_action_zones(current_price=100.0)
        zones["position_sizing_range"]["dollars_for_account"] = None
        sizing = _build_sizing(zones, {"current_price": 100.0})
        assert sizing["fractional_shares"] is None
        assert sizing["starter_dollars"] is None


def _card(fixture: dict) -> dict:
    """The card for a fixture: its gates from _compute_hard_gates (fundamentals and liquidity clean), its
    decision from decide(), its binding conditions from build_conditions (gates only: no candidate factors)."""
    hard_gates = _compute_hard_gates(
        events_data=fixture["events_data"],
        data_quality={"fundamentals_status": "available"},
        technicals_data={},
        fundamentals_summary={"valuation": {"valuation_note": None}, "burn_metrics": {"status": "not_applicable"}},
        risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
    )
    policy_action = fixture.pop("policy_action")
    decision = decide_on(policy_action, hard_gates["blocking"])
    conditions = build_conditions(decision=decision, candidate=policy_action, factors=[], inputs=None, downgrade=[],
                                  support=None)
    return build_decision_card(**fixture, hard_gates=hard_gates, decision=decision, conditions=conditions,
                               data_quality={"fundamentals_status": "available", "staleness_warnings": []}, why=[])


class TestDecisionCardSchema:
    """Top-level shape is stable and includes all expected fields."""

    @pytest.fixture
    def fixture(self) -> dict:
        return {
            "summary": {"current_price": 100.0},
            "verdict": {"confidence": "moderate"},
            "action_zones": _empty_action_zones(),
            "policy_action": _empty_policy_action("buy"),
            "events_data": {"earnings": {"days_until": 60, "next_date": "2026-07-15"}},
            "dislocation_framework": {"action": {"add_only_if": ["fcf_turns_positive"]}},
            "now": ET.localize(datetime(2026, 5, 16, 10, 0)),
        }

    def test_full_card_has_all_top_level_keys(self, fixture: dict) -> None:
        card = _card(fixture)
        for key in (
            "horizon",
            "new_money",
            "if_held",
            "long_term",
            "action_now",
            "hard_gates",
            "why",
            "exit_triggers",
            "data_flags",
            "sizing",
            "entry",
            "exit",
            "conditions",
            "next_review",
            "confidence",
        ):
            assert key in card, f"missing key: {key}"
        assert not {"rationale", "horizon_fit"} & set(card)
        assert card["long_term"]["new_money"]["conditions"] == {"buy_only_if": []}

    def test_clean_inputs_yield_buy_action(self, fixture: dict) -> None:
        card = _card(fixture)
        assert card["action_now"] == card["new_money"]["action"] == "buy"
        assert card["hard_gates"]["any_blocking"] is False

    def test_earnings_blackout_blocks_buy(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 2, "next_date": "2026-05-14"}}
        card = _card(fixture)
        assert card["action_now"] == card["new_money"]["action"] == "wait"
        assert card["hard_gates"]["any_blocking"] is True
        # do_not_buy_if should contain the blocking reason
        assert any("earnings" in r.lower() for r in card["conditions"]["do_not_buy_if"])

    def test_a_clean_buy_lists_no_buy_or_add_condition(self, fixture: dict) -> None:
        """OC-M6: the plan §7e buy row: nothing binds, so no buy or add condition (the dislocation framework's
        lists are no longer copied). M."""
        card = _card(fixture)
        assert (card["conditions"]["buy_only_if"], card["conditions"]["add_only_if"]) == ([], [])

    def test_next_review_uses_near_earnings_when_within_90_days(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 30, "next_date": "2026-06-11"}}
        card = _card(fixture)
        assert card["next_review"]["date"] == "2026-06-11"
        assert "earnings" in card["next_review"]["reason"]

    def test_next_review_defaults_when_earnings_far_out(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 200, "next_date": "2026-11-30"}}
        card = _card(fixture)
        # Defaults to 30-day re-evaluation reason
        assert "default" in card["next_review"]["reason"]

    def test_sizing_includes_fractional_shares(self, fixture: dict) -> None:
        card = _card(fixture)
        assert "fractional_shares" in card["sizing"]
        assert card["sizing"]["fractional_shares"] is not None

    def test_a_binding_gate_lists_its_clears_when(self, fixture: dict) -> None:
        """OC-M6: a blackout that tightens buy to wait binds at both horizons: its clears_when is the buy
        condition, and the add condition while if_held is hold (plan §7e). M."""
        fixture["events_data"] = {"earnings": {"days_until": 2, "next_date": "2026-05-14"}}
        card = _card(fixture)
        clears = GATES["earnings_blackout"].clears_when
        assert (card["conditions"]["buy_only_if"], card["conditions"]["add_only_if"]) == ([clears], [clears])
        assert card["long_term"]["new_money"]["conditions"] == {"buy_only_if": [clears]}

    def test_both_buy_only_if_and_add_only_if_present_in_schema(self, fixture: dict) -> None:
        card = _card(fixture)
        assert "buy_only_if" in card["conditions"]
        assert "add_only_if" in card["conditions"]
