"""Tests for the small-account decision card and its hard gates."""

import pytest

from stock_analysis.tools.analyze.decision_card import (
    _build_sizing,
    _compute_hard_gates,
    _resolve_action_now,
    build_decision_card,
)


def _empty_policy_action(mid_term: str = "buy") -> dict:
    return {
        "mid_term": mid_term,
        "long_term": "ok",
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
            dip_assessment={"dip_classification": {"type": "healthy_pullback"}},
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
            dip_assessment=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["earnings_blackout"] is True
        assert gates["any_blocking"] is True

    def test_earnings_blackout_does_not_fire_at_6_days(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 6}},
            data_quality={},
            dip_assessment=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["earnings_blackout"] is False

    def test_data_quality_critical_fires(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={"fundamentals_status": "missing"},
            dip_assessment=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["data_quality_critical"] is True
        assert gates["any_blocking"] is True

    def test_falling_knife_fires(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            dip_assessment={"dip_classification": {"type": "falling_knife"}},
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 50_000_000}},
        )
        assert gates["checks"]["falling_knife"] is True

    def test_missing_runway_only_when_unprofitable_and_burn_unavailable(self) -> None:
        # Unprofitable + burn unavailable → fires
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            dip_assessment=None,
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
            dip_assessment=None,
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
            dip_assessment=None,
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
            dip_assessment=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 500_000}},
        )
        assert gates["checks"]["weak_liquidity"] is True

    def test_weak_liquidity_does_not_fire_above_threshold(self) -> None:
        gates = _compute_hard_gates(
            events_data={},
            data_quality={},
            dip_assessment=None,
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 2_000_000}},
        )
        assert gates["checks"]["weak_liquidity"] is False

    def test_multiple_gates_all_listed(self) -> None:
        gates = _compute_hard_gates(
            events_data={"earnings": {"days_until": 3}},
            data_quality={"fundamentals_status": "missing"},
            dip_assessment={"dip_classification": {"type": "falling_knife"}},
            fundamentals_summary={},
            risk_data={"liquidity": {"avg_dollar_volume": 100_000}},
        )
        assert gates["any_blocking"] is True
        gate_ids = {b["id"] for b in gates["blocking"]}
        assert "earnings_blackout" in gate_ids
        assert "data_quality_critical" in gate_ids
        assert "falling_knife" in gate_ids
        assert "weak_liquidity" in gate_ids


class TestActionNowResolution:
    """Hard gates override bullish actions; non-bullish actions pass through."""

    def test_buy_downgraded_to_wait_for_data_when_gate_fires(self) -> None:
        action, rationale = _resolve_action_now(
            _empty_policy_action("buy"),
            {
                "any_blocking": True,
                "blocking": [{"id": "earnings_blackout", "reason": "..."}],
                "checks": {},
            },
        )
        assert action == "wait_for_data"
        assert any("hard_gates fired" in r for r in rationale)

    def test_starter_downgraded_to_wait_for_data_when_gate_fires(self) -> None:
        action, _ = _resolve_action_now(
            _empty_policy_action("speculative_small_position"),
            {
                "any_blocking": True,
                "blocking": [{"id": "falling_knife", "reason": "..."}],
                "checks": {},
            },
        )
        assert action == "wait_for_data"

    def test_hold_not_downgraded_by_gate(self) -> None:
        """Holding through a gate is fine; we only block new entries."""
        action, _ = _resolve_action_now(
            _empty_policy_action("hold"),
            {
                "any_blocking": True,
                "blocking": [{"id": "earnings_blackout", "reason": "..."}],
                "checks": {},
            },
        )
        assert action == "hold"

    def test_avoid_not_changed_by_gate(self) -> None:
        action, _ = _resolve_action_now(
            _empty_policy_action("avoid"),
            {"any_blocking": True, "blocking": [{"id": "x", "reason": "y"}], "checks": {}},
        )
        assert action == "avoid"

    def test_buy_passes_through_when_no_gate(self) -> None:
        action, _ = _resolve_action_now(
            _empty_policy_action("buy"),
            {"any_blocking": False, "blocking": [], "checks": {}},
        )
        assert action == "buy"


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


class TestDecisionCardSchema:
    """Top-level shape is stable and includes all expected fields."""

    @pytest.fixture
    def fixture(self) -> dict:
        return {
            "summary": {"current_price": 100.0},
            "verdict": {"confidence": "moderate"},
            "action_zones": _empty_action_zones(),
            "policy_action": _empty_policy_action("buy"),
            "fundamentals_summary": {
                "valuation": {"valuation_note": None},
                "burn_metrics": {"status": "not_applicable"},
            },
            "risk_data": {"liquidity": {"avg_dollar_volume": 50_000_000}},
            "events_data": {"earnings": {"days_until": 60, "next_date": "2026-07-15"}},
            "data_quality": {"fundamentals_status": "available"},
            "dip_assessment": {"dip_classification": {"type": "healthy_pullback"}},
            "dislocation_framework": {"action": {"add_only_if": ["fcf_turns_positive"]}},
        }

    def test_full_card_has_all_top_level_keys(self, fixture: dict) -> None:
        card = build_decision_card(**fixture)
        for key in (
            "action_now",
            "rationale",
            "hard_gates",
            "sizing",
            "entry",
            "exit",
            "conditions",
            "next_review",
            "confidence",
            "horizon_fit",
        ):
            assert key in card, f"missing key: {key}"

    def test_clean_inputs_yield_buy_action(self, fixture: dict) -> None:
        card = build_decision_card(**fixture)
        assert card["action_now"] == "buy"
        assert card["hard_gates"]["any_blocking"] is False

    def test_earnings_blackout_blocks_buy(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 2, "next_date": "2026-05-14"}}
        card = build_decision_card(**fixture)
        assert card["action_now"] == "wait_for_data"
        assert card["hard_gates"]["any_blocking"] is True
        # do_not_buy_if should contain the blocking reason
        assert any("earnings" in r.lower() for r in card["conditions"]["do_not_buy_if"])

    def test_dislocation_add_only_if_propagated(self, fixture: dict) -> None:
        card = build_decision_card(**fixture)
        assert "fcf_turns_positive" in card["conditions"]["add_only_if"]

    def test_next_review_uses_near_earnings_when_within_90_days(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 30, "next_date": "2026-06-11"}}
        card = build_decision_card(**fixture)
        assert card["next_review"]["date"] == "2026-06-11"
        assert "earnings" in card["next_review"]["reason"]

    def test_next_review_defaults_when_earnings_far_out(self, fixture: dict) -> None:
        fixture["events_data"] = {"earnings": {"days_until": 200, "next_date": "2026-11-30"}}
        card = build_decision_card(**fixture)
        # Defaults to 30-day re-evaluation reason
        assert "default" in card["next_review"]["reason"]

    def test_sizing_includes_fractional_shares(self, fixture: dict) -> None:
        card = build_decision_card(**fixture)
        assert "fractional_shares" in card["sizing"]
        assert card["sizing"]["fractional_shares"] is not None
