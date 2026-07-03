"""Tests for the trade plan builder (entries, targets, sizing, time stop)."""

from datetime import date, datetime

import pytz

from stock_analysis.tools.trade_setup.plan import add_trading_days, build_plan

ET = pytz.timezone("America/New_York")
NOW = ET.localize(datetime(2026, 3, 10, 18, 0))  # Tuesday evening


def make_setup(**overrides):
    base = {
        "type": "pullback_in_uptrend",
        "quality": "B",
        "thesis": [], "invalidation": [],
        "trigger_price": 101.0,
        "trigger_satisfied": False,
        "trigger_condition": "price reclaims prior-day high 101.00",
        "stop_price": 97.0,
        "stop_basis": "swing_low",
        "target_primary": {"price": 105.0, "basis": "prior_20d_high"},
    }
    base.update(overrides)
    return base


def test_add_trading_days_skips_weekends() -> None:
    assert add_trading_days(date(2026, 3, 10), 10) == date(2026, 3, 24)
    assert add_trading_days(date(2026, 3, 13), 1) == date(2026, 3, 16)  # Fri -> Mon


class TestTargets:
    def test_structural_primary_plus_2r(self) -> None:
        plan = build_plan(make_setup(), action="enter_on_trigger", session="closed",
                          account_size=10_000.0, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        # risk/share = 4.0; primary 105 -> 1.0R; 2R = 109
        assert plan["targets"] == [
            {"price": 105.0, "r_multiple": 1.0, "basis": "prior_20d_high"},
            {"price": 109.0, "r_multiple": 2.0, "basis": "r_multiple"},
        ]
        assert plan["reward_risk"] == 1.0
        prices = [t["price"] for t in plan["targets"]]
        assert prices == sorted(prices) and len(set(prices)) == len(prices)

    def test_1r_fallback_without_structural_target(self) -> None:
        plan = build_plan(make_setup(target_primary=None), action="enter_on_trigger",
                          session="closed", account_size=None, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        assert plan["targets"][0] == {"price": 105.0, "r_multiple": 1.0, "basis": "r_multiple"}
        assert plan["targets"][1] == {"price": 109.0, "r_multiple": 2.0, "basis": "r_multiple"}

    def test_structural_target_beyond_2r_sorts_after(self) -> None:
        plan = build_plan(make_setup(target_primary={"price": 112.0, "basis": "prior_20d_high"}),
                          action="enter_on_trigger", session="closed", account_size=None,
                          risk_per_trade_pct=1.0, max_position_pct=10.0, now=NOW)
        assert [t["price"] for t in plan["targets"]] == [109.0, 112.0]
        assert plan["reward_risk"] == 2.0


class TestSizing:
    def test_r_based_sizing_with_account(self) -> None:
        plan = build_plan(make_setup(), action="enter_on_trigger", session="closed",
                          account_size=10_000.0, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        # risk budget $100, risk/share $4 -> 25 shares; cap 10% = $1000/101 = 9.9 shares
        assert plan["fractional_shares"] == 9.901
        assert plan["shares"] == 9
        assert plan["position_dollars"] == round(9.901 * 101.0, 2)
        assert plan["max_loss_dollars"] == round(9.901 * 4.0, 2)
        assert plan["position_pct"] == 10.0

    def test_null_sizing_without_account(self) -> None:
        plan = build_plan(make_setup(), action="enter_on_trigger", session="closed",
                          account_size=None, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        assert plan["max_loss_dollars"] is None
        assert plan["shares"] is None
        assert plan["fractional_shares"] is None
        assert plan["position_dollars"] is None
        # percent outputs remain populated
        assert plan["position_pct"] == 10.0
        assert plan["stop"]["distance_pct"] == round(4.0 / 101.0, 4)


class TestEntry:
    def test_trade_now_enters_at_market(self) -> None:
        plan = build_plan(make_setup(trigger_satisfied=True), action="trade_now",
                          session="regular", account_size=None, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        assert plan["entry"]["type"] == "market"
        assert plan["entry"]["valid"] == "good_till_time_stop"

    def test_off_session_trigger_valid_next_session(self) -> None:
        plan = build_plan(make_setup(), action="enter_on_trigger", session="closed",
                          account_size=None, risk_per_trade_pct=1.0,
                          max_position_pct=10.0, now=NOW)
        assert plan["entry"]["type"] == "buy_stop"
        assert plan["entry"]["valid"] == "next_session"
        assert plan["entry"]["trigger_price"] == 101.0

    def test_time_stop_uses_setup_days(self) -> None:
        plan = build_plan(make_setup(type="oversold_mean_reversion"),
                          action="enter_on_trigger", session="closed", account_size=None,
                          risk_per_trade_pct=1.0, max_position_pct=10.0, now=NOW)
        assert plan["time_stop"] == {"trading_days": 5, "date": "2026-03-17"}
