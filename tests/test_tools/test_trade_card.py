"""Invariant tests for the trade-setup card: precedence, blockers, plan presence."""

from datetime import datetime
from typing import Any

import pytest
import pytz

from stock_analysis.tools.trade_setup.card import build_trade_setup_card
from tests.test_tools.test_setup_detection import make_features, make_technicals

ET = pytz.timezone("America/New_York")
NOW_REGULAR = ET.localize(datetime(2026, 3, 10, 10, 30))
NOW_EVENING = ET.localize(datetime(2026, 3, 10, 18, 30))

FRESH = {"as_of": "2026-03-10T14:25:00+00:00", "basis": "bar_timestamp",
         "session": "regular", "quote_age_seconds": 300, "stale": False}
EOD_FRESH = {"as_of": "2026-03-10", "basis": "bar_timestamp",
             "session": "after_hours", "quote_age_seconds": None, "stale": False}
STALE = {"as_of": "2026-03-10T13:00:00+00:00", "basis": "bar_timestamp",
         "session": "regular", "quote_age_seconds": 5400, "stale": True}
UNVERIFIABLE = {"as_of": None, "basis": "unverifiable", "session": "regular",
                "quote_age_seconds": None, "stale": True}


def build_card(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "symbol": "TEST",
        "summary_data": {"currency": "USD", "name": "Test Co"},
        "technicals_data": make_technicals(),
        "risk_data": {"liquidity": {"avg_dollar_volume": 50_000_000}},
        "events_data": {"earnings": {"days_until": 40, "next_date": "2026-04-19"}},
        "freshness": FRESH,
        "features": make_features(),
        "actionable_price": 100.0,
        "session": "regular",
        "now": NOW_REGULAR,
        "account_size": 10_000.0,
        "risk_per_trade_pct": 1.0,
        "max_position_pct": 10.0,
        "tool_failures": [],
    }
    kwargs.update(overrides)
    return build_trade_setup_card(**kwargs)


def blocker_ids(card: dict[str, Any]) -> set[str]:
    return {b["id"] for b in card["blockers"]}


class TestPrecedence:
    def test_armed_pullback_is_enter_on_trigger(self) -> None:
        card = build_card()
        assert card["action"] == "enter_on_trigger"
        assert card["setup"]["type"] == "pullback_in_uptrend"
        assert card["plan"] is not None

    def test_satisfied_trigger_fresh_regular_is_trade_now(self) -> None:
        card = build_card(actionable_price=101.5)  # above 101 trigger
        assert card["action"] == "trade_now"
        assert card["plan"]["entry"]["type"] == "market"
        # sizing anchored to actionable price (101.5), not the 101.0 trigger level
        assert card["plan"]["stop"]["distance_pct"] == round((101.5 - 97.0) / 101.5, 4)

    def test_stale_data_wins_over_everything(self) -> None:
        card = build_card(freshness=STALE, actionable_price=101.5)
        assert card["action"] == "wait_for_data"
        assert "stale_data" in blocker_ids(card)
        assert card["plan"] is None and card["setup"] is None

    def test_unverifiable_freshness_is_wait_for_data(self) -> None:
        card = build_card(freshness=UNVERIFIABLE)
        assert card["action"] == "wait_for_data"
        assert "freshness_unverifiable" in blocker_ids(card)

    def test_critical_tool_failure_is_wait_for_data(self) -> None:
        card = build_card(tool_failures=[{"tool": "technicals", "error": "boom"}])
        assert card["action"] == "wait_for_data"
        assert "data_quality_critical" in blocker_ids(card)

    def test_missing_features_is_wait_for_data(self) -> None:
        card = build_card(features=None)
        assert card["action"] == "wait_for_data"
        assert "data_quality_critical" in blocker_ids(card)

    def test_falling_knife_is_avoid(self) -> None:
        t = make_technicals()
        t["moving_averages"]["rules"]["death_cross"] = {"triggered": True}
        t["moving_averages"]["rules"]["above_sma200"] = {"triggered": False}
        t["moving_averages"]["sma_200_slope_pct_per_day"] = -0.002
        t["returns"]["return_3m"] = -0.35
        card = build_card(technicals_data=t)
        assert card["action"] == "avoid"
        assert "falling_knife" in blocker_ids(card)
        assert card["setup"] is None and card["plan"] is None

    def test_weak_liquidity_is_avoid(self) -> None:
        card = build_card(risk_data={"liquidity": {"avg_dollar_volume": 400_000}})
        assert card["action"] == "avoid"
        assert "weak_liquidity" in blocker_ids(card)

    def test_boring_chart_is_no_setup(self) -> None:
        t = make_technicals(rsi={"value": 60.0, "bullish_divergence": False})
        f = make_features(high_20d_prior=120.0)
        card = build_card(technicals_data=t, features=f)
        assert card["action"] == "no_setup"
        assert card["setup"] is None and card["plan"] is None

    def test_earnings_blackout_is_watch_with_setup(self) -> None:
        card = build_card(events_data={"earnings": {"days_until": 3, "next_date": "2026-03-13"}})
        assert card["action"] == "watch"
        assert "earnings_blackout" in blocker_ids(card)
        assert card["setup"] is not None  # blackout watch carries the setup
        assert card["plan"] is None
        assert card["next_review"]["date"] == "2026-03-13"
        assert card["confidence"] == "low"

    def test_events_calendar_failure_caps_at_watch_with_setup(self) -> None:
        card = build_card(
            events_data={},
            tool_failures=[{"tool": "events_calendar", "error": "fetch_error"}],
        )
        assert card["action"] == "watch"
        assert "earnings_unverifiable" in blocker_ids(card)
        assert card["setup"] is not None
        assert card["plan"] is None


class TestSessionInvariant:
    @pytest.mark.parametrize("session,now", [
        ("pre_market", ET.localize(datetime(2026, 3, 10, 8, 0))),
        ("after_hours", NOW_EVENING),
        ("closed", ET.localize(datetime(2026, 3, 8, 12, 0))),
    ])
    def test_never_trade_now_outside_regular(self, session: str, now: datetime) -> None:
        fresh = {**EOD_FRESH, "session": session}
        card = build_card(freshness=fresh, session=session, now=now,
                          actionable_price=101.5)  # trigger satisfied
        assert card["action"] == "enter_on_trigger"
        assert "market_closed" in blocker_ids(card)
        assert card["plan"]["entry"]["valid"] == "next_session"


class TestSchemaInvariants:
    def test_plan_iff_actionable(self) -> None:
        actionable = {"trade_now", "enter_on_trigger"}
        cases = [
            build_card(),
            build_card(actionable_price=101.5),
            build_card(freshness=STALE),
            build_card(events_data={"earnings": {"days_until": 2, "next_date": "2026-03-12"}}),
            build_card(risk_data={"liquidity": {}}),
        ]
        for card in cases:
            assert (card["plan"] is not None) == (card["action"] in actionable)

    def test_required_keys_and_enum(self) -> None:
        card = build_card()
        for key in ("symbol", "freshness", "action", "setup", "plan", "blockers",
                    "event_risk", "notes", "confidence", "next_review"):
            assert key in card
        assert card["action"] in {"trade_now", "enter_on_trigger", "watch",
                                  "no_setup", "avoid", "wait_for_data"}
        assert card["event_risk"]["expected_move_pct"] is None

    def test_pdt_note_for_small_account(self) -> None:
        card = build_card(account_size=5_000.0)
        assert any("day trade" in n for n in card["notes"])

    def test_pdt_note_when_account_omitted(self) -> None:
        card = build_card(account_size=None)
        assert any("day trade" in n for n in card["notes"])
        assert card["plan"]["shares"] is None

    def test_confidence_low_when_not_actionable(self) -> None:
        card = build_card(freshness=STALE)
        assert card["confidence"] == "low"


def test_v_recovery_is_not_a_falling_knife() -> None:
    # HOOD 2026-07-06 regression: recovery chart must not be gated as a knife.
    t = make_technicals(rsi={"value": 68.0, "bullish_divergence": False})
    t["moving_averages"]["rules"]["death_cross"] = {"triggered": True}
    t["moving_averages"]["rules"]["golden_cross"] = {"triggered": False}
    t["moving_averages"]["sma_200_slope_pct_per_day"] = -0.000514
    t["price_position"]["days_since_52w_high"] = 186
    t["returns"]["return_3m"] = 0.66
    f = make_features(high_20d_prior=120.0)  # far from high: no setup expected
    card = build_card(technicals_data=t, features=f)
    assert card["action"] == "no_setup"
    assert "falling_knife" not in blocker_ids(card)
