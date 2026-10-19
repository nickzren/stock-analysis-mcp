"""The scan promotes with caller limits and whole-share amount consistency."""

import json

import pytest

from tests.support.sizing_scan import build_scan_case


async def test_scan_promotes_real_card_and_forwards_sizing_limits():
    case = await build_scan_case()
    scan, card = case["scan"], case["promoted_card"]
    assert scan["symbols_scanned"] == scan["full_cards"] == 1
    assert scan["errors"] == []
    assert card["symbol"] == scan["rows"][0]["symbol"] == "EXMP"
    assert scan["rows"][0]["source"] == "card"
    assert card["limits"] == case["inputs"] == {
        "account_size": 10_000.0, "risk_per_trade_pct": 0.25, "max_position_pct": 8.25}
    assert card["clock"] == scan["scanned_at"] == case["clock"] == "2026-03-10T08:00:00-04:00"
    assert card["action"] == scan["rows"][0]["action"] == "enter_on_trigger"
    assert card["setup"]["type"] == "breakout"
    assert scan["rows"][0]["blockers"] == ["market_closed"]
    plan = card["plan"]
    assert (plan["shares"], plan["fractional_shares"], plan["position_dollars"], plan["max_loss_dollars"]) == (
        7, None, 783.72, 7.84)
    entry, stop = plan["entry"]["trigger_price"], plan["stop"]["price"]
    assert (entry, stop, plan["position_pct"]) == (111.96, 110.84, 7.8372)
    assert plan["max_loss_dollars"] == plan["planned_loss_at_stop"]
    assert plan["sizing"]["limits"]["risk_origin"] == "caller"
    assert plan["sizing"]["limits"]["max_position_origin"] == "caller"
    assert plan["shares"] * entry == pytest.approx(783.72)
    assert plan["shares"] * (entry - stop) == pytest.approx(7.84)


async def test_scan_recording_is_deterministic_and_uses_fresh_isolated_state():
    first, second = await build_scan_case(), await build_scan_case()
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    assert first["scan"]["changes"] == []
    assert [warning["id"] for warning in first["scan"]["warnings"]] == ["first_scan"]
