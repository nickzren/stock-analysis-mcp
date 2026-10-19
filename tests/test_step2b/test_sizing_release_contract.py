"""Real registered-tool responses preserve research decisions under sizing changes.

Numeric schema/coercion/access guards live in test_numeric_baseline. These cases
continue through the real handlers, projections, arithmetic and JSON serializer;
only market data and the clock are stubbed, and scan state stays in pytest's dir.
"""

import copy
import json
from dataclasses import replace
from datetime import datetime

import pytest
from fastmcp import Client

from stock_analysis import SCHEMA_VERSION, server
from stock_analysis.tools import watchlist
from stock_analysis.tools.analyze import orchestrator
from stock_analysis.utils.normalize import SNAPSHOT_VERSION
from tests.support import decision_matrix as dm
from tests.support import sizing_matrix as sizing
from tests.support import swing_matrix as swing

TIERS = ("decision", "standard", "full")


def pin_clock(monkeypatch, module, now):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(module, "datetime", FixedClock)


async def payload(client, tool, **arguments):
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert not result.is_error, result.content
    body = json.loads(result.content[0].text)
    assert not body.get("error"), body
    return body


def decisions(body):
    card = body["decision_card"]
    return {key: card[key] for key in ("new_money", "if_held", "long_term", "hard_gates", "action_now")}


def test_sizing_response_migration_keeps_snapshot_version():
    assert SCHEMA_VERSION == "7"
    assert SNAPSHOT_VERSION == "3.0.0"


@pytest.mark.parametrize("action", ["buy", "starter"])
async def test_client_missing_atr_explains_null_size_without_changing_decisions(monkeypatch, action):
    case = sizing.scenario("balanced_wide" if action == "buy" else "starter")
    pin_clock(monkeypatch, orchestrator, case.now)
    missing = copy.deepcopy(case.risk)
    missing["atr"] = {"value": None, "as_pct_of_price": None}
    async with Client(server.mcp) as client:
        with dm.stubbed(case):
            reference = await payload(client, "analyze", symbol="ACME", account_size=10_000, detail="full")
        with dm.stubbed(replace(case, risk=missing)):
            for tier in TIERS:
                body = await payload(client, "analyze", symbol="ACME", account_size=10_000, detail=tier)
                card = body["decision_card"]
                assert body["meta"]["schema_version"] == "7"
                assert card["new_money"]["action"] == action
                assert card["sizing"] is None
                assert card["sizing_status"] == {"status": "unavailable", "reason": "stop_price_missing"}
                if tier != "decision":
                    assert "No size estimate:" in body["executive_summary"]
                    assert "planning stop" in body["executive_summary"]
                    assert f"no entry size: new_money is {action}" not in body["executive_summary"].lower()
                assert decisions(body) == decisions(reference)
                assert body["watchlist_snapshot"] == reference["watchlist_snapshot"]


@pytest.mark.parametrize("name", ["starter", "earnings_blocked", "no_account", "small_account"])
async def test_client_all_tiers_keep_visibility_and_per_scenario_quantity_status(monkeypatch, name):
    case = sizing.scenario(name)
    pin_clock(monkeypatch, orchestrator, case.now)
    expected_snapshot = json.loads((sizing.FIXTURE_DIR / "before.json").read_text())["analyze"][name]["snapshot"]
    with dm.stubbed(case):
        async with Client(server.mcp) as client:
            outputs = [await payload(client, "analyze", symbol="ACME", account_size=case.account_size, detail=tier)
                       for tier in TIERS]
    for tier, body in zip(TIERS, outputs, strict=True):
        card = body["decision_card"]
        size = card["sizing"]
        assert body["meta"]["schema_version"] == "7"
        assert body["watchlist_snapshot"] == expected_snapshot
        assert card["sizing_status"] == outputs[0]["decision_card"]["sizing_status"]
        assert size == outputs[0]["decision_card"]["sizing"]
        if name == "earnings_blocked":
            assert card["new_money"]["action"] == "wait"
            assert size is None
            assert card["sizing_status"]["status"] == "not_allowed"
            assert "earnings_blackout" in card["sizing_status"]["reason"]
            if tier != "decision":
                assert "blocked by earnings_blackout" in body["executive_summary"]
                assert "No size estimate:" not in body["executive_summary"]
            continue
        assert size["horizon"] == "mid_term"
        if name == "starter":
            assert card["new_money"]["action"] == "starter"
            assert size["full"] is size["full_pct_cap"] is size["full_dollars"] is None
            assert size["starter"]["whole_shares"] == 2
        elif name == "no_account":
            assert card["sizing_status"]["status"] == "percentage_only"
            for scenario in (size["starter"], size["full"]):
                assert scenario["quantity_status"] == "percentage_only"
                assert all(scenario[key] is None for key in ("whole_shares", "position_dollars", "planned_loss_at_stop"))
            if tier != "decision":
                assert "account size not provided" in body["executive_summary"]
                assert "allocation ceiling" in body["executive_summary"]
        else:
            assert card["new_money"]["action"] == "buy"
            assert size["starter"]["quantity_status"] == "below_one_whole_share"
            assert size["full"]["quantity_status"] == "available"
            assert (size["starter"]["whole_shares"], size["full"]["whole_shares"]) == (0, 1)
            assert (size["starter"]["position_dollars"], size["full"]["position_dollars"]) == (0, 50)
            if tier != "decision":
                assert "Starter: no whole-share position fits this budget" in body["executive_summary"]
                assert "Full — Position: $50" in body["executive_summary"]


async def test_client_sizing_inputs_preserve_complete_snapshots_and_decision_surfaces(monkeypatch):
    case = sizing.scenario("balanced_wide")
    pin_clock(monkeypatch, orchestrator, case.now)
    variants = [
        {}, {"account_size": None, "risk_per_trade_pct": None, "max_position_pct": None},
        {"account_size": 10_000, "profile": "core"},
        {"account_size": 10_000, "profile": "speculative"},
        {"account_size": 10_000, "risk_per_trade_pct": 0.25},
        {"account_size": 10_000, "max_position_pct": 2},
        {"account_size": 10_000, "risk_per_trade_pct": 0.25, "max_position_pct": 0.5},
        {"account_size": " 1e4 ", "risk_per_trade_pct": "1e-3", "max_position_pct": "0.06"},
    ]
    with dm.stubbed(case):
        async with Client(server.mcp) as client:
            reference = await payload(client, "analyze", symbol="ACME", account_size=10_000, detail="full")
            expected = dm.digest(reference)
            for arguments in variants:
                body = await payload(client, "analyze", symbol="ACME", detail="full", **arguments)
                assert body["watchlist_snapshot"] == reference["watchlist_snapshot"], arguments
                assert decisions(body) == decisions(reference), arguments
                actual = dm.digest(body)
                for key in ("verdict", "policy", "modes", "dip", "dislocation", "price_zone", "valuation_gate"):
                    assert actual[key] == expected[key], (arguments, key)
                assert body["summary"] == reference["summary"], arguments
            numeric = await payload(client, "analyze", symbol="ACME", account_size=10_000,
                                    risk_per_trade_pct=1.5, max_position_pct=1.5, detail="full")
            text = await payload(client, "analyze", symbol="ACME", account_size=" 1e4 ",
                                 risk_per_trade_pct="1.5", max_position_pct=" +1.5e0 ", detail="full")
    assert text["decision_card"] == numeric["decision_card"]
    assert text["watchlist_snapshot"] == numeric["watchlist_snapshot"]


async def test_client_swing_and_scan_reach_real_sizing_with_isolated_state(monkeypatch, tmp_path):
    swing.install(monkeypatch, swing.SCAN_SCENARIOS["fresh_breakout"])
    pin_clock(monkeypatch, swing.trade_setup, swing.PRE_MARKET)
    pin_clock(monkeypatch, watchlist, swing.PRE_MARKET)
    data_dir = tmp_path / "client-scan"
    monkeypatch.setenv("STOCK_ANALYSIS_DATA_DIR", str(data_dir))
    await watchlist.manage_watchlist("add", [swing.SYMBOL])
    promoted = []
    real_card = watchlist.analyze_trade_setup

    async def capture_card(*args, **kwargs):
        card = await real_card(*args, **kwargs)
        promoted.append(card)
        return card

    monkeypatch.setattr(watchlist, "analyze_trade_setup", capture_card)
    variants = [
        {"account_size": 10_000, "risk_per_trade_pct": 1.5, "max_position_pct": 1.5},
        {"account_size": " 1e4 ", "risk_per_trade_pct": "1.5", "max_position_pct": " +1.5e0 "},
        {}, {"account_size": None}, {"account_size": 1},
    ]
    cards, scans = [], []
    async with Client(server.mcp) as client:
        for arguments in variants:
            card = await payload(client, "analyze_trade_setup", symbol=swing.SYMBOL, **arguments)
            scan = await payload(client, "scan_watchlist", **arguments)
            cards.append(card)
            scans.append(scan)
            assert scan["symbols_scanned"] == scan["full_cards"] == 1
            assert not scan["errors"] and not scan["changes"]
            assert scan["rows"][0]["source"] == "card"
            assert scan["rows"][0]["action"] == card["action"] == "enter_on_trigger"
            assert promoted[-1]["plan"] == card["plan"]
            assert card["plan"]["planned_loss_at_stop"] == card["plan"]["max_loss_dollars"]
            assert card["plan"]["sizing"]["horizon"] == "swing"
            assert "plan" not in scan["rows"][0]
    assert cards[0]["plan"] == cards[1]["plan"]
    assert cards[2]["plan"] == cards[3]["plan"]
    for card in cards[2:4]:
        assert card["plan"]["sizing_status"]["status"] == "percentage_only"
        assert all(card["plan"][key] is None for key in ("shares", "position_dollars", "planned_loss_at_stop"))
    assert cards[4]["plan"]["sizing"]["full"]["quantity_status"] == "below_one_whole_share"
    assert cards[4]["plan"]["shares"] == cards[4]["plan"]["planned_loss_at_stop"] == 0
    assert cards[4]["plan"]["entry"]["condition_text"] == (
        "The setup qualifies, but no whole-share position fits this budget.")
    for card in cards[1:]:
        for key in ("action", "setup", "blockers", "unevaluated", "decided_by", "freshness"):
            assert card[key] == cards[0][key], key
    for scan in scans[1:]:
        assert scan["rows"] == scans[0]["rows"]
    saved = json.loads((data_dir / "scan_state.json").read_text())
    assert saved["symbols"][swing.SYMBOL]["action"] == "enter_on_trigger"
