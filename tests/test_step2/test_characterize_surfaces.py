"""WP2.0: decision-surface contradictions S-C1-S-C16 (docs/plans/step2-research-decision-surfaces.md §4).

Each test records 0609fde behavior; plan §11 assigns each S-C id to the WP or Step that fixes it.
Where each S-C id is pinned (matrix rows are tests/fixtures/decision_matrix/before.json under DM-CH1):

- S-C1: DM-CH1 rows S4, C2, S9 (action_now vs horizon_fit, modes, snapshot); SC-CH1 (executive summary).
- S-C2: DM-CH1 rows S6, C (dip, core, policy long_term, add_only_if); OC-CH3.
- S-C3: OC-CH3; DM-CH1 rows S6, C, B (dip type with entry_actions).
- S-C4: OC-CH1 (S3 avoid), OC-CH4 (S7 ETF).
- S-C5: OC-CH4 (card and snapshot halves); DM-CH1 row S7.
- S-C6: GC-C3 test_gate_resolution_matrix (analyze) vs GC-C1 test_frozen_rows (card).
- S-C7: GC-C3 weak_liq row (hold passes through) vs GC-C1 weak_liq and GC-C4 liq_* rows.
- S-C8: SC-CH8.
- S-C9: DM-CH1 rows S6, C (buy_now small_starter) vs GC-C1 knife (card avoid).
- S-C10: SC-CH10.
- S-C11: PH-CH4.
- S-C12: SC-CH12 (rows S6 and C show it too, but their dislocation override would give it anyway).
- S-C13: SC-CH13. S-C14: SC-CH14. S-C15: SC-CH15. S-C16: SC-CH16.

Ported from the research repro scripts repro_cross_tool.py case 1 (SC-CH8, SC-CH16) and
repro_dislocation.py (SC-CH10); S-C12-S-C15 were found by code reading.
Analyze runs go through tests/support/decision_matrix.py and swing runs through
tests/support/swing_matrix.py; every run pins its clock.
"""

import copy
import re
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Self

import pytest

import stock_analysis.tools.position as position_module
import stock_analysis.tools.watchlist as watchlist
from stock_analysis.prompts.templates import get_prompt
from stock_analysis.resources.analyze_guide import ANALYZE_RENDERING_GUIDE
from stock_analysis.server import mcp
from stock_analysis.tools import compare as compare_module
from stock_analysis.tools import portfolio as portfolio_module
from stock_analysis.tools.analyze.decision import IF_HELD_ACTIONS
from stock_analysis.utils.watch_store import load_scan_state
from tests.support import decision_matrix as dm
from tests.support import swing_matrix as sm
from tests.test_tools.test_setup_detection import make_technicals
from tests.test_tools.test_trade_card import build_card


def _pullback_chart() -> dict[str, Any]:
    """The trade-card fixture's uptrend pullback, with the MACD, returns and 52-week range analyze reads."""
    technicals = make_technicals()
    technicals["macd"] = {"rules": {"bullish_cross": {"triggered": True}, "bearish_cross": {"triggered": False}}}
    technicals["returns"].update({"return_1m": 0.04, "return_1y": 0.25})
    technicals["price_position"].update({"week_52_low": 70.0, "week_52_high": 108.0, "from_52w_high": -0.07})
    return technicals


PULLBACK = _pullback_chart()
TRIGGER_CROSSED = 101.5  # above the pullback's 101.0 trigger: the card says trade_now in session
# The research repro's unprofitable payload also fired missing_runway; POOR_BUSINESS keeps gates out.
PULLBACK_POOR_BUSINESS = dm.Scenario("S-C8 repro_cross_tool 1", dm.at_price(PULLBACK, TRIGGER_CROSSED),
                                     dm.POOR_BUSINESS, dm.MODERATE)


def pullback_card() -> dict[str, Any]:
    """The trade card on the same chart (build_card pins a regular session); it reads no fundamentals."""
    return build_card(technicals_data=copy.deepcopy(PULLBACK), actionable_price=TRIGGER_CROSSED)


class TestGatedSummary:
    @pytest.mark.parametrize("name", ["S4_earnings_gate_on_hold_or_add", "S9_weak_liquidity"])
    async def test_a_gated_hold_or_add_leads_with_its_gated_decision(self, name: str) -> None:
        """SC-CH1 (flipped by WP2.3): with new_money wait (S4, event) or avoid (S9, weak liquidity), the executive
        summary no longer copies S1's (starter, no gate): its new-money segment names the gated action, the gate
        and its reassessment and carries no size, while S1's names the starter size only; no "Policy:" clause
        remains, and the descriptive sentences after the three segments are unchanged. The action clause flipped
        in WP2.2 (D-2, D-5a). G from WP2.3 (PD D-9: executive-summary segments, plan §7)."""
        fund = {**dm.PROFITABLE, "financial_health": {"debt_to_equity": 0.5}}
        gated = await dm.run(name, fundamentals=fund)
        clean = await dm.run("S1_hold_or_add_clean", fundamentals=fund)
        gated_sentences = gated["executive_summary"].split(". ")
        clean_sentences = clean["executive_summary"].split(". ")

        assert (gated["decision_card"]["action_now"], clean["decision_card"]["action_now"]) == (
            {"S4_earnings_gate_on_hold_or_add": "wait", "S9_weak_liquidity": "avoid"}[name], "starter")
        assert gated_sentences[0] == {
            "S4_earnings_gate_on_hold_or_add": "Decision (mid-term, new money): wait; blocked by earnings_blackout; "
                                               "reassess on or after Tue Oct 13 (not a buy date)",
            "S9_weak_liquidity": "Decision (mid-term, new money): avoid; blocked by weak_liquidity; "
                                 "no reassessment "
                                 "date: re-run analyze after 1-year average daily dollar volume >= $1M"}[name]
        assert clean_sentences[0].startswith("Decision (mid-term, new money): starter; Starter — Position: $200 ")
        assert "estimated loss:" in clean_sentences[0]
        assert "size ~" not in gated["executive_summary"]
        assert "Policy:" not in gated["executive_summary"] + clean["executive_summary"]
        assert clean_sentences[3:5] == [
            "Mid-term sizing only: This planning stop is not monitored or an exit instruction",
            "Gaps and slippage can make the actual loss larger"]
        assert gated_sentences[3:] == clean_sentences[5:]


class TestSwingVsMidTerm:
    async def test_analyze_avoids_where_the_card_trades_now(self) -> None:
        """SC-CH8: on one uptrend pullback with a poor business, analyze's mid-term new_money is avoid
        [thesis_broken] with if_held review_exit (speculative avoid: its own wait capped by new_money; core avoid)
        while the trade card, which reads no fundamentals, says trade_now (swing, WP2.1). The action clause flipped
        in WP2.2 (PD D-1c: the poor business breaks the thesis, so not D-2's wait); the speculative clause in WP2.3
        (PD D-3, PH-RF19). G from WP2.3: the swing/mid-term disagreement is horizons, labelled (S-C8)."""
        result = await dm.run(PULLBACK_POOR_BUSINESS)
        modes = result["decision_modes"]
        fit = result["verdict"]["horizon_fit"]
        card = pullback_card()

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == ("avoid", ["thesis_broken"])
        assert result["decision_card"]["if_held"]["action"] == "review_exit"
        assert (modes["speculative"]["action"], modes["speculative"]["capped_by"], modes["core"]["action"]) == (
            "avoid", "new_money_avoid", "avoid")
        assert (fit["mid_term"], fit["long_term"]) == ("ok", "avoid")
        assert (card["action"], card["setup"]["type"], card["blockers"]) == ("trade_now", "pullback_in_uptrend", [])

    async def test_analyze_decision_card_states_its_horizon(self) -> None:
        """SC-CH8: analyze's decision_card states mid_term and its long_term block long_term at every tier; the
        card and scan-row clauses are S-C8's (test_wp21_card.py, scan rows with stage 1).
        G from WP2.2 (plan §2, D-3; flipped from "no horizon key")."""
        shown = dm.tiers(await dm.run(PULLBACK_POOR_BUSINESS))

        for tier, payload in shown.items():
            card = payload["decision_card"]
            assert (card["horizon"], card["long_term"]["horizon"]) == ("mid_term", "long_term"), tier


class TestStandardTierCarriesTheDecision:
    async def test_dislocation_check_carries_the_decision(self) -> None:
        """SC-CH10 (flipped by WP2.3): at the standard tier (analyze's default), S6's dislocation_check is the
        five-question read plus decision = {new_money, blocked_by ids} (avoid, [falling_knife]); the full tier's
        dislocation action follows new_money (can_start_now false, buy_now null). G from WP2.3 (S-C10, PD D-9)."""
        result = await dm.run("S6_knife_dislocation")
        standard = dm.tiers(result)["standard"]
        check = standard["dislocation_check"]

        assert set(check) == {"overall_status", "overall_read", "questions", "decision"}
        assert check["decision"] == {"new_money": "avoid", "blocked_by": ["falling_knife"]}
        assert check["overall_status"] == "price_broken_more_than_business"
        assert [q["id"] for q in check["questions"]] == [
            "down_enough", "business_still_good", "balance_sheet_safe", "long_term_thesis_intact",
            "price_broken_more_than_business"]
        assert "dislocation_framework" not in standard
        action = result["dislocation_framework"]["action"]
        assert (action["can_start_now"], action["buy_now"]) == (False, None)

    async def test_the_guide_answers_what_to_do_from_the_card_at_every_tier(self) -> None:
        """SC-CH10 (flipped by WP2.4): analyze still defaults to standard, and the rendering guide no longer
        assumes detail="full": its second step answers "What should I do now?" from decision_card, present at
        every detail level, in three labelled horizon segments (new money mid-term, if held, long term), never
        upgrading past blocked_by; can_start_now is no longer the source (plan §9). G from WP2.4 (S-C10, BL-F1)."""
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        steps = re.split(r"(?m)^(?=\d+\. )", ANALYZE_RENDERING_GUIDE)
        card = " ".join(next(s for s in steps if s.startswith("2. ")).split())

        assert tools["analyze"].parameters["properties"]["detail"]["default"] == "standard"
        assert 'assumes `detail="full"`' not in ANALYZE_RENDERING_GUIDE
        assert "If `action.can_start_now` is true" not in ANALYZE_RENDERING_GUIDE
        assert [t.strip() for _, t in re.findall(r"(?m)^(\d+)\. ([^:(\n]+)", ANALYZE_RENDERING_GUIDE)][:2] == [
            "Header", "Decision card"]
        assert "every detail level" in card and "What should I do now?" in card and "`decision_card`" in card
        assert card.index("New money, mid-term") < card.index("If held") < card.index("Long term")
        assert "Never upgrade past `blocked_by`" in card


class TestSpeculativeMode:
    async def test_a_neutral_tilt_no_longer_offers_a_starter_under_avoid(self) -> None:
        """SC-CH12 (flipped by WP2.3): S3's chart with QUALITY fundamentals is a plain avoid (mid fit avoid, no
        gate, not a dislocation candidate) with a neutral tilt and a buy_the_dip dip (now capped to watch), and
        speculative's own starter_position_only is mapped to starter, then capped by new_money: avoid; S3 itself
        (bearish tilt, own watch) gives avoid too. G from WP2.3 (PD D-3, PH-RF19)."""
        neutral = await dm.run("S3_plain_avoid", fundamentals=dm.QUALITY)
        bearish = await dm.run("S3_plain_avoid")

        assert (neutral["decision_card"]["action_now"], dm.gate_ids(neutral)) == ("avoid", [])
        assert (neutral["verdict"]["tilt"], neutral["verdict"]["horizon_fit"]["mid_term"]) == ("neutral", "avoid")
        assert neutral["dip_assessment"]["assessment"]["dip_quality"] == "good"  # buy_the_dip before the cap
        assert neutral["dip_assessment"]["assessment"]["recommendation"] == "watch"
        assert neutral["decision_modes"]["balanced"]["suitability"] == "avoid"  # starter_only only under starter
        assert neutral["decision_modes"]["speculative"]["action"] == "avoid"
        assert bearish["verdict"]["tilt"] == "bearish"
        assert bearish["decision_modes"]["speculative"]["action"] == "avoid"


# QUALITY's shape with every value compare ranks worse, and no metric QUALITY lacks.
FAIR_BUSINESS = {"valuation": {"pe_trailing": 20.0, "trailing_eps": 5.0}, "growth": {"revenue_yoy": 0.06},
                 "profitability": {"net_margin": 0.1}, "yield_metrics": {"fcf_yield": 0.04},
                 "cash_flow": {"rules": {"positive_fcf": {"triggered": True}}},
                 "financial_health": {"debt_to_equity": 0.5, "rules": {"net_cash_positive": {"triggered": True}}}}
STRONG_FAIR_BUSINESS = dm.Scenario("S-C13 partner", dm.STRONG, FAIR_BUSINESS, dm.CALM)


class TestCompareHasNoGates:
    async def test_compare_ranks_first_a_stock_analyze_avoids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SC-CH13: compare puts S6 (analyze: avoid, falling_knife) at composite_position 1 ahead of a
        bullish, ungated stock, and its rows carry no decision or gate field.
        CH: flipped by Step 5 (compare, plan §1, §11); holds through Step 2."""
        scenarios = {"ACME": dm.SCENARIOS["S6_knife_dislocation"], "BETA": STRONG_FAIR_BUSINESS}
        avoided, partner = [await dm.run(scenario) for scenario in scenarios.values()]
        payloads = {symbol: scenario.payloads() for symbol, scenario in scenarios.items()}

        def tool(name: str) -> Any:
            async def fake(symbol: str, **kwargs: Any) -> dict[str, Any]:
                return {"symbol": symbol, **copy.deepcopy(payloads[symbol][name] or {})}
            return fake

        for name in ("stock_summary", "fundamentals_snapshot", "technicals"):
            monkeypatch.setattr(compare_module, name, tool(name))
        compared = await compare_module.compare_stocks(["BETA", "ACME"])
        rows = compared["comparison"]

        assert (avoided["decision_card"]["action_now"], dm.gate_ids(avoided)) == ("avoid", ["falling_knife"])
        assert (partner["verdict"]["tilt"], partner["verdict"]["horizon_fit"]["mid_term"], dm.gate_ids(partner)) == (
            "bullish", "ok", [])
        assert {symbol: (row["composite_rank"], row["composite_position"]) for symbol, row in rows.items()} == {
            "ACME": (1.44, 1), "BETA": (1.56, 2)}
        for row in rows.values():
            assert set(row) == {"name", "sector", "market_cap", "metrics", "composite_rank", "composite_position"}


class TestPortfolioThreshold:
    async def test_a_15_percent_holding_is_not_flagged_above_every_analyze_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SC-CH14: analyze caps a full position at 6-10% (profile cap; the card's full_pct_cap is lower
        or equal), yet analyze_portfolio flags only weights above 20%, so a 15% holding passes
        unflagged. CH: flipped by Step 3 (portfolio threshold, plan §11); holds through Step 2."""
        caps: dict[str, tuple[float, float]] = {}
        for profile in ("core", "balanced", "speculative"):
            result = await dm.run("S2_buy_clean", profile=profile, fundamentals={
                **dm.PROFITABLE, "financial_health": {"debt_to_equity": 0.5}})
            caps[profile] = (result["action_zones"]["position_sizing_range"]["profile_cap_pct"],
                             result["decision_card"]["sizing"]["full_pct_cap"])

        async def info(symbol: str) -> dict[str, Any]:
            return {}

        async def no_history(params: Any) -> Any:
            raise RuntimeError("no history offline")

        monkeypatch.setattr(portfolio_module, "fetch_info", info)
        monkeypatch.setattr(portfolio_module, "fetch_history", no_history)
        exposure = await portfolio_module.portfolio_exposure([
            {"symbol": "ACME", "value": 15_000}, {"symbol": "BETA", "value": 25_000},
            {"symbol": "GAMMA", "value": 60_000}])

        assert caps == {"core": (8.0, 8.0), "balanced": (10.0, 8.0), "speculative": (6.0, 6.0)}
        assert {p["symbol"]: (p["weight"], p["is_concentrated"]) for p in exposure["concentration"]["positions"]} == {
            "GAMMA": (0.6, True), "BETA": (0.25, True), "ACME": (0.15, False)}
        assert max(cap for pair in caps.values() for cap in pair) < 15.0
        assert exposure["liquidity"]["illiquid_positions"] == []


class _FridayEvening(datetime):
    """analyze_position's holding period reads datetime.now(); pin it to the matrix clock's date."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> Self:
        return cls(2026, 10, 9, 20, 0)


POSITION_DECISION_ARGS = {"symbol": "ACME", "cost_basis": "80", "purchase_date": "2025-01-02"}


def position_decision_text() -> str:
    prompt = get_prompt("position_decision", POSITION_DECISION_ARGS)
    assert prompt is not None
    content: str = prompt["messages"][0]["content"]
    return content


class TestPositionDecisionPrompt:
    def test_the_prompt_decides_from_analyze(self) -> None:
        """SC-CH15 (flipped by WP2.4): position_decision calls analyze first for the decision, then
        analyze_my_position for P/L and tax, and asks for decision_card.if_held.action in the holder vocabulary,
        not HOLD / SELL / TRIM and not get_technicals (plan §9). G from WP2.4."""
        text = position_decision_text()

        assert re.findall(r"^\d+\. (\w+)\(", text, flags=re.MULTILINE) == ["analyze", "analyze_my_position"]
        assert "`decision_card.if_held.action`" in text
        assert [a for a in IF_HELD_ACTIONS if not re.search(rf"\b{a}\b", text)] == []
        assert "HOLD / SELL / TRIM" not in text and "get_technicals" not in text

    async def test_the_prompt_decision_tool_tells_a_broken_business_from_a_quality_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SC-CH15 (flipped by WP2.4): A (broken business) and D (quality business) share the BEARISH chart.
        analyze_my_position reads technicals only and returns the same signals for both, so the prompt takes the
        decision from analyze's if_held, which tells them apart (review_exit business_broken vs hold
        no_reduction_evidence). G from WP2.4 (plan §9)."""
        def technicals(body: dict[str, Any] | None) -> Any:
            async def fake(symbol: str, **kwargs: Any) -> dict[str, Any]:
                return {"symbol": symbol, **copy.deepcopy(body or {})}
            return fake

        monkeypatch.setattr(position_module, "datetime", _FridayEvening)
        reads: dict[str, dict[str, Any]] = {}
        analyzed: dict[str, tuple[str, str]] = {}
        for name in ("A_broken_bearish", "D_quality_attractive_poor_chart"):
            scenario = dm.SCENARIOS[name]
            monkeypatch.setattr(position_module, "technicals", technicals(scenario.payloads()["technicals"]))
            read = await position_module.analyze_position("ACME", cost_basis=80.0, purchase_date="2025-01-02")
            reads[name] = {key: value for key, value in read.items() if key != "meta"}
            result = await dm.run(scenario)
            analyzed[name] = (result["decision_card"]["if_held"]["action"], result["decision_card"]["if_held"]["basis"])

        assert re.findall(r"^\d+\. (\w+)\(", position_decision_text(), flags=re.MULTILINE)[0] == "analyze"
        assert analyzed == {"A_broken_bearish": ("review_exit", "business_broken"),
                            "D_quality_attractive_poor_chart": ("hold", "no_reduction_evidence")}
        assert reads["A_broken_bearish"] == reads["D_quality_attractive_poor_chart"]
        assert reads["A_broken_bearish"]["sell_signals"] == {
            "broke_sma50": True, "broke_sma200": True, "death_cross": False, "rsi_overbought": False,
            "macd_bearish": True, "active_count": 3}
        assert set(reads["A_broken_bearish"]) == {"data_provenance", "symbol", "position", "holding", "tax",
                                                  "current_signals", "sell_signals", "support_levels"}


class TestSavedRecords:
    async def test_the_snapshot_and_the_scan_state_disagree_for_one_symbol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """SC-CH16 (flipped by WP2.4): for ACME on the S-C8 chart and business, analyze's 3.0.0 watchlist_snapshot
        saves the mid-term new_money_action avoid, if_held_action review_exit and long_term_action avoid
        [thesis_broken] (WP2.2: D-1c) while the scan state saves the swing card's trade_now; the snapshot still
        names no horizon (plan §8). The docs clause (the snapshot is mid-term and long-term, the scan state swing)
        is tests/test_docs.py::test_readme_and_agents_carry_the_decision_tables_and_horizons and the
        detect_changes / scan_watchlist descriptions (S-C16.d). G from WP2.4."""
        snapshot = (await dm.run(PULLBACK_POOR_BUSINESS))["watchlist_snapshot"]
        card = pullback_card()

        async def trade_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {**copy.deepcopy(card), "symbol": symbol}

        sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"])  # stage 1 promotes, so the card runs
        monkeypatch.setattr(watchlist, "analyze_trade_setup", trade_card)
        await watchlist.manage_watchlist("add", ["ACME"], _today=sm.PRE_MARKET.date(), data_dir=tmp_path)
        scan = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)
        state, warnings = load_scan_state(tmp_path)

        assert (scan["full_cards"], warnings) == (1, [])
        assert (snapshot["symbol"], snapshot["new_money_action"], snapshot["if_held_action"],
                snapshot["long_term_action"], snapshot["hard_gate_ids"]) == (
            "ACME", "avoid", "review_exit", "avoid", ["thesis_broken"])
        assert "decision_action_now" not in snapshot
        assert state["symbols"]["ACME"]["action"] == "trade_now"
        assert "horizon" not in snapshot
