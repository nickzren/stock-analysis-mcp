"""Step 2b guidance uses returned availability and scenario assumptions."""

import pytest

from tests.support import decision_matrix as dm
from tests.support import sizing_matrix as sm
from tests.test_step2.test_wp24_guidance import descriptions, flat, texts

NULL_RULE = ("When sizing is null, report `decision_card.sizing_status` and its reason; "
             "do not infer the reason from the action alone.")


@pytest.mark.parametrize("key", ["instructions", "guide", "tool:analyze", "prompt:full_analysis",
                                  "prompt:position_decision"])
async def test_null_guidance_uses_status_and_preserves_research_action(key):
    text = flat((await texts())[key])
    assert NULL_RULE in text
    assert "Missing sizing does not change the research action." in text
    assert "no entry size: new_money is" not in text
    assert "mid-term new money only" in text


async def test_guide_separates_availability_quantity_and_conditional_loss():
    text = flat((await texts())["guide"])
    for needle in (
        "`not_allowed` names the action and applicable blockers",
        "`unavailable` names the missing or invalid input",
        "No size estimate: a valid planning stop is unavailable.",
        "`percentage_only`: show allocation and hypothetical loss ceilings",
        "`below_one_whole_share`: no whole-share position fits this budget",
        "Check each scenario independently",
        "Never invent fractional-share support",
        "`decision_card.sizing.assumptions.entry_price` (E)",
        "`decision_card.sizing.assumptions.stop_price` (S)",
        "`decision_card.sizing.assumptions.price_provenance`",
        "`decision_card.sizing.assumptions.stop_basis`",
        "`decision_card.sizing.limits`",
        "`risk_origin` and `max_position_origin`",
        "Position: $P (Y% of the account); estimated loss: X% of the account if sold at $S.",
        "Estimated loss if sold at this stop price. Gaps and slippage can make the actual loss larger.",
        "This planning stop is not monitored or an exit instruction.",
        "standalone allocation",
        "Existing holdings, outstanding orders and available buying power are unknown.",
        "never remaining add capacity or an enforced aggregate portfolio cap",
        "recomputed per request",
        "not a saved holder stop",
        "`entry_band_unavailable`",
        "When sizing is null, use `decision_card.entry.current_price` as the reference only if returned.",
        "If the reference is unavailable, say so; never invent a size or stop.",
    ):
        assert needle in text, needle


async def test_position_prompt_keeps_sizing_separate_from_holder_decision():
    text = flat((await texts())["prompt:position_decision"])
    for needle in ("never attach it to a long-term or holder answer",
                   "This planning stop is not monitored or an exit instruction.",
                   "Gaps and slippage can make the actual loss larger.",
                   "standalone allocation", "remaining add capacity"):
        assert needle in text, needle


@pytest.mark.parametrize("tool", ["analyze", "analyze_trade_setup", "scan_watchlist"])
async def test_numeric_input_contract_is_advertised(tool):
    text = flat((await descriptions())[tool])
    for needle in ("finite numbers or decimal/scientific strings", "surrounding whitespace and a sign",
                   "never booleans", "NaN/infinity", "overflow or underflow to zero",
                   "percentage points", "1 means 1%; 0.01 means 0.01%",
                   "account-value assumption, not available buying power",
                   "before symbol lookup, provider reads or watchlist access"):
        assert needle in text, (tool, needle)
    if tool == "analyze":
        assert "Omitted/null limits use the selected profile" in text
    else:
        assert "Omitted limits use 1% risk and 10% position cap; explicit null limits are rejected" in text
    assert "Omitted/null account gives percentage-only sizing" in text
    if tool == "scan_watchlist":
        assert "Sizing inputs are forwarded to promoted cards; scan rows contain no plan or sizing." in text
        assert "percentage-only sizing in promoted cards" in text


@pytest.mark.parametrize(("name", "action", "status", "reason"), [
    ("no_atr", "buy", "unavailable", "stop_price_missing"),
    ("negative_stop", "starter", "unavailable", "stop_price_invalid"),
    ("earnings_blocked", "wait", "not_allowed", "new_money is wait; blocked by: earnings_blackout"),
])
async def test_null_size_reasons_survive_every_tier_with_the_research_action(name, action, status, reason):
    raw = await sm.run_case(name)
    for tier, payload in dm.tiers(raw).items():
        card = payload["decision_card"]
        assert card["new_money"]["action"] == action, tier
        assert card["sizing"] is None, tier
        assert card["sizing_status"] == {"status": status, "reason": reason}, tier
        assert card["if_held"] == raw["decision_card"]["if_held"], tier
        assert card["long_term"] == raw["decision_card"]["long_term"], tier
        if "executive_summary" in payload:
            summary = payload["executive_summary"]
            assert f"no entry size: new_money is {action}" not in summary
            if status == "unavailable":
                assert "No size estimate: a valid planning stop is unavailable" in summary
            else:
                assert "earnings" in summary.lower()


@pytest.mark.parametrize("name", ["no_account", "small_account"])
async def test_non_null_scenarios_retain_their_distinct_quantity_status(name):
    raw = await sm.run_case(name)
    for tier, payload in dm.tiers(raw).items():
        card = payload["decision_card"]
        size = card["sizing"]
        assert size is not None, tier
        if name == "no_account":
            assert card["sizing_status"]["status"] == "percentage_only", tier
            for scenario in (size["starter"], size["full"]):
                assert scenario["quantity_status"] == "percentage_only", tier
                assert scenario["whole_shares"] is scenario["position_dollars"] is None, tier
                assert scenario["planned_loss_at_stop"] is None, tier
        else:
            assert card["sizing_status"]["status"] == "available", tier
            assert size["starter"]["quantity_status"] == "below_one_whole_share", tier
            assert size["starter"]["whole_shares"] == 0, tier
            assert size["full"]["quantity_status"] == "available", tier
            assert size["full"]["whole_shares"] == 1, tier
