"""Guidance for ambiguous horizons and non-actionable supporting fields."""

from pathlib import Path

from stock_analysis.prompts.templates import get_prompt
from stock_analysis.resources.analyze_guide import ANALYZE_RENDERING_GUIDE as GUIDE
from stock_analysis.server import mcp
from stock_analysis.tools.analyze.decision import EVIDENCE_BASES, NON_EVIDENCE_BASES
from tests.support import decision_matrix as dm


def flat(text: str) -> str:
    return " ".join(text.split())


def test_one_year_defaults_to_mid_term_unless_long_term_is_explicit() -> None:
    for text in (mcp.instructions or "", GUIDE):
        assert "12 months / one year means mid-term unless long term is explicit" in flat(text)
    assert '"Analyze HOOD" or a bare holder question: mid-term (3-12 months), no question' in (mcp.instructions or "")


def test_long_term_holder_decision_is_named_without_substituting_new_money() -> None:
    prompt = get_prompt("position_decision", {"symbol": "ACME", "cost_basis": "80",
                                              "purchase_date": "2025-01-02"})
    assert prompt is not None
    readme = Path(__file__).resolve().parents[2].joinpath("README.md").read_text()
    for text in (mcp.instructions or "", GUIDE, prompt["messages"][0]["content"], readme):
        assert "There is no long-term holder decision" not in flat(text)
    assert "decision_card.long_term.if_held" in GUIDE
    assert "Never borrow the new-money action as a holder answer" in prompt["messages"][0]["content"]


async def test_holder_basis_does_not_grant_permission_to_add() -> None:
    card = (await dm.run("S4_earnings_gate_on_hold_or_add", fundamentals=dm.QUALITY))["decision_card"]
    assert card["if_held"]["basis"] == "candidate_admits_new_money"
    assert card["if_held"]["add_allowed"] is False
    assert card["new_money"]["action"] == "wait"
    guide = flat(GUIDE)
    assert "`candidate_admits_new_money` describes the pre-gate candidate, not permission to add" in guide
    assert "Adding to a holding is allowed only when `decision_card.if_held.add_allowed` is true" in guide


def test_guide_explains_each_holder_basis_including_unknown_thesis() -> None:
    for basis in (*EVIDENCE_BASES, *NON_EVIDENCE_BASES):
        assert f"`{basis}`" in GUIDE
    assert "`thesis_unknown`: required thesis evidence is unavailable or unverified" in flat(GUIDE)


async def test_entry_band_is_a_reference_not_a_new_price_gate() -> None:
    card = (await dm.run("S1_hold_or_add_clean", fundamentals={
        **dm.PROFITABLE, "financial_health": {"debt_to_equity": 0.5}}))["decision_card"]
    assert card["new_money"]["action"] == "starter"
    assert card["sizing"]["at_price"] > card["entry"]["buy_zone_high"]
    guide = flat(GUIDE)
    assert "`decision_card.entry` is a reference band" in guide
    assert "use the returned action and `decision_card.sizing.at_price`, even outside that band" in guide
    assert "Never invent an entry-price gate from the band" in guide


def test_instructions_disclose_unknown_checks_alongside_entry() -> None:
    assert "List unevaluated gates beside entry as not checked, never cleared" in flat(mcp.instructions or "")
    assert "List `unevaluated` gates as not checked" in flat(GUIDE)


async def test_scan_changes_include_screen_and_card_transitions() -> None:
    tools = {tool.name: tool.description or "" for tool in await mcp.list_tools()}
    assert "`changes` lists screen/card transitions" in flat(tools["scan_watchlist"])
