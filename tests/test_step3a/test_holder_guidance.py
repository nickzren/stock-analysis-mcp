"""The real MCP metadata routes holders without a second decision or data fetch."""

import pytest
from fastmcp import Client

from stock_analysis.prompts.templates import get_prompt
from stock_analysis.resources.analyze_guide import ANALYZE_RENDERING_GUIDE as GUIDE
from stock_analysis.server import mcp


def flat(text):
    return " ".join(text.split())


async def metadata():
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        prompts = {prompt.name: prompt for prompt in await client.list_prompts()}
        assert client.initialize_result is not None
        return client.initialize_result.instructions, tools, prompts


async def test_client_receives_holder_routes_and_protected_guidance():
    instructions, tools, _ = await metadata()
    assert len(instructions) <= 2000
    assert len(tools["analyze"].description) <= 1200
    for text in (instructions, tools["analyze"].description, GUIDE):
        assert "decision_card.long_term.if_held" in text
        assert "There is no long-term holder decision" not in text
    for clause in (
        "Personal context: analyze_my_position(horizon=...) once; inputs optional",
        'Label review_status "Allocation check"',
        "Hold is not endorsement: retain fit concerns/reasons",
        "not_assessed means the instrument is outside the company rubric, not missing data",
        "Adding requires add_allowed; a basis is no permission to add",
        "Disclose an exceeded ceiling or unchecked capacity",
        "Horizons are independent; never substitute new money for a holder answer",
    ):
        assert clause in flat(instructions).replace("`", "")


async def test_client_horizon_defaults_do_not_ask_redundant_questions():
    instructions, tools, _ = await metadata()
    text = flat(instructions)
    assert '"Analyze HOOD" or a bare holder question: mid-term (3-12 months), no question' in text
    assert "12 months / one year means mid-term unless long term is explicit" in text
    assert "Five years: long term (1-5 years), no question" in text
    assert "Ask only for conflicting or unsupported horizons" in text
    assert "one year: mid-term unless long term is explicit" in tools["analyze"].description


async def test_client_position_prompt_only_requires_the_symbol():
    _, _, prompts = await metadata()
    arguments = prompts["position_decision"].arguments
    assert {arg.name for arg in arguments if arg.required} == {"symbol"}
    assert {arg.name for arg in arguments} == {
        "symbol", "horizon", "cost_basis", "purchase_date", "shares", "portfolio_value", "max_holding_pct",
    }
    async with Client(mcp) as client:
        result = await client.get_prompt("position_decision", {"symbol": "SPY", "horizon": "long_term"})
    text = result.messages[0].content.text
    assert 'analyze_my_position("SPY", horizon="long_term")' in text
    assert 'analyze("SPY"' not in text
    assert "Cost basis: $" not in text and "cost_basis=" not in text
    assert "Tax Implications" not in text
    assert "position_card.holder_assessment" in text
    assert "This tool does not assess long-term ETF ownership; its company rules do not apply." in text


@pytest.mark.parametrize("arguments, supplied", [
    ({"symbol": "ACME"}, 'analyze_my_position("ACME", horizon="mid_term")'),
    ({"symbol": "ACME", "cost_basis": "0", "shares": "1.5", "portfolio_value": "1000",
      "max_holding_pct": "10"},
     'cost_basis="0", shares="1.5", portfolio_value="1000", max_holding_pct="10"'),
    ({"symbol": "ACME", "purchase_date": "2026-01-02"}, 'purchase_date="2026-01-02"'),
])
def test_position_prompt_keeps_optional_inputs_as_given(arguments, supplied):
    result = get_prompt("position_decision", arguments)
    text = result["messages"][0]["content"]
    assert supplied in text
    assert text.count("analyze_my_position(") == 1
    assert "Optional inputs are user-reported USD context; do not ask for missing inputs unless needed" in text
    assert "No tax estimate, holder stop or sale quantity" in text
    assert 'review_status as "Allocation check"' in text
    assert "hold, reduce, review_exit, insufficient_data or not_assessed" in text
    assert "Never borrow the new-money action as a holder answer" in text


def test_guide_separates_coverage_fit_allocation_and_personal_numbers():
    text = flat(GUIDE)
    for clause in (
        "This tool does not assess long-term ETF ownership; its company rules do not apply.",
        "Do not ask for ETF cash runway",
        "The checks have not demonstrated a thesis failure, but the long-term suitability assessment is avoid because",
        "the actual reasons from suitability.reasons",
        "verdict.horizon_fit.reasons",
        "not a stock all-clear",
        "your own ceiling governs this single-position check",
        "Stale or fallback prices are historical context",
        "No implicit currency conversion",
        "cost basis of zero permits dollar P&L but no return fraction",
        "thesis_unknown",
        "applicability_unverified",
        "not contradictory provider observations",
        "Technical references are unmonitored diagnostics, not holder exit instructions",
    ):
        assert clause in text, clause


async def test_change_description_distinguishes_shape_checks_from_symbol_resolution():
    _, tools, _ = await metadata()
    description = flat(tools["detect_changes"].description)
    assert "Invalid symbol, types" not in description
    assert "Shapes, types or missing required" in description
    assert "symbol is checked against the resolved ticker afterward" in description
