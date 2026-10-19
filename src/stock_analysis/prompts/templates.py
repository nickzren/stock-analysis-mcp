"""Prompt templates for stock analysis."""

import json
from typing import Any

# Prompt definitions
PROMPTS = {
    "full_analysis": {
        "description": "Comprehensive reproducible stock analysis with consistent JSON output",
        "arguments": [{"name": "symbol", "required": True}],
    },
    "growth_memo": {
        "description": "Generate a growth investment analysis memo",
        "arguments": [{"name": "symbol", "required": True}],
    },
    "value_memo": {
        "description": "Generate a value investment analysis memo",
        "arguments": [{"name": "symbol", "required": True}],
    },
    "position_decision": {
        "description": "Holder assessment for a selected horizon, with optional personal USD context",
        "arguments": [
            {"name": "symbol", "required": True},
            {"name": "cost_basis", "required": False},
            {"name": "purchase_date", "required": False},
            {"name": "shares", "required": False},
            {"name": "horizon", "required": False},
            {"name": "portfolio_value", "required": False},
            {"name": "max_holding_pct", "required": False},
        ],
    },
}


def list_prompts() -> list[dict[str, Any]]:
    """List available prompts."""
    return [
        {
            "name": name,
            "description": info["description"],
            "arguments": info["arguments"],
        }
        for name, info in PROMPTS.items()
    ]


def get_prompt(name: str, arguments: dict[str, str]) -> dict[str, Any] | None:
    """
    Get a prompt by name with arguments filled in.

    Returns dict with 'messages' key for MCP GetPromptResult.
    """
    if name not in PROMPTS:
        return None

    if name == "full_analysis":
        symbol = arguments.get("symbol", "")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Analyze {symbol} with the single `analyze` tool first.

Use `analyze("{symbol}", detail="full")` by default.
If the user gives an account size, include it:
`analyze("{symbol}", account_size=<value>, detail="full")`

Do not manually reconstruct the report from secondary tools unless `analyze` fails.

If you need detailed presentation rules, read:
`stock-analysis://guides/analyze-rendering`

When you present the result, lead with:
1. `decision_card`: `decision_card.new_money` (mid-term) and `decision_card.long_term.new_money`, each action verbatim with its `blocked_by`, then `decision_card.if_held` action verbatim with its `basis` (mid-term), and `decision_card.long_term.if_held` action with its `basis` (long-term)
2. `executive_summary`, verbatim
3. `decision_card.sizing`, mid-term new money only; never attach it to a long-term or holder answer
4. `decision_modes.core`, `decision_modes.balanced` and `decision_modes.speculative`, with any `capped_by`
5. `dislocation_framework` (its `action` mirrors the card)
6. `verdict`, `action_zones`, `dip_assessment` and the catalyst-rich `news_summary`, as diagnostics, timing and third-party context

When sizing is null, report `decision_card.sizing_status` and its reason; do not infer the reason from the action alone.
Missing sizing does not change the research action. Follow the rendering guide for separate percentage-only and zero-share scenarios, assumptions and conditional loss disclosures.

Never upgrade past `blocked_by`. Keep the response decision-focused and use the exact fields from the tool output instead of inventing a new schema.""",
                }
            ]
        }

    if name == "growth_memo":
        symbol = arguments.get("symbol", "")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Analyze {symbol} as a growth investment.

Use `analyze("{symbol}")` first. Only call secondary tools if something critical is missing.

Then provide:
1. **Thesis** (2-3 sentences): Why this is/isn't a good growth investment
2. **Key Metrics**: Revenue growth, EPS growth, momentum signals, next catalyst
3. **Risks**: Top 3 risks
4. **Decision**: `decision_card.new_money.action` verbatim (mid-term new money: buy, starter, wait, wait_for_data, insufficient_data, watch or avoid) and `decision_card.long_term.new_money.action` (long term), each with its `blocked_by` and the specific reasoning

Be direct. Never upgrade past `blocked_by`.""",
                }
            ]
        }

    if name == "value_memo":
        symbol = arguments.get("symbol", "")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Analyze {symbol} as a value investment.

Use `analyze("{symbol}")` first. Only call secondary tools if valuation support is unclear.

Then provide:
1. **Thesis** (2-3 sentences): Why this is/isn't undervalued
2. **Key Metrics**: P/E, P/B, debt levels, cash flow
3. **Margin of Safety**: Current price vs intrinsic value estimate
4. **Risks**: Top 3 risks
5. **Decision**: `decision_card.new_money.action` verbatim (mid-term new money: buy, starter, wait, wait_for_data, insufficient_data, watch or avoid) and `decision_card.long_term.new_money.action` (long term), each with its `blocked_by` and the specific reasoning

Be direct. Never upgrade past `blocked_by`.""",
                }
            ]
        }

    if name == "position_decision":
        symbol = arguments.get("symbol", "")
        horizon = arguments.get("horizon") or "mid_term"
        call_args = [json.dumps(symbol), f"horizon={json.dumps(horizon)}"]
        for key in ("cost_basis", "purchase_date", "shares", "portfolio_value", "max_holding_pct"):
            value = arguments.get(key)
            if value not in (None, ""):
                call_args.append(f"{key}={json.dumps(value)}")
        call = f"analyze_my_position({', '.join(call_args)})"
        return {
            "messages": [{
                "role": "user",
                "content": f"""Review my holding in {symbol}, using horizon {horizon}.

Use one tool call:
1. {call}

This call supplies the canonical holder assessment and personal context together; do not call analyze or fetch technicals again.
Optional inputs are user-reported USD context; do not ask for missing inputs unless needed. Omitted cost, date, shares, portfolio value or ceiling stay unknown; never invent them. No implicit currency conversion.

Then provide:
1. **Holder assessment**: requested horizon and `position_card.holder_assessment` action verbatim with its basis and reasons (hold, reduce, review_exit, insufficient_data or not_assessed). Hold is not endorsement: preserve the returned suitability fit and its actual reasons, including avoid, caution or unknown.
2. **Allocation check**: label review_status as "Allocation check". It reflects only the supplied ceiling, never changes or replaces the stock action, and is not a stock all-clear. No ceiling means not requested; missing or stale inputs mean not evaluated. Report any personal review trigger before addition eligibility.
3. **Position context**: returned P&L, portfolio weight, price provenance and elapsed days only when available. Show missing-input reasons and historical price qualifications; unknown is not zero. Portfolio value includes this holding and is not buying power.
4. **Addition status**: `position_card.new_money` verbatim with blocked_by; adding requires the holder's add_allowed. Report an exceeded ceiling or unchecked capacity beside it. No remaining add capacity is assessed.

Never borrow the new-money action as a holder answer. For a long-term ETF, say: "This tool does not assess long-term ETF ownership; its company rules do not apply." Do not request company evidence or substitute the mid-term answer. Show any personal allocation check separately.
No tax estimate, holder stop or sale quantity. Technical references are unmonitored diagnostics, not holder exit instructions. Do not attach entry sizing or ATR planning stops to this answer. For a separate mid-term new-money sizing question, follow `stock-analysis://guides/analyze-rendering`; this position call supplies no entry size.

`candidate_admits_new_money` is a pre-gate basis, not permission to add; add_allowed governs additions. Never upgrade past blocked_by.""",
            }],
        }

    return None
