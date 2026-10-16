"""Prompt templates for stock analysis."""

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
        "description": "Holder decision for an existing position (analyze's if_held, with P/L and tax)",
        "arguments": [
            {"name": "symbol", "required": True},
            {"name": "cost_basis", "required": True},
            {"name": "purchase_date", "required": True},
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
1. `decision_card`: `decision_card.new_money` (mid-term) and `decision_card.long_term.new_money`, each action verbatim with its `blocked_by`, then `decision_card.if_held` action verbatim with its `basis`
2. `executive_summary`, verbatim
3. `decision_card.sizing`, mid-term new money only (when null: no entry size); never attach it to a long-term or holder answer
4. `decision_modes.core`, `decision_modes.balanced` and `decision_modes.speculative`, with any `capped_by`
5. `dislocation_framework` (its `action` mirrors the card)
6. `verdict`, `action_zones`, `dip_assessment` and the catalyst-rich `news_summary`, as diagnostics, timing and third-party context

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
        cost_basis = arguments.get("cost_basis", "")
        purchase_date = arguments.get("purchase_date", "")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Help me decide what to do with my position in {symbol}.

Position details:
- Cost basis: ${cost_basis}
- Purchase date: {purchase_date}

Use these tools:
1. analyze("{symbol}")
2. analyze_my_position("{symbol}", cost_basis={cost_basis}, purchase_date="{purchase_date}")

Then provide:
1. **Decision**: `decision_card.if_held.action` verbatim (hold, reduce, review_exit or insufficient_data) with its `basis`; adding is allowed only when `decision_card.if_held.add_allowed` is true
2. **Position Status**: Current P/L and holding period
3. **Tax Implications**: Short-term vs long-term status and impact
4. **Exit Plan**: `decision_card.exit`, `decision_card.exit_triggers` and `decision_card.conditions.reduce_if`
5. **New Money**: `decision_card.new_money.action` with its `blocked_by`, for any addition

The decision comes from analyze, not from P/L, tax or technical signals alone. Report it verbatim.""",
                }
            ]
        }

    return None
