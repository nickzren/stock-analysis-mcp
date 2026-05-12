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
        "description": "Sell/hold analysis for existing position",
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
1. `executive_summary`
2. `dislocation_framework`
3. `policy_action`
4. `decision_modes.core`
5. `decision_modes.balanced`
6. `decision_modes.speculative`
7. `verdict`, `action_zones`, `dip_assessment`, and the catalyst-rich `news_summary`

Keep the response decision-focused and use the exact fields from the tool output instead of inventing a new schema.""",
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
4. **Action**: Buy / Wait / Pass with specific reasoning

Be direct. No hedging.""",
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
5. **Action**: Buy / Wait / Pass with specific reasoning

Be direct. No hedging.""",
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
                    "content": f"""Help me decide whether to hold or sell my position in {symbol}.

Position details:
- Cost basis: ${cost_basis}
- Purchase date: {purchase_date}

Use these tools:
1. analyze_my_position("{symbol}", cost_basis={cost_basis}, purchase_date="{purchase_date}")
2. get_technicals("{symbol}")

Then provide:
1. **Position Status**: Current P/L and holding period
2. **Tax Implications**: Short-term vs long-term status and impact
3. **Technical Signals**: Current sell signals active
4. **Recommendation**: HOLD / SELL / TRIM with specific reasoning
5. **If Holding**: Key levels to watch for potential exit
6. **If Selling**: Suggested execution approach

Be decisive. Give a clear recommendation.""",
                }
            ]
        }

    return None
