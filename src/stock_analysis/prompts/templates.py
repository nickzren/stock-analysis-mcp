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
                    "content": f"""Analyze {symbol}.

Execute these tools in order:
1. get_stock_summary("{symbol}")
2. get_news("{symbol}")
3. get_technicals("{symbol}")
4. get_fundamentals("{symbol}")
5. get_events("{symbol}")

Then return your analysis as a JSON object with this exact structure:

```json
{{
  "symbol": "{symbol}",
  "analysis_date": "YYYY-MM-DD",
  "company": {{
    "name": "...",
    "sector": "...",
    "industry": "...",
    "market_cap": "...",
    "description": "1-2 sentence summary"
  }},
  "price": {{
    "current": 0.00,
    "change_1d": 0.00,
    "change_1m": 0.00,
    "change_ytd": 0.00,
    "vs_52w_high": 0.00,
    "vs_52w_low": 0.00
  }},
  "news_summary": {{
    "sentiment": "positive|neutral|negative|mixed",
    "key_themes": ["theme1", "theme2"],
    "notable_events": ["event1", "event2"]
  }},
  "technicals": {{
    "trend": "bullish|bearish|neutral",
    "signals": {{
      "bullish": ["signal1", "signal2"],
      "bearish": ["signal1", "signal2"]
    }},
    "support_levels": [0.00, 0.00],
    "resistance_levels": [0.00, 0.00]
  }},
  "fundamentals": {{
    "valuation": "undervalued|fair|overvalued",
    "pe_ratio": 0.00,
    "peg_ratio": 0.00,
    "profit_margin": 0.00,
    "revenue_growth": 0.00,
    "debt_to_equity": 0.00
  }},
  "risk": {{
    "level": "low|medium|high",
    "volatility_annual": 0.00,
    "beta": 0.00,
    "max_drawdown": 0.00,
    "var_95": 0.00
  }},
  "events": {{
    "next_earnings": "YYYY-MM-DD or null",
    "days_to_earnings": 0,
    "dividend_yield": 0.00
  }},
  "thesis": {{
    "summary": "2-3 sentence investment thesis",
    "bull_case": "Key reason to be bullish",
    "bear_case": "Key reason to be bearish",
    "catalysts": ["catalyst1", "catalyst2"]
  }},
  "verdict": {{
    "action": "BUY|HOLD|SELL|WATCH",
    "confidence": "high|medium|low",
    "reasoning": "1-2 sentence justification"
  }}
}}
```

IMPORTANT:
- Use null for any unavailable data
- Round numbers to 2 decimal places
- Keep text fields concise
- The JSON must be valid and parseable
- Do not add any fields not in the schema above""",
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

Use these tools in order:
1. get_stock_summary("{symbol}")
2. analyze("{symbol}")

Then provide:
1. **Thesis** (2-3 sentences): Why this is/isn't a good growth investment
2. **Key Metrics**: Revenue growth, EPS growth, momentum signals
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

Use these tools in order:
1. get_stock_summary("{symbol}")
2. analyze("{symbol}")

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
