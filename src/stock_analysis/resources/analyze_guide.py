"""On-demand guide resources for analysis rendering."""

ANALYZE_RENDERING_GUIDE = """# Analyze Rendering Guide

Use this guide when presenting the result of the `analyze` tool.
This guide assumes `detail="full"`; with `detail="standard"`, render only
sections whose fields are present.

## Order

1. Header: symbol, price, market cap, sector.

2. Executive summary:
   - Render `executive_summary` verbatim.
   - Do not paraphrase or rewrite it.
   - If FCF is mentioned, include value and period from
     `fundamentals_summary.cash_flow.free_cash_flow_label`.

3. Dislocation framework:
   - Answer:
     - Is it down enough?
     - Is the business still good?
     - Is the balance sheet safe?
     - Is the long-term thesis intact?
     - Has price broken more than business?
     - What should I do now?
   - Use `setup`, `business_integrity`, `balance_sheet_safety`,
     `thesis_integrity`, `mismatch_verdict`, and `action`.
   - If `action.can_start_now` is true, say a small starter is acceptable now
     and render `action.add_only_if`.
   - Otherwise render `action.buy_only_if`.
   - Always render `action.sell_without_attachment_if`.

4. Verdict summary:
   - Tilt and confidence.
   - Decomposed setup, business quality, and risk.
   - Horizon fit with reasons.

5. Horizon drivers:
   - Render `decision_context.horizon_drivers` only if non-empty.
   - Format as `[horizon] gate: reason`.

6. Score math:
   - Use `verdict.score_display.formula`.
   - Use `verdict.score_display.component_breakdown`.
   - Keep 6-decimal precision where the payload does.
   - Show component exclusions when coverage is below 1.0.

7. Confidence path:
   - Current blockers.
   - Upgrade conditions that are not yet satisfied.
   - Downgrade conditions.

8. Top drivers:
   - Use `decision_context.top_triggers`.
   - Show category, direction, reason, and `score_delta`.
   - For neutral tilt, show top two bearish plus top one bullish trigger.
   - For bullish tilt, still show the top bearish driver for balance.
   - Do not show duplicate drivers from the same category.

9. Signals:
   - Bullish pros.
   - Bearish cons.

10. Key metrics:
    - P/E or P/S with source.
    - FCF using `free_cash_flow_label`.
    - Cash runway with basis.
    - Quarterly FCF/OCF burn when present.
    - Beta, volatility, drawdown.

11. Action zones:
    - Current zone and valuation gate.
    - Price levels with distances.
    - Stop-loss distance and basis.
    - For unprofitable companies, show P/S-based valuation gate.
    - If valuation gate is unknown, mention reduced valuation confidence.

12. Dip assessment:
    - Dip classification and explanation.
    - Dip depth from highs/lows.
    - Oversold composite and key components.
    - Closest support levels.
    - Volume signal and interpretation.
    - Bounce potential.
    - Entry signals and wait-for conditions.
    - Dip confidence.
    - Overall assessment and `add_only_if` when present.

13. Position sizing:
    - Percentage range.
    - Dollar range when account size is available.
    - Share range at current price.
    - Stop-implied max size when available.

14. Decision context:
    - Render categories separately: fundamentals, valuation, risk, technicals,
      news, and next catalyst.
    - For unprofitable companies, show path-to-profitability and cash runway
      checkpoints with current values.

15. Market context:
    - SPY trend.
    - Provenance.
    - Sanity warnings when present.

16. News:
    - Recent headlines when available.

## Unprofitable Companies

- Show P/S instead of P/E.
- Show burn metrics: liquidity, runway, burn rates, and dilution risk.
- If burn metrics are unavailable, show the status reason.
- Emphasize path-to-profitability triggers in fundamentals.
"""


def read_analyze_rendering_guide() -> tuple[str, str]:
    """Return the analyze rendering guide and MIME type."""
    return ANALYZE_RENDERING_GUIDE, "text/markdown"
