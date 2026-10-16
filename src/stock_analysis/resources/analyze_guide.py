"""On-demand guide resources for analysis rendering."""

ANALYZE_RENDERING_GUIDE = """# Analyze Rendering Guide

Use this guide when presenting the result of the `analyze` tool.
`analyze` returns `detail="standard"` by default; `detail="decision"` is
smaller and `detail="full"` adds the audit sections. Render only the sections
present in the response. The decision card is present at every detail level.

## Order

1. Header: symbol, price, market cap, sector. Give the price's as-of from
   `summary.price_provenance`, and say so when it is `info_quote_fallback`
   or `summary.price_provenance.staleness.stale` is true. When
   `data_quality.freshness.codes` is not empty, say the price or daily history
   could not be verified as current.

2. Decision card (every detail level). Answer "What should I do now?" from
   `decision_card` first, in three labelled horizon segments:
   - New money, mid-term (3-12 months): `decision_card.new_money.action`
     verbatim, with every `decision_card.new_money.blocked_by` id and reason.
   - If held: `decision_card.if_held.action` verbatim, with its `basis`.
     Adding to a holding is allowed only when `decision_card.if_held.add_allowed`
     is true.
   - Long term (1-5 years): `decision_card.long_term.new_money.action`
     verbatim, with its own `decision_card.long_term.new_money.blocked_by`.
   - New-money actions: `buy`, `starter`, `wait`, `wait_for_data`,
     `insufficient_data`, `watch`, `avoid`. Holder actions: `hold`, `reduce`,
     `review_exit`, `insufficient_data`.
   - Never upgrade past `blocked_by`: report each action as given. No
     narrative, zone, dip, mode, rating or news item makes it looser (`wait`
     is never shown as `starter`, nor `starter` as `buy`).
   - `decided_by` names the gate that set the action. List `unevaluated` gates
     as not checked.
   - `decision_card.action_now` is a deprecated alias of
     `decision_card.new_money.action`, kept for one release.
   - Render `why`, `exit`, `exit_triggers`, and `data_flags` when not empty.

3. Reassessment date (`earliest_reassessment`, one per horizon):
   - It is not a buy date; re-run analyze on or after it and act only on the
     new decision. It never changes an action.
   - Dated: "Reassess on or after <Ddd Mon D>. This is not a buy date; re-run
     analyze. Still required: <outstanding clears_when…>."
   - No date: "No reassessment date: re-run analyze after <clears_when…>."
   - Basis `unsupported`: "Outside supported scope: no reassessment date."
   - Never say "now". When the field is null, give no date.
   - `decision_card.next_review` is a monitoring date, not a buy date.

4. Conditions (`decision_card.conditions` for mid-term,
   `decision_card.long_term.new_money.conditions` for long term):
   - They are necessary, not sufficient: meeting them never permits a buy by
     itself; re-run analyze.
   - Under `starter`, `buy_only_if` lists what the full position needs.
   - Render `add_only_if` only for a holder.
   - Render `do_not_buy_if`, `reduce_if`, and `monitor` as written.

5. Executive summary (standard and full):
   - Render `executive_summary` verbatim. It opens with the same three horizon
     segments.
   - Do not paraphrase or rewrite it.
   - If FCF is mentioned, include value and period from
     `fundamentals_summary.cash_flow.free_cash_flow_label`.

6. Position sizing (mid-term new money only):
   - Size only from `decision_card.sizing`.
   - When it is null, say "no entry size: new_money is <action>".
   - Under a mid-term `starter`, show the starter size only; the full-size
     fields are null.
   - There is no long-term or holder size: never attach `decision_card.sizing`
     or `decision_card.entry` to a long-term or holder answer.
   - Dollar and share amounts need `account_size`.
   - Never derive a size from levels, zones, presets, modes, or ratings.
   - `decision_card.entry` is the price band for a new entry; it is null unless
     mid-term new money is `buy` or `starter`.

7. Verdict summary:
   - Tilt and confidence.
   - Decomposed setup, business quality, and risk.
   - Horizon fit and its reasons (full) are diagnostics, not the decision.

8. Diagnostics and timing (context, not advice):
   - `action_zones.price_zone` (`near_52w_low`, `below_sma200`, `above_sma50`,
     `at_or_below_sma50`, `near_52w_high`, `undetermined`; null without a
     price) and `action_zones.levels` describe where price sits. Never call
     them buy or sell zones.
   - Stop-loss distance and basis from `action_zones.stop_calculation`.
   - Valuation gate from `action_zones.valuation_assessment`. For unprofitable
     companies, show the sales-multiple gate (EV/S, else P/S; `unknown` when
     enterprise value is not positive). If the gate is unknown, mention reduced
     valuation confidence.
   - `dislocation_check` (standard): render the five answers; its `decision`
     mirrors the card.
   - Analyst coverage, news, ownership, short interest, and options data are
     third-party: attribute them and never present them as advice.
   - `diagnostics` (full) holds pre-gate inputs, marked
     `pre_gate_input_not_advice`; the card supersedes them (`superseded_by`).
     Never present them as advice.
   - A field with `capped_by` (for example `new_money_avoid`) was capped by the
     decision: show the capped value and name the cap.

9. Dislocation framework (full):
   - Answer:
     - Is it down enough?
     - Is the business still good?
     - Is the balance sheet safe?
     - Is the long-term thesis intact?
     - Has price broken more than business?
   - Use `setup`, `business_integrity`, `balance_sheet_safety`,
     `thesis_integrity`, and `mismatch_verdict`.
   - `dislocation_framework.action` mirrors the card. Its `can_start_now` is
     deprecated and, like `buy_now`, never the source of a decision.
   - Always render `dislocation_framework.action.sell_without_attachment_if`.

10. Decision modes (full):
    - `decision_modes.core` follows the long-term decision (capped by the
      long-term fit), `decision_modes.balanced` equals new money, and
      `decision_modes.speculative` is capped by new money. Show `capped_by`.
    - The profile changes sizing only, never the decision.

11. Dip assessment (full; timing context):
    - Dip classification and explanation.
    - Dip depth from highs/lows.
    - Oversold composite and key components.
    - Closest support levels.
    - Volume signal and interpretation.
    - Bounce potential and dip confidence.
    - The recommendation and entry signals are timing context, never a buy
      call; when one carries `capped_by`, say the decision capped it.
    - `dip_assessment.entry_timing.wait_for` lists timing conditions.

12. Horizon drivers (full):
    - Render `decision_context.horizon_drivers` only if non-empty.
    - Format as `[horizon] gate: reason`.

13. Score math (full):
    - Use `verdict.score_display.formula`.
    - Use `verdict.score_display.component_breakdown`.
    - Keep 6-decimal precision where the payload does.
    - Show component exclusions when coverage is below 1.0.

14. Confidence path (full):
    - Current blockers.
    - Upgrade conditions that are not yet satisfied.
    - Downgrade conditions.

15. Top drivers (full):
    - Use `decision_context.top_triggers`.
    - Show category, direction, reason, and `score_delta`.
    - For neutral tilt, show top two bearish plus top one bullish trigger.
    - For bullish tilt, still show the top bearish driver for balance.
    - Do not show duplicate drivers from the same category.

16. Key metrics:
    - P/E or P/S with source.
    - FCF using `free_cash_flow_label`.
    - Cash runway with basis.
    - Quarterly FCF/OCF burn when present.
    - Beta, volatility, drawdown.

17. Decision context (full):
    - Render categories separately: fundamentals, valuation, risk, technicals,
      news, and next catalyst.
    - For unprofitable companies, show path-to-profitability and cash runway
      checkpoints with current values.

18. Market context (full):
    - SPY trend.
    - Provenance.
    - Sanity warnings when present.

19. News:
    - Recent headlines when available, attributed to their source.

## Field classes

| Class | Meaning | Examples |
|---|---|---|
| A | Actionable: only from the decision, with its horizon | `decision_card.new_money.action`, `decision_card.if_held.action`, `decision_card.long_term.new_money.action` |
| S | Sizing, mid-term new money only: present only when it is `buy` or `starter`; never attach it to a long-term or holder answer | `decision_card.sizing`, `decision_card.entry` |
| C | Conditional: predicates, necessary, not sufficient | `decision_card.conditions`, `decision_card.new_money.earliest_reassessment` |
| D | Diagnostic: describes state, never advice | `action_zones.price_zone`, `dislocation_check.questions`, `verdict.score` |
| N | Our narrative, three horizon segments first | `executive_summary` |
| P | Pre-gate input (full only), superseded by the card | `diagnostics` |
| T | Third-party data, attributed, never advice | `analyst_coverage`, `news_summary` |
| E | Static profile labels | `decision_modes.balanced.label` |

## Unprofitable Companies

- Show the sales multiple (EV/S, else P/S) instead of P/E.
- Show burn metrics: liquidity, runway, burn rates, and dilution risk.
- If burn metrics are unavailable, show the status reason.
- Emphasize path-to-profitability triggers in fundamentals.
"""


def read_analyze_rendering_guide() -> tuple[str, str]:
    """Return the analyze rendering guide and MIME type."""
    return ANALYZE_RENDERING_GUIDE, "text/markdown"
