"""On-demand guide resources for analysis rendering."""

ANALYZE_RENDERING_GUIDE = """# Analyze Rendering Guide

Use this guide when presenting the result of the `analyze` tool.
`analyze` returns `detail="standard"` by default; `detail="decision"` is
smaller and `detail="full"` adds the audit sections. Render only the sections
present in the response. The decision card is present at every detail level.
12 months / one year means mid-term unless long term is explicit.
"Analyze HOOD" and a bare "I own X; hold or sell?" use the labelled mid-term
answer without a question. "Hold for five years" selects long term without a
question. Ask one focused clarification only for conflicting or unsupported horizons.
Stock-only holder questions use the matching analyze holder field below. Personal
context questions use analyze_my_position(horizon=...) once, not analyze followed
by a second analysis; its inputs are optional.

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
   - If held, mid-term: `decision_card.if_held.action` verbatim, with its `basis`.
     Adding to a holding is allowed only when `decision_card.if_held.add_allowed`
     is true.
   - Long term (1-5 years): `decision_card.long_term.new_money.action`
     verbatim, with its own `decision_card.long_term.new_money.blocked_by`, then
     `decision_card.long_term.if_held.action` verbatim with its basis and reasons.
     Label new money and holder separately; neither implies the other. Long-term
     adding requires `decision_card.long_term.if_held.add_allowed`; that is entry
     eligibility, not personal capacity.
   - New-money actions: `buy`, `starter`, `wait`, `wait_for_data`,
     `insufficient_data`, `watch`, `avoid`. Holder actions: `hold`, `reduce`,
     `review_exit`, `insufficient_data` for mid-term. Long-term holder actions:
     `hold`, `review_exit`, `insufficient_data`, `not_assessed`.
   - Never upgrade past `blocked_by`: report each action as given. No
     narrative, zone, dip, mode, rating or news item makes it looser (`wait`
     is never shown as `starter`, nor `starter` as `buy`).
   - `decided_by` names the gate that set the action. List `unevaluated` gates
     as not checked.
   - `decision_card.action_now` is a deprecated alias of
     `decision_card.new_money.action`, kept for one release.
   - Render `why` and `data_flags` when not empty. Existing `exit`, `exit_triggers`
     and reduce/add conditions are mid-term; never relabel them as long-term exits.

   Mid-term holder `basis` explains the action, not an instruction to add:
   - `candidate_admits_new_money` describes the pre-gate candidate, not permission
     to add; `add_allowed` still governs additions.
   - `runway_critical` / `business_broken`: established thesis failure.
   - `thesis_unknown`: required thesis evidence is unavailable or unverified.
   - `price_broken_more_than_business`: the dislocation classifier still supports
     holding. `tilt_bearish` is the residual bearish-score rule.
   - `tilt_missing` / `candidate_insufficient`: too little evidence for a holder
     decision. `no_reduction_evidence` means no reduction rule fired, not proof
     that the investment is safe.

   Long-term holder coverage and qualifications:
   - `not_assessed` means the known instrument is outside the company rubric,
     not that company data are missing. For SPY or VTI: "This tool does not assess
     long-term ETF ownership; its company rules do not apply." Do not ask for ETF
     cash runway or borrow the mid-term holder/long-term new-money answer.
   - `insufficient_data` means required applicability or business/runway evidence
     cannot be assessed. Show checks, reasons, watch_factors and review_when;
     missing evidence is never a pass. A known failure can coexist with an unknown.
   - `review_exit` prompts review of a demonstrated thesis failure, not an order.
   - `hold` means no demonstrated thesis failure under these checks, not endorsement.
     Preserve caution/avoid/unknown horizon fit with actual reasons. For example:
     "The checks have not demonstrated a thesis failure, but the long-term suitability
     assessment is avoid because [actual reasons]. This is not an endorsement to
     keep or add without considering those concerns." In analyze use the reasons
     from verdict.horizon_fit.reasons for the long-term fit; in a position answer use the actual
     reasons from suitability.reasons. Do not replace them with a generic warning.
   - Mid-term `thesis_unknown` and long-term `applicability_unverified` can describe
     the same missing inputs under different horizon rules; they are not contradictory
     provider observations. Do not rename the returned basis.

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
   - Render `add_only_if` only for a mid-term holder.
   - Render `do_not_buy_if`, `reduce_if`, and `monitor` as written and with their
     returned horizon; never turn mid-term reduce_if into a long-term exit rule.

5. Executive summary (standard and full):
   - Render `executive_summary` verbatim. It opens with the same three horizon
     segments; the long-term segment separates new money from ownership.
   - Do not paraphrase or rewrite it.
   - If FCF is mentioned, include value and period from
     `fundamentals_summary.cash_flow.free_cash_flow_label`.

6. Position sizing (mid-term new money only):
   - Size only from `decision_card.sizing`.
   - When sizing is null, report `decision_card.sizing_status` and its reason;
     do not infer the reason from the action alone. `not_allowed` names the
     action and applicable blockers; `unavailable` names the missing or invalid
     input, or `numeric_range_unavailable` when the estimate cannot be represented
     without losing numerical precision. For example: "No size estimate: a valid planning stop is unavailable."
     Missing sizing does not change the research action. Keep the returned
     new-money, holder and long-term actions; never invent a size or imply zero risk.
   - Under a mid-term `starter`, show the starter size only; the full-size
     fields are null. Sizing requires an allowed entry and valid inputs.
   - There is no long-term or holder size: never attach `decision_card.sizing`
     or `decision_card.entry` to a long-term or holder answer.
   - Beside the size show `decision_card.sizing.assumptions.entry_price` (E),
     `decision_card.sizing.assumptions.stop_price` (S),
     `decision_card.sizing.assumptions.price_provenance` and
     `decision_card.sizing.assumptions.stop_basis`. The ATR planning reference is
     recomputed per request; it is not a saved holder stop and does not apply
     automatically to every price in the reference band.
   - Show effective budgets from `decision_card.sizing.limits`, including
     `risk_origin` and `max_position_origin` (caller, profile or default).
     `max_position_pct` caps this standalone allocation. Existing holdings,
     outstanding orders and available buying power are unknown. This estimate is
     never remaining add capacity or an enforced aggregate portfolio cap.
   - Render `decision_card.sizing.starter` and `decision_card.sizing.full`
     separately when present. Check each scenario independently: one may fit a
     whole share while the other does not. Show `allocation_cap_pct` as a ceiling,
     plus `binding_constraints`; never call a ceiling an actual filled position.
     For `available`, use that scenario's `whole_shares`, `position_dollars`,
     `position_pct`, `planned_loss_at_stop` and `effective_risk_pct`:
     "Position: $P (Y% of the account); estimated loss: X% of the account if sold at $S."
     X is effective risk after whole-share rounding, not the input budget.
   - For every non-null sizing result state:
     "This planning stop is not monitored or an exit instruction."
     "Estimated loss if sold at this stop price. Gaps and slippage can make the actual loss larger."
     Never promise a maximum loss. Shortened numbers need `~`; never round a cap
     upward or hide a positive amount as zero. Use scientific notation when needed.
   - `percentage_only`: show allocation and hypothetical loss ceilings
     (`allocation_cap_pct`, `ceiling_risk_pct`), conditional on sale at S.
     Account size was not provided: do not invent shares, dollars or a filled position.
   - `below_one_whole_share`: no whole-share position fits this budget.
     The scenario's shares and derived dollars/loss are zero; preserve its research
     action and do not imply an affordable order. Never invent fractional-share support.
     Percentage-only and zero-share results are non-null scenarios, not missing sizing.
   - Never derive a size from levels, zones, presets, modes, or ratings.
   - `decision_card.entry` is a reference band; it is null unless
     mid-term new money is `buy` or `starter`.
     If `entry_band_unavailable` is non-null, report its reason: both bounds are null,
     independently of the reference price and sizing availability. For a valid size,
     use the returned action and `decision_card.sizing.at_price`, even outside that band
     (the same E). Never invent an entry-price gate from the band. When sizing is null, use
     `decision_card.entry.current_price` as the reference only if returned.
     If the reference is unavailable, say so; never invent a size or stop.

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
   - `dislocation_framework.action.sell_without_attachment_if` is mid-term
     diagnostic context, not the long-term holder answer or an exit instruction.

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

## Position answers

Use analyze_my_position once when personal context matters. Lead with the requested
horizon and position_card.holder_assessment action/reason, then any personal review
trigger, then separate addition eligibility. Cost, purchase date, shares, portfolio
value and ceiling are optional; do not request data irrelevant to the question.
The position response supplies its own canonical assessment; do not run analyze again.

| Field/status | Rendering rule |
|---|---|
| position_card.review_status | Label "Allocation check", not a stock all-clear. It depends only on concentration_check, never the holder action. |
| review_allocation | Above the user's stated ceiling: show the actual weight and ceiling beside addition eligibility. |
| no_allocation_trigger | At or below the supplied ceiling, not an endorsement of the stock or permission to add. |
| not_evaluated | A requested allocation check lacks usable inputs/current price; capacity was not checked. |
| not_requested | No ceiling supplied; no default ceiling or conclusion. |
| context_status | Describes personal inputs only; inconsistent inputs are not a new stock decision. |

The holder action stands even when allocation has no trigger. Stock add_allowed can
be true while a ceiling is exceeded: show both, never claim remaining add capacity.
If another portfolio tool supplies fixed concentration flags, your own ceiling governs
this single-position check; do not substitute those fixed thresholds.

Personal numbers require verified USD per-share equity/ETF units. No implicit currency
conversion. Cost and shares are user-reported on the current split-adjusted basis.
A known ETF can have P&L and allocation numbers while its long-term ownership is
not_assessed. Unknown units suppress dependent values with calculation_reasons.
A cost basis of zero permits dollar P&L but no return fraction. Missing values stay
unknown, never zero; display each reason. The portfolio total includes this holding,
not buying power. A holding larger than that total is inconsistent; don't clamp it.
Stale or fallback prices are historical context: label position.price_context.as_of and provenance and
never report a current allocation check from them. No reconciliation is promised.

Tax is not_assessed: no rates, tax-saving estimates, legal long-term category or tax
recommendation. Supplied purchase date and elapsed days are descriptive only.
Technical references are unmonitored diagnostics, not holder exit instructions.
No ATR planning stop, entry size or sale quantity belongs to a long-term holder
answer. diagnostics.technicals retains technical context without a competing sell
count. Entry sizing remains the separate mid-term contract above.

## Field classes

| Class | Meaning | Examples |
|---|---|---|
| A | Actionable: only from the decision, with its horizon | `decision_card.new_money.action`, `decision_card.if_held.action`, `decision_card.long_term.new_money.action`, `decision_card.long_term.if_held.action` |
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
