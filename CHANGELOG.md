# Changelog

## 0.6.0 (2026-10-22): Python 3.12 minimum

**Package:** `meta.server_version` goes from `0.5.2` to `0.6.0`.
**Python:** the minimum rises from 3.11 to **3.12**; Python 3.11 is no longer
supported. Response schema `"8"` and stock snapshot `"4.0.0"`
are unchanged; there is no intended analysis, decision or sizing behavior change.

- Target Python 3.12 in package metadata, Ruff and mypy. Internal generic helpers
  use Python 3.12 type-parameter syntax; their annotation metadata changes, while
  application behavior is preserved.
- Keep all retained locked dependency versions unchanged. Remove only the
  Python-3.11-only dependency and artifacts that cannot serve supported versions.
- Python 3.11 users can keep their Git install pinned to `v0.5.2`, the last release
  declaring 3.11 support; its runtime compatibility on 3.11 was not verified.
  Under default settings, `uvx` can select or download a compatible interpreter.
  Changing a live MCP registration remains a separate step.

## 0.5.2 (2026-10-21): dependency compatibility

**Package:** `meta.server_version` goes from `0.5.1` to `0.5.2`.
Response schema `"8"` and stock snapshot `"4.0.0"` are unchanged. No decision
policy or dependency bounds change. Inputs that previously failed on a missing
vendor volume, or were reported as `invalid_symbol` for a non-symbol failure,
can now return different results.

- Fix daily history with missing vendor volume under pandas 3 by combining
  masks without modifying a read-only array.
- Preserve the five existing symbol/no-data cases as `invalid_symbol` using a
  dedicated exception. Other structured fetch failures report
  `data_unavailable`, and analyze honors those error types even when their
  messages resemble a missing symbol. Legacy hints for raw exceptions and the
  company-name search error path remain unchanged.
- Make transport and MCP metadata tests work across supported dependency
  versions while retaining checks for error rejection, fresh retries and cache
  cleanup. Add a manual fresh-resolution release check alongside the locked
  suite; failed checks remain recorded as failures.

## 0.5.1 (2026-10-21): maintenance

**Package:** `meta.server_version` goes from `0.5.0` to `0.5.1`.
Response schema `"8"` and stock snapshot `"4.0.0"` are unchanged. Decisions,
gates, sizing, holder rules and snapshots are unchanged.

- Populate the insider and institutional ownership fractions from yfinance's
  indexed `major_holders` layout. Values remain unscaled, including fractions
  above 1; invalid values remain null with warnings. Legacy two-column
  percentage data is still supported. `insider_activity.insider_pct` is now
  always present in `get_ownership` and analyze's `ownership_flow` (null when
  unknown); previously the key was absent. These are third-party context, not
  decision inputs.
- Remove false verdict invariant warnings when scoring coverage is unavailable,
  retaining diagnostic component scores and all existing analysis output.
- Normalize invalid internal holder-check states to `not_evaluated`, preserving
  their evidence and the existing decision rules.
- Omit package-version labels from current holder recordings and compare client
  metadata with the running version, so package bumps do not churn fixtures.
  Frozen baselines, response schemas and snapshot version pins are unchanged.
- Update the locked NumPy from yanked 2.4.0 to 2.4.6. No other dependency
  changes; `pyproject.toml` still requires `numpy>=1.24.0`.

## 0.5.0 (2026-10-20): longer-term holders and position context

**Package:** `meta.server_version` goes from `0.4.0` to `0.5.0`.
**Response schema:** `meta.schema_version` goes from `"7"` to `"8"`.
**Stock snapshot:** `snapshot_version` goes from `"3.0.0"` to `"4.0.0"`.
Mid-term holder actions, both new-money decisions, gates and entry sizing are
unchanged.
The schema-7 sizing fields and compatibility aliases remain in place.

### Long-term ownership and routing

`decision_card.long_term.if_held` is a separate 1–5-year ownership assessment,
identical at every response tier. It does not reuse long-term entry permission
as an exit rule. Applicability is checked before company evidence:

| `action` | Meaning |
|---|---|
| `not_assessed` | Known instrument outside the company rubric, including ETFs and preferred shares. This is a coverage boundary, not missing company data. |
| `insufficient_data` | Classification is unverified, or required business/runway evidence is unavailable with no demonstrated failure. |
| `review_exit` | Applicable company has cash runway below four quarters or a broken business under the existing rules. A known failure wins over another missing check. |
| `hold` | Available checks have not demonstrated a thesis failure. This is not an endorsement or a promise of returns; retain known concerns and the separate suitability assessment. |

`basis` is the first matching reason; `reasons` keeps other failures and missing
checks. `assessment_status` is `evaluated`, `not_evaluated` or `not_applicable`.
`checks.business` and `checks.cash_runway` retain `state`, `classification`,
`reason`, `evidence`, `periods` and `periods_reason`. Financial periods belong to
each input; unknown periods stay unknown. `watch_factors` lists known concerns
with evidence; `review_when` states what deserves another look. `add_allowed`
follows long-term new money only, not personal capacity.

The `checks` keys are `business` and `cash_runway`. Evaluated business labels are
`intact`, `mixed`, `broken`; runway labels are `adequate`, `watch`, `critical`.
Unknown labels do not pass a check. Existing non-burner/runway exemptions remain;
they do not assert positive cash flow. Neither chart signals, entry gates,
valuation, the purchase price nor profile selection triggers a long-term exit.

“Analyze HOOD” and a bare holder question keep the labelled 3–12-month default.
One year/12 months defaults to mid-term unless long term is explicit; “hold for
five years” selects long term without a question. Stock-only holder questions
use the matching `analyze` field. With personal context, use
`analyze_my_position(horizon=...)` once, without a preceding duplicate analysis.
Contradictory or unsupported horizons still need clarification.

The executive summary keeps three segments; its long-term segment now separates
new money from ownership. Preserve a caution/avoid/unknown long-term fit with
its actual reasons beside `hold`. For ETFs: “This tool does not assess long-term
ETF ownership; its company rules do not apply.” Do not borrow the mid-term
answer or ask for company cash-runway data. Mid-term `thesis_unknown` and
long-term `applicability_unverified` can describe the same unavailable input:
the long-term policy checks applicability first and names that narrower cause.

### Position inputs and calculations

`analyze_my_position` keeps the existing argument names/order, makes
`cost_basis`, `purchase_date` and `shares` optional, and adds `horizon`
(`mid_term` default, `long_term`), `portfolio_value` and `max_holding_pct`.

- Monetary inputs are USD. `portfolio_value` is the caller's total portfolio
  **including this holding**, not cash or buying power. Cost and shares describe
  the current split-adjusted position; no broker, lot or corporate-action
  reconciliation is performed. Do not silently convert non-USD amounts.
- Numeric inputs accept JSON numbers or decimal strings, including scientific
  notation. Cost must be finite and nonnegative; genuine zero is accepted.
  Shares and portfolio value must be finite and positive; the holding ceiling
  is in `(0, 100]` percentage points. Reject booleans before framework coercion,
  malformed strings, NaN/infinity, overflow and positive underflow to zero.
  Omitted/null optional context is accepted. FastMCP serializes Python
  NaN/infinity as JSON null before sending; text `"nan"`/`"inf"` is rejected.
- Validate numeric/horizon inputs before data access. Bad dates return
  `invalid_parameters` before analysis; the date must be a real `YYYY-MM-DD`,
  no later than today in Eastern Time. `holding.days_held` counts elapsed
  calendar days, never a tax category.
- The tool runs one analysis and uses its chosen price/provenance for both the
  stock assessment and personal arithmetic. This adds source work compared with
  the former technicals-only call; unchanged latency is not promised. Technical
  diagnostics use the already-fetched input, with no second price fetch.
- Personal amounts require verified USD per-share `EQUITY`/`ETF` units. This
  permits ETF, preferred-share and USD ADR arithmetic independently of company
  assessment or entry eligibility. Unknown units or non-USD prices suppress
  dependent amounts with reasons, not an invented exchange rate.
- Gain/loss is price-only: no dividends, fees, tax or realized trades. Zero cost
  permits dollar gain/loss but no return fraction. Missing inputs null only
  dependent values. Stale/fallback prices may show labelled historical amounts,
  but never evaluate the allocation check. Unrepresentable calculations return
  null with `numeric_range_unavailable`.
- Allocation compares unrounded values with the caller's ceiling; equality
  meets it. A holding above the supplied portfolio total is inconsistent, not
  clamped to 100%. There is no default ceiling, remaining add capacity, trim
  amount, long-term size, monitoring or saved holding state.

### Schema 7 → 8 response inventory

**Analyze:** add the complete `decision_card.long_term.if_held` object described
above. Standard and decision tiers now retain `verdict.horizon_fit` with its
actual reasons, so a qualified holder answer does not lose suitability concerns.
The executive-summary long-term segment now includes that assessment
and any caution/avoid/unknown fit with its reasons, except for `not_assessed`
coverage answers. Those explain the coverage limit; company suitability remains
in structured diagnostics. Existing mid-term exits and
technical levels retain their horizon; they do not become long-term stops.

**Position:** the following blocks are new or changed:

| Path / block | Schema 8 meaning |
|---|---|
| `position_card` | `horizon`, `holder_assessment` and `new_money` copy the requested canonical horizon. `context_status`, `review_status`, `concentration_check` and `addition_context` are personal context only. |
| `context_status` | `{status, reasons}`: `available`, `partial`, `not_supplied` or `inconsistent`; it never changes the stock decision. |
| `concentration_check` | `{status, weight_pct, ceiling_pct, threshold_basis, reasons}`. `threshold_basis` is `user_supplied` or null; there is no profile ceiling. |
| `review_status` | Label as **Allocation check**: `above_limit` maps to `review_allocation`, `within_limit` to `no_allocation_trigger`; `not_evaluated` and `not_requested` remain as named. A stock `reduce`/`review_exit` never becomes an allocation all-clear. |
| `addition_context` | Keeps addition eligibility separate from an exceeded personal ceiling or an unevaluated requested allocation check. No remaining buying capacity is implied. |
| `position` | Existing `cost_basis`, `current_price`, `shares`, `current_value`, `gain_loss` (fraction) and `gain_loss_dollars` now use the selected analysis price and validated units. Add `portfolio_value`, `max_holding_pct`, `portfolio_weight_pct` (percentage points), `calculation_reasons`, `price_provenance` and `price_context`. |
| `price_context` | `{status, as_of, qualification}` distinguishes `current`, `historical`, `unavailable`; these are research prices, not verified account values. |
| `holding` | `purchase_date` and `days_held` are optional descriptive context. |
| `assumptions` | Lists user-reported USD/split-adjusted units, portfolio-total meaning, price-only P&L and allocation limitations. |
| `suitability` | `{horizon, fit, reasons, note}` retains the requested horizon's fit and actual reasons independently of the holder action. |
| `data_quality` | Carries the shared analysis disclosures, alongside its `data_provenance`. |
| `tax.status`, `tax.reason` | `not_assessed` and an explanation; no personalized tax or sale-quantity estimate. |

If a portfolio tool is also shown, its generic 20% single-position / 30% sector
flags remain separate. The position tool's explicitly supplied ceiling governs
this allocation check; neither tool's flags override the other's threshold.

**Removed or relocated position paths:**

| Previous path | Replacement / reason |
|---|---|
| `current_signals` | `diagnostics.technicals.current_signals`; diagnostic labels only. |
| `sell_signals` | Boolean/null flags move to `diagnostics.technicals.flags`, including actual bearish MACD evidence. |
| `sell_signals.active_count` | Removed; no competing technical sell score. |
| `support_levels` | `diagnostics.technicals.reference_levels` (`sma_50`, `sma_200`, `atr_1x_below`, `recent_low_1m`), with `quote_currency` and a diagnostic note. These references are not monitored and are not holder exit instructions. |
| `holding.is_long_term`, `holding.days_to_long_term` | Removed; elapsed days do not establish tax treatment. |
| `tax.type`, `tax.rate_if_sold_now`, `tax.rate_if_long_term`, `tax.tax_on_gain_now`, `tax.tax_if_waited`, `tax.potential_savings` | Removed; use `tax.status`/`tax.reason`. The fixed rates and waiting-for-tax-savings advice are retired. |

The position prompt's personal inputs are optional. It leads with the requested
holder action/reason, then the Allocation check and separate addition status;
no tax details or a second analyze call are required. Mid-term new-money sizing
remains the only analyze size: do not attach its ATR planning stop to ownership.

### Snapshot 3.0.0 → 4.0.0

Only `long_term_if_held_action` is added: `hold`, `review_exit`,
`insufficient_data`, `not_assessed`, or null when unavailable. The new key and
version appear together. Its value enters the hash; personal inputs, P&L,
allocation status and holder reasons do not. `if_held_action` remains mid-term,
and `long_term_action` remains long-term **new money**. The old fields' enums
and all released field sets stay frozen.

- **4→4:** holder appearance/disappearance/change is material, including evidence
  loss/recovery. A `not_assessed` transition is described as coverage/action
  change, not recovered financial data or proven investment deterioration.
- **3→4:** compare the existing fields with 3.0.0 semantics, including its three
  actions, `price_zone` and `unevaluated_gate_ids`. The new holder field is
  `not_comparable`, not a phantom change. One warning explains the addition;
  a version-only change is not material.
- **1.x/2→4:** retain the earlier restrictions on action/zone comparisons,
  pre-2.0.0 signals and unevaluated gates; the new holder field is also
  `not_comparable`. Do not treat these as the compatible 3→4 pair.
- A complete 4.0.0 snapshot requires the new key and its valid enum/null before
  analysis. Older validators keep their own rules. All detail levels share one
  snapshot; `detect_changes` returns it unchanged. Ordinary clock/age exclusions
  remain. A reason-only change without a tracked action/gate change is not
  promised as an alert.

**Rollback:** v0.4.0 cannot read 4.0.0 snapshots. Preserve older snapshots or
start a new baseline after rollback; **never relabel** a 4.0.0 snapshot as 3.0.0.
Upgrade the MCP configuration only after a verified v0.5.0 tagged install. Saved
holdings, anchored holder stops, bank/REIT models and swing-trade semantics remain
outside this change.

## 0.4.0 (2026-10-19): sizing and caps

The package version (`meta.server_version`) goes from `0.3.0` to `0.4.0`.
**Response schema:** `meta.schema_version` goes from `"6"` to `"7"`.
Snapshot version remains `"3.0.0"`; install pins target v0.4.0. The 0.3.0 history
below still describes schema 6.

### Behavior and inputs

Sizing enforces both effective risk and position ceilings, including omitted
limits. The selected profile still supplies the same defaults:

| Tool/profile | Risk budget | Maximum allocation | Starter preset |
|---|---:|---:|---:|
| `analyze`, core | 0.5% | 8% | 2% |
| `analyze`, balanced (default) | 1% | 10% | 2.5% |
| `analyze`, speculative | 1.5% | 6% | 1% |
| `analyze_trade_setup` / `scan_watchlist` | 1% | 10% | No starter scenario |

- `account_size` is an optional account-value assumption, **not available buying
  power**. `max_position_pct` caps this standalone allocation; existing holdings,
  outstanding orders and remaining add capacity are unknown. It does not enforce
  an aggregate portfolio cap.
- `risk_per_trade_pct` and `max_position_pct` use percentage points: `1` means 1%,
  `0.01` means 0.01%. Each must be finite in `(0, 100]`; supplied `account_size`
  must be finite and positive. Small positive values are not rounded into defaults.
- All three tools accept JSON numbers or decimal strings, including trimmed,
  signed and scientific forms such as `" +1.5 "` and `"1e-3"`. They reject
  booleans before coercion, NaN/infinity, empty or malformed strings, `%`/currency
  suffixes, separators, underscores, hex notation, lists/objects, and conversion
  overflow or underflow to zero. Validation precedes search, provider and
  watchlist access. MCP validation can return a framework tool error naming the
  field, rather than application `invalid_parameters` JSON. FastMCP's client
  serializes Python NaN/infinity to JSON null before sending; the server cannot
  distinguish that from intentional null. Text `"nan"` / `"inf"` is rejected.
- Analyze's omitted/null risk and maximum each use the selected profile; an
  explicit override replaces only that limit. Swing/scan omission uses 1%/10%,
  but explicit null risk or maximum is rejected. Omitted/null account is accepted
  in all three tools.
- Public `limits.risk_origin` / `limits.max_position_origin` are `caller`,
  `profile` (analyze defaults), or `default` (swing/scan defaults). Explicit values
  equal to a default still say `caller`; internal `SwingSizingDefault` markers
  preserve omission through the MCP boundary and scan promotion. Analyze's
  internal `limit_origins` is mapped into these public fields. `scan_watchlist`
  forwards validated limits to promoted swing cards but returns no plan or sizing
  in scan rows; scan plan columns remain deferred.

Default sizes can shrink. With a $10,000 account, balanced/medium, entry $100 and
stop $80, the default 1% risk budget reduces full allocation from 8%/$800 to
5%/$500 (5 shares, $100 planned loss). Risk-only override 0.25% retains the 10%
profile maximum but allows a 1.25% ceiling: 1 share/$100, $20 planned loss. A 2%
maximum alone retains default 1% risk and allows 2 shares/$200, $40 planned loss.
Ceilings are not minimum purchases.

Decisions, scores, gates, horizons, holder guidance and snapshot fields/hashes
are unchanged. `decision_card.action_now` remains the compatibility alias.
Alternative modes may reduce the selected profile's canonical size, never raise
it or replace its risk budget. **Speculative mode can no longer show a larger
size than the selected profile's canonical card, including calls without
overrides.** Its action and horizon are unchanged. Core mode has no allocation
estimate because its decision is long term; selecting the core *profile* still
permits mid-term card sizing.

### Canonical sizing and availability

`decision_card.sizing_status` is `{status, reason}` at every detail tier, outside
nullable `decision_card.sizing`. When sizing is null, report this status and its
reason; do not infer the reason from the action alone or change the research
action.

| Outer `status` | `sizing` | Meaning |
|---|---|---|
| `not_allowed` | null | New money is blocked; `reason` names the action and applicable blockers. Takes precedence over missing sizing inputs. |
| `unavailable` | null | Entry is allowed but a required input or numeric representation is unusable; read `reason`. |
| `percentage_only` | Present | No account assumption; allocation and hypothetical loss ceilings only. `reason` is null. |
| `available` | Present | Account and inputs are valid; each scenario can still fit zero shares. `reason` is null. |

Missing/nonfinite/nonpositive entry, stop or ATR suppress size; `0 < stop < entry`
is required. Reasons include `entry_price_missing` / `entry_price_invalid`,
`stop_price_missing` / `stop_price_invalid`, `atr_missing` / `atr_invalid`, and
`stop_not_below_entry`. The shared calculator can report
`entry_price_unavailable`, `stop_price_unavailable` or `atr_unavailable`;
`sizing_inputs_unavailable` is the card-builder fallback.
`numeric_range_unavailable` covers positive values outside numeric JSON range or
whole-share monetary products that cannot round-trip exactly as numeric JSON.
It suppresses the estimate instead of emitting infinity, zero risk or a rounded
share-product amount.

Canonical `sizing` contains `horizon` (`mid_term` for analyze, `swing` for trade
plans), `loss_disclosure`, `assumptions`, `limits`, `starter` and `full`:

| Block | Fields and meaning |
|---|---|
| `assumptions` | `entry_price`, `stop_price`: the selected reference and actual returned stop; `stop_distance_fraction` = (entry − stop) / entry. `stop_basis` is `atr` for analyze and `provided` for swing; `atr` is null for a provided stop. `account_size` is null when omitted. Analyze also carries `price_provenance`, copied from `summary.price_provenance`. |
| `limits` | `risk_per_trade_pct`, `max_position_pct`, `risk_origin`, `max_position_origin`: effective inputs and origins. `regime_max_pct`, `regime_starter_pct`, `preset_starter_pct`: unchanged analyze policy caps; all three are null for swing. `risk_cap_pct` = risk budget percentage / stop-distance fraction. `risk_budget_dollars` = account × risk percentage / 100, or null without an account. |
| Each scenario | `quantity_status`; `allocation_cap_pct`, `allocation_cap_dollars`: enforced allocation ceilings. `ceiling_risk_pct`: hypothetical loss percentage at that ceiling. `whole_shares`, `position_dollars`, `planned_loss_at_stop`, `position_pct`, `effective_risk_pct`: actual whole-share quantity, allocation and conditional loss. `binding_constraints` lists tied binding caps in deterministic order; `limitations` separately reports quantity granularity. |

For analyze, full allocation is the minimum of risk, maximum-position and regime
caps; starter also obeys regime and preset starter caps. `full` is null under a
`starter` action. Swing has only `full`; `starter` is null. Constraint names are
`caller_risk_budget`, `profile_risk_budget` or `default_risk_budget`;
`caller_max_position`, `profile_max_position` or `default_max_position`;
`regime_maximum`, `regime_starter`, and `preset_starter`. Reduced mode scenarios
use the applicable `<mode>_<starter|full>_cap` constraint.

Each scenario's `quantity_status` is independent:

- `available`: one or more whole shares fit. Quantity is floored under **both**
  allocation and risk budgets; `position_dollars` = shares × entry and
  `planned_loss_at_stop` = shares × (entry − stop). `position_pct` and
  `effective_risk_pct` divide those actual amounts by the account and multiply
  by 100. Monetary values retain sub-cent precision when needed.
- `below_one_whole_share`: quantity and actual allocation/loss fields are zero;
  the ceilings remain valid. Starter can be zero while full fits a share. This
  is not a null estimate or an affordable fractional-share order.
- `percentage_only`: `whole_shares`, `allocation_cap_dollars`, `position_dollars`,
  `planned_loss_at_stop`, `position_pct` and `effective_risk_pct` are null.
  `allocation_cap_pct` and `ceiling_risk_pct` remain available; no filled position
  or dollar loss is implied.

`limitations` contains `whole_share_granularity` when flooring reduces the
permitted quantity, otherwise `[]`; it is `[]` for percentage-only scenarios.
The stop is a current-reference planning assumption, recomputed per request,
not a monitored stop, holder exit instruction or a stop for every buy-zone price.
**Estimated loss if sold at this stop price. Gaps and slippage can make the actual
loss larger.** It is never a guaranteed maximum loss.

### Schema 6 → 7 compatibility inventory

Flat sizing fields remain adapters for one release. Migrate to the canonical
scenario fields above. Null-size status replaces action-only explanations.

**Card — paths below `decision_card.sizing`:**

| Legacy field | Schema 7 value / replacement |
|---|---|
| `starter_pct`, `full_pct_cap` | `starter.allocation_cap_pct`, `full.allocation_cap_pct`: ceilings, now risk-enforced; full stays null for starter-only actions. |
| `starter_dollars`, `full_dollars` | Corresponding scenario `position_dollars`, now whole-share amounts rather than percentage-based allocations. Full stays null for starter-only actions. |
| `whole_shares` | Starter scenario only: use `starter.whole_shares` or `full.whole_shares` explicitly. |
| `fractional_shares` | Deprecated, always null; use the scenario's `whole_shares`. No fractional-lot support is implied. |
| `at_price` | `assumptions.entry_price`. |
| `unavailable_reason` | Still null whenever sizing is present. Use outer `decision_card.sizing_status` and its reason when sizing is null. |

**Zones — paths below `action_zones.position_sizing_range`:** the entire block is
null unless new money is `buy` **and** canonical sizing is non-null (including
`percentage_only`). A starter has no zone sizing range.

| Legacy field | Schema 7 value / replacement |
|---|---|
| `suggested_pct_range`, `starter_pct`, `max_pct` | Starter/full `allocation_cap_pct` pair, starter cap and full cap. Both range ends are ceilings, not a minimum/maximum purchase instruction. |
| `risk_per_trade_pct`, `profile_cap_pct` | Canonical `limits.risk_per_trade_pct`, `limits.max_position_pct`; explicit caller values are retained, despite the legacy profile label. |
| `dollars_for_account` | `min` / `max` now use starter/full `position_dollars`; `portfolio_assumption` is `assumptions.account_size`. Entire block null without an account. |
| `shares_range` | `min` / `max` are starter/full `whole_shares`; `at_price` is `assumptions.entry_price`. Entire block null without an account. |
| `stop_implied_max` | Despite its legacy name, `pct` is the **final enforced full allocation cap**, not the uncapped risk-implied cap; `dollars_for_account` is full `position_dollars`, not cap dollars. `risk_per_trade_pct` is the effective budget. Use canonical `limits.risk_cap_pct` for the risk-only ceiling. |
| `stop_implied_max.stop_distance_pct` | Percentage points from `assumptions.stop_distance_fraction × 100`, no longer rounded to one decimal. |
| `rationale` | Regime rationale unchanged. |

**Modes — paths below `decision_modes`:** `core.starter_position` and
`core.max_position` are always null. Balanced/speculative `starter_position`
requires both canonical and mode entry permission plus valid sizing;
`max_position` additionally requires both actions to be `buy` (speculative's
remains null). Smaller mode presets still apply.

| Legacy field | Schema 7 value / replacement |
|---|---|
| `starter_position` / `max_position`: `pct`, `dollars`, `shares` | Mode-capped scenario `allocation_cap_pct`, whole-share `position_dollars`, `whole_shares`. Blocks now also contain every canonical scenario field above and `horizon: mid_term`. Compare actual `position_pct` with the cap explicitly. |
| `speculative.capital_at_risk` (`capital_at_risk`) | Deprecated, always null; previously echoed the account. Read the relevant scenario's `planned_loss_at_stop` for conditional loss and card `assumptions.account_size` for account context. No public `account_size_assumption` field is introduced. |

**Swing — paths below `analyze_trade_setup.plan` (`plan`):** the existing setup,
entry anchor, stop, targets and action remain. New `sizing_status` and `sizing`
follow the canonical contract above. The returned stop supplies the calculation,
including its existing two-decimal quantization.

| Legacy field | Schema 7 value / replacement |
|---|---|
| `shares`, `position_dollars` | `sizing.full.whole_shares`, `sizing.full.position_dollars`; dollars now use whole shares rather than the rounded fractional quantity. |
| `fractional_shares` | Deprecated, always null. |
| `max_loss_dollars` | Deprecated equal-value alias of new `planned_loss_at_stop` for one release; both mirror `sizing.full.planned_loss_at_stop`, now based on whole shares. They are null without an account or valid estimate. |
| `position_pct` | Actual whole-share `sizing.full.position_pct` with an account; `sizing.full.allocation_cap_pct` without one; null if sizing is unavailable. Previously a rounded allocation ceiling in both cases. |
| `entry.condition_text` | Zero shares: “The setup qualifies, but no whole-share position fits this budget.” Unavailable sizing: “The setup qualifies, but a size estimate is unavailable.” The setup action is retained. |
| `stop.distance_pct` (`distance_pct`) | Still a fraction, rounded to four decimals; use canonical `assumptions.stop_distance_fraction` for sizing. |

### Entry bands, units and summaries

- `decision_card.entry` stays present for `buy`/`starter` even when sizing is
  unavailable. Its `buy_zone_low` / `buy_zone_high` are now **both null** if either
  bound is missing, nonfinite, nonpositive or inverted. New
  `entry_band_unavailable` is `{reason: "missing" | "invalid" | "inverted"}` or
  null for a valid band. `current_price`, source levels and research decisions
  remain. A valid current-reference size does not prove the band usable.
- The two legacy `stop_distance_pct` fields keep their units: analyze
  `action_zones.stop_calculation.stop_distance_pct` is a fraction rounded to four
  decimals (`0.04` means 4%); zone `stop_implied_max.stop_distance_pct` is percentage
  points (`4` means 4%) without the former one-decimal rounding. Swing
  `plan.stop.distance_pct` is also a four-decimal fraction. These ambiguous paths
  are deprecated in favor of canonical `assumptions.stop_distance_fraction`.
  `stop_too_tight` now compares the unrounded selected distance.
- Missing ATR's 2% fallback band is labelled
  `action_zones.basis.strong_buy_below: 52w_low_plus_2pct_fallback` and
  `action_zones.basis.reduce_above: 52w_high_minus_2pct_fallback`. A fallback band
  does not invent a planning stop or enable sizing.
- `executive_summary` keeps its three horizon segments, but replaces the old
  “size ~a-b%” clause with separate starter/full allocation and conditional-loss
  descriptions, effective limits and origins. Percentage-only and zero-share
  scenarios have their own wording. Allowed entries with null sizing name the
  unavailable input; blocked summaries state the action and blockers once.
  The planning-stop disclosure follows valid sizing. Human prose uses `~` only
  for rounded values, rounds ceilings down and keeps tiny positives visible in
  scientific notation; canonical numeric fields do not follow display rounding.

## 0.3.0 (2026-10-16): Step 2, one canonical decision

The package version (`meta.server_version`) goes from `0.2.0` to `0.3.0`. Two other versions change independently:

- **Response schema:** `meta.schema_version` goes from `"5"` to `"6"`.
- **Watchlist snapshot:** `watchlist_snapshot.snapshot_version` goes from `"2.0.0"` to `"3.0.0"`. `detect_changes` accepts 1.0.0, 1.1.0, 2.0.0 and 3.0.0 snapshots.

### Read this first

- **One decision, gates first.** `analyze` decides once, in `decision_card`:
  - `new_money`: the mid-term (3–12 months) action for new money, the labelled default;
  - `if_held`: mid-term guidance for an existing holder;
  - `long_term.new_money`: the long-term (1–5 years) action for new money.

  The executive summary, decision modes, dislocation, dip and snapshot now follow it.
- **New-money actions** (loosest to tightest): `buy`, `starter`, `wait`, `wait_for_data`, `insufficient_data`, `watch`, `avoid`. `decision_card.action_now` mirrors `new_money.action` for one release.
- **Holder actions:** `if_held.action` is `hold`, `reduce`, `review_exit` or `insufficient_data`, with a `basis`. Gates never change it.
- **`policy_action` is removed**, with no alias. The pre-gate candidate is `diagnostics.pre_gate_policy` (`detail="full"` only), marked `role: "pre_gate_input_not_advice"`.
- **Sizing only for an entry.** `decision_card.sizing` and `decision_card.entry` are mid-term new-money sizing only (no long-term or holder size) and null unless new money is `buy` or `starter`; a starter hides the full size.
- **Snapshots:** the first `detect_changes` against a saved 2.0.0 snapshot compares no action fields and no zone, and can report one round of rule-driven gate changes ([One-time alerts](#one-time-alerts-after-upgrading)).
- **Horizons:** In `watchlist_snapshot`, `new_money_action` and `if_held_action` are the mid-term decision and `long_term_action` the long-term one; `scan_watchlist` state holds swing-card actions. When the two disagree, they describe different horizons, not a contradiction.

**Examples (v0.2.0 → v0.3.0):** `hold_or_add` maps to `starter` before gates. An earnings blackout imposes `wait`, missing required thesis evidence `wait_for_data`, and weak liquidity `avoid`; the strongest restriction wins. An ETF's `hold` becomes `watch`; a broken business with a bullish chart goes from `hold_or_add` to `avoid` (`thesis_broken`).

### Decision rules

- **Combination:** at each horizon, `new_money.action` = the most restrictive of the mapped candidate and the outcome of every fired gate, in the order avoid > watch > insufficient_data > wait_for_data > wait > starter > buy. A gate never loosens an action and can tighten one that already denied new money (a support gate turns `wait` into `watch`). v0.2.0 turned any gate on `buy`, `starter` or `hold_or_add` into `wait_for_data` (`watch` with a support gate) and passed every other action through.
- **Candidate mapping:** mid-term `buy` → buy; `hold_or_add`, `speculative_small_position` → starter; `hold`, `hold_or_reduce`, `wait_for_entry` → wait; `avoid`, `insufficient_data` unchanged. Long-term `accumulate` → buy; `hold_or_add`, `small_position_with_stops`, `speculative_with_thesis`, `speculative_only` → starter; `hold`, `hold_size_conservatively`, `wait_or_avoid` → wait; `avoid_core_position` → avoid. An unknown code now raises. The candidate no longer carries the falling-knife override (`dip_type=falling_knife`); the knife is the `falling_knife` gate.
- **Gates (analyze), in rank order:**

  | Gate | Class (rank) | New money, mid / long term | Change |
  |---|---|---|---|
  | `stale_data`, `freshness_unverifiable` | data (1) | wait_for_data | New in analyze |
  | `data_quality_critical` | data (1) | wait_for_data | A missing beta alone no longer fires it |
  | `missing_runway` | data (1) | wait_for_data | |
  | `thesis_unverifiable` | data (1), analyze only | wait_for_data | New: required thesis evidence cannot be assessed |
  | `liquidity_missing` | data (1), analyze only | wait_for_data | Data class in analyze; the card and scan keep avoid |
  | `falling_knife` | price_risk (2) | avoid | Was `wait` (knife override) or `wait_for_data`; now read from the technicals rubric for every instrument, unsupported ones included |
  | `weak_liquidity` | liquidity (2) | avoid | Was wait_for_data on an admitting candidate |
  | `thesis_broken` | thesis (3) | avoid | New: business broken, or cash runway under 4 quarters |
  | `unsupported_instrument`, `unsupported_currency` | support (4) | watch | |
  | `earnings_blackout` | event (5) | wait | Was wait_for_data |
  | `earnings_unverifiable` | event (5) | wait | New in analyze: the events tool failed or `verification` is `unverifiable` |
  | `near_52w_high` | zone (6) | wait, mid-term only | New: `price_zone` is `near_52w_high` |

  Every gate except `near_52w_high` also reaches `long_term.new_money` and the core mode.
- **`decided_by`:** the highest-ranked fired gate whose outcome is the shown action and tighter than the mapped candidate; null when the candidate decided.
- **`unevaluated`:** applicable gates that could not be computed, each `{id, reason}`: `falling_knife` (inputs cannot establish a trigger or clearance), `missing_runway` (fundamentals missing or applicability unverified), `thesis_broken` (required business/runway evidence missing or applicability unverified), `stale_data` (required price/history age unavailable, unless another leg already proves staleness), `weak_liquidity` (dollar volume or stock/USD applicability unverified; missing volume still fires `liquidity_missing` for a supported USD stock), `earnings_blackout` (events tool failed), `near_52w_high` (`price_zone` null or undetermined).
- **`if_held` (mid-term, first match):** runway under 4 quarters → `review_exit` (`runway_critical`); business broken → `review_exit` (`business_broken`); required thesis evidence unavailable → `insufficient_data` (`thesis_unknown`); candidate admits new money → `hold`; dislocation candidate → `hold` (`price_broken_more_than_business`); tilt missing → `insufficient_data`; candidate `insufficient_data` → `insufficient_data`; tilt bearish → `reduce` (`tilt_bearish`); otherwise `hold` (`no_reduction_evidence`). `add_allowed` is true only when new money is `buy` or `starter`.
- **Partial fundamentals:** unknown instrument applicability is `unverified`, not `not_applicable`. Missing currency does not discard company evidence. Missing business or applicable cash-runway evidence makes the thesis `unknown` and removes unsupported exit templates; proven failures still win. A known non-burner needs no numeric runway. The four-quarter threshold uses the unrounded runway, while displayed values and dilution arithmetic retain their rounding. The existing unprofitability predicate still determines whether burn metrics are computed.
- **Unverifiable thesis:** in analyze, an applicable thesis check that cannot be evaluated fires the separate `thesis_unverifiable` data gate at both horizons; `thesis_broken` remains unevaluated. The gate imposes `wait_for_data` and appears in `blocked_by`, `hard_gates` and snapshot `hard_gate_ids`. Existing `watch` (including unverified support) and `avoid` outcomes remain stronger; `decided_by` names it only when it determines the action. No entry means no sizing and `add_allowed: false`; the holder's `insufficient_data` / `thesis_unknown` answer is unchanged. Its clear-when condition is "the stock type is confirmed and the required business and cash-runway data are available"; re-run analyze afterwards, without treating clearance as an automatic buy. Known unsupported instruments, including ETFs, are exempt. Other unevaluated checks retain their existing policies; the trade card and scan do not gain this gate.
- **Modes:** balanced = `new_money.action`, and `balanced.long_term_view` = `long_term.new_money.action`. Speculative = its own code (`starter_then_add`, `starter_position_only` → starter; `wait`, `watch` → wait) capped by new money. Core = the most restrictive of its long-term fit (`ok` buy, `caution` wait, `unknown` insufficient_data, `avoid` avoid) and `long_term.new_money.action`. A capped mode carries `capped_by` (`new_money_<action>` or `long_term_<action>`). The `profile` argument changes sizing only, never an action.
- **Freshness gate:** two legs; a fresh quote never vouches for the history.
  - Leg Q (the selected price) blocks on the info-quote fallback; in the regular session on any rejected intraday quote (unavailable, older than 15 min, future-dated); off-hours on a daily close at least one session behind the last completed session.
  - Leg H (`technicals.daily_history`) blocks on missing or malformed history, an invalid latest row, a newest completed bar at least one session behind, a future-dated row, or a valid row after the last completed session other than today's forming row.
  - `stale_data` codes: `intraday_quote_stale`, `daily_close_stale`, `daily_history_stale`. `freshness_unverifiable` codes: `intraday_quote_unavailable`, `intraday_future_timestamp`, `info_quote_fallback`, `daily_latest_bar_invalid`, `daily_history_unavailable`, `daily_future_bar`, `daily_bar_after_last_session`. No price at all stays `data_quality_critical`.
  - Disclosure only: flags `includes_forming_bar` and `calendar_unverified`.
  - `data_quality.component_freshness.price` now ages technicals' own last completed valid bar, so the warning and the gate agree, and `component_freshness.risk` ages risk_metrics' own history; both used to read the merged `price` provenance.
- **`earliest_reassessment`** (per horizon, on `new_money` and `long_term.new_money`): `{date, basis, earnings_risk_through, session_method, outstanding: [{id, clears_when}]}`.
  - Present only when a gate fired and the mapped candidate admits new money at that horizon; otherwise null.
  - `basis`: `earnings_clear` (`date` is the first NYSE session no earnings window blacks out), `conditions_only` (date null), `unsupported` (date null; `clears_when` is "never: outside this tool's supported scope").
  - `session_method`: `calendar_static`, or `clock_only_no_holidays_fallback` outside the vendored calendar; null without a date.
  - `outstanding` lists the undated gates in rank order; any event gate adds one `earnings_reverify` entry, because the calendar goes stale after the event.
  - It is a lower bound and a date to re-run analyze, never permission to buy; it never changes an action. Known limits: a second earnings-page row the page hides can block again after the date, and the union of info dates is conservative.
- **Conditions:** only what binds, per horizon.
  - `decision_card.conditions` is mid-term: `buy_only_if` lists each binding non-support gate's `clears_when`, then each candidate input whose change alone would admit new money (under `starter`: the inputs that would make it `buy`, i.e. for the full position); `add_only_if` is the same list while `if_held` is `hold`; `do_not_buy_if` lists the `blocked_by` reasons; `monitor` the remaining upgrade conditions. A permanent support gate empties both horizons.
  - New `long_term.new_money.conditions.buy_only_if` is the long-term list; core's `required_before_full_size` mirrors it, and balanced, speculative, dislocation and dip conditionals mirror the mid-term lists.
  - Conditions are necessary, not sufficient: meeting them never permits a buy by itself; re-run `analyze`.
  - Text: valuation conditions quote the gate's own thresholds (`P/E <= 50x`, `PEG <= 3.0`, `EV/S <= 10x` or `P/S <= 10x`, each "(currently headwind)"; v0.2.0 said `P/E < 25x`, `EV/S < 6x`, `P/S < 6x`); risk conditions name their drivers (e.g. `risk_regime <= high (volatility < 60% and max_drawdown_1y > -50%)`); a poor business no longer lists `net_margin > 0%`; a negative-FCF long-term gate adds `free_cash_flow_ttm >= 0`.
- **Earnings window:** one helper serves the blackout, the warning, the reassessment date and analyze's `next_review`.
  - The blackout reason names an active window ("earnings window Oct 7–15 active") or "earnings today", else "earnings in N days (<= 5)", in analyze and on the trade card.
  - `data_quality.warnings` now has `earnings_within_7_days` on earnings day (`days_until` 0), and `decision_context.next_catalyst` keeps `days_until: 0`.
  - Analyze's `next_review` inside an active window is the first session after it (reason "after earnings (Oct 15)"), never today; an undatable window gives "default 30-day re-evaluation (earnings window undatable)". The trade card's `next_review` is unchanged.

### Schema 6 migration

**Added**

- `decision_card.horizon` (`mid_term`), `new_money`, `if_held`, `long_term` (with `horizon: long_term`); `earliest_reassessment`, `unevaluated` and `decided_by` per horizon; `why` (the deciding gate and the mid-term evidence), `exit_triggers` (the thesis stop triggers) and `data_flags` (fundamentals status, staleness and zone warnings).
- Blocker `class` and `effect` (`veto`, `cap` or `conditional`) on `new_money.blocked_by` and the trade card's `blockers`; `codes` on freshness and `thesis_broken` blockers (`business_broken`, `runway_critical`), also in `hard_gates.blocking`.
- `decision_card.hard_gates.checks` grows from 8 to 14 keys (`stale_data`, `freshness_unverifiable`, `thesis_unverifiable`, `thesis_broken`, `earnings_unverifiable`, `near_52w_high`), in rank order; `blocking` is in rank order too.
- `diagnostics` (`detail="full"` only): `pre_gate_policy` (`mid_term`, `long_term`, `rationale`, `conditions_to_upgrade`, `conditions_to_downgrade`, `inputs: {zone, valuation_gate, valuation_basis, is_unprofitable}`, `support_gates` when support capped it, `role`, `superseded_by: {new_money, if_held, long_term, blocked_by}`) and `decision_trace`.
- `data_quality.freshness` at every detail level: `session`, `expected_session`, `price_source`, `daily_last_valid_date`, `daily_completed_session`, `daily_sessions_behind`, `codes`, `flags`.
- `action_zones.price_zone`: `near_52w_low`, `below_sma200`, `above_sma50`, `at_or_below_sma50`, `near_52w_high`, `undetermined` or null, from the uncapped classification.
- `dislocation_check.decision` (standard tier): `{new_money, blocked_by}`.
- `capped_by` on capped `decision_modes.core` / `speculative`, `dislocation_framework.action.capped_by`, `dip_assessment.assessment` and dip `entry_timing.signals`.
- `long_term.new_money.conditions.buy_only_if`.
- `data_gaps` value `beta_unavailable`.
- `get_technicals` `daily_history`: `last_row_date`, `last_valid_date`, `last_row_valid`, `completed_valid_date`.
- Trade card: `horizon: "swing"`, `decided_by`, `unevaluated`. Scan rows: `horizon: "swing"`, `source` (`screen` or `card`), `unevaluated`; scan response and state: `rules_version`; warning `rules_changed`.
- Snapshot 3.0.0: `new_money_action`, `if_held_action`, `long_term_action`, `price_zone`, `unevaluated_gate_ids`.

**Removed or moved**

- Top-level `policy_action`, including `sizing_guidance` → `diagnostics.pre_gate_policy` (no sizing), no alias. Its `valuation_gate`, `valuation_basis` and `is_unprofitable` → `diagnostics.pre_gate_policy.inputs`; the snapshot's `valuation_gate` and `is_unprofitable` now read `action_zones.valuation_assessment` (same values).
- `decision_card.horizon_fit` (use `new_money` and `long_term.new_money`).
- `decision_card.rationale` → `diagnostics.decision_trace` (`pre_gate_policy.mid_term=<code> -> <action>`, then `decided_by=<id>`).
- `decision_card.entry.current_zone`, and `action_zones.current_zone` → `action_zones.price_zone` (a neutral, uncapped vocabulary; the capped label survives only as `diagnostics.pre_gate_policy.inputs.zone`).
- `unavailable_reason` in `action_zones.position_sizing_range` and the decision modes' `starter_position` / `max_position`: those blocks are null whenever sizing is withheld. `decision_card.sizing.unavailable_reason` stays in the shape but is always null.
- `data_quality.missing_critical` no longer lists `beta`.
- Snapshot 3.0.0: `action_mid_term`, `action_long_term`, `decision_action_now`, `zone`.

**Kept**

- `decision_card.hard_gates`: `blocking` lists every fired gate (the source of snapshot `hard_gate_ids`, a superset of `blocked_by`), `checks` stays booleans.
- `decision_card.exit`, `conditions`, `next_review`, `confidence`; `action_zones.levels` key names.

**Vocabulary changes**

- `decision_card.action_now`: `buy`, `starter`, `wait`, `wait_for_data`, `insufficient_data`, `watch`, `avoid` (no more `hold`, `hold_or_add`, `hold_or_reduce`).
- `decision_modes.*.action`, `balanced.long_term_view` and `dislocation_framework.action.{core, balanced, speculative}`: the new-money set (v0.2.0 core: `build_position`, `not_core_yet`, `needs_more_data`; speculative: `starter_then_add`, `starter_position_only`).
- `decision_modes.balanced.why` holds the card's `why` plus the `business_quality` and `risk_regime` facts (it held the pre-gate rationale); `balanced.suitability` is `starter_only` on a dislocation only when new money is `starter`; the modes' `add_on_only_if` mirror the mid-term `add_only_if` (they held the first three upgrade conditions and, for speculative, the dip `wait_for` signals).
- `dislocation_framework.action.buy_now`: `buy` or `starter` (was `small_starter`); `can_start_now` is true under `buy` too and is deprecated (it equals new money being `buy` or `starter`).
- `dip_assessment.assessment.recommendation` and `entry_timing.signals[*].action` follow new money: under `starter`, `strong_buy_the_dip`, `buy_the_dip` and `cautious_accumulation` become `starter_only`, and `accumulate` and `add_position` become `starter_position`; under a non-entry action every entry code becomes `watch` with the rationale "No new-money entry: new_money is <action>.". The dislocation override (`starter_only_wait_for_stabilization`, `scope`, `timing_only`, `add_only_if`) applies only under a `starter`, and its `add_only_if` is no longer cut to 4.
- `action_zones.zone_warnings`: `extreme_risk_regime:prefer_small_position` → `extreme_risk_regime`, `high_risk_regime:size_conservatively` → `high_risk_regime`. `position_sizing_range.rationale`: `extreme_risk_requires_minimal_exposure` → `extreme_risk_regime`, `high_risk_warrants_conservative_sizing` → `high_risk_regime`, `moderate_risk_standard_sizing` → `medium_risk_regime`, `low_risk_allows_larger_positions` → `low_risk_regime`.
- Diagnostic text no longer gives sizing advice: `verdict.horizon_fit.reasons` (e.g. "mid_term: extreme risk regime", "mid_term: setup favorable"), `decision_context.horizon_drivers[*].reason`, the dip type explanations and rationales.
- `executive_summary` opens with three segments: "Decision (mid-term, new money): <action>[; blocked by <ids>][; size ~a-b%][; <reassessment>]. If held: <action>. Long-term (1-5y): <action>[; …]." The closing "Policy: …" clause is gone, and the long-term-avoid lead reads "<name>: long-term fit avoid (<gates>)." The reassessment reads "reassess on or after Tue Oct 13 (not a buy date)", "no reassessment date: re-run analyze after <conditions>" or "outside supported scope: no reassessment date".
- Trade card and `get_technicals` `intraday.freshness`: `reason_code` gains `nonpositive_close`.
- Scan row blockers: freshness ids carry their leg (`stale_data@daily`, `freshness_unverifiable@quote`).

**Sizing presence**

- `decision_card.sizing` and `entry`: only when new money is `buy` or `starter` (a starter nulls `full_pct_cap` and `full_dollars`). `action_zones.position_sizing_range`: only under `buy`. Mode `starter_position`: only when new money and the mode's action are `buy` or `starter`; `max_position`: only when both are `buy`, so speculative's is always null. `speculative.capital_at_risk` is null without a speculative starter. The summary's "size ~" part appears only in the new-money segment of an entry. The numbers themselves are unchanged.

**Aliases**

- `decision_card.action_now` (= `new_money.action`), for one release.

### Snapshots and `detect_changes` (3.0.0)

- **Fields:** 3.0.0 is 2.0.0 without `action_mid_term`, `action_long_term`, `decision_action_now` and `zone`, plus `new_money_action`, `if_held_action`, `long_term_action` (= `long_term.new_money.action`), `price_zone` and `unevaluated_gate_ids`. They are hashed and diffed; any change is material. `price_zone` follows the uncapped bands, so a capped name (falling knife, extended decline, extreme risk, valuation headwind) can now report zone moves the capped `zone` did not; they are context, not advice. `hard_gate_ids` lists every fired gate and `unevaluated_gate_ids` every gate whose input failed (`new_money.unevaluated` at either horizon). A gate whose input failed is reported as not evaluated (`+unevaluated_gate`, "Gate(s) not evaluated", `changes.hard_gates.not_evaluated`), never as cleared, also against an older snapshot: "Cleared hard gate(s)" means the gate was evaluated and did not fire. A gate evaluated again is `-unevaluated_gate` ("Gate(s) evaluated again", `changes.unevaluated_gates`). Signal lists are unknown when instrument applicability is unverified, avoiding invented removals on classification loss. Reasons, the pre-gate candidate, `earliest_reassessment` and the clock stay out of the hash.
- **Validation:** a 3.0.0 previous snapshot must carry every 3.0.0 key (null allowed), else `invalid_snapshot`. Older versions validate only their own keys and ignore stray ones.
- **Unreleased snapshots:** intermediate 3.0.0 snapshots saved before `unevaluated_gate_ids` was added are rejected as incomplete. Re-run analyze to save a complete snapshot; released 2.0.0 snapshots remain accepted.
- **Across versions:** fields both versions carry compare as before, 2.0.0 signals included. The three action fields, `price_zone` and `unevaluated_gate_ids` are not compared, and the summary names them ("Not compared with the previous 2.0.0 snapshot: …"); a 1.1.0 snapshot therefore loses the `action_now` alert v0.2.0 gave it, for one cycle. A version warning names the retired fields and the 3.0.0 rule changes: new analyze gates `earnings_unverifiable`, `stale_data`, `freshness_unverifiable`, `thesis_unverifiable`, `thesis_broken` and `near_52w_high`; knife and weak liquidity → avoid; earnings → wait; `liquidity_missing` is data class; the knife is evaluated for every instrument; a missing beta alone no longer fires `data_quality_critical`; `zone` → `price_zone`. Rule-driven gate changes are reported, not suppressed. `hash_changed` is reported only between two 3.0.0 snapshots.
- **Freshness:** a new-money change caused only by a freshness gate is material. A growing age alone is never an alert (`price_stale` and ages stay unhashed).
- **Renamed in the response:** the `changes` keys `action_zone` → `price_zone` and `decision_action_now` → `new_money_action`; `action_mid_term` and `action_long_term` are gone (the long-term decision is `long_term_action`); `if_held_action` is new. The material lines read `price_zone: a -> b`, `new_money_action: a -> b` and so on; the summary says "Price zone moved from a to b." and "New money: a -> b.", "If held: a -> b.", "Long term: a -> b." (was "Zone moved …" and "Action now: …").

### Agent guidance

- **Server instructions:** the server now sends MCP `instructions`: start with `analyze`; a bare ticker never triggers a question and gets the mid-term decision, labelled as mid-term; ask for a horizon only when the request names two horizons, or a holding period that fits none (a period that maps to one horizon, such as "hold for five years", gets it; exactly one year uses mid-term unless long-term is explicitly requested); where each horizon's answer lives; never upgrade past `blocked_by`; size only from `decision_card.sizing`.
- **Tool descriptions** start with "Use when …". `analyze_my_position` is described as P/L, holding period and tax, the hold/reduce decision being `analyze`'s `decision_card.if_held` (it was "hold/sell decision"); `compare` applies no decision gates; `analyze_trade_setup` and `scan_watchlist` are the swing horizon; `detect_changes` names its snapshot's horizons. The `profile` argument is documented as changing sizing only.
- **Rendering guide** (`stock-analysis://guides/analyze-rendering`), rewritten for every detail level (it assumed `detail="full"`): the card first, in three horizon segments; never upgrade past `blocked_by`; sizing only from `decision_card.sizing` ("no entry size: new_money is <action>"); `price_zone`, levels, the dip assessment, `diagnostics` and third-party data are context, not advice; `can_start_now` is deprecated; the reassessment date is not a buy date; a field-class table.
- **Interpretation:** report holder `basis` separately from new-money `blocked_by` and `add_allowed`; there is no long-term holder decision. Disclose unevaluated gates without inventing a new blocking policy. `decision_card.entry` is a reference band, not a recommendation to pay above market; `sizing.at_price` is its sizing reference.
- **Prompts:** `full_analysis` leads with `decision_card` (`policy_action` is gone); `growth_memo` and `value_memo` ask for `decision_card.new_money.action` and `decision_card.long_term.new_money.action` instead of "Buy / Wait / Pass"; `position_decision` calls `analyze` for `decision_card.if_held` and `analyze_my_position` for P/L and tax, without `get_technicals` or HOLD / SELL / TRIM.

### Trade card and scan

- **Trade card (`analyze_trade_setup`):** `earnings_unverifiable` is disclosed on every outcome and still caps only a detected setup at `watch`. Duplicate blocker ids merge (reasons joined by "; "). A close that is not finite and positive is never the actionable price: in session a non-positive 5m close is `freshness_unverifiable` (`nonpositive_close`); a non-positive daily close fails the daily leg (`stale_data` in session, `freshness_unverifiable` off-hours). The resolution order is unchanged.
- **Scan stage 1:** it evaluates every gate it can and resolves rank-first like the card. Data first: under 50 bars, no features, or off-session no valid close is `data_quality_critical`, and the card's daily freshness leg applies (both `wait_for_data`); then the knife and liquidity on the card's basis (1-year mean volume × last close; was a 20-bar dollar mean) give `avoid`. An unsupported instrument or currency with a setup is `watch` with the setup kept; unverified support is `unevaluated` and a setup promotes, so the card decides. Screen rows report the detector's `trigger_satisfied` (was always false).
- **Swing uncertainty:** card and screen also disclose incomplete falling-knife inputs and missing liquidity/applicability as unevaluated. A proven knife still fires with other inputs missing. Missing-volume actions retain their existing precedence.
- **Scan diffs:** blockers compare as (id, leg) pairs that neither side left unevaluated; an entry saved before this change compares at id level. A state from other rules adds a `rules_changed` warning.

### One-time alerts after upgrading

- **`detect_changes` with a saved 2.0.0 (or 1.x) snapshot:** the version warning, the "not compared" summary for the action fields and `price_zone`, and `±hard_gate` lines for gates the new rules fire or clear: `+earnings_unverifiable`, `+stale_data` / `+freshness_unverifiable`, `+thesis_unverifiable`, `+thesis_broken`, `+near_52w_high`, `+falling_knife` on ETFs and other unsupported instruments, `-data_quality_critical` where only beta was missing. Save the new 3.0.0 snapshot to restore full comparison.
- **First `scan_watchlist`:** a `rules_changed` warning, plus one round of `changes` from the new rules: `earnings_unverifiable` on non-setup cards, and rows whose stage-1 outcome or `trigger_satisfied` changed. Leg suffixes and merged duplicate ids alone are not reported.

### Upgrade and rollback

- Pin the install to `git+https://github.com/nickzren/stock-analysis-mcp@v0.3.0`. Keep a copy of existing 2.0.0 snapshots and watchlist/scan state before upgrading. Use a separate `STOCK_ANALYSIS_DATA_DIR` when trying development versions.
- **Rollback:** a v0.2.0 scan on a data dir last written by v0.3.0 reports one round of rule-driven changes with no warning (v0.2.0 has no `rules_version` check), and v0.2.0's `detect_changes` rejects a 3.0.0 snapshot as `invalid_snapshot` (`unsupported snapshot_version: '3.0.0'`). Use the saved 2.0.0 snapshots and state when returning to v0.2.0.

### Known limitations (0.3.0)

- **Resolved from 0.2.0:** earnings verification in analyze (`earnings_unverifiable`); the earnings-day warning; the core profile mismatch (core follows `long_term.new_money`).
- **Intraday price basis, narrowed:** a stale or missing in-session quote now blocks new money (`wait_for_data`). Zones and stops still use the last completed close while technicals include the forming bar (flag `includes_forming_bar`), and `if_held` is unchanged.
- **`if_held` is stateless:** it does not know your cost, size or holding period, and there is no long-term holder read yet.
- **Trade card and scan:** the card reads the raw last row where analyze walks back past an invalid bar; scan stage 1 detects on the daily close while the card can use an intraday price, so their setups can differ; the card's `next_review` and `market_closed` handling are unchanged.
- **Reassessment date:** see its known limits above; the blackout is not yet anchored on the next fillable session.

## 0.2.0 (2026-10-12): Step 1, correctness foundation

The package version (`meta.server_version`) goes from `0.1.0` to `0.2.0`. Two other versions change, each on its own scale:

- **Response schema:** `meta.schema_version` goes from `"4"` to `"5"`.
- **Watchlist snapshot:** `watchlist_snapshot.snapshot_version` goes from `"1.1.0"` to `"2.0.0"`. `detect_changes` still accepts 1.0.0 and 1.1.0 snapshots.

Upstream is now `yfinance>=1.0,<2`. `beautifulsoup4` is a direct dependency, because the earnings page is now parsed directly.

### Read this first: fields whose meaning or units changed

Old consumers should re-check every item below.

**Dividend yield**

- **One forward value on three paths:** `get_fundamentals` `yield_metrics.dividend_yield`, `get_stock_summary` `dividend_yield` and `get_events` `dividends.yield` all carry the same value: `dividendRate / price`, as a fraction (0.005 = 0.5%).
  - v4 passed Yahoo's `dividendYield` through. It was in percent on the fundamentals path, and divided by 100 only when above 1 on the other two.
- **Edge cases:**
  - Non-payers get `0.0` (basis `no_dividend`). v4 passed Yahoo's value through: `0.0` for some non-payers, null where Yahoo omitted it.
  - A missing price gives null.
  - An unknown or minor-unit currency gives null (GBp, ZAc and similar).
- **New companions:** `dividend_yield_basis` (`forward`, `no_dividend` or `unverified_currency`) and `dividend_yield_trailing`. In events they are `dividends.yield_basis` and `dividends.yield_trailing`.
- **`compare`:** it ranks the new fraction, and non-payers are always ranked (0.0); v4 left them out where Yahoo omitted the field.
- **`compare` leverage:** a negative D/E or negative equity is null and ranked last; v4 ranked it best.

**`get_fundamentals`**

- **Debt to equity:** `financial_health.debt_to_equity` is always Yahoo's percent divided by 100, i.e. a ratio. v4 left raw values ≤ 10, and all negatives, in percent.
  - `rules.low_debt` is now False for negative equity.
- **Valuation ratios are null when undefined or unverified:**
  - **P/E:** `pe_trailing` / `pe_forward` are null when their own EPS ≤ 0.
  - **EV/EBITDA:** null when EBITDA ≤ 0.
  - **Currency:** P/E, P/S, EV/S, EV/EBITDA, P/B, FCF yield, the payout ratios, the trailing dividend yield and the P/E and P/S history are null when the trading currency differs from `financialCurrency`, or is unknown or a minor unit. A USD ADR reporting in TWD loses all of them.
  - **Share units:** P/B, the trailing dividend yield, FCF payout and the P/E and P/S history are also null when `marketCap ≈ price × sharesOutstanding` fails, e.g. multi-class shares.
  - **`earnings_yield`:** uses only the usable P/E.
  - **`compare`:** P/E, P/S or PEG ≤ 0 are null and ranked last, where v4 ranked them best.
- **Cash flow periods:** `cash_flow.operating_cf_ttm` / `free_cash_flow_ttm` keep their names but can hold an annual value.
  - A TTM is used only when the 4 newest quarter columns are consecutive and all finite.
  - Otherwise the value comes from the newest finite annual column, then from Yahoo's info figure.
  - `*_period`, `*_period_end` and `*_source` say which was used. An annual label comes from the column actually used, not from `lastFiscalYearEnd`.
  - FCF rebuilt from OCF + capex always comes from one period.
- **Valuation history:**
  - `valuation_context.pe_5y_avg`, `pe_percentile_5y` and `ps_5y_avg` are null unless that metric's own history spans 5 years. With Yahoo's usual quarterly depth they are usually null.
  - `*_stats` need 8 or more points.
  - `current_percentile` now ranks the current usable ratio, not the newest history point.
- **Trends:** in `fundamental_trends`, the QoQ, YoY, margin-trend and 3-year CAGR comparisons are left out unless the quarters they compare are consecutive.

**`get_technicals` and price history**

- **Empty sessions are kept:** daily-or-longer frames now include sessions Yahoo sends with no data, as null OHLC. A missing 1d volume stays null instead of becoming 0, and a row with no prices has no volume. This reaches `get_price_history` rows, previews and the CSV resource, where whole volumes still print as integers.
  - `data_provenance.price.last_bar_date` (technicals, risk metrics, price history) and `check_data_quality`'s `price_last_bar` give the newest bar with a valid close, not an empty row. `check_data_quality` reports no price when no bar has one.
  - `analyze_my_position` uses the latest valid completed-session close when the latest row is empty.
- **Returns count sessions:**
  - **Return fields:** `returns.return_*` are null when either endpoint close is missing or non-finite. The 1-week z-score is null when the latest week is undefined.
  - **Counts and comparisons:** `days_since_52w_*` count sessions, and `price_action` compares consecutive sessions.
  - **SMA200 slope:** compares the SMA 20 sessions apart.
  - **RSI divergence:** keeps each close with its own session's RSI.
  - **SMA and Bollinger:** they use the last N valid closes, all N required, so an older empty session no longer voids the window. A day with no close has no value, so an empty latest session still nulls them.
  - **Unchanged indicators:** EMA, MACD, RSI and ATR behave as before. `bar_validity` discloses the gaps in every indicator window.
- **Volume and the latest bar:**
  - `volume.current` / `ratio` and all `obv` fields are null when the latest volume is missing.
  - `volume.avg_20d` averages only valid volumes.
  - `current_price` is null when the latest row is an empty session.
- **`get_price_history` summary:** `start_price` / `end_price` / `total_return` use the first and last rows, matching `start_date` / `end_date`. A missing endpoint gives null.

**`get_events` earnings**

- **`next_date` and `days_until`:**
  - `next_date` is the earliest risk date across all sources, verified or not, and is today while an estimate window is open.
  - `days_until` is never negative.
  - A past info `earningsTimestamp` (the last report) is ignored, so it can't widen the next quarter's estimate window back to today.
  - `next_date_source` no longer has the value `calendar_dataframe`.
- **Dates are ET:** calendar dates use exchange time (ET). `dividends.ex_date` / `pay_date` are UTC dates of the raw epoch; v4 used the host's local timezone.
- **`sources_failed` widened:** it now also covers HTTP errors, malformed payloads and an earnings page with no table.
- **History:** `history` lists reported quarters only.

**`analyze`**

- **Price:** `summary.current_price` is a fresh regular-session 5m bar, else the last completed-session close, else the cached info quote as a labelled fallback. v4 used the info quote, which could be up to 4 h old. Zones, stops, sizing, analyst upside and the snapshot price all follow this price. Its market-data time is `summary.price_provenance.price_as_of`; every tool's `data_provenance.*.as_of` stays the fetch time.
- **Freshness:**
  - `data_quality.component_freshness` price/risk entries count sessions from the last bar date (`last_bar_date`, `expected_session`, `sessions_behind`). v4 counted hours from the fetch time.
  - `staleness_warnings` `price_stale_<N>h` / `risk_stale_<N>h` become `…_stale_<N>_sessions`.
- **Ratios:** `fundamentals_summary.valuation` ratios are null unless usable. `metric_status` explains the excluded ones.
- **Dilution:** `fundamentals_summary.burn_metrics.dilution_analysis` is null when the price and statement currencies differ or are unverified, with the burn warning `dilution_currency_unverified`. v4 divided a raise in the statement currency by the market cap in the price currency.
- **Valuation gate (`action_zones.valuation_assessment`):**
  - It reads only usable metrics.
  - For unprofitable companies it uses EV/S first, then P/S, and "attractive" needs 0 < multiple < 3; v4 rated a negative EV/S as attractive. With EV ≤ 0 there is no P/S fallback, so the gate is `unknown`, also when there is no revenue to compute EV/S (`metric_status.ev_to_sales` reads `ev_nonpositive` either way).
  - PEG ≤ 0 is never attractive.
  - `reasons` name the excluded metrics, e.g. `pe_trailing_unverified_currency` or `ev_to_sales_ev_nonpositive`.
  - For an unsupported instrument the gate is `not_applicable`. This also reaches `policy_action.valuation_gate` and the snapshot.
- **`data_quality`:** `fundamentals_status` can be `not_applicable`. An unverified P/E gives the reason `pe_<reason>` (e.g. `pe_unverified_currency`), and the same value replaces `pe_not_meaningful` in `data_gaps`.
- **Unsupported instruments and currencies:** anything other than USD-priced common equity, or a symbol whose support can't be verified, gets new-money actions capped at `watch` everywhere. That covers `policy_action`, `decision_card.action_now`, `decision_modes` and the trade card. There is no share or dollar sizing and no dollar-liquidity gate.
  - For an unsupported instrument, the fundamentals rubric, dip and dislocation blocks read `not_applicable`. For an unsupported currency, a new-money `dip_assessment.assessment.recommendation` and the new-money `entry_timing.signals` actions read `watch`, with `support_gates` on the assessment.
- **Missing tilt:** a verdict tilt that is missing (absent or null) makes the tilt-decided actions `insufficient_data`, with rationale `tilt=missing`. In v4 an absent tilt read as neutral, and a null one as bearish; a failed fundamentals fetch could say `hold_or_reduce`.
- **Balance sheet:** in the dislocation framework, `balance_sheet_safety` is `unsafe` only when the cash runway fails (under 4 quarters). Net debt, exactly zero net cash or negative equity is `watch`, so the thesis is no longer `broken` from net debt alone.

**`get_news` and `get_options_signals`**

- **News:** `get_news` returns a normal body when the feed fails. The failure shows in `source_status` and `warnings`; v4 returned an error.
- **Options:** if `get_options_signals` can't read the expirations, it returns `data_unavailable`. v4 reported `no_options`.

### Added

**`analyze`**

- **Summary:**
  - `summary.support`: `{instrument: {status, quote_type, reason}, currency: {status, code, reason}}`; status is `supported`, `unsupported` or `unverified`.
  - `summary.price_provenance`: `price_source`, `price_as_of`, `as_of_kind` (`session_close`, `bar_time` or `quote_time`), `as_of_precision`, `fetched_at`, `cache_hit`, `fallback_reason`, and `staleness` (sessions behind, or quote age in hours).
- **Data quality:**
  - `data_quality.in_flight_jobs`.
  - `data_quality.bar_validity`.
  - Warnings `latest_daily_bar_invalid`, `latest_daily_volume_invalid` and `invalid_bars_in_indicator_windows`.
  - A `news_source_failed` tool failure.
  - `news_unverified` in `data_gaps`. `data_gaps` is now in every `detail` level; v4 had it only in `full`.
  - `fundamentals_summary.status` at the `standard` level.
- **Fundamentals summary:** `pe_forward`, `metric_status`, `health.leverage_status`, and `cash_flow.free_cash_flow_ttm_status` / `free_cash_flow_period_lag`.
- **Decision card and zones:**
  - `decision_card.hard_gates` adds `unsupported_instrument` / `unsupported_currency`.
  - `sizing.unavailable_reason`, `action_zones.position_sizing_range.unavailable_reason` and the decision modes' position `unavailable_reason`.
  - `policy_action.support_gates` when the support cap applied.
- **Events and dislocation:**
  - `events_summary.next_catalyst.window` (`{start, end}`).
  - `dislocation_framework.balance_sheet_safety.reasons`; the `summary` names the causes.

**`get_fundamentals`**

- **Valuation:** `valuation.metric_status`, `forward_eps`, `enterprise_value`.
- **Leverage:** `financial_health.leverage_status` (`reported`, `unavailable` or `negative_equity`).
- **Cash flow periods:**
  - `cash_flow.operating_cf_period`, `_period_end`, `_source`, `_ttm_status` and `_period_lag`.
  - `free_cash_flow_ttm_status` / `_period_lag`.
- **Yields:** `yield_metrics.metric_status`, `dividend_yield_basis`, `dividend_yield_trailing`.
- **History:**
  - `valuation_history.status`, `history_points`, `history_span_years` and `non_consecutive_windows`.
  - `*_stats.span_years`.
  - `valuation_context.history_points` / `history_span_years`.
- **Trends:** `fundamental_trends.quarter_gaps`.

**Other tools**

- **`get_stock_summary`:** `financial_currency`, `quote_type`, `support`, `quote_time`, `fetched_at`, `cache_hit`, and the dividend basis and trailing fields. `currency` no longer defaults to "USD".
- **`get_technicals`:**
  - `session_close`: the latest valid completed-session close, with `session_date` and `fetched_at`.
  - `bar_validity`: `latest_bar_invalid`, `latest_volume_invalid`, `invalid_bars_in_window` per indicator window, and `invalid_return_endpoints` (the `returns.return_*` that are null although the history reaches back far enough: an endpoint the return used is missing or non-finite, or its start is zero; `return_1y` follows its first-bar fallback).
- **`get_events` earnings:** `verification` (`verified_date`, `verified_none` or `unverifiable`), `next_window`, `windows` (each with `window_basis` and `metadata_conflict`), `candidate_dates`, `next_date_is_estimate`, `sources_stale`.
- **`get_news`:** `source_status` (`ok`, `empty`, `failed` or `unverified`), `source` (`news` or the `search` fallback), `rejected_count`.
- **`analyze_trade_setup`:**
  - `support`, with `unsupported_instrument` / `unsupported_currency` blockers that demote actionable cards to `watch`.
  - `event_risk.earnings_verification`, `earnings_window`, `earnings_windows`, `verified_earnings_dates`.
- **`scan_watchlist`:** a new row action `watch` when only support blocks a row.

**Snapshots and `detect_changes`**

- **One snapshot, version 2.0.0:**
  - `watchlist_snapshot` is built once per analysis and is identical at every `detail` level. `detect_changes` uses it as is.
  - It adds `zone`, `mid_term_fit`, `long_term_fit`, `bullish_signals` / `bearish_signals`, `support_instrument` / `support_currency`, and `price_source` / `price_as_of` / `as_of_kind` / `price_stale`.
  - An unknown list is null; a known-empty list is `[]`. Signal lists are null when a tool feeding them failed, the latest daily bar is empty, or `return_3m` is null because its start bar is empty.
- **Clock-driven values:**
  - The snapshot's `next_catalyst` no longer carries `days_until` (1.1.0 did). It also drops `date` when an earnings `window` is present, because inside a window that date is just today.
  - `snapshot_hash` covers the full set except `snapshot_as_of_date` (fetch dates) and `price_stale` (an age past a threshold).
- **`detect_changes` validation:** a `previous_snapshot` with another symbol, a missing `snapshot_version` / `symbol`, or a wrong-typed field returns `error_type: "invalid_snapshot"` with `problems`.
- **`detect_changes` output:**
  - It compares only fields both snapshots carry and lists the rest in `not_comparable`. Older snapshots therefore show no phantom changes, and their signal lists are not compared.
  - It adds `hash_changed`, reported only within one snapshot version.
  - **`changes` shape:**
    - Keys appear only for comparable fields.
    - It adds `confidence`, `action_mid_term`, `action_long_term`, `valuation_gate`, `support_instrument`, `support_currency` and `price_source`. `price_source` has `material: false` for a bar ↔ close switch.
    - `signals` adds `bullish_changed` / `bearish_changed`.
    - `hard_gates.previous` / `current` can be null (unknown).
  - `material_changes` now covers confidence, actions, valuation gate, support and a price-source change involving the fallback.

### Failures and errors

- **Read failures raise:** yfinance reads no longer hide failures as empty data.
  - **Reported as failed, not "none":** news, earnings sources, symbol search, option expirations and ownership.
  - **Still degrade to missing fields, as before, with no failure flag:** fundamentals statement, estimate and dividend-history reads, and the splits read in `get_events`.
  - **Provider statuses:** `ok`, `empty`, `failed`, `unverified`.
  - **`error_type` values:** `rate_limited`, `http_<status>`, `timeout`, `queued_for_admission`, `queued_behind_symbol_lock`, `unreadable_response`, `malformed`, and others.
- **History errors:**
  - An unknown symbol is still `invalid_symbol`.
  - An unreadable body (e.g. an HTML 5xx page) is retried, then reported as `data_unavailable`.
  - A 5xx or 429 that reaches the client as an HTTP error is retried with backoff, and 401 is retried once; other 4xx are not. A 5xx carrying Yahoo's JSON error body is not: see Known limitations.
- **Search:** `search_symbol` returns `data_unavailable` when Search fails upstream. In `analyze`, a company name whose search fails is reported as `invalid_parameters` rather than "no match".
- **Ownership:** a failed read gives a `*_error` warning, kept distinct from `*_unavailable`.
- **analyze timeouts:** each tool's reads stop at 90% of its deadline, so a slow read comes back as a structured failure. Worker threads that are still running are counted in `in_flight_jobs`.

### Operators

- **New environment variables:**
  - `YF_READ_DEADLINE_SECONDS` (default 8): one read's time from admission, including retries.
  - `YF_REQUEST_TIMEOUT_SECONDS` (default 5): caps every yfinance HTTP request, including cookie, crumb and consent requests.
- **`YF_MAX_WORKERS`** now gates every yfinance read. Property reads for one symbol run one at a time. A timed-out read keeps its slot until its thread finishes.
- **Retries:**
  - Retries stop when the next backoff would pass the read deadline.
  - A 429 starts a process-wide cooldown. A waiting read waits out every extension another read makes, and a cooldown that begins while a read is queued for a worker slot is checked again when the slot is granted: the read gives the slot back, waits and queues again. It fails `rate_limited` when the cooldown would outlast its budget.
- **Caches:** verified results are cached in process: calendar 1 h, earnings page 6 h, news 15 min, symbol search 1 h. Failures are never cached.

### Known limitations

- **Stale latest bar:** when the latest daily bar is missing, ATR% and dollar liquidity in `risk_metrics` are null. Using the last valid close there is a separately approved fill-policy change.
- **Empty latest rows:** how often Yahoo sends an empty latest daily row is not established. With one, `get_technicals` `current_price` and `compare`'s `position_in_52w_range` and `return_1m` are null (ranked last), and `scan_watchlist` returns `wait_for_data`, as it also does when the previous session is empty.
- **Earnings verification in analyze:** `analyze`'s earnings gate reads `days_until` only. An `unverifiable` calendar caps the trade card at `watch` but not `analyze`; Step 2 owns that gate.
- **Intraday price basis:** in the regular session, when the 5m bar is stale or missing, `analyze` zones and stops use the last completed close while technicals include today's forming daily bar. Step 2 owns stale-price gating.
- **Chart errors:** a price-history 5xx whose body is Yahoo's JSON error envelope is reported as `invalid_symbol` and not retried, as in v4.
- **Earnings day:** the `earnings_within_7_days` warning is skipped when `days_until` is 0, as in v4 (the blackout gate still fires).
- **Core profile mismatch:** with a missing tilt, the core decision mode can still read `build_position` while the policy long-term action is `insufficient_data`. Aligning action surfaces belongs to Step 2.
- **Non-USD prices:** prose and sizing fields that print `$` are only correct for USD prices. Other currencies are unsupported, and FX conversion is not planned yet.
