# Changelog

## Unreleased: Step 2, one canonical decision

Not released yet. The README install commands still install v0.2.0; keep a live MCP client on v0.2.0 until the release (see [Running main before the release](#running-main-before-the-release)).

Two versions change; the package version (`meta.server_version`) stays `0.2.0` until the release:

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

**Examples from the decision matrix (v0.2.0 → main):** a plain `hold_or_add` becomes `starter`; an earnings blackout on it goes from `wait_for_data` to `wait`; weak liquidity from `wait_for_data` to `avoid`; an ETF's `hold` becomes `watch`; a broken business with a bullish chart goes from `hold_or_add` to `avoid` (`thesis_broken`).

### Decision rules

- **Combination:** at each horizon, `new_money.action` = the most restrictive of the mapped candidate and the outcome of every fired gate, in the order avoid > watch > insufficient_data > wait_for_data > wait > starter > buy. A gate never loosens an action and can tighten one that already denied new money (a support gate turns `wait` into `watch`). v0.2.0 turned any gate on `buy`, `starter` or `hold_or_add` into `wait_for_data` (`watch` with a support gate) and passed every other action through.
- **Candidate mapping:** mid-term `buy` → buy; `hold_or_add`, `speculative_small_position` → starter; `hold`, `hold_or_reduce`, `wait_for_entry` → wait; `avoid`, `insufficient_data` unchanged. Long-term `accumulate` → buy; `hold_or_add`, `small_position_with_stops`, `speculative_with_thesis`, `speculative_only` → starter; `hold`, `hold_size_conservatively`, `wait_or_avoid` → wait; `avoid_core_position` → avoid. An unknown code now raises. The candidate no longer carries the falling-knife override (`dip_type=falling_knife`); the knife is the `falling_knife` gate.
- **Gates (analyze), in rank order:**

  | Gate | Class (rank) | New money, mid / long term | Change |
  |---|---|---|---|
  | `stale_data`, `freshness_unverifiable` | data (1) | wait_for_data | New in analyze |
  | `data_quality_critical` | data (1) | wait_for_data | A missing beta alone no longer fires it |
  | `missing_runway` | data (1) | wait_for_data | |
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
- **`unevaluated`:** applicable gates that could not be computed, each `{id, reason}`: `falling_knife` (no technicals), `missing_runway` and `thesis_broken` (fundamentals missing), `earnings_blackout` (events tool failed), `near_52w_high` (`price_zone` null or undetermined).
- **`if_held` (mid-term, first match):** runway under 4 quarters → `review_exit` (`runway_critical`); business broken → `review_exit` (`business_broken`); candidate admits new money → `hold`; dislocation candidate → `hold` (`price_broken_more_than_business`); tilt missing → `insufficient_data`; candidate `insufficient_data` → `insufficient_data`; tilt bearish → `reduce` (`tilt_bearish`); otherwise `hold` (`no_reduction_evidence`). `add_allowed` is true only when new money is `buy` or `starter`.
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
- `decision_card.hard_gates.checks` grows from 8 to 13 keys (`stale_data`, `freshness_unverifiable`, `thesis_broken`, `earnings_unverifiable`, `near_52w_high`), in rank order; `blocking` is in rank order too.
- `diagnostics` (`detail="full"` only): `pre_gate_policy` (`mid_term`, `long_term`, `rationale`, `conditions_to_upgrade`, `conditions_to_downgrade`, `inputs: {zone, valuation_gate, valuation_basis, is_unprofitable}`, `support_gates` when support capped it, `role`, `superseded_by: {new_money, if_held, long_term, blocked_by}`) and `decision_trace`.
- `data_quality.freshness` at every detail level: `session`, `expected_session`, `price_source`, `daily_last_valid_date`, `daily_completed_session`, `daily_sessions_behind`, `codes`, `flags`.
- `action_zones.price_zone`: `near_52w_low`, `below_sma200`, `above_sma50`, `at_or_below_sma50`, `near_52w_high`, `undetermined` or null, from the uncapped classification.
- `dislocation_check.decision` (standard tier): `{new_money, blocked_by}`.
- `capped_by` on capped `decision_modes.core` / `speculative`, `dislocation_framework.action.capped_by`, `dip_assessment.assessment` and dip `entry_timing.signals`.
- `long_term.new_money.conditions.buy_only_if`.
- `data_gaps` value `beta_unavailable`.
- `get_technicals` `daily_history`: `last_row_date`, `last_valid_date`, `last_row_valid`, `completed_valid_date`.
- Trade card: `horizon: "swing"`, `decided_by`, `unevaluated`. Scan rows: `horizon: "swing"`, `source` (`screen` or `card`), `unevaluated`; scan response and state: `rules_version`; warning `rules_changed`.
- Snapshot 3.0.0: `new_money_action`, `if_held_action`, `long_term_action`, `price_zone`.

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

- **Fields:** 3.0.0 is 2.0.0 without `action_mid_term`, `action_long_term`, `decision_action_now` and `zone`, plus `new_money_action`, `if_held_action`, `long_term_action` (= `long_term.new_money.action`), `price_zone` and `unevaluated_gate_ids`. They are hashed and diffed; any change is material. `price_zone` follows the uncapped bands, so a capped name (falling knife, extended decline, extreme risk, valuation headwind) can now report zone moves the capped `zone` did not; they are context, not advice. `hard_gate_ids` lists every fired gate and `unevaluated_gate_ids` every gate whose input failed (`new_money.unevaluated` at either horizon). A gate whose input failed is reported as not evaluated (`+unevaluated_gate`, "Gate(s) not evaluated", `changes.hard_gates.not_evaluated`), never as cleared, also against an older snapshot: "Cleared hard gate(s)" means the gate was evaluated and did not fire. A gate evaluated again is `-unevaluated_gate` ("Gate(s) evaluated again", `changes.unevaluated_gates`). Reasons, the pre-gate candidate, `earliest_reassessment` and the clock stay out of the hash.
- **Validation:** a 3.0.0 previous snapshot must carry every 3.0.0 key (null allowed), else `invalid_snapshot`. Older versions validate only their own keys and ignore stray ones.
- **Across versions:** fields both versions carry compare as before, 2.0.0 signals included. The three action fields, `price_zone` and `unevaluated_gate_ids` are not compared, and the summary names them ("Not compared with the previous 2.0.0 snapshot: …"); a 1.1.0 snapshot therefore loses the `action_now` alert v0.2.0 gave it, for one cycle. A version warning names the retired fields and the 3.0.0 rule changes: new analyze gates `earnings_unverifiable`, `stale_data`, `freshness_unverifiable`, `thesis_broken` and `near_52w_high`; knife and weak liquidity → avoid; earnings → wait; `liquidity_missing` is data class; the knife is evaluated for every instrument; a missing beta alone no longer fires `data_quality_critical`; `zone` → `price_zone`. Rule-driven gate changes are reported, not suppressed. `hash_changed` is reported only between two 3.0.0 snapshots.
- **Freshness:** a new-money change caused only by a freshness gate is material. A growing age alone is never an alert (`price_stale` and ages stay unhashed).
- **Renamed in the response:** the `changes` keys `action_zone` → `price_zone` and `decision_action_now` → `new_money_action`; `action_mid_term` and `action_long_term` are gone (the long-term decision is `long_term_action`); `if_held_action` is new. The material lines read `price_zone: a -> b`, `new_money_action: a -> b` and so on; the summary says "Price zone moved from a to b." and "New money: a -> b.", "If held: a -> b.", "Long term: a -> b." (was "Zone moved …" and "Action now: …").

### Agent guidance

- **Server instructions:** the server now sends MCP `instructions`: start with `analyze`; a bare ticker never triggers a question and gets the mid-term decision, labelled as mid-term; ask for a horizon only when the request names two horizons, or a holding period that fits none (a period that maps to one horizon, such as "hold for five years", gets it); where each horizon's answer lives; never upgrade past `blocked_by`; size only from `decision_card.sizing`.
- **Tool descriptions** start with "Use when …". `analyze_my_position` is described as P/L, holding period and tax, the hold/reduce decision being `analyze`'s `decision_card.if_held` (it was "hold/sell decision"); `compare` applies no decision gates; `analyze_trade_setup` and `scan_watchlist` are the swing horizon; `detect_changes` names its snapshot's horizons. The `profile` argument is documented as changing sizing only.
- **Rendering guide** (`stock-analysis://guides/analyze-rendering`), rewritten for every detail level (it assumed `detail="full"`): the card first, in three horizon segments; never upgrade past `blocked_by`; sizing only from `decision_card.sizing` ("no entry size: new_money is <action>"); `price_zone`, levels, the dip assessment, `diagnostics` and third-party data are context, not advice; `can_start_now` is deprecated; the reassessment date is not a buy date; a field-class table.
- **Prompts:** `full_analysis` leads with `decision_card` (`policy_action` is gone); `growth_memo` and `value_memo` ask for `decision_card.new_money.action` and `decision_card.long_term.new_money.action` instead of "Buy / Wait / Pass"; `position_decision` calls `analyze` for `decision_card.if_held` and `analyze_my_position` for P/L and tax, without `get_technicals` or HOLD / SELL / TRIM.

### Trade card and scan

- **Trade card (`analyze_trade_setup`):** `earnings_unverifiable` is disclosed on every outcome and still caps only a detected setup at `watch`. Duplicate blocker ids merge (reasons joined by "; "). A close that is not finite and positive is never the actionable price: in session a non-positive 5m close is `freshness_unverifiable` (`nonpositive_close`); a non-positive daily close fails the daily leg (`stale_data` in session, `freshness_unverifiable` off-hours). The resolution order is unchanged.
- **Scan stage 1:** it evaluates every gate it can and resolves rank-first like the card. Data first: under 50 bars, no features, or off-session no valid close is `data_quality_critical`, and the card's daily freshness leg applies (both `wait_for_data`); then the knife and liquidity on the card's basis (1-year mean volume × last close; was a 20-bar dollar mean) give `avoid`. An unsupported instrument or currency with a setup is `watch` with the setup kept; unverified support is `unevaluated` and a setup promotes, so the card decides. Screen rows report the detector's `trigger_satisfied` (was always false).
- **Scan diffs:** blockers compare as (id, leg) pairs that neither side left unevaluated; an entry saved before this change compares at id level. A state from other rules adds a `rules_changed` warning.

### One-time alerts after upgrading

- **`detect_changes` with a saved 2.0.0 (or 1.x) snapshot:** the version warning, the "not compared" summary for the action fields and `price_zone`, and `±hard_gate` lines for gates the new rules fire or clear: `+earnings_unverifiable`, `+stale_data` / `+freshness_unverifiable`, `+thesis_broken`, `+near_52w_high`, `+falling_knife` on ETFs and other unsupported instruments, `-data_quality_critical` where only beta was missing. On the decision matrix, 3 of 28 recorded snapshots alert (`+hard_gate: thesis_broken`). Save the new 3.0.0 snapshot to restore full comparison.
- **First `scan_watchlist`:** a `rules_changed` warning, plus one round of `changes` from the new rules: `earnings_unverifiable` on non-setup cards, and rows whose stage-1 outcome or `trigger_satisfied` changed. Leg suffixes and merged duplicate ids alone are not reported.

### Running main before the release

- Keep a live MCP client on v0.2.0 (the README commands pin it). If you run main, give it its own `STOCK_ANALYSIS_DATA_DIR`.
- **Rollback:** a v0.2.0 scan on a data dir last written by main reports one round of rule-driven changes with no warning (v0.2.0 has no `rules_version` check), and v0.2.0's `detect_changes` rejects a 3.0.0 snapshot as `invalid_snapshot` (`unsupported snapshot_version: '3.0.0'`). Keep the 2.0.0 snapshot until you upgrade.

### Known limitations (main)

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
