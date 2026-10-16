# Repository Instructions

## Purpose
- This repository provides the `stock-analysis` MCP server for single-stock analysis.
- The public contract is the MCP tool surface and the JSON schemas returned by those tools.

## Key Paths
- `src/stock_analysis/server.py`: MCP server entrypoint
- `src/stock_analysis/tools/`: tool implementations
- `src/stock_analysis/data/`: yfinance client and caching; `provider.py` reads the sources whose failures yfinance would otherwise hide (calendar, earnings page, news, Search news and quotes) and the `Ticker` property reads; `jobs.py` runs every blocking read as an executor job
- `src/stock_analysis/resources/`: MCP resources
- `tests/`: schema, invariants, and utility coverage
- `README.md`: install, usage, and public-facing behavior

## Working Rules
- Keep changes consistent with the existing MCP tool surface unless the task explicitly changes it.
- Preserve response-shape stability for `analyze`, `compare`, and other public tools. If a schema changes, update tests and docs in the same change.
- Prefer extending existing analysis and normalization code over adding duplicate logic paths.
- Surface missing, stale, or low-confidence data through provenance and data-quality fields instead of hiding it.
- Keep tests deterministic. Do not hardcode live market values or rely on current dates unless the test is explicitly about date handling.
- Treat output as informational software, not financial advice. Do not weaken existing disclaimers.

## Validation
- Install dev dependencies with `uv pip install -e ".[dev]"`.
- Run `uv run pytest` after behavior changes.
- Run `uv run ruff check` after code edits.
- Run `uv run mypy` when touching typed Python code or schemas. `mypy` is available via the `dev` extra in `pyproject.toml`.

## Test Characteristics
- The suite is unit-test focused and runs offline. An autouse guard in `tests/conftest.py` fails any test that sends a request through a `curl_cffi` or `requests` session (yfinance's transport), or makes an unrouted stub request, even when app code swallows the error. It also points `STOCK_ANALYSIS_DATA_DIR` at a per-test temp dir and yfinance's cookie/tz caches at a session temp dir, disables retry sleeps, and clears provider source caches, job permits/locks and in-flight singleflight entries between tests. Concurrency tests (`tests/test_jobs.py`) block jobs on `threading.Event` barriers and must release them; `FakeSession.hold(source)` blocks a transport request the same way.
- To test yfinance behavior itself (error suppression, caching, parsing), drive the real pinned library with only its HTTP session stubbed: the `yf_transport` fixture or `stubbed_yfinance_transport()` in `tests/support/transport.py`, with synthetic fixtures under `tests/fixtures/transport/`. `tests/support/fake_ticker.py` is an app-level fake that skips yfinance internals; `tests/support/fake_source.py` (`FakeSource`) stands in for one `provider` read in tool-level tests.
- `tests/test_yfinance_transport_contract.py` pins current library behavior, including known defects; a change that alters how a source is read updates those tests deliberately.
- `scripts/record_provider_fixtures.py` records real responses (manual, networked) to the gitignored `tests/fixtures/_recorded/`; never commit recordings.
- Keep the suite fast enough for routine local validation.

## When Changing Analysis Logic
- Update schema and invariant tests under `tests/test_schemas.py` and `tests/test_tools/` when score math, section ordering, or required fields change.
- Keep score math internally consistent and auditable.
- If you change user-visible output semantics, update the relevant examples in `README.md`.

## Decision Contract (schema 6)
- `analyze` decides once, gates first: `tools/analyze/decision.py` (`decide()`,
  pure) over the `GATES` registry in `tools/analyze/gates.py`. Every other
  action-like surface (executive summary, decision modes, dislocation, dip,
  snapshot) derives from its output and never feeds back into it.
- `decision_card` (every `detail` level):
  ```
  horizon: mid_term
  new_money: {action, blocked_by: [{id, reason, class, effect, codes?}],
              decided_by, unevaluated: [{id, reason}], earliest_reassessment}
  if_held: {action, add_allowed, basis}                    # mid-term
  long_term: {horizon: long_term, new_money: {action, blocked_by, decided_by,
              unevaluated, earliest_reassessment, conditions: {buy_only_if}}}
  hard_gates: {any_blocking, blocking: [every fired gate], checks: {id: bool}}
  action_now                                               # = new_money.action, one release
  why, exit_triggers, data_flags
  sizing | null       # mid-term new money only: buy or starter; a starter nulls full_pct_cap and full_dollars
  entry | null        # {current_price, buy_zone_low, buy_zone_high}, same predicate
  exit, conditions (mid-term), next_review, confidence
  ```
  `diagnostics` (full tier only) holds `pre_gate_policy` (the support-capped
  candidate: codes, rationale, conditions, `inputs`, `role:
  "pre_gate_input_not_advice"`, `superseded_by`) and `decision_trace`. Never copy
  a pre-gate value into the card, a mode, the summary or the snapshot.

| `new_money.action` | Meaning |
|---|---|
| `buy` | Entry allowed, up to full size. |
| `starter` | Entry allowed at starter size only. |
| `wait` | No entry now: the candidate, an event gate or `near_52w_high`. |
| `wait_for_data` | No entry: a data-class gate fired. |
| `insufficient_data` | No entry: the candidate lacks data. |
| `watch` | No entry: a support gate fired. |
| `avoid` | No entry: the candidate, `falling_knife`, `weak_liquidity` or `thesis_broken`. |

- Combination: per horizon, `action = most_restrictive(map(candidate), outcome of
  every fired gate)` in the order avoid > watch > insufficient_data >
  wait_for_data > wait > starter > buy; a gate never loosens. The trade card and
  scan stage 1 resolve rank-first instead (the card's frozen order); the
  differences are pinned in tests. `decided_by` is the highest-ranked fired gate
  whose outcome equals the action and is tighter than `map(candidate)`.
- Gates, by rank (analyze's `blocking`, `blocked_by`, `checks` and
  `outstanding` use this order): data 1 (`stale_data`, `freshness_unverifiable`,
  `data_quality_critical`, `missing_runway`, and in analyze `liquidity_missing`)
  → wait_for_data; price_risk / liquidity 2 (`falling_knife`, `weak_liquidity`)
  → avoid; thesis 3 (`thesis_broken`) → avoid; support 4 → watch; event 5
  (`earnings_blackout`, `earnings_unverifiable`) → wait in analyze, watch on the
  card; zone 6 (`near_52w_high`) → wait, mid-term only; session 7
  (`market_closed`) and plan 8 (`plan_invariant_violation`) are card-only. A new
  gate id must be a `GATES` entry with the same class and rank in every tool
  (only `liquidity_missing` differs: data class in analyze).
- `if_held` (first match): runway < 4 quarters or business broken →
  `review_exit` (`runway_critical` / `business_broken`); candidate admits new
  money → `hold`; dislocation candidate → `hold` (`price_broken_more_than_business`);
  tilt missing or candidate `insufficient_data` → `insufficient_data`; tilt
  bearish → `reduce` (`tilt_bearish`); else `hold` (`no_reduction_evidence`).
  Gates never change `action` or `basis`; `add_allowed` is new money `buy` or
  `starter`.
- `earliest_reassessment` is non-null only when a gate fired and the mapped
  candidate admits new money at that horizon; only `earnings_blackout` is dated
  (first NYSE session no window blacks out); undated gates go to `outstanding`
  in rank order, event gates as one `earnings_reverify`. It never changes an
  action and is not in the snapshot hash.
- Conditions are per horizon and binding only: `decision_card.conditions` is
  mid-term, `long_term.new_money.conditions` long-term; core's conditionals
  follow long term, balanced / speculative / dislocation / dip follow mid-term.
  Values are predicates: necessary, not sufficient.
- Derived surfaces: balanced = `new_money.action`; speculative and core are
  their own codes capped by new money / `long_term.new_money` (`capped_by`);
  dislocation `can_start_now` (deprecated) = new money `buy` or `starter`; dip
  codes are capped by new money (`capped_by`). The `profile` argument changes
  sizing only, never an action.
- Field classes (closed registry `FIELD_REGISTRY` in
  `tests/support/output_contract.py`; every leaf at every tier matches one row,
  so a new output field needs a row):

| Class | Meaning |
|---|---|
| A | Actionable: only from `decide()`, with one horizon. |
| S | Sizing (a subset of A): only when new money is `buy` or `starter`. |
| C | Conditional: buy/add words only in keys; values are predicates. |
| D | Diagnostic: no entry codes, no advice phrases. |
| N | Our narrative: the executive summary's three horizon segments. |
| P | Pre-gate: full tier only, under `diagnostics`. |
| T | Third-party (analyst, news, insider, options): never in `decision_card`. |
| E | Static config echo. |

- In `watchlist_snapshot`, `new_money_action` and `if_held_action` are the
  mid-term decision and `long_term_action` the long-term one; `scan_watchlist`
  state holds swing-card actions. The snapshot has no `horizon` field.
- Agent guidance: `server.INSTRUCTIONS` (MCP `instructions=`) carries the horizon
  rule; every tool description starts "Use when …"; the rendering guide
  (`resources/analyze_guide.py`) and the prompts (`prompts/templates.py`) name
  only existing fields and the canonical vocabulary
  (`tests/test_step2/test_wp24_guidance.py`).
- Decision matrix (`tests/support/decision_matrix.py`): after a decision or
  derived-surface change, re-record with
  `uv run --frozen python -m tests.support.decision_matrix --after` (writes
  `after.json` / `after.md` only) and extend `ROW_CHANGES`; never edit those
  files by hand. `before.json`, `README.md` and `snapshots_2_0_0/` are the frozen
  v0.2.0 baseline.

## Common Schema Gotchas
- Two versions move independently: `meta.schema_version` (response schema,
  `SCHEMA_VERSION` in `src/stock_analysis/__init__.py`, now "6"; v0.2.0 shipped
  "5") and `watchlist_snapshot.snapshot_version` (`normalize.SNAPSHOT_VERSION`,
  now "3.0.0"; v0.2.0 shipped "2.0.0"). CHANGELOG.md lists every field that
  changed meaning or units since schema 4, unreleased changes under
  "Unreleased"; update it with any further output change.
- `technicals` returns `current_price` at the top level, not under a `summary` object.
- `analyze` returns price, name, sector, market cap, and currency under `summary`, not as duplicated top-level fields.
- Support boundaries (WP4, D5): `summary.support` (and the trade card's
  `support`) is `{instrument: {status, quote_type, reason}, currency: {status,
  code, reason}}`, status `supported` / `unsupported` / `unverified`. Only
  `EQUITY` (not a detected preferred: Yahoo `-P` / `-P<letter>` suffix, or a
  name with `Pfd`, `Preferred Stock/Shares/Securities` or `Non-Cumulative`;
  ordinary ADRs named "American Depositary Shares" stay supported) priced in
  `USD` is supported. In `analyze` and the trade card unknown values fail
  closed: unsupported or unverified (instrument or currency) fires the
  `unsupported_instrument` / `unsupported_currency` support gate, which caps new
  money at `watch` at both horizons (most-restrictive: `wait` becomes `watch`,
  `avoid` stays `avoid`); `if_held` never reads support. There is no
  share/dollar sizing (`watch` is not an entry, so every sizing block is null)
  and no dollar-liquidity gate (liquidity gates are not_applicable unless
  supported). The emitted `diagnostics.pre_gate_policy` is support-capped too
  (`support_gates`); every other surface follows `decide()`. An unsupported
  instrument also skips the fundamentals rubric, dip and dislocation
  (`not_applicable`, including the verdict's fundamentals exclusion,
  `decision_context.fundamentals`, its `thesis_checkpoints` and the standard
  tier's `dislocation_check`), with no stock-only conditions (a permanently
  unsupported gate empties both horizons' conditions); the falling-knife gate
  still reads its technicals. An unsupported currency's dip recommendation and
  entry-signal actions read `watch` with `capped_by` (the assessment also lists
  `support_gates`), and
  burn `dilution_analysis` needs the price
  and statement currencies to match (else null, warning
  `dilution_currency_unverified`). Scan stage 1 reads currency and instrument
  type from the daily history's chart metadata (`df.attrs`, no extra request):
  unsupported caps the row at `watch` (setup kept) and never promotes it;
  unverified is listed in the row's `unevaluated`, and a setup promotes so the
  card decides. `stock_summary.currency` no longer
  defaults to USD; it adds `quote_type`, `financial_currency` and `support`.
- Valuation basis (WP4): ratios stay scalars; each section carrying them has a
  sibling `metric_status` `{metric: {basis_status, scoring_applicable,
  reason}}`. A ratio mixing price and statement data (P/E, P/S, P/B, EV/S,
  EV/EBITDA, FCF yield, trailing dividend, valuation history) is kept only when
  the price and financial currencies are identical (`GBp` != `GBP`), else null
  with `unverified_currency` (a USD ADR reporting in TWD loses all of them);
  P/B and the trailing dividend also need `marketCap ≈ price ×
  sharesOutstanding`. P/E is undefined when its own EPS <= 0 (forward P/E
  survives a negative trailing EPS). Yahoo's PEG is basis `provider` and only
  PEG > 0 is scored. EV can be negative: EV/S is kept but unscored when EV <=
  0, and the unprofitable gate is then `unknown` (no P/S fallback; P/S stands in
  only when EV/S is absent); EV/EBITDA is undefined when EBITDA <= 0. P/B <= 0
  is kept, unscored. A gate with no usable metric names each excluded one
  (e.g. `pe_trailing_unverified_currency`). A P/E that Yahoo omits is absent
  (no status), not unverified. FCF payout and the dividend safety score's FCF
  coverage need one currency and consistent share units.
  Consumers (valuation gate, verdict, prose, decision context, sector
  comparison, `compare`) read ratios only through
  `valuation_basis.usable_metric`. D/E is always Yahoo's percent / 100;
  `financial_health.leverage_status` is `negative_equity` when D/E < 0 or D/E is
  null with P/B < 0, and is never low debt. Dividend yield is one forward
  fraction (`dividendRate / price`) on all three paths, with
  `dividend_yield_basis` (`forward`; `no_dividend` with 0.0; `unverified_currency`
  with null for a minor-unit or unknown price currency; null when there is no
  dividend data or no price) and
  `dividend_yield_trailing` (`dividends.yield_basis` / `yield_trailing` in
  events). Valuation-history stats need >= 8 points (`insufficient_history`),
  each metric's `*_5y` fields need that metric's own 5-year span
  (`pe_stats.span_years` / `ps_stats.span_years`), and `current_percentile`
  ranks the current usable ratio; `history_points` and `history_span_years`
  are always exposed. Valuation conditions and thesis milestones follow the
  gate's own basis (no P/S condition when EV <= 0 left the gate `unknown`).
  EV <= 0 sets `metric_status.ev_to_sales` to `ev_nonpositive` even with no
  revenue to compute EV/S, so P/S never stands in.
- Statement periods (WP6, `fundamentals._resolve_cashflow`): `cash_flow`
  keeps `operating_cf_ttm` / `free_cash_flow_ttm` (key names kept, whatever
  the period), each with `_period`, `_period_end`, `_source`, `_ttm_status`
  and `_period_lag`. A TTM is the 4 most recent quarter columns only when they
  are consecutive (77-125 days apart: calendar, 13-week and 12/16/17-week
  fiscal quarters count, a skipped quarter, at least 168 days, does not) and all
  finite; otherwise `ttm_status` is `ttm_incomplete` (`unavailable` with no
  usable quarterly row), never a sum that skips the gap. Statement rows are read as
  plain floats first, so a `pd.NA` in a nullable (`Float64` or object) row is
  a gap too. Next comes the newest finite
  annual value labelled from its own column (`FY <year>`, its period end) with
  `period_lag` = newer empty columns, then the info figure labelled `TTM` /
  `info`. FCF rebuilt as OCF + capex uses one period (the same 4 quarters, or
  the same annual column; the newest annual candidate wins). Rolling P/E and
  P/S history skips non-consecutive 4-quarter windows and lists them in
  `valuation_history.non_consecutive_windows` (None when an unverified share
  basis stopped the history before any window was checked); trend QoQ / YoY / margin trend
  / 3Y CAGR need an unbroken chain of quarters (`fundamental_trends.quarter_gaps`
  lists quarters whose older neighbour isn't the previous quarter). Analyze's
  `fundamentals_summary.cash_flow` carries `free_cash_flow_ttm_status` and
  `free_cash_flow_period_lag`.
- Classification inputs (WP9): the dislocation balance-sheet check is `unsafe`
  only when the cash runway fails (< 4 quarters), negative equity included;
  net debt (`net_cash_positive` False, exactly zero included) or negative
  equity is `watch`, and `balance_sheet_safety.reasons` names why
  (`runway_under_4q`, `runway_under_8q`, `net_debt`, `net_cash_unknown`,
  `negative_equity`, `high_leverage`, `dilution_risk`). The thesis reads `broken` only when the
  business is broken or the runway fails, and `watch` never makes a dislocation
  candidate (that needs `safe`). In `build_policy_action` an absent or null
  tilt is missing, not neutral or bearish: the branches it would decide become
  `insufficient_data` with `tilt=missing` in the rationale, while avoid fits and
  valuation- and dislocation-driven actions are unchanged (with a known
  mid-term fit of `ok`, a failed fundamentals fetch used to read as
  `hold_or_reduce`). Since Step 2 the candidate has no falling-knife override:
  the knife is the `falling_knife` gate. `balance_sheet_safety.summary` names the
  causes, and a missing tilt with an `ok` mid-term fit makes the balanced
  mode's suitability `needs_more_data`.
- `analyze` returns the `standard` projection by default; deep audit fields such as `decision_context`, `dip_assessment`, `diagnostics`, and full `verdict` / `data_quality` require `detail="full"`.
- `options_signals` returns put/call data at `put_call_ratio.volume_based` and `put_call_ratio.oi_based`, not under `summary.*`.
- In `analyze`, raw options output is nested under `options_signals`, while derived narrative fields appear elsewhere such as `section_summaries`, `signals`, and `decision_context`.
- `risk_metrics` keeps volatility at `volatility.annualized`; the synthesized `analyze` output exposes the normalized view under `risk_summary.annualized_volatility`.
- `analyze_trade_setup` returns `plan` only when `action` is `trade_now` or
  `enter_on_trigger`; `watch` keeps `setup` with a null `plan`.
- `analyze_trade_setup` reuses the `wait_for_data` action code from the decision
  card; there is no `wait_for_fresh_data`.
- `analyze_trade_setup` sizing is R-based (risk budget ÷ stop distance) under
  `plan`. `analyze` sizing is `decision_card.sizing`, present only when new money
  is `buy` or `starter` (a starter hides the full size); the accumulation-style
  `action_zones.position_sizing_range` is null unless new money is `buy`.
  Dollar/share fields are `null` (not `0`) when `account_size` is omitted.
- `analyze_trade_setup.freshness.as_of` is a market-data bar timestamp, never
  fetch time; `basis="unverifiable"` forces `wait_for_data`.
- When `analyze_trade_setup.action` is not `trade_now`/`enter_on_trigger`, do not
  render entry/stop/target/sizing structure and do not mix `analyze` fields into
  the card; forward-looking levels are conditions to watch, not a plan.
- `analyze_trade_setup` lists an `earnings_unverifiable` blocker on every
  outcome when `events_calendar` fails or `earnings.verification` is
  `unverifiable`, and caps a detected setup at `watch`; `verified_date` and
  `verified_none` avoid the blocker. When `events_calendar` fails and no
  blackout fired, `earnings_blackout` is listed in `unevaluated`. The
  blocker reason names the failed and stale sources and an assumed window.
  `event_risk` carries `earnings_verification`, `earnings_window` (the window
  behind `next_earnings_date`, if any), `earnings_windows` (every source
  window) and `verified_earnings_dates`, so an assumed window's start is never
  shown without its uncertainty.
- `analyze_trade_setup` cards carry `horizon: "swing"`, `decided_by` (the
  highest-ranked fired gate whose outcome is the shown action and tighter than
  the uncapped outcome; `market_closed` never decides), `unevaluated`
  (`falling_knife` without technicals, `earnings_blackout` when events failed),
  and blockers with their registry `class` and `effect`; duplicate ids merge
  (reasons joined by "; "). The card's rank-first resolution order is frozen.
- `get_technicals` default response gained one additive `short_term` key;
  `intraday` exists only under `timeframe="swing"`. `prior_*` level fields
  exclude the current bar.
- `short_term.rvol.value` excludes the current bar from its 20-day average;
  the legacy `volume.ratio` includes it — the two legitimately differ.
- `intraday.freshness` is disclosure, not gating: stale/missing intraday data
  nulls dependent fields and adds `{id, reason}` warnings instead of blocking.
- Breakout `target_primary` is the nearest of measured-move / 52-week-high
  strictly above the effective entry anchor; `None` means the 1R fallback.
- `get_market_state().method` is `"calendar_static"` within the vendored
  NYSE calendar (2025–2030) and `"clock_only_no_holidays_fallback"` outside
  it. Early-close days run regular 9:30–13:00 with after-hours ending at
  17:00 (approximation). To extend coverage: add the year's dates to
  `utils/market_calendar.py`, bump `CALENDAR_LAST_YEAR`, update the count
  tests.
- Freshness treats any close that is not finite and positive (NaN/±inf/<= 0,
  `helpers.valid_price`) or a bar timestamp more than 5s in the future as
  unverifiable — a quote without a trustworthy price cannot verify the
  actionable price, and a daily close that fails `valid_price` is never the
  card's actionable price. The session-specific
  daily-frame rules (regular two-leg vs off-hours) live in the
  `analyze_trade_setup.freshness` bullet below.
- `events_calendar` earnings: `days_until` counts ET calendar days with day-of
  earnings = `0` (same-day earnings count as day 0). The calendar
  (quoteSummary `calendarEvents`) and the earnings page are read through
  `data/provider.py` with HTTP status and envelope checks, not yfinance's
  properties. `sources_failed` lists a source whose read failed (HTTP error,
  error or malformed envelope, parse failure) or could not be verified: an
  earnings page with no table (an error page or no data) does not verify (D2).
  A verified absence -- a valid calendar envelope without earnings, an earnings
  table with no future rows -- is NOT a failure. Calendar and info earnings
  epochs convert to ET dates (never the host time zone, unlike
  `Ticker.calendar`); dividend epochs are UTC midnight, so they convert to UTC
  dates. The earnings page is parsed with BeautifulSoup in page order (newest
  first), all or nothing (any unparseable row fails the page), and the earliest
  future row wins. Naive earnings timestamps are treated as exchange-local ET.
- `events_calendar` earnings verification (WP2 truth table): `verification` is
  `verified_date` when any source has a verified future date, `verified_none`
  when all three sources (calendar, earnings page, info earnings dates)
  verify no future date, else `unverifiable`. A calendar whose dates are all
  past is stale (`sources_stale`) and does not verify. Two calendar entries are
  an interval (`window_basis` `confirmed` only when every present
  `isEarningsDateEstimate` flag -- calendar and info -- is the boolean `true`
  and info's `earningsTimestampStart/End` match; a non-boolean flag or bad range
  is malformed and leaves it `assumed`; contradicting metadata or disagreeing
  flags give `assumed` with `metadata_conflict`). `windows` lists every source
  window (calendar, info) with its `source`; `next_window` is the one behind
  `next_date`, or `null` when a point date drives it.
  An assumed window still sets `next_date`/`days_until` (to its start, or `0`
  while `window_active`) but never verifies on its own. Reversed epochs fail
  the calendar. Info and calendarEvents carry the same Yahoo record, so an info
  date inside an unconfirmed calendar window does not verify it. Info's range
  is read on its own (with or without `earningsTimestamp`; the window spans the
  earliest to the latest of the three, and a past `earningsTimestamp` is the
  last report, so it is ignored); a multi-day range is an unverified
  window that still drives the blackout while it covers today, and a malformed
  range (one end missing, not an epoch, reversed) fails info. `next_date` is the
  earliest risk date across all sources, verified or not; `candidate_dates`
  discloses each verified source's date. `history` holds reported quarters
  only, newest first.
- `get_news` adds `source_status` (`ok` | `empty` | `failed` | `unverified`),
  `source` (`news` feed or the `search` fallback) and `rejected_count` (ads and
  items without a usable publish date). The feed is read with status and
  envelope checks and falls back to Search unless it returned items. A failed
  or unverified feed warns instead of reporting "No news articles found";
  `analyze` lists a failed feed in `tool_failures`, and a failed or unverified
  feed does not count toward news coverage (`news_unverified` data gap).
- Data layer: yfinance's `hide_exceptions` is set to `False` once, at import of
  `data/yfinance_client.py` (never per call). Retries classify `curl_cffi`
  HTTP status (401 once; 5xx with backoff; 429 and `YFRateLimitError` with
  backoff and a shared cooldown; 404 never), timeouts, connection errors and
  other transport errors; an unparseable history body is retried, not reported
  as `invalid_symbol`, and every retry attempt clears yfinance's `cache_get`
  LRU so it does fresh I/O. Provider reads cap retry backoff at 4 s (at most
  three attempts). They cache only `ok`/`empty` results
  (calendar 1h, earnings page 6h, news and Search 15 min, symbol search 1h).
  Nested fields are validated too: a malformed calendar earnings/dividend
  field, or a non-object news/Search item, makes the read `failed`. Symbol
  search reads Search quotes the same way, so a failed search is a
  `data_unavailable` error, never a cached empty result. An info quoteSummary
  404 falls back to the v7 quote endpoint like yfinance's hidden mode did. A `Ticker` or info cache miss clears yfinance's `cache_get`
  LRU so the refetch does fresh I/O; a failed `Ticker` attribute read evicts
  that Ticker. Supported range: `yfinance>=1.0,<2`.
- Concurrency (WP3, `data/jobs.py`): no yfinance read runs on the event loop.
  Each attempt is one executor job that takes an admission permit (at most
  `YF_MAX_WORKERS`, default 4) -- plus the symbol's lock for `Ticker` property
  reads, since `Ticker` state is not thread-safe -- when submitted, and gives
  them back only when its thread finishes. A cancelled or timed-out waiter just
  stops waiting; the orphaned job keeps its permit and lock, so the limit holds
  and one symbol's property reads never overlap. Freed slots go to waiters in
  arrival order. Backoff and rate-limit cooldown sleeps hold neither. Both retry
  loops run jobs through `yfinance_client.run_job_after_cooldown`: it re-waits
  each cooldown extension made while it slept, and `run_job(defer=...)` rechecks
  at admission, so a cooldown that began while the read was queued makes it give
  the slot back (`JobDeferredError`), wait and queue again; it fails
  `rate_limited` once the cooldown would outlast its clock (a shared flight has
  no clock, so its callers' own deadlines bound it). A read's
  `jobs.ReadClock` allows `YF_READ_DEADLINE_SECONDS` (default 8) from its first
  admission, capped by any enclosing `jobs.deadline_scope`; time queued for a
  slot counts only against that scope, so batch tools' healthy reads just wait
  their turn -- without a scope that queue wait is unbounded, so 8 s is not a
  total response limit. A read whose deadline passed while it was queued never
  starts, even if the loop wakes it late. Every yfinance HTTP request (data,
  cookie, CSRF/basic crumb, consent POST) is capped at
  `YF_REQUEST_TIMEOUT_SECONDS` (default 5; `yfinance_client` wraps each
  request-sending `YfData` method once at import); that bounds each request,
  not a job that makes several. A provider read that runs out of time is
  `failed` with `error_type` `timeout`, `queued_behind_symbol_lock` or
  `queued_for_admission` (`rate_limited` only while a cooldown is actually
  active); a job that finished is never reported as timed out. `Ticker` property reads (statements, estimates, dividends,
  splits, holders, options) go through `provider.read_property(ticker, source,
  read)`, where `read` takes the Ticker: inside the job any HTTP error response
  yfinance would not raise fails the read as it arrives, before yfinance parses
  or caches anything from it (`preserving_http_status`), and retries follow its
  status -- a 503 is retried, an HTML 404 or an unparseable 200 is not. A retry
  evicts the Ticker and reads a fresh one (yfinance caches parsed state on the
  Ticker) after clearing yfinance's response cache. Results are a `SourceRead`
  (`ok`, `unverified` when empty, `failed`), with each site keeping its
  existing degraded output; a failed read also evicts the Ticker and clears the
  response cache, so the next read fetches again. Concurrent reads of one provider
  source, and concurrent `fetch_info` calls for one symbol, share a single
  in-flight fetch (`jobs.singleflight`): a detached task with no read budget,
  so the source stays owned until the fetch really finishes, that every caller
  awaits through `asyncio.shield`, each only within its own clock (while the
  fetch is queued for a slot only the caller's scope counts; its read budget
  starts when the fetch is admitted). A caller already out of time may join a
  fetch in flight but never starts a new one (so a news feed whose caller's
  scope has run out starts no Search fallback; a feed that only spent its own
  read budget still falls back within the caller's remaining scope). The entry is removed when the fetch lands, so a failure
  is shared only while in flight and never cached. `analyze` runs each tool under `TIMEOUT_SECONDS`
  (10) with its reads capped at 90% of it, so a hung read fails structured and
  the tool still returns; `data_quality.in_flight_jobs`, kept at every
  `detail` level, (and a timed-out tool's `tool_failures` entry) reports threads
  still running, since they can't be killed.
- Price provenance (WP5, `utils/price_basis.py`): `analyze` zones, stops,
  sizing, analyst upside and the snapshot price use `select_price`: in the
  regular session a 5m bar whose own UTC timestamp is at most 15 min old (and
  at most 5 s ahead), from `technicals(..., intraday_quote=True)`, which fetches
  it only then (the clock is the later of analyze's start and the bar's
  `fetched_at`); otherwise the latest valid close of a completed session
  (today's forming bar never counts as `session_close`; off-hours the completed
  close wins; in the regular session `fallback_reason` says why the bar was not
  used: `intraday_quote_unavailable` / `intraday_quote_stale` /
  `intraday_future_timestamp`); otherwise the info quote, labelled
  `info_quote_fallback` (it can sit in the 4h info cache). `summary.price_provenance` (kept at every `detail`
  level) carries `price_source`, `price_as_of`, `as_of_kind`
  (`session_close` date / `bar_time` / `quote_time` = `regularMarketTime`, UTC),
  `as_of_precision`, `fetched_at`, `cache_hit` (history is never cached
  in-app), `fallback_reason` and `staleness`: daily closes in sessions behind
  `market_calendar.last_completed_session`, timestamped quotes in hours
  (regular session: stale past 15 min; off-hours: stale when it predates the
  last completed session's close, 16:00 or 13:00 ET; when it was fetched never
  makes it current). Quote ages use the later of analyze's start and the
  quote's `fetched_at`. `stock_summary` reports `quote_time`, `fetched_at`
  and `cache_hit`. Analyze's freshness gate (`gates.check_price_freshness`)
  reads this provenance as leg Q and `technicals.daily_history` as leg H, and
  fires `stale_data` / `freshness_unverifiable` (codes on the blocker and in
  `data_quality.freshness`, kept at every `detail` level), which cap new money
  at `wait_for_data` at both horizons; `if_held` is unchanged.
  `data_quality.component_freshness` ages price from technicals' own last
  completed valid bar and risk from risk_metrics' own provenance, in sessions
  (warnings `price_stale_N_sessions` / `risk_stale_N_sessions`), and the rest
  in UTC hours.
- Snapshots (WP7 and Step 2, `normalize.build_watchlist_snapshot`, version
  3.0.0; `normalize.SNAPSHOT_FIELDS` freezes each version's field set): analyze
  builds one `watchlist_snapshot` before projection (identical at every
  `detail` level) and `detect_changes` uses it as is. It holds the compact,
  enriched (`price_zone`, horizon fit, signals) and decision fields
  (`new_money_action`, `if_held_action`, `long_term_action`, and
  `hard_gate_ids` = every fired gate; 3.0.0 retired `action_mid_term`,
  `action_long_term`, `decision_action_now` and `zone`) plus
  `support_instrument` / `support_currency` and `price_source` /
  `price_as_of` / `as_of_kind` / `price_stale`; fetch time, cache hit, ages and
  `next_catalyst.days_until` are left out, and so is `next_catalyst.date` when
  analyze reports the earnings `window` behind it (inside an open window the
  date is today; the bounds are the data). `snapshot_hash` covers the full set
  except clock-driven fields (`snapshot_as_of_date`, `price_stale`). Lists are
  read before normalization: unknown is null, known-empty is `[]`; the signal
  lists are null when a tool feeding them failed (`no_options` is a known
  state), ownership discloses `insider_transactions_error` /
  `insider_transactions_columns_unrecognized`, or the latest daily bar is
  invalid (`bar_validity.latest_bar_invalid`: the SMA rules have no value that day),
  or `return_3m` is null because its start bar is invalid
  (`bar_validity.invalid_return_endpoints`, which follows `calculate_returns`:
  a missing, non-finite or zero start, and `return_1y`'s first-bar fallback; it
  feeds the 3m momentum signals). Non-dict sections read as empty. `detect_changes` checks the
  shape of `previous_snapshot` before analysis (required `snapshot_version` in
  1.0.0/1.1.0/2.0.0/3.0.0 and `symbol`; a 3.0.0 snapshot must carry every
  3.0.0 key, null allowed, while an older one may omit fields and its stray
  keys are ignored; known fields are type-checked, `next_catalyst`'s keys and
  its `window` bounds included) and matches its symbol against the ticker
  analyze resolved, returning `invalid_snapshot` with `problems`. It compares
  only fields both snapshots carry (the rest are `not_comparable`; across
  versions the three actions and `price_zone` are never compared and the
  summary names them; a pre-2.0.0
  snapshot's signal lists are treated as absent, since the old producers wrote
  `[]` or partial lists when a source failed); null -> value is a change, and
  an unknown list on either side is one change, never new or removed items.
  Material: price > 5% (judged before rounding, with a 1e-9 tolerance so an
  exact 5% or 0.10 is not crossed by float error) or a
  price/score appearing or disappearing, score > 0.10, gates, `price_zone`,
  horizon fit, signals, tilt, confidence, risk regime, the three actions (a
  new-money change caused only by a freshness gate included), valuation gate,
  support, and a price-source change involving the info fallback (a
  bar <-> close switch, at the close or when the intraday bar goes stale or
  fails, is recorded, not material, per WP5's disclosure-only policy);
  `is_unprofitable` and `price_stale` are kept but not compared.
  `hash_changed` is reported only within one version, separately from
  `material_changes`; a version change alone is not material, but an older
  previous snapshot adds a version warning naming the retired fields and the
  3.0.0 rule changes, and rule-driven gate changes are reported, not
  suppressed.
- Session-indexed bars (WP5): rows are never dropped. Daily-or-longer
  `fetch_history` frames pass `keepna=True` (yfinance otherwise drops empty
  session rows); `1d` frames also restore missing volumes: yfinance zero-fills a
  null volume after dating the sessions, so `yfinance_client` wraps
  `yfinance.utils.fix_Yahoo_dst_issue` once at import to note the exchange-local
  session dates with a null volume on every raw row (a separate live row with a
  volume keeps it; contract test pins the seam; a Sydney fixture pins the
  dating). A row with no OHLC has a null volume (yfinance adds one for a
  dividend after the last price row, after the recorder runs). CSV and preview
  rows print whole volumes as integers. Provenance `last_bar_date` and
  `check_data_quality`'s `price_last_bar` use `price_basis.last_valid_bar_date`
  (the newest valid close, not an empty row), and `analyze_my_position` falls
  back to `session_close` when the latest row is empty. Intraday frames keep yfinance's defaults. Return
  series use `indicators.session_returns` (NaN unless both endpoint prices are
  finite). N-session returns
  (technicals returns, the 1w z-score's latest week, risk benchmark returns,
  `get_price_history` total return, the SMA200 slope, YTD) need both endpoints
  finite, else null; `price_action` and `days_since_52w_*` count sessions, and RSI
  divergence pairs each close with its own session's RSI. SMA and Bollinger
  (D7, `indicators._rolling_over_finite`) use the last N finite closes, all N
  required, placed back on their own session (a session without a finite close
  has no value, so an invalid latest bar still nulls them); before D7 one empty
  session nulled SMA200 for ~200 sessions and silenced the falling-knife gate.
  The other indicators keep their masking (EMA and ATR carry on; RSI treats a
  NaN delta as no change; pivots, bandwidth percentile and RSI divergence still
  skip invalid bars). An empty previous session still gives the scan
  `wait_for_data` (separate from D7); `technicals.bar_validity`
  discloses `latest_bar_invalid`, `latest_volume_invalid` and
  `invalid_bars_in_window` per indicator (its period; MACD 34 bars; ATR checks
  H/L/C, OBV close and volume). Analyze keeps it as `data_quality.bar_validity`
  at every `detail` level and warns `latest_daily_bar_invalid`,
  `latest_daily_volume_invalid` and `invalid_bars_in_indicator_windows`. A
  missing latest volume nulls the volume metrics (current, ratio, OBV; an
  all-missing window nulls `avg_20d`) but never a valid close.
- `analyze_trade_setup.freshness` in regular session requires BOTH a fresh
  intraday bar and a current daily frame (`daily_bar_date >=
  daily_expected_date`, the previous completed session); daily-leg failure is
  `stale_data`, not unverifiable. Off-hours, unusable daily data (non-finite
  or non-positive close, future date) IS unverifiable. `reason_code` is a
  stable enum: no_bar_timestamp | nonfinite_close | nonpositive_close |
  future_timestamp | daily_unusable | stale_quote | stale_daily. Bar timestamps more than 5s in the future are
  unverifiable. `get_technicals` `intraday.freshness` shares these rules
  (disclosure only, never gates).
- Detectors only emit setups with `0 < stop < trigger` on 2-dp rounded
  values; `build_plan` independently raises `PlanInvariantError` on geometry
  violations and the card downgrades to `watch` with a
  `plan_invariant_violation` blocker. Sizing inputs must be finite.
- `add_trading_days` / time stops / `next_review` skip NYSE FULL_HOLIDAYS
  within calendar coverage (weekday-only outside coverage). Inside an active
  earnings window, `analyze`'s `next_review` is the first session after it
  (reason `after earnings (<Mon D>)`), never today; the card's is unchanged.
- `manage_watchlist` and `scan_watchlist` persist to `$STOCK_ANALYSIS_DATA_DIR` (or `$XDG_DATA_HOME/stock-analysis` or `~/.local/share/stock-analysis`); watchlist.json stores symbols with `added` date, scan_state.json stores `rules_version` (`watchlist.SCAN_RULES_VERSION`; bump it when stage-1, row or diff rules change), scanned_at and per-symbol row summaries; corrupt files degrade to empty with a warning, never fail.
- `scan_watchlist` stage 1 (cheap daily screen for all symbols) runs `screen_symbol`: it evaluates every gate the daily frame allows and resolves rank-first like the card: data (fewer than 50 bars, no features, off-session no valid close; the card's daily freshness leg) → `wait_for_data`; `falling_knife` or liquidity on the card's basis → `avoid`; then `no_setup`, `watch` (unsupported, setup kept) or a promoted candidate. It nulls `earnings_in_days` for screen rows (no events fetch); the full card is fetched only for candidates (`screen["promote"]`) and previously-actionable symbols.
- Rows carry `horizon: "swing"`, `source` (`screen` or `card`) and `unevaluated`; freshness ids carry their leg (`stale_data@daily`, `stale_data@quote`). Screen rows always list `earnings_blackout`, `earnings_unverifiable`, `market_closed` and `plan_invariant_violation` as unevaluated (plus the quote-leg freshness ids in session and unverified support ids). Screen rows carry screen blockers only; they do not have `entry`, `stop`, or `plan` fields from a full card — these are not omitted, they do not exist in screen rows.
- `scan_watchlist` `changes` lists every symbol whose action, setup_type, trigger_satisfied, blockers or trigger_price changed since its saved row; blockers compare as (id, leg) pairs that neither side left unevaluated (an entry saved by v0.2.0, with no `source`, compares at id level). A state saved under other rules adds a `rules_changed` warning. The first scan always returns empty `changes` with a `first_scan` warning; changes appear starting from the second scan.
- `expected_move_pct` populates only within 21 days of earnings AND when the
  options chain is usable; `basis` (in the internal result) discloses
  mid-vs-last pricing; it never gates an action and never causes a scan
  transition.

## Multi-Agent Note
- `AGENTS.md` is the canonical shared instruction file for this repo.
- `CLAUDE.md` should stay as a thin wrapper and only contain Claude-specific deltas.
