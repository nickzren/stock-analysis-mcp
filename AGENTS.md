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

## Common Schema Gotchas
- `technicals` returns `current_price` at the top level, not under a `summary` object.
- `analyze` returns price, name, sector, market cap, and currency under `summary`, not as duplicated top-level fields.
- `analyze` returns the `standard` projection by default; deep audit fields such as `decision_context`, `dip_assessment`, and full `verdict` / `data_quality` require `detail="full"`.
- `options_signals` returns put/call data at `put_call_ratio.volume_based` and `put_call_ratio.oi_based`, not under `summary.*`.
- In `analyze`, raw options output is nested under `options_signals`, while derived narrative fields appear elsewhere such as `section_summaries`, `signals`, and `decision_context`.
- `risk_metrics` keeps volatility at `volatility.annualized`; the synthesized `analyze` output exposes the normalized view under `risk_summary.annualized_volatility`.
- `analyze_trade_setup` returns `plan` only when `action` is `trade_now` or
  `enter_on_trigger`; `watch` keeps `setup` with a null `plan`.
- `analyze_trade_setup` reuses the `wait_for_data` action code from the decision
  card; there is no `wait_for_fresh_data`.
- `analyze_trade_setup` sizing is R-based (risk budget ÷ stop distance) under
  `plan`, not the accumulation-style `position_sizing_range` used by `analyze`.
  Dollar/share fields are `null` (not `0`) when `account_size` is omitted.
- `analyze_trade_setup.freshness.as_of` is a market-data bar timestamp, never
  fetch time; `basis="unverifiable"` forces `wait_for_data`.
- When `analyze_trade_setup.action` is not `trade_now`/`enter_on_trigger`, do not
  render entry/stop/target/sizing structure and do not mix `analyze` fields into
  the card; forward-looking levels are conditions to watch, not a plan.
- `analyze_trade_setup` caps the action at `watch` with an `earnings_unverifiable`
  blocker when `events_calendar` fails or `earnings.verification` is
  `unverifiable`; `verified_date` and `verified_none` avoid the blocker. The
  blocker reason names the failed and stale sources and an assumed window.
  `event_risk` carries `earnings_verification`, `earnings_window` (the window
  behind `next_earnings_date`, if any), `earnings_windows` (every source
  window) and `verified_earnings_dates`, so an assumed window's start is never
  shown without its uncertainty.
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
- Freshness treats any non-finite close (NaN/±inf) or a bar timestamp more
  than 5s in the future as unverifiable — a quote without a trustworthy
  finite price cannot verify the actionable price. The session-specific
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
  earliest to the latest of the three); a multi-day range is an unverified
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
  arrival order. Backoff and rate-limit cooldown sleeps hold neither. A read's
  `jobs.ReadClock` allows `YF_READ_DEADLINE_SECONDS` (default 8) from its first
  admission, capped by any enclosing `jobs.deadline_scope`; time queued for a
  slot counts only against that scope, so batch tools' healthy reads just wait
  their turn. Every yfinance HTTP request, cookie and crumb fetches included,
  is capped at `YF_REQUEST_TIMEOUT_SECONDS` (default 5; `yfinance_client`
  wraps `YfData._make_request` and `_get_cookie_and_crumb` once at import). A
  provider read that runs out of time is `failed` with `error_type` `timeout`,
  `queued_behind_symbol_lock` or `queued_for_admission` (`rate_limited` only
  while a cooldown is actually active); a job that finished is never reported
  as timed out. `Ticker` property reads (statements, estimates, dividends,
  splits, holders, options) go through `provider.read_property`: retried (an
  error page hidden behind a JSON/parse error counts as retryable) and returned
  as a `SourceRead` (`ok`, `unverified` when empty, `failed`), with each site
  keeping its existing degraded output. Concurrent reads of one provider
  source, and concurrent `fetch_info` calls for one symbol, share a single
  in-flight fetch (`jobs.singleflight`): a detached task on its own read budget
  that every caller awaits through `asyncio.shield`, each caller only until its
  own scope ends; it is removed when it lands, so a failure is shared only while
  in flight and never cached. `analyze` runs each tool under `TIMEOUT_SECONDS`
  (10) with its reads capped at 90% of it, so a hung read fails structured and
  the tool still returns; `data_quality.in_flight_jobs` (and a timed-out tool's
  `tool_failures` entry) reports threads still running, since they can't be
  killed.
- `analyze_trade_setup.freshness` in regular session requires BOTH a fresh
  intraday bar and a current daily frame (`daily_bar_date >=
  daily_expected_date`, the previous completed session); daily-leg failure is
  `stale_data`, not unverifiable. Off-hours, unusable daily data (non-finite
  close, future date) IS unverifiable. `reason_code` is a stable enum:
  no_bar_timestamp | nonfinite_close | future_timestamp | daily_unusable |
  stale_quote | stale_daily. Bar timestamps more than 5s in the future are
  unverifiable. `get_technicals` `intraday.freshness` shares these rules
  (disclosure only, never gates).
- Detectors only emit setups with `0 < stop < trigger` on 2-dp rounded
  values; `build_plan` independently raises `PlanInvariantError` on geometry
  violations and the card downgrades to `watch` with a
  `plan_invariant_violation` blocker. Sizing inputs must be finite.
- `add_trading_days` / time stops / `next_review` skip NYSE FULL_HOLIDAYS
  within calendar coverage (weekday-only outside coverage).
- `manage_watchlist` and `scan_watchlist` persist to `$STOCK_ANALYSIS_DATA_DIR` (or `$XDG_DATA_HOME/stock-analysis` or `~/.local/share/stock-analysis`); watchlist.json stores symbols with `added` date, scan_state.json stores scanned_at and per-symbol card summaries; corrupt files degrade to empty with a warning, never fail.
- `scan_watchlist` phase 1 (cheap daily screen for all symbols) runs `screen_symbol`, which nulls `earnings_in_days` for screened-out rows (no events fetch in phase 1); full card is fetched only for candidates (`screen["promote"]`) and previously-actionable symbols.
- Phase-1 `avoid` rows carry screen blockers only (e.g., `"falling_knife"`, `"weak_liquidity"`); they do not have `entry`, `stop`, or `plan` fields from a full card — these are not omitted, they do not exist in phase-1 rows.
- `scan_watchlist` `changes` lists only action transitions for symbols with a prior scan state; the first scan always returns empty `changes` with a `first_scan` warning; transitions appear starting from the second scan.
- `expected_move_pct` populates only within 21 days of earnings AND when the
  options chain is usable; `basis` (in the internal result) discloses
  mid-vs-last pricing; it never gates an action and never causes a scan
  transition.

## Multi-Agent Note
- `AGENTS.md` is the canonical shared instruction file for this repo.
- `CLAUDE.md` should stay as a thin wrapper and only contain Claude-specific deltas.
