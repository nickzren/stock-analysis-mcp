# Repository Instructions

## Purpose
- This repository provides the `stock-analysis` MCP server for single-stock analysis.
- The public contract is the MCP tool surface and the JSON schemas returned by those tools.

## Key Paths
- `src/stock_analysis/server.py`: MCP server entrypoint
- `src/stock_analysis/tools/`: tool implementations
- `src/stock_analysis/data/`: yfinance client and caching; `provider.py` reads the sources whose failures yfinance would otherwise hide
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
- The suite is unit-test focused and runs offline. An autouse guard in `tests/conftest.py` fails any test that sends a request through a `curl_cffi` or `requests` session (yfinance's transport), or makes an unrouted stub request, even when app code swallows the error. It also points `STOCK_ANALYSIS_DATA_DIR` at a per-test temp dir and yfinance's cookie/tz caches at a session temp dir, disables retry sleeps, and clears provider source caches between tests.
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
  blocker when the earnings calendar cannot be fetched or a missing date is found
  inside a successful calendar response with non-empty `sources_failed` (some
  source failed or could not be verified); a verified-empty calendar (successful
  response, empty `sources_failed`) avoids the blocker.
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
  table with no future rows -- is NOT a failure. Calendar dates still use the
  host time zone like yfinance did (WP2 switches to ET); the earnings page is
  parsed with BeautifulSoup in page order (newest first), all or nothing (any
  unparseable row fails the page), and the earliest future row wins. Naive
  earnings timestamps are treated as exchange-local ET.
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
  as `invalid_symbol`. Provider reads stop retrying once backoff would exceed
  4 s (at most three attempts), so a failing read returns `failed` inside
  analyze's per-tool deadline, and cache only `ok`/`empty` results (calendar
  1h, earnings page 6h, news and Search 15 min). An info quoteSummary 404 falls
  back to the v7 quote endpoint like yfinance's hidden mode did. A `Ticker` or info cache miss clears yfinance's `cache_get`
  LRU so the refetch does fresh I/O; a failed `Ticker` attribute read evicts
  that Ticker. Supported range: `yfinance>=1.0,<2`.
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
