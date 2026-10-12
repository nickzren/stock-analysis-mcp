# Changelog

## Unreleased: Step 1, correctness foundation

Two versions change, separately:

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

- **Price:** `summary.current_price` is a fresh regular-session 5m bar, else the last completed-session close, else the cached info quote as a labelled fallback. v4 used the info quote, which could be up to 4 h old. Zones, stops, sizing, analyst upside and the snapshot price all follow this price.
- **Freshness:**
  - `data_quality.component_freshness` price/risk entries count sessions from the last bar date (`last_bar_date`, `expected_session`, `sessions_behind`). v4 counted hours from the fetch time.
  - `staleness_warnings` `price_stale_<N>h` / `risk_stale_<N>h` become `…_stale_<N>_sessions`.
- **Ratios:** `fundamentals_summary.valuation` ratios are null unless usable. `metric_status` explains the excluded ones.
- **Dilution:** `fundamentals_summary.burn_metrics.dilution_analysis` is null when the price and statement currencies differ or are unverified, with the burn warning `dilution_currency_unverified`. v4 divided a raise in the statement currency by the market cap in the price currency.
- **Valuation gate (`action_zones.valuation_assessment`):**
  - It reads only usable metrics.
  - For unprofitable companies it uses EV/S first, then P/S, and "attractive" needs 0 < multiple < 3; v4 rated a negative EV/S as attractive. With EV ≤ 0 there is no P/S fallback, so the gate is `unknown`.
  - PEG ≤ 0 is never attractive.
  - `reasons` name the excluded metrics, e.g. `pe_trailing_unverified_currency` or `ev_to_sales_ev_nonpositive`.
  - For an unsupported instrument the gate is `not_applicable`. This also reaches `policy_action.valuation_gate` and the snapshot.
- **`data_quality`:** `fundamentals_status` can be `not_applicable`. An unverified P/E gives the reason `pe_<reason>` (e.g. `pe_unverified_currency`), and the same value replaces `pe_not_meaningful` in `data_gaps`.
- **Unsupported instruments and currencies:** anything other than USD-priced common equity, or a symbol whose support can't be verified, gets new-money actions capped at `watch` everywhere. That covers `policy_action`, `decision_card.action_now`, `decision_modes` and the trade card. There is no share or dollar sizing and no dollar-liquidity gate.
  - For an unsupported instrument, the fundamentals rubric, dip and dislocation blocks read `not_applicable`. For an unsupported currency, a new-money `dip_assessment.assessment.recommendation` reads `watch`, with `support_gates`.
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
  - `bar_validity`: `latest_bar_invalid`, `latest_volume_invalid`, and `invalid_bars_in_window` per indicator window.
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
  - An unknown list is null; a known-empty list is `[]`. Signal lists are null when a tool feeding them failed or the latest daily bar is empty.
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
  - A 429 starts a process-wide cooldown.
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
