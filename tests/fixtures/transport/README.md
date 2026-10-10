# Transport fixtures

Synthetic, minimal HTTP responses for contract tests that drive the real pinned
yfinance with only its HTTP session stubbed (`tests/support/transport.py`).
They contain no real Yahoo data: symbol `EXMP`, `example.com` links, made-up
numbers.

Each `<source>/<name>.json` holds `status` plus exactly one body:

- `json`: an object served as a JSON body
- `body_file`: a sibling file (HTML pages), with optional `content_type` (default `text/html`)
- `body`: inline text, with optional `content_type` (default `text/plain`)

Sources are matched by URL fragment (`ROUTES` in `tests/support/transport.py`):
`calendar` (any `/v10/finance/quoteSummary/` request, since the module is a query
param), `earnings_page`, `news` (`/xhr/ncp`), `search`, `timeseries`
(statement time series, which yfinance reads through its `cache_get` LRU) and
`quote_v7` (the v7 quote endpoint used as the info fallback), `options` (the v7
options endpoint), `consent_page` (Yahoo's cookie-consent page) and `chart` (the v8 chart endpoint behind history and the time-zone probe). A spec may add `final_url` to simulate a redirect,
e.g. to the consent page.

Real responses recorded by `scripts/record_provider_fixtures.py` go to the
gitignored `tests/fixtures/_recorded/` and must not be committed.
