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
param), `earnings_page`, `news` (`/xhr/ncp`) and `search`.

Real responses recorded by `scripts/record_provider_fixtures.py` go to the
gitignored `tests/fixtures/_recorded/` and must not be committed.
