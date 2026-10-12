# Stock Analysis MCP

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Stock research for Claude Code, Codex, and other MCP clients. Get buy, wait, or
avoid guidance, plus hold/sell signals for positions you own—with valuation,
chart signals, risks, and earnings context.

Decision-card hard gates can block a buy for a falling knife, earnings risk, weak
liquidity, poor data quality, or unsupported instruments—even if the narrative is
positive.

Yahoo Finance data via `yfinance`. No market-data API key required. Does not place orders.

## How to use

Once connected, just ask your agent:

```text
Analyze HOOD
```

No extra parameters needed. For more specific questions:

| Question | What you get |
|---|---|
| “Analyze MSFT. What supports buying, and what would make you wait?” | A decision card, supporting evidence, blockers, and conditions to revisit. |
| “I own 20 AAPL shares bought at $150 on 2025-01-02. Should I hold or sell?” | Position P&L, technical sell signals, and holding-period context. |
| “Compare MSFT, AAPL, and GOOGL.” | Valuation, growth, profitability, and momentum side by side. |
| “Any swing setup on HOOD? Account $3,000, risk 1%, position cap 10%.” | Entry, stop, targets, and stop-based sizing when a setup is actionable. |
| “Add AAPL and MSFT to my watchlist, then scan for setups.” | A saved watchlist and setup screen; later scans highlight changes. |

**Real example:** shortened `analyze` output for RIVN, recorded after the close
on **2026-10-12**. Historical output, not a current recommendation.

```text
RIVN — $13.54
Action: wait
Blocker: falling_knife
Reason: severe decline with broken trends — wait for stabilization
Price: daily_close, as of 2026-10-12
Data gap: pe_not_meaningful
```

## Set up

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
uv can [download Python automatically](https://docs.astral.sh/uv/concepts/python-versions/#installing-python-versions).
Choose your client:

**Claude Code**

```bash
claude mcp add -s user stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp@v0.2.0 stock-analysis
```

**Codex**

```bash
codex mcp add stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp@v0.2.0 stock-analysis
```

These commands are pinned to **v0.2.0**. Other MCP clients can use the same `uvx`
command as a stdio server.

## For agents

Start with `analyze(symbol="MSFT")`. It accepts a ticker or company name;
use `search_symbol` to resolve names before calling other tools.

| Task | Tool |
|---|---|
| General stock decision | `analyze`: `detail="standard"` by default; `"decision"` for a smaller response, `"full"` for the audit detail. |
| Long-only swing trade, days to weeks | `analyze_trade_setup` |
| Existing holding | `analyze_my_position`: provide `symbol`, `cost_basis`, `purchase_date`; `shares` is optional. |
| Compare 2–5 stocks / review portfolio exposure | `compare` / `analyze_portfolio` |
| Recheck a stock | Save `watchlist_snapshot`; pass it as `previous_snapshot` to `detect_changes`. |
| Maintain and scan up to 25 symbols | `manage_watchlist` / `scan_watchlist` |
| Inspect a specific input | `get_stock_summary`, `get_price_history`, `get_technicals`, `get_fundamentals`, `get_events`, `get_news`, `get_ownership`, `get_options_signals`, `check_data_quality` |

Read `decision_card` with its hard gates, `summary.price_provenance`, and
`data_quality`. Keep timestamps, blockers, and missing-data warnings in the answer;
do not promote a positive narrative into a buy when a gate blocks it.

For swing cards, preserve `action` and `blockers`. Show the returned `plan` only
for `trade_now` or `enter_on_trigger`. `trade_now` requires fresh regular-session
data; unverifiable earnings cap the card at `watch`.

`account_size` enables dollar/share sizing. `analyze` sizes are estimates, not a
guarantee that every suggestion fits the risk budget. Use `analyze_trade_setup`
for stop-based risk sizing; `risk_per_trade_pct=1` means 1% of the account.

Responses use schema **5**; snapshots use **2.0.0**.
[CHANGELOG.md](CHANGELOG.md) covers field meanings, migration, and known limitations.

## Data and limits

- **Supported decisions:** USD-priced common stocks, including USD ADRs. Unsupported
  or unverified instruments/currencies block new buying and dollar/share sizing;
  existing-holder actions can remain. ETFs, preferreds, and non-USD listings are
  outside the supported decision rules.
- **`analyze` price basis:** uses a fresh intraday bar during regular hours when
  available, otherwise the completed daily close, then a labelled quote fallback.
  Yahoo data can be delayed or incomplete; check the reported source and age.
- **Units and gaps:** dividend yields are fractions (`0.005 = 0.5%`); debt-to-equity
  is a ratio (`1.2 = 120%`). Undefined or unverified ratios are null;
  `metric_status` marks ratios excluded from scoring. Read cash-flow period labels:
  a field named `*_ttm` can contain an annual fallback.
- **Limits remain:** `analyze` does not apply all the swing card's freshness and
  earnings-verification gates. Some statement reads become missing fields after
  failure. Scores and sizing are rule-based estimates. See the [known limitations](CHANGELOG.md#known-limitations).
- **Local state:** watchlist and scan state persist in `$STOCK_ANALYSIS_DATA_DIR`,
  otherwise `$XDG_DATA_HOME/stock-analysis`, otherwise `~/.local/share/stock-analysis`.
  Price-history `price://` resources last only in the server process that fetched
  them; use `include_preview=true` for the last five inline bars if resource reads fail.

## Development

[Tests](tests/) run offline against synthetic data and a stubbed Yahoo transport.
They cover failures, earnings dates, price freshness, decision gates, and snapshots.

```bash
uv sync --extra dev --locked
uv run --frozen pytest
uv run --frozen ruff check src tests
```

## Disclaimer

This tool is for **informational purposes only**. It is not financial advice. Always do your own research and consult a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this tool.
