# Stock Analysis MCP

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Stock research for Claude Code, Codex, and other MCP clients. Get buy, wait, or
avoid guidance, plus holder assessments for positions you own—with valuation,
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
| “I own AAPL and plan to hold for five years. What supports holding, and what would make me reconsider?” | Long-term holder evidence, concerns, and reasons to review. |
| “I own 20 AAPL shares bought at $150. My portfolio is $20,000; my holding ceiling is 10%.” | The holder assessment, P&L, and a separate check against your ceiling. |
| “Compare MSFT, AAPL, and GOOGL.” | Valuation, growth, profitability, and momentum side by side. |
| “Any swing setup on HOOD? Account $3,000, risk 1%, position cap 10%.” | Entry, stop, targets, and stop-based sizing when a setup is actionable. |
| “Add AAPL and MSFT to my watchlist, then scan for setups.” | A saved watchlist and setup screen; later scans highlight changes. |

**Real example:** shortened v0.2.0 `analyze` output for RIVN, recorded after the close
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
claude mcp add -s user stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp@v0.5.2 stock-analysis
```

**Codex**

```bash
codex mcp add stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp@v0.5.2 stock-analysis
```

These commands are pinned to **v0.5.2**. Other MCP clients can use the same `uvx`
command as a stdio server.

## For agents

Start with `analyze(symbol="MSFT")`. It accepts a ticker or company name;
use `search_symbol` to resolve names before calling other tools.

| Task | Tool |
|---|---|
| General stock decision | `analyze`: `detail="standard"` by default; `"decision"` for a smaller response, `"full"` for the audit detail. |
| Long-only swing trade, days to weeks | `analyze_trade_setup` |
| Existing holding | Stock-only: `analyze` for the requested horizon’s `if_held`; with personal context: one `analyze_my_position` call, with optional inputs. |
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

Sizing uses whole shares within caller or profile/default risk and allocation
caps; `risk_per_trade_pct=1` means 1% of the account. `account_size` enables dollar
amounts; a valid estimate without it is `percentage_only`. A scenario can fit
zero shares (`below_one_whole_share`), even when an entry is allowed.

Read `decision_card.sizing_status` and its reason when sizing is null; a `buy` or
`starter` can lack a valid size. Allocation and estimated loss at the planning
stop are separate. The planning stop is not monitored or a holder exit instruction;
gaps and slippage can increase the loss. Existing holdings and buying power are unknown.

Version 0.5.2 responses use schema **8** and snapshots **4.0.0**, unchanged since 0.5.0. See the
[schema 7 → 8 migration](CHANGELOG.md#050-2026-10-20-longer-term-holders-and-position-context)
for position fields, holder guidance, and snapshot compatibility.

## Decision

`decision_card` holds one decision, made after the hard gates:

- `new_money`: what to do with new money over 3–12 months (one year defaults here unless long term is explicit);
- `if_held`: what to do if you already own it: `hold`, `reduce`, `review_exit` or `insufficient_data`;
- `long_term.new_money`: new money over 1–5 years;
- `long_term.if_held`: ownership over 1–5 years: `hold`, `review_exit`, `insufficient_data` or `not_assessed`.

Read `decision_card.long_term.if_held` for the 1–5-year holder answer. A bare
holder question and one year default to mid-term unless long term is explicit.
`hold` means no demonstrated thesis failure, not an endorsement; keep the separate
suitability fit and its actual reasons visible.

For personal context, call `analyze_my_position` once with the chosen horizon.
Cost, date, shares, portfolio value and your holding ceiling are optional;
monetary inputs are USD. **Allocation check** reports only your supplied ceiling,
not the stock decision or remaining buying power. No tax estimate is provided;
technical reference levels are not monitored holder stops.

Known ETFs/preferreds return `not_assessed` for long-term ownership. For ETFs:
“This tool does not assess long-term ETF ownership; its company rules do not
apply.” Verified USD share prices can still support personal P&L/allocation.

`blocked_by` lists the gates that fired and `decided_by` the one that set the action;
never upgrade past them. Read sizing only from `decision_card.sizing`: mid-term new
money only (no long-term or holder size). It is null when entry is blocked or
sizing inputs are unusable; a starter hides the full size. `policy_action` moved
to `diagnostics.pre_gate_policy` (`detail="full"`): pre-gate input, not advice.

| `new_money.action` | Meaning |
|---|---|
| `buy` | Entry allowed, up to full size. |
| `starter` | Entry allowed at starter size only. |
| `wait` | No entry now: the evidence, earnings risk or the 52-week-high band says wait. |
| `wait_for_data` | No entry: required data is missing or stale. |
| `insufficient_data` | No entry: too little data to decide. |
| `watch` | No entry: outside the supported instruments or currencies. |
| `avoid` | No entry: a poor setup, falling knife, weak liquidity or broken thesis. |

Gates, by rank: `stale_data`, `freshness_unverifiable`, `data_quality_critical`,
`missing_runway`, `thesis_unverifiable`, `liquidity_missing` → `wait_for_data`; `falling_knife`,
`weak_liquidity`, `thesis_broken` → `avoid`; `unsupported_instrument`,
`unsupported_currency` → `watch`; `earnings_blackout`, `earnings_unverifiable` →
`wait`; `near_52w_high` → `wait` (mid-term only). The `profile` argument (`core`,
`balanced`, `speculative`) changes sizing only, never an action.

If required business or cash-runway evidence cannot be checked, `thesis_unverifiable`
blocks new money at both horizons. Re-run `analyze` when the evidence is available;
clearing that check alone does not mean buy. Known unsupported instruments do not get this gate.

| Class | Fields | Read as |
|---|---|---|
| A | `decision_card` actions, modes, dislocation and dip actions | The decision, per horizon. |
| S | `decision_card.sizing`, `entry`, sizing ranges | Mid-term new-money size, only for an entry. |
| C | `conditions`, `earliest_reassessment`, `next_review` | Necessary, not sufficient; re-run `analyze`. |
| D | `price_zone`, levels, verdict, data quality | Context, not advice. |
| N | `executive_summary` | Our summary: new money, if held, long term. |
| P | `diagnostics` | Pre-gate input, not advice. |
| T | Analyst, news, ownership and options data | Third-party data, attributed. |
| E | Mode labels and descriptions | Static configuration. |

In `watchlist_snapshot`, `new_money_action` / `if_held_action` mean mid-term
new money / holder; `long_term_action` / `long_term_if_held_action` mean long-term
new money / holder. `scan_watchlist` state holds swing-card actions.

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
- **Limits remain:** some statement reads become missing fields after failure.
  Scores and sizing are rule-based estimates; holder guidance is stateless.
  See the [known limitations](CHANGELOG.md#known-limitations-030).
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
