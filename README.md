# Stock Analysis MCP Server

[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Server-00ADD8?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMyA3VjE3TDEyIDIyTDIxIDE3VjdMMTIgMloiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L3N2Zz4=)](https://modelcontextprotocol.io)
[![Data: yfinance](https://img.shields.io/badge/Data-yfinance-720e9e)](https://github.com/ranaroussi/yfinance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An MCP server that gives AI agents (Claude Code, Codex, etc.) single-stock analysis capabilities.
Informational only — not financial advice.

## Philosophy

**One command, complete picture.** Just say "Analyze AAPL" and get a comprehensive,
human-readable report with a consistent JSON schema, plus default `core`,
`balanced`, and `speculative` decision modes so non-technical users can act
based on style without changing any code.

## Architecture

Read left to right: request -> orchestration -> parallel data fetch -> synthesized response.

### Quick Flow

```mermaid
flowchart LR
    C["AI Agent"] --> A["analyze"]
    A --> O["Orchestrator"]
    O --> T["Core Tools"]
    T --> D["Data Client + Cache"]
    D --> Y["yfinance"]
    O --> R["Response JSON"]
```

### Detailed Flow

```mermaid
flowchart LR
    classDef client fill:#EAF3FF,stroke:#2F6DB3,color:#12345B,stroke-width:1px;
    classDef entry fill:#E9FFF4,stroke:#2E9B64,color:#0E4F32,stroke-width:1px;
    classDef engine fill:#FFF7E8,stroke:#C58C2A,color:#5B3A00,stroke-width:1px;
    classDef data fill:#F3EEFF,stroke:#6E4EC9,color:#2F1F63,stroke-width:1px;
    classDef output fill:#FFEDEF,stroke:#C3445A,color:#5E1E2A,stroke-width:1px;

    subgraph L1["Client"]
        C["AI Agent<br/>Claude or Codex"]
    end

    subgraph L2["MCP Layer"]
        S["MCP Server"]
        A["analyze<br/>primary tool"]
        OT["other tools<br/>compare, detect_changes<br/>get_ownership, get_options_signals"]
    end

    subgraph L3["Analysis Engine"]
        O["Orchestrator"]
        M["Modules<br/>scoring, actions, context"]
    end

    subgraph L4["Core Data Layer"]
        T["Core tools<br/>technicals, fundamentals<br/>risk, events, news"]
        D["Data client + cache"]
        Y["yfinance"]
    end

    R["Response JSON"]

    C -->|"Analyze NVDA"| S
    S --> A
    S --> OT
    A --> O
    O -->|"parallel calls"| T
    OT --> T
    T --> D --> Y
    O --> M --> R

    class C client
    class S,A,OT entry
    class O,M engine
    class T,D,Y data
    class R output
```

Color legend: `blue=client`, `green=entry`, `orange=engine`, `purple=data`, `red=output`.

Node map: `S=src/stock_analysis/server.py`, `O=src/stock_analysis/tools/analyze/orchestrator.py`, `D=src/stock_analysis/data/`.

Output shape details are documented in `Usage -> Full Analysis (Recommended)`.

### Request Lifecycle

```mermaid
sequenceDiagram
    participant U as AI Agent
    participant M as MCP Server
    participant A as analyze
    participant O as Orchestrator
    participant T as Core Tools
    participant D as Data Client
    participant Y as yfinance

    U->>M: "Analyze NVDA"
    M->>A: call analyze
    A->>O: start orchestration
    par parallel fetch
        O->>T: technicals/fundamentals/risk/events/news
    end
    T->>D: fetch data
    D->>Y: request market data
    Y-->>D: raw data
    D-->>T: normalized data
    T-->>O: tool outputs
    O-->>M: unified response JSON
    M-->>U: analysis result
```

## Quick Start

### Claude Code

```bash
claude mcp add stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp stock-analysis
```

### Codex

```bash
codex mcp add stock-analysis -- uvx --from git+https://github.com/nickzren/stock-analysis-mcp stock-analysis
```

## Usage

### Full Analysis (Recommended)

The primary way to use this server—just say:

```
"Analyze NVDA"
```

If you want dollar sizing, you can still pass account context:

```
analyze("ASTS", account_size=3000)
```

This returns a comprehensive JSON report covering:
- **Executive summary** — materiality-first narrative (leads with what matters most)
- **Dislocation framework** — plain-English “broken price vs broken business” view: drawdown setup, business integrity, balance-sheet safety, thesis integrity, mismatch verdict, and what to do now
- **Section summaries** — 1–2 sentence takeaways per major section
- **Verdict** — score, tilt (bullish/neutral/bearish), confidence, horizon fit
- **Policy action** — mid/long-term decision framing with position sizing ranges (informational)
- **Decision modes** — separate `core`, `balanced`, and `speculative` decision blocks from the same underlying analysis
- **Action zones** — ATR-based price levels (accumulate/reduce/stop)
- **Dip assessment** — oversold metrics, support levels, bounce potential, entry timing
- **Decision context** — what would change the verdict (bullish_if/bearish_if)
- **Catalyst intelligence** — structured bullish/bearish news catalysts (guidance, dilution, approvals, contracts, etc.)
- **Analyst coverage** — consensus rating + targets
- **Ownership & short interest** — insiders/institutions/float, days-to-cover
- **Governance** — audit/board/comp/rights risk scores
- **Company profile** — description, industry, employees, website
- Technical signals, fundamental metrics, risk regime, news sentiment, events

The output follows a consistent schema, making it easy to compare multiple stocks or track changes over time.
For invalid/delisted symbols, `analyze` returns a top-level error (`error=true`) with diagnostics in `data_quality.tool_failures`.

### Optional Inputs

`analyze` works without any extra parameters. Optional inputs only refine sizing:

- `account_size`: portfolio/account size used for dollar sizing output
- `risk_per_trade_pct`: percent of account you are willing to lose if a stop is hit
- `max_position_pct`: hard cap for a single position

If `account_size` is omitted, the analysis still returns the default `core` / `balanced` / `speculative` modes, but keeps sizing percent-based.

## Available Tools

### Primary

| Tool | Description |
|------|-------------|
| `analyze` | Comprehensive single-stock analysis with default Core/Balanced/Speculative modes, optional dollar sizing, human-readable summaries, and company-name resolution for common inputs |

### Comparative Analysis

| Tool | Description |
|------|-------------|
| `compare` | Side-by-side comparison for 2-5 symbols with rankings |
| `detect_changes` | Diff current analysis vs prior snapshot (material changes) |

### Market Data & Signals

| Tool | Description |
|------|-------------|
| `search_symbol` | Search for stock symbols by company name or ticker |
| `get_stock_summary` | Basic stock info (name, sector, price, market cap) |
| `get_price_history` | Historical price data with summary and resource URI |
| `get_technicals` | Technical indicators (SMA, EMA, RSI, MACD, Bollinger, Fibonacci, OBV) |
| `get_fundamentals` | Financial metrics, valuation history, analyst estimates, dividends |
| `get_events` | Earnings dates, dividends, splits |
| `get_news` | Recent news headlines, earnings surprise data, and structured catalyst tags |
| `get_ownership` | Insider transactions and institutional ownership trends |
| `get_options_signals` | Options-derived signals (IV, put/call, unusual activity) |

### Portfolio & Utilities

| Tool | Description |
|------|-------------|
| `analyze_my_position` | Hold/sell analysis for existing positions |
| `analyze_portfolio` | Concentration, sector exposure, correlation |
| `check_data_quality` | Verify data availability for symbols |

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run python -m pytest tests/ -v

# Linting
ruff check src/ tests/
```

## Disclaimer

This tool is for **informational purposes only**. It is not financial advice. Always do your own research and consult a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this tool.
