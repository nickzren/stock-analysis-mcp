# Stock Analysis MCP Server

[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Server-00ADD8?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMyA3VjE3TDEyIDIyTDIxIDE3VjdMMTIgMloiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L3N2Zz4=)](https://modelcontextprotocol.io)
[![Data: yfinance](https://img.shields.io/badge/Data-yfinance-720e9e)](https://github.com/ranaroussi/yfinance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An MCP server that gives AI agents (Claude Code, Codex, etc.) stock analysis capabilities.

## Philosophy

**One command, complete picture.** Just say "Analyze AAPL" and get news, technicals, fundamentals, risk metrics, and events in a consistent JSON schema.

## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/nickzren/stock-analysis-mcp.git
cd stock-analysis-mcp

# Install dependencies
uv pip install -e .
```

### Claude Code (VS Code)

Ask Claude Code to add the server:

> "Add the stock-analysis MCP server from this directory"

Or run manually:

```bash
claude mcp add stock-analysis -- uv --directory /path/to/stock-analysis-mcp run stock-mcp
```

### Codex (VS Code)

Ask Codex to add the server:

> "Add the stock-analysis MCP server from this directory"

Or run manually:

```bash
codex mcp add stock-analysis -- uv --directory /path/to/stock-analysis-mcp run stock-mcp
```

## Usage

### Full Analysis (Recommended)

The primary way to use this server—just say:

```
"Analyze NVDA"
```

This returns a comprehensive JSON report covering:
- Company overview and current price
- Recent news and sentiment
- Technical signals (trend, support/resistance)
- Fundamental metrics (valuation, growth, margins)
- Risk profile (volatility, beta, drawdown)
- Upcoming events (earnings, dividends)
- Investment thesis with bull/bear cases
- Verdict with action recommendation

The output follows a consistent schema, making it easy to compare multiple stocks or track changes over time.

## Available Tools

| Tool | Description |
|------|-------------|
| `search_symbol` | Search for stock symbols by company name or ticker |
| `get_stock_summary` | Basic stock info (name, sector, price, market cap) |
| `get_price_history` | Historical price data with summary and resource URI |
| `get_technicals` | Technical indicators (SMA, EMA, RSI, MACD, ATR) |
| `get_fundamentals` | Financial metrics (P/E, margins, growth, debt) |
| `get_events` | Earnings dates, dividends, splits |
| `get_news` | Recent news headlines and earnings surprise data |
| `analyze` | Comprehensive analysis with news, technicals, fundamentals, risk, events |
| `analyze_my_position` | Hold/sell analysis for existing positions |
| `analyze_portfolio` | Concentration, sector exposure, correlation |
| `check_data_quality` | Verify data availability for symbols |

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run python -m pytest tests/ -v

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Disclaimer

This tool is for **informational and educational purposes only**. It is not financial advice. Always do your own research and consult a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this tool.