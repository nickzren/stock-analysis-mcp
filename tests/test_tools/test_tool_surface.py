"""Pin the MCP tool surface: count and names."""

import pytest

from stock_analysis.server import mcp

EXPECTED_TOOLS = {
    "analyze", "analyze_my_position", "analyze_portfolio", "analyze_trade_setup",
    "check_data_quality", "compare", "detect_changes", "get_events",
    "get_fundamentals", "get_news", "get_options_signals", "get_ownership",
    "get_price_history", "get_stock_summary", "get_technicals", "search_symbol",
    "manage_watchlist", "scan_watchlist",
}


@pytest.mark.asyncio
async def test_tool_surface_is_pinned() -> None:
    tools = await mcp.list_tools()
    assert {tool.name for tool in tools} == EXPECTED_TOOLS  # 18 tools
