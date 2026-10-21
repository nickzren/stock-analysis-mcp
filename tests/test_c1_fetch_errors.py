"""C.1: daily masks and the existing symbol/no-data error classifications."""

import importlib
import math
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFInvalidPeriodError, YFTickerMissingError

from stock_analysis.data import yfinance_client
from tests.support.fake_ticker import FakeTicker, http_error
from tests.support.transport import FakeSession

history_tool = importlib.import_module("stock_analysis.tools.price_history")
summary_tool = importlib.import_module("stock_analysis.tools.stock_summary")
technicals_tool = importlib.import_module("stock_analysis.tools.technicals")
SYMBOL = "EXMP"


@pytest.fixture(autouse=True)
def clear_info_cache() -> Iterator[None]:
    yfinance_client.info_cache.clear()
    yield
    yfinance_client.info_cache.clear()


def raising_history(error: Exception) -> Any:
    def read(**kwargs: Any) -> pd.DataFrame:
        raise error

    return read


async def test_missing_volume_fetch_works_with_copy_on_write(yf_transport: FakeSession) -> None:
    """Pandas 2 opts in; pandas 3 already requires copy-on-write."""
    context = pd.option_context("mode.copy_on_write", True) if int(pd.__version__.split(".")[0]) < 3 else nullcontext()
    yf_transport.route("chart", "daily_gaps")

    with context:
        result = await history_tool.price_history(SYMBOL)

    assert result.get("error") is not True, result
    assert result["resource_rows"] == 7
    assert math.isnan(result["preview"][-1]["volume"])


@pytest.mark.parametrize("tool", [history_tool.price_history, technicals_tool.technicals])
async def test_plain_fetch_value_error_is_not_an_invalid_symbol(
    monkeypatch: pytest.MonkeyPatch, tool: Any
) -> None:
    error = ValueError("array math failed")
    monkeypatch.setattr(
        yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, history=raising_history(error))
    )

    result = await tool(SYMBOL)

    assert result["error_type"] == "data_unavailable"
    assert result["message"] == "Failed to fetch data: array math failed"


@pytest.mark.parametrize(
    "case",
    ["missing_ticker", "invalid_period", "history_404", "empty_history", "info_404", "empty_info"],
)
async def test_existing_symbol_no_data_sites_keep_their_labels(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    if case == "missing_ticker":
        attributes = {"history": raising_history(YFTickerMissingError(SYMBOL, "no timezone found"))}
    elif case == "invalid_period":
        attributes = {"history": raising_history(YFInvalidPeriodError(SYMBOL, "1y", "1mo"))}
    elif case == "history_404":
        attributes = {"history": raising_history(http_error(404))}
    elif case == "empty_history":
        attributes = {"history": lambda **kwargs: pd.DataFrame()}
    elif case == "info_404":
        attributes = {"info": http_error(404)}
        monkeypatch.setattr(yfinance_client, "_info_from_quote_endpoint", lambda symbol: None)
    else:
        attributes = {"info": {}}
    monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, **attributes))

    tool = summary_tool.stock_summary if case in {"info_404", "empty_info"} else history_tool.price_history
    result = await tool(SYMBOL)

    assert result["error_type"] == "invalid_symbol"
    expected = f"Invalid symbol: {SYMBOL}" if case in {"info_404", "empty_info"} else f"No data returned for {SYMBOL}"
    assert result["message"].startswith(expected)
