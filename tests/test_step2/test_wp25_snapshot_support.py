"""Losing instrument classification must not look like fundamental signals disappeared."""

import importlib
from typing import Any

import pytest

from tests.support import decision_matrix as dm

diff_analysis = importlib.import_module("stock_analysis.tools.diff_analysis")


async def test_unknown_instrument_signals_are_unknown_until_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    before = await dm.run("E_poor_business_bullish_chart")
    unknown = await dm.run("E_poor_business_bullish_chart", summary={**dm.EQUITY, "quote_type": None})
    snapshot = unknown["watchlist_snapshot"]
    assert before["watchlist_snapshot"]["bearish_signals"]
    assert snapshot["bullish_signals"] is None
    assert snapshot["bearish_signals"] is None

    for previous, current in ((before["watchlist_snapshot"], unknown), (snapshot, before)):
        async def analyze(*args: Any, current: dict[str, Any] = current, **kwargs: Any) -> dict[str, Any]:
            return current

        monkeypatch.setattr(diff_analysis, "analyze_stock", analyze)
        delta = await diff_analysis.what_changed("ACME", previous_snapshot=previous)
        assert not any(line.startswith(("-bullish", "-bearish", "+bullish", "+bearish"))
                       for line in delta["material_changes"])
        assert delta["changes"]["signals"]["bearish_changed"]


@pytest.mark.parametrize("summary", [
    dm.EQUITY,
    {**dm.EQUITY, "currency": None},
    {**dm.EQUITY, "quote_type": "ETF"},
])
async def test_known_instrument_lists_are_not_erased_by_currency_or_inapplicability(summary: dict[str, Any]) -> None:
    result = await dm.run("E_poor_business_bullish_chart", summary=summary)
    assert result["watchlist_snapshot"]["bullish_signals"] is not None
    assert result["watchlist_snapshot"]["bearish_signals"] is not None
