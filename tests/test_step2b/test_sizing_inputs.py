"""Sizing normalization is strict before provider or watchlist access."""

import math
from unittest.mock import AsyncMock, Mock

import pytest

from stock_analysis.tools import watchlist
from stock_analysis.tools.trade_setup import orchestrator as swing
from stock_analysis.utils.sizing_inputs import normalize_positive_number, normalize_sizing_params


@pytest.mark.parametrize("raw", [True, False, None, math.nan, math.inf, -math.inf, 0, -1,
    "", " ", "true", "false", "NaN", "Infinity", "1e309", "1e-400", "1_000", "1,000",
    "$1", "1%", "0x10", "1.2.3", "1 2", "1e", "١", [], {}, 1 + 2j, 10 ** 400])
def test_invalid_numbers_are_rejected_with_field_name(raw):
    with pytest.raises(ValueError, match="account_size"):
        normalize_positive_number(raw, field="account_size")


@pytest.mark.parametrize(("raw", "expected"), [
    (1, 1.0), (1.5, 1.5), ("1.5", 1.5), (" +1e-3 ", 0.001), (".5", 0.5),
    ("1.", 1.0), ("1E+2", 100.0), ("\t+0001.2500\r\n", 1.25), ("\u00a0.5\f\v", 0.5),
    ("0.000000123456789", 0.000000123456789), (1e-300, 1e-300),
])
def test_valid_numbers_are_normalized_without_rounding(raw, expected):
    actual = normalize_positive_number(raw, field="risk_per_trade_pct", maximum=100)
    assert type(actual) is float
    assert actual == expected


def test_percentage_bounds_and_nullable_account_are_distinct():
    assert normalize_sizing_params(None, "100", 100) == (None, 100.0, 100.0)
    assert normalize_sizing_params("1000000", "0.001", "0.06") == (1000000.0, 0.001, 0.06)
    for raw in (None, 100.0001, "101", 1e300):
        for field in ("risk_per_trade_pct", "max_position_pct"):
            with pytest.raises(ValueError, match=field):
                normalize_sizing_params(None, **{"risk_per_trade_pct": 1, "max_position_pct": 10, field: raw})


@pytest.mark.parametrize("tool", ["swing", "scan"])
@pytest.mark.parametrize("field", ["account_size", "risk_per_trade_pct", "max_position_pct"])
async def test_direct_tools_reject_before_provider_or_watchlist_access(monkeypatch, tool, field):
    readers = []
    for name in ("stock_summary", "technicals", "risk_metrics", "events_calendar", "fetch_history"):
        reader = AsyncMock(side_effect=AssertionError("must not read providers"))
        monkeypatch.setattr(swing, name, reader)
        readers.append(reader)
    for name in ("resolve_data_dir", "load_watchlist", "load_scan_state", "save_scan_state"):
        reader = Mock(side_effect=AssertionError("must not access watchlist"))
        monkeypatch.setattr(watchlist, name, reader)
        readers.append(reader)
    scan_provider = AsyncMock(side_effect=AssertionError("must not fetch history"))
    monkeypatch.setattr(watchlist, "fetch_history", scan_provider)
    readers.append(scan_provider)
    for raw in (True, "nan", "1e-400", "1_000", -1):
        kwargs = {field: raw}
        result = (await swing.analyze_trade_setup("ACME", **kwargs) if tool == "swing"
                  else await watchlist.scan_watchlist(**kwargs))
        assert result["error_type"] == "invalid_parameters"
        assert field in result["message"]
    for reader in readers:
        reader.assert_not_called()


async def test_direct_swing_normalizes_strings_before_forwarding_to_card(monkeypatch):
    for name in ("stock_summary", "technicals", "risk_metrics", "events_calendar"):
        monkeypatch.setattr(swing, name, AsyncMock(return_value={}))
    monkeypatch.setattr(swing, "fetch_history", AsyncMock(return_value=None))
    card = Mock(return_value={})
    monkeypatch.setattr(swing, "build_trade_setup_card", card)
    result = await swing.analyze_trade_setup(
        "ACME", account_size="10000.123456789", risk_per_trade_pct="1e-3", max_position_pct="0.06",
    )
    assert not result.get("error")
    card.assert_called_once()
    for key, expected in {"account_size": 10000.123456789, "risk_per_trade_pct": 0.001,
                          "max_position_pct": 0.06}.items():
        assert type(card.call_args.kwargs[key]) is float
        assert card.call_args.kwargs[key] == expected


async def test_direct_scan_normalizes_strings_before_forwarding_to_promoted_card(monkeypatch, tmp_path):
    monkeypatch.setattr(watchlist, "load_watchlist", Mock(return_value=({"ACME": {}}, [])))
    monkeypatch.setattr(watchlist, "load_scan_state", Mock(return_value=({}, [])))
    monkeypatch.setattr(watchlist, "save_scan_state", Mock())
    monkeypatch.setattr(watchlist, "_screen_one", AsyncMock(return_value={"promote": True}))
    card = AsyncMock(return_value={"action": "no_setup"})
    monkeypatch.setattr(watchlist, "analyze_trade_setup", card)
    result = await watchlist.scan_watchlist(
        account_size="10000.123456789", risk_per_trade_pct="1e-3", max_position_pct="0.06",
        data_dir=tmp_path,
    )
    assert result["full_cards"] == 1
    card.assert_awaited_once()
    for key, expected in {"account_size": 10000.123456789, "risk_per_trade_pct": 0.001,
                          "max_position_pct": 0.06}.items():
        assert type(card.await_args.kwargs[key]) is float
        assert card.await_args.kwargs[key] == expected
    assert list(tmp_path.iterdir()) == []
