"""Unit tests for analysis helper logic."""

import numpy as np
import pandas as pd

from stock_mcp.tools.analyze.decision_context import build_relative_performance
from stock_mcp.tools.fundamentals import _compute_valuation_history


class _FakeTicker:
    """Minimal ticker stub for fundamentals helper tests."""

    def __init__(self, income_stmt: pd.DataFrame, history_df: pd.DataFrame) -> None:
        self.quarterly_income_stmt = income_stmt
        self._history_df = history_df

    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        return self._history_df


def test_build_relative_performance_uses_benchmark_returns() -> None:
    tech_data = {
        "returns": {
            "return_1m": 0.05,
            "return_3m": 0.12,
            "return_1y": 0.30,
        }
    }
    risk_data = {
        "beta": {"value": 1.1},
        "benchmark_returns": {
            "return_1m": 0.02,
            "return_3m": 0.08,
            "return_1y": 0.20,
        },
    }

    result = build_relative_performance(tech_data=tech_data, risk_data=risk_data)

    assert result["benchmark_return_1y"] == 0.2
    assert result["alpha_1y"] == 0.1
    assert result["outperformed_1y"] is True
    assert result["warnings"] == []


def test_compute_valuation_history_reflects_historical_price_levels() -> None:
    cols = pd.to_datetime(
        ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31"]
    )
    income_stmt = pd.DataFrame(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0],  # Net Income
            [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],  # Total Revenue
        ],
        index=["Net Income", "Total Revenue"],
        columns=cols,
    )

    history_index = pd.bdate_range("2023-01-02", "2025-01-10")
    history_df = pd.DataFrame(
        {"Close": np.linspace(100.0, 180.0, len(history_index))},
        index=history_index,
    )

    ticker = _FakeTicker(income_stmt=income_stmt, history_df=history_df)
    info = {"sharesOutstanding": 1_000_000}

    result = _compute_valuation_history(ticker=ticker, info=info)
    assert result is not None

    pe_history = result.get("pe_history") or []
    ps_history = result.get("ps_history") or []
    assert len(pe_history) >= 2
    assert len(ps_history) >= 2

    pe_values = [entry["pe"] for entry in pe_history if entry.get("pe") is not None]
    ps_values = [entry["ps"] for entry in ps_history if entry.get("ps") is not None]
    assert len(set(pe_values)) > 1
    assert len(set(ps_values)) > 1
