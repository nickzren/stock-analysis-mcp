"""Risk metrics tool."""

import operator
from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.data.yfinance_client import fetch_history
from stock_mcp.utils.indicators import (
    calculate_atr,
    calculate_beta,
    calculate_current_drawdown,
    calculate_max_drawdown,
    calculate_sma,
    calculate_var,
    calculate_volatility,
)
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.validators import FetchParams, check_rule


async def risk_metrics(
    symbol: str,
    benchmark: str = "SPY",
    portfolio_value: float = 50000,
    risk_per_trade: float = 0.02,
) -> dict[str, Any]:
    """
    Calculate risk metrics for a symbol.

    Args:
        symbol: Stock ticker symbol
        benchmark: Benchmark symbol for beta calculation (default: SPY)
        portfolio_value: Portfolio value for position sizing (default: 50000)
        risk_per_trade: Risk per trade as decimal (default: 0.02 = 2%)

    Returns:
        Dict with volatility, beta, drawdown, VaR, ATR, liquidity, position sizing
    """
    start_time = perf_counter()

    # Fetch symbol data
    params = FetchParams(
        symbol=symbol,
        period="1y",
        interval="1d",
        adjusted=True,
    )

    try:
        df = await fetch_history(params)
    except ValueError as e:
        return build_error_response(
            error_type="invalid_symbol",
            message=str(e),
            symbol=symbol,
        )
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch data: {e}",
            symbol=symbol,
        )

    if len(df) < 50:
        return build_error_response(
            error_type="insufficient_data",
            message=f"Need at least 50 data points, got {len(df)}",
            symbol=symbol,
        )

    # Fetch benchmark data for beta calculation
    benchmark_params = FetchParams(
        symbol=benchmark,
        period="1y",
        interval="1d",
        adjusted=True,
    )

    benchmark_df = None
    beta_data: dict[str, Any] = {"value": None, "warning": "benchmark_fetch_failed"}

    try:
        benchmark_df = await fetch_history(benchmark_params)
    except Exception:
        pass  # Beta will show warning

    # Extract price series
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    # Set index to date for alignment
    df_indexed = df.copy()
    df_indexed["date"] = pd.to_datetime(df_indexed["date"])
    df_indexed = df_indexed.set_index("date")

    current_price = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else None

    # Calculate daily returns
    returns = close.pct_change().dropna()

    # Volatility
    daily_std = float(returns.std()) if len(returns) > 0 else None
    annualized_vol = calculate_volatility(returns, annualize=True)

    volatility = {
        "daily_std": round(daily_std, 6) if daily_std is not None else None,
        "annualized": round(annualized_vol, 4) if annualized_vol is not None else None,
        "rules": {
            "high_volatility": {
                "triggered": check_rule(annualized_vol, 0.40, operator.gt),
                "threshold": 0.40,
            },
        },
    }

    # Beta (with proper alignment)
    if benchmark_df is not None and len(benchmark_df) > 0:
        benchmark_indexed = benchmark_df.copy()
        benchmark_indexed["date"] = pd.to_datetime(benchmark_indexed["date"])
        benchmark_indexed = benchmark_indexed.set_index("date")
        benchmark_close = pd.to_numeric(benchmark_indexed["close"], errors="coerce")
        benchmark_returns = benchmark_close.pct_change().dropna()

        # Create returns series with date index
        symbol_returns = pd.to_numeric(df_indexed["close"], errors="coerce").pct_change().dropna()

        beta_data = calculate_beta(symbol_returns, benchmark_returns, min_overlap=200)

    beta = {
        "value": beta_data.get("value"),
        "overlap_days": beta_data.get("overlap_days"),
        "rules": {
            "high_beta": {
                "triggered": check_rule(beta_data.get("value"), 1.3, operator.gt),
                "threshold": 1.3,
            },
            "low_beta": {
                "triggered": check_rule(beta_data.get("value"), 0.7, operator.lt),
                "threshold": 0.7,
            },
        },
    }
    if beta_data.get("warning"):
        beta["warning"] = beta_data["warning"]

    # Drawdown
    max_dd = calculate_max_drawdown(close)
    current_dd, days_since_high = calculate_current_drawdown(close)

    drawdown = {
        "max_1y": round(max_dd, 4) if max_dd is not None else None,
        "current": round(current_dd, 4) if current_dd is not None else None,
        "days_since_high": days_since_high,
    }

    # VaR
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)

    var = {
        "daily_95": round(var_95, 4) if var_95 is not None else None,
        "daily_99": round(var_99, 4) if var_99 is not None else None,
    }

    # ATR
    atr_series = calculate_atr(high, low, close, 14)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None
    atr_pct = atr_val / current_price if atr_val and current_price else None

    atr = {
        "value": round(atr_val, 2) if atr_val is not None else None,
        "as_pct_of_price": round(atr_pct, 4) if atr_pct is not None else None,
    }

    # Liquidity
    avg_volume = float(volume.mean()) if not volume.isna().all() else None
    avg_dollar_volume = avg_volume * current_price if avg_volume and current_price else None

    liquidity = {
        "avg_dollar_volume": round(avg_dollar_volume, 2) if avg_dollar_volume else None,
        "rules": {
            "liquid": {
                "triggered": check_rule(avg_dollar_volume, 10_000_000, operator.gt),
                "threshold": 10_000_000,
            },
        },
    }

    # Stop suggestions
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    sma_50_val = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
    sma_200_val = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None

    stop_suggestions = _build_stop_suggestions(current_price, atr_val, sma_50_val, sma_200_val)

    # Position sizing
    position_sizing = _build_position_sizing(
        portfolio_value=portfolio_value,
        risk_per_trade=risk_per_trade,
        current_price=current_price,
        avg_dollar_volume=avg_dollar_volume,
        stop_suggestions=stop_suggestions,
    )

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("risk_metrics", duration_ms),
        "data_provenance": {
            "price": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
                last_bar_date=df["date"].iloc[-1] if len(df) > 0 else None,
            ),
        },
        "symbol": params.symbol,
        "benchmark": benchmark,
        "volatility": volatility,
        "beta": beta,
        "drawdown": drawdown,
        "var": var,
        "atr": atr,
        "liquidity": liquidity,
        "stop_suggestions": stop_suggestions,
        "position_sizing": position_sizing,
    }


def _build_stop_suggestions(
    current_price: float | None,
    atr_val: float | None,
    sma_50_val: float | None,
    sma_200_val: float | None,
) -> dict[str, dict[str, float | None]]:
    """Build stop loss suggestions based on ATR and SMAs."""
    suggestions: dict[str, dict[str, float | None]] = {}

    if current_price and atr_val:
        for multiplier, label in [(1.0, "atr_1x"), (1.5, "atr_1_5x"), (2.0, "atr_2x")]:
            stop_price = current_price - (atr_val * multiplier)
            distance = (stop_price - current_price) / current_price
            suggestions[label] = {
                "price": round(stop_price, 2),
                "distance": round(distance, 4),
            }
    else:
        for label in ["atr_1x", "atr_1_5x", "atr_2x"]:
            suggestions[label] = {"price": None, "distance": None}

    if current_price and sma_50_val:
        distance = (sma_50_val - current_price) / current_price
        suggestions["sma_50"] = {
            "price": round(sma_50_val, 2),
            "distance": round(distance, 4),
        }
    else:
        suggestions["sma_50"] = {"price": None, "distance": None}

    if current_price and sma_200_val:
        distance = (sma_200_val - current_price) / current_price
        suggestions["sma_200"] = {
            "price": round(sma_200_val, 2),
            "distance": round(distance, 4),
        }
    else:
        suggestions["sma_200"] = {"price": None, "distance": None}

    return suggestions


def _build_position_sizing(
    portfolio_value: float,
    risk_per_trade: float,
    current_price: float | None,
    avg_dollar_volume: float | None,
    stop_suggestions: dict[str, dict[str, float | None]],
    max_concentration: float = 0.10,
) -> dict[str, Any]:
    """Build position sizing recommendations."""
    inputs = {
        "portfolio_value": portfolio_value,
        "risk_per_trade": risk_per_trade,
        "current_price": current_price,
    }

    # Constraints
    max_by_concentration = portfolio_value * max_concentration
    max_by_liquidity = (
        avg_dollar_volume * 0.01 if avg_dollar_volume and avg_dollar_volume > 0 else None
    )

    constraints = {
        "max_by_concentration": round(max_by_concentration, 2),
        "max_by_liquidity": round(max_by_liquidity, 2) if max_by_liquidity else None,
    }

    # Position size by stop level
    by_stop_level: dict[str, dict[str, float | int]] = {}
    recommended: dict[str, Any] = {
        "stop_level": None,
        "position_dollars": 0,
        "shares": 0,
        "binding_constraint": "no_valid_constraint",
    }

    best_position = 0.0
    best_stop_level = None
    best_binding = "no_valid_constraint"

    for stop_level in ["atr_1x", "atr_1_5x", "atr_2x"]:
        stop_data = stop_suggestions.get(stop_level, {})
        stop_distance = stop_data.get("distance")

        if stop_distance is None or stop_distance >= 0:
            # Invalid stop (at or above current price)
            by_stop_level[stop_level] = {"position_dollars": 0, "shares": 0}
            continue

        # stop_distance is negative (e.g., -0.05 for 5% below)
        risk_amount = portfolio_value * risk_per_trade
        position_by_risk = risk_amount / abs(stop_distance)

        # Apply constraints (fixed priority order for tie-breaking)
        constraint_list = [
            ("risk_budget", position_by_risk),
            ("concentration", max_by_concentration),
        ]
        if max_by_liquidity is not None:
            constraint_list.append(("liquidity", max_by_liquidity))

        binding_label, position_dollars = min(constraint_list, key=lambda x: x[1])

        shares = int(position_dollars / current_price) if current_price else 0

        by_stop_level[stop_level] = {
            "position_dollars": round(position_dollars, 2),
            "shares": shares,
        }

        # Track best position (prefer larger positions, i.e., wider stops)
        if position_dollars > best_position:
            best_position = position_dollars
            best_stop_level = stop_level
            best_binding = binding_label

    if best_stop_level:
        recommended = {
            "stop_level": best_stop_level,
            "position_dollars": round(best_position, 2),
            "shares": int(best_position / current_price) if current_price else 0,
            "binding_constraint": best_binding,
        }

    return {
        "inputs": inputs,
        "by_stop_level": by_stop_level,
        "constraints": constraints,
        "recommended": recommended,
    }
