"""Portfolio exposure tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.data.yfinance_client import fetch_history, fetch_info
from stock_mcp.utils.indicators import calculate_pairwise_correlations
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.validators import FetchParams


async def portfolio_exposure(
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze portfolio exposure and concentration risk.

    Args:
        positions: List of position dicts with 'symbol' and 'value' keys

    Returns:
        Dict with concentration, sector exposure, correlation, liquidity analysis
    """
    start_time = perf_counter()

    if not positions:
        return build_error_response(
            error_type="invalid_parameters",
            message="positions list cannot be empty",
            symbol=None,
        )

    # Calculate total value
    total_value = sum(p.get("value", 0) for p in positions)
    if total_value <= 0:
        return build_error_response(
            error_type="invalid_parameters",
            message="Total portfolio value must be positive",
            symbol=None,
        )

    # Normalize symbols
    normalized_positions = []
    for p in positions:
        symbol = p.get("symbol", "").upper().strip()
        value = p.get("value", 0)
        normalized_positions.append({"symbol": symbol, "value": value})

    # Fetch info for each symbol (for sector data)
    symbol_info: dict[str, dict[str, Any]] = {}
    for p in normalized_positions:
        try:
            info = await fetch_info(p["symbol"])
            symbol_info[p["symbol"]] = info
        except Exception:
            symbol_info[p["symbol"]] = {}

    # Concentration analysis
    position_details: list[dict[str, Any]] = []
    for p in normalized_positions:
        weight = p["value"] / total_value
        info = symbol_info.get(p["symbol"], {})
        sector = info.get("sector")

        # Consider concentrated if > 20% of portfolio
        is_concentrated = weight > 0.20 if weight is not None else None

        position_details.append(
            {
                "symbol": p["symbol"],
                "value": p["value"],
                "weight": round(weight, 4),
                "sector": sector,
                "is_concentrated": is_concentrated,
            }
        )

    # Sort by weight descending
    position_details.sort(key=lambda x: x["weight"], reverse=True)

    # Top 5 weight
    top_5_weight = sum(p["weight"] for p in position_details[:5])

    # HHI (Herfindahl-Hirschman Index)
    hhi = sum(p["weight"] ** 2 for p in position_details)

    concentration = {
        "positions": position_details,
        "top_5_weight": round(top_5_weight, 4),
        "hhi": round(hhi, 4),
    }

    # Sector exposure
    sector_weights: dict[str, float] = {}
    for p in position_details:
        sector = p.get("sector") or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0) + p["weight"]

    sector_exposure = []
    for sector, weight in sorted(sector_weights.items(), key=lambda x: -x[1]):
        # Consider overweight if > 30%
        is_overweight = weight > 0.30 if weight is not None else None
        sector_exposure.append(
            {
                "sector": sector,
                "weight": round(weight, 4),
                "is_overweight": is_overweight,
            }
        )

    # Correlation analysis
    returns_dict: dict[str, pd.Series] = {}
    for p in normalized_positions:
        try:
            params = FetchParams(
                symbol=p["symbol"],
                period="1y",
                interval="1d",
                adjusted=True,
            )
            df = await fetch_history(params)
            df["date"] = pd.to_datetime(df["date"])
            close = pd.to_numeric(df["close"], errors="coerce")
            # Calculate returns and align with dates
            returns = close.pct_change()
            returns.index = df["date"]
            returns = returns.dropna()
            returns_dict[p["symbol"]] = returns
        except Exception:
            pass

    correlation_data: dict[str, Any] = {
        "pairs": [],
        "high_correlation_pairs": [],
        "avg_correlation": None,
        "avg_abs_correlation": None,
        "high_correlation_risk": None,
    }

    if len(returns_dict) >= 2:
        corr_results = calculate_pairwise_correlations(returns_dict, min_overlap=100)
        correlation_data = {
            "pairs": corr_results["pairs"],
            "high_correlation_pairs": corr_results["high_correlation_pairs"],
            "avg_correlation": corr_results["avg_correlation"],
            "avg_abs_correlation": corr_results["avg_abs_correlation"],
            "high_correlation_risk": (
                len(corr_results["high_correlation_pairs"]) > 0
                if corr_results["high_correlation_pairs"] is not None
                else None
            ),
        }

    # Liquidity analysis
    illiquid_positions: list[dict[str, Any]] = []
    for p in normalized_positions:
        info = symbol_info.get(p["symbol"], {})
        avg_volume = info.get("averageVolume")
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")

        if avg_volume and current_price:
            avg_dollar_volume = avg_volume * current_price
            # Calculate days to exit (assuming 1% of ADV per day)
            if avg_dollar_volume > 0:
                days_to_exit = p["value"] / (avg_dollar_volume * 0.01)
                if days_to_exit > 5:  # More than 5 days to exit
                    illiquid_positions.append(
                        {
                            "symbol": p["symbol"],
                            "days_to_exit": round(days_to_exit, 1),
                        }
                    )

    liquidity = {
        "illiquid_positions": illiquid_positions,
    }

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("portfolio_exposure", duration_ms),
        "data_provenance": {
            "fundamentals": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
            "price": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "total_value": total_value,
        "concentration": concentration,
        "sector_exposure": sector_exposure,
        "correlation": correlation_data,
        "liquidity": liquidity,
    }
