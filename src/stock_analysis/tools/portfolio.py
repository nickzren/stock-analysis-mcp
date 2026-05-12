"""Portfolio exposure tool."""

from time import perf_counter
from typing import Any

import pandas as pd

from stock_analysis.data.yfinance_client import fetch_history, fetch_info
from stock_analysis.utils.helpers import current_price_from_info
from stock_analysis.utils.indicators import calculate_pairwise_correlations
from stock_analysis.utils.provenance import (
    build_error_response,
    build_meta,
    build_provenance,
    utcnow_isoformat_z,
)
from stock_analysis.utils.validators import FetchParams


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

    normalized_positions = _normalize_positions(positions)
    symbol_info = await _fetch_symbol_info(normalized_positions)
    concentration = _build_concentration(
        normalized_positions=normalized_positions,
        symbol_info=symbol_info,
        total_value=total_value,
    )
    sector_exposure = _build_sector_exposure(concentration["positions"])
    correlation_data = await _build_correlation_data(normalized_positions)
    liquidity = _build_liquidity(
        normalized_positions=normalized_positions,
        symbol_info=symbol_info,
    )

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("portfolio_exposure", duration_ms),
        "data_provenance": {
            "fundamentals": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
            ),
            "price": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
            ),
        },
        "total_value": total_value,
        "concentration": concentration,
        "sector_exposure": sector_exposure,
        "correlation": correlation_data,
        "liquidity": liquidity,
    }


def _normalize_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize symbols while preserving position values."""
    normalized_positions = []
    for position in positions:
        symbol = position.get("symbol", "").upper().strip()
        value = position.get("value", 0)
        normalized_positions.append({"symbol": symbol, "value": value})
    return normalized_positions


async def _fetch_symbol_info(
    normalized_positions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fetch yfinance info for each position."""
    symbol_info: dict[str, dict[str, Any]] = {}
    for position in normalized_positions:
        try:
            symbol_info[position["symbol"]] = await fetch_info(position["symbol"])
        except Exception:
            symbol_info[position["symbol"]] = {}
    return symbol_info


def _build_concentration(
    normalized_positions: list[dict[str, Any]],
    symbol_info: dict[str, dict[str, Any]],
    total_value: float,
) -> dict[str, Any]:
    """Build concentration metrics."""
    position_details: list[dict[str, Any]] = []
    for position in normalized_positions:
        weight = position["value"] / total_value
        info = symbol_info.get(position["symbol"], {})
        position_details.append(
            {
                "symbol": position["symbol"],
                "value": position["value"],
                "weight": round(weight, 4),
                "sector": info.get("sector"),
                "is_concentrated": weight > 0.20,
            }
        )

    position_details.sort(key=lambda item: item["weight"], reverse=True)
    top_5_weight = sum(position["weight"] for position in position_details[:5])
    hhi = sum(position["weight"] ** 2 for position in position_details)

    return {
        "positions": position_details,
        "top_5_weight": round(top_5_weight, 4),
        "hhi": round(hhi, 4),
    }


def _build_sector_exposure(position_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sector exposure rows."""
    sector_weights: dict[str, float] = {}
    for position in position_details:
        sector = position.get("sector") or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0) + position["weight"]

    return [
        {
            "sector": sector,
            "weight": round(weight, 4),
            "is_overweight": weight > 0.30,
        }
        for sector, weight in sorted(sector_weights.items(), key=lambda item: -item[1])
    ]


async def _build_correlation_data(
    normalized_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build pairwise correlation metrics."""
    returns_dict: dict[str, pd.Series] = {}
    for position in normalized_positions:
        try:
            params = FetchParams(
                symbol=position["symbol"],
                period="1y",
                interval="1d",
                adjusted=True,
            )
            df = await fetch_history(params)
            df["date"] = pd.to_datetime(df["date"])
            close = pd.to_numeric(df["close"], errors="coerce")
            returns = close.pct_change()
            returns.index = df["date"]
            returns_dict[position["symbol"]] = returns.dropna()
        except Exception:
            pass

    if len(returns_dict) < 2:
        return {
            "pairs": [],
            "high_correlation_pairs": [],
            "avg_correlation": None,
            "avg_abs_correlation": None,
            "high_correlation_risk": None,
        }

    corr_results = calculate_pairwise_correlations(returns_dict, min_overlap=100)
    return {
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


def _build_liquidity(
    normalized_positions: list[dict[str, Any]],
    symbol_info: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build liquidity exit-risk fields."""
    illiquid_positions: list[dict[str, Any]] = []
    for position in normalized_positions:
        info = symbol_info.get(position["symbol"], {})
        avg_volume = info.get("averageVolume")
        current_price = current_price_from_info(info)

        if avg_volume and current_price:
            avg_dollar_volume = avg_volume * current_price
            if avg_dollar_volume > 0:
                days_to_exit = position["value"] / (avg_dollar_volume * 0.01)
                if days_to_exit > 5:
                    illiquid_positions.append(
                        {
                            "symbol": position["symbol"],
                            "days_to_exit": round(days_to_exit, 1),
                        }
                    )

    return {"illiquid_positions": illiquid_positions}
