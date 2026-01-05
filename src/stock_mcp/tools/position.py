"""Analyze position tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.tools.technicals import technicals
from stock_mcp.utils.provenance import build_error_response, build_meta


async def analyze_position(
    symbol: str,
    cost_basis: float,
    purchase_date: str,
    shares: float | None = None,
) -> dict[str, Any]:
    """
    Analyze an existing position for hold/sell decision.

    Args:
        symbol: Stock ticker symbol
        cost_basis: Cost basis per share
        purchase_date: Purchase date (YYYY-MM-DD)
        shares: Number of shares (optional)

    Returns:
        Dict with position details, holding info, tax implications, signals
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()

    # Parse purchase date
    try:
        purchase_dt = datetime.strptime(purchase_date, "%Y-%m-%d")
    except ValueError:
        return build_error_response(
            error_type="invalid_parameters",
            message=f"Invalid purchase_date format: {purchase_date}. Expected YYYY-MM-DD",
            symbol=symbol,
        )

    # Get current technicals
    tech_result = await technicals(normalized_symbol)

    if tech_result.get("error"):
        return tech_result

    current_price = tech_result.get("current_price")
    if current_price is None:
        return build_error_response(
            error_type="data_unavailable",
            message="Could not get current price",
            symbol=symbol,
        )

    # Position calculations
    gain_loss = (current_price - cost_basis) / cost_basis if cost_basis != 0 else None
    gain_loss_dollars = (
        (current_price - cost_basis) * shares if shares is not None else None
    )
    current_value = current_price * shares if shares is not None else None

    position = {
        "cost_basis": cost_basis,
        "current_price": current_price,
        "shares": shares,
        "current_value": _safe_round(current_value, 2),
        "gain_loss": _safe_round(gain_loss, 4),
        "gain_loss_dollars": _safe_round(gain_loss_dollars, 2),
    }

    # Holding period calculations
    now = datetime.now()
    days_held = (now - purchase_dt).days
    is_long_term = days_held >= 365

    days_to_long_term = None
    if not is_long_term:
        days_to_long_term = 365 - days_held

    holding = {
        "purchase_date": purchase_date,
        "days_held": days_held,
        "is_long_term": is_long_term,
        "days_to_long_term": days_to_long_term,
    }

    # Tax calculations (simplified US tax rates)
    SHORT_TERM_RATE = 0.37  # Top marginal rate
    LONG_TERM_RATE = 0.20  # Long-term capital gains rate

    tax_type = "long_term" if is_long_term else "short_term"
    rate_now = LONG_TERM_RATE if is_long_term else SHORT_TERM_RATE

    tax_on_gain_now = None
    tax_if_waited = None
    potential_savings = None

    if gain_loss_dollars is not None and gain_loss_dollars > 0:
        tax_on_gain_now = gain_loss_dollars * rate_now
        tax_if_waited = gain_loss_dollars * LONG_TERM_RATE
        potential_savings = tax_on_gain_now - tax_if_waited if not is_long_term else 0

    tax = {
        "type": tax_type,
        "rate_if_sold_now": rate_now,
        "rate_if_long_term": LONG_TERM_RATE,
        "tax_on_gain_now": _safe_round(tax_on_gain_now, 2),
        "tax_if_waited": _safe_round(tax_if_waited, 2),
        "potential_savings": _safe_round(potential_savings, 2),
    }

    # Extract current signals from technicals
    ma = tech_result.get("moving_averages", {})
    rsi = tech_result.get("rsi", {})
    macd = tech_result.get("macd", {})
    returns = tech_result.get("returns", {})

    bullish: list[str] = []
    bearish: list[str] = []

    # Bullish signals
    if _get_rule_triggered(ma, "above_sma50") is True:
        bullish.append("above_sma50")
    if _get_rule_triggered(ma, "above_sma200") is True:
        bullish.append("above_sma200")
    if _get_rule_triggered(ma, "golden_cross") is True:
        bullish.append("golden_cross")
    if _get_rule_triggered(rsi, "oversold") is True:
        bullish.append("rsi_oversold")
    if _get_rule_triggered(macd, "bullish_cross") is True:
        bullish.append("macd_bullish")

    ret_1m = returns.get("return_1m")
    if ret_1m is not None and ret_1m > 0.05:
        bullish.append("positive_1m_momentum")

    # Bearish signals
    if _get_rule_triggered(ma, "above_sma50") is False:
        bearish.append("below_sma50")
    if _get_rule_triggered(ma, "above_sma200") is False:
        bearish.append("below_sma200")
    if _get_rule_triggered(ma, "death_cross") is True:
        bearish.append("death_cross")
    if _get_rule_triggered(rsi, "overbought") is True:
        bearish.append("rsi_overbought")
    if _get_rule_triggered(macd, "bearish_cross") is True:
        bearish.append("macd_bearish")

    if ret_1m is not None and ret_1m < -0.05:
        bearish.append("negative_1m_momentum")

    current_signals = {
        "bullish": bullish,
        "bearish": bearish,
    }

    # Sell signals (specific triggers)
    # Use _invert_nullable to flip True/False while preserving None
    above_sma50 = _get_rule_triggered(ma, "above_sma50")
    above_sma200 = _get_rule_triggered(ma, "above_sma200")
    broke_sma50 = _invert_nullable(above_sma50)  # True if below, None if unknown
    broke_sma200 = _invert_nullable(above_sma200)  # True if below, None if unknown
    death_cross = _get_rule_triggered(ma, "death_cross")
    rsi_overbought = _get_rule_triggered(rsi, "overbought")
    macd_bearish = _get_rule_triggered(macd, "bearish_cross")

    # Count active sell signals (only count True, not None)
    sell_signal_values = [broke_sma50, broke_sma200, death_cross, rsi_overbought, macd_bearish]
    active_count = sum(1 for v in sell_signal_values if v is True)

    sell_signals = {
        "broke_sma50": broke_sma50,
        "broke_sma200": broke_sma200,
        "death_cross": death_cross,
        "rsi_overbought": rsi_overbought,
        "macd_bearish": macd_bearish,
        "active_count": active_count,
    }

    # Support levels
    sma_50 = ma.get("sma_50")
    sma_200 = ma.get("sma_200")

    atr = tech_result.get("atr", {})
    atr_val = atr.get("value")
    atr_1x_below = current_price - atr_val if atr_val else None

    # Get recent low (1 month)
    price_pos = tech_result.get("price_position", {})
    recent_low_1m = price_pos.get("low_1m")

    support_levels = {
        "sma_50": sma_50,
        "sma_200": sma_200,
        "atr_1x_below": _safe_round(atr_1x_below, 2),
        "recent_low_1m": recent_low_1m,
    }

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("analyze_position", duration_ms),
        "data_provenance": tech_result.get("data_provenance", {}),
        "symbol": normalized_symbol,
        "position": position,
        "holding": holding,
        "tax": tax,
        "current_signals": current_signals,
        "sell_signals": sell_signals,
        "support_levels": support_levels,
    }


def _get_rule_triggered(data: dict[str, Any], rule_name: str) -> bool | None:
    """Extract triggered value from a rules dict."""
    rules = data.get("rules", {})
    rule = rules.get(rule_name, {})
    return rule.get("triggered")


def _invert_nullable(value: bool | None) -> bool | None:
    """Invert a boolean while preserving None."""
    if value is None:
        return None
    return not value


def _safe_round(value: float | None, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    return round(value, decimals)
