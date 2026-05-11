"""Shared helper functions for type conversion and formatting."""

import re
from typing import Any

import pandas as pd

# --- Type conversion helpers ---


def safe_float(value: Any) -> float | None:
    """Convert to float or return None. Handles NaN."""
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def safe_round(value: float | None, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    return round(value, decimals)


def safe_last_float(series: pd.Series) -> float | None:
    """Return the last value of a pandas Series as float, or None if empty/NaN."""
    if len(series) == 0:
        return None
    last = series.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)


def safe_int(value: Any) -> int | None:
    """Convert to int or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_str(value: Any) -> str | None:
    """Convert to stripped string or return None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --- Numeric formatting helpers ---


def round_or_none(x: float | None, ndigits: int = 0) -> float | None:
    """Round to ndigits or return None. Handles 0 correctly (unlike truthiness)."""
    if x is None:
        return None
    return round(x, ndigits)


def format_pct(value: float | None, decimals: int = 1) -> str | None:
    """Format a decimal ratio as a percent string."""
    if value is None:
        return None
    return f"{value * 100:.{decimals}f}%"


def format_compact_number(value: float | int | None) -> str | None:
    """Format large numbers with compact suffixes."""
    if value is None:
        return None
    abs_val = abs(float(value))
    if abs_val >= 1e12:
        return f"{value / 1e12:.1f}T"
    if abs_val >= 1e9:
        return f"{value / 1e9:.1f}B"
    if abs_val >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs_val >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def format_price(value: float | None, currency: str | None = None) -> str | None:
    """Format price with 2 decimals and optional currency code."""
    if value is None:
        return None
    label = f"{value:.2f}"
    if currency and currency != "USD":
        return f"{currency} {label}"
    return f"${label}"


def format_cashflow_value(value: float | None, currency: str | None = None) -> str | None:
    """Format cash flow values with sign, scale, and currency."""
    if value is None:
        return None
    sign = "+" if value > 0 else "-" if value < 0 else ""
    abs_val = abs(value)

    if abs_val >= 1e9:
        scaled = abs_val / 1e9
        unit = "B"
        decimals = 1
    elif abs_val >= 1e6:
        scaled = abs_val / 1e6
        unit = "M"
        decimals = 0 if abs_val >= 1e8 else 1
    elif abs_val >= 1e3:
        scaled = abs_val / 1e3
        unit = "K"
        decimals = 0
    else:
        scaled = abs_val
        unit = ""
        decimals = 0

    number = f"{scaled:.{decimals}f}{unit}"
    if currency and currency != "USD":
        return f"{sign}{currency} {number}"
    return f"{sign}${number}"


def format_fcf_label(
    value: float | None,
    period: str | None,
    currency: str | None,
    period_end: str | None = None,
) -> str | None:
    """Format FCF value with period for reporting."""
    value_str = format_cashflow_value(value, currency)
    if value_str is None:
        return None
    period_label = period or "TTM"
    end_label = f" (end {period_end})" if period_end else ""
    return f"FCF ({period_label}): {value_str}{end_label}"


def fcf_label_from_cashflow(cash_flow: dict[str, Any] | None) -> str | None:
    """Format the FCF label from a cash_flow section dict."""
    if not cash_flow:
        return None
    return format_fcf_label(
        cash_flow.get("free_cash_flow_ttm"),
        cash_flow.get("free_cash_flow_period"),
        cash_flow.get("currency"),
        cash_flow.get("free_cash_flow_period_end"),
    )


def format_level_distance_label(pct: float | None) -> str | None:
    """Format level distance relative to current price."""
    if pct is None:
        return None
    if abs(pct) < 0.0005:
        return "at current"
    direction = "above" if pct > 0 else "below"
    return f"{abs(pct) * 100:.1f}% {direction} current"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def first_sentences(text: str | None, max_sentences: int = 2) -> str | None:
    """Return the first N sentences from text."""
    if not text:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return " ".join(parts[:max_sentences]).strip()
