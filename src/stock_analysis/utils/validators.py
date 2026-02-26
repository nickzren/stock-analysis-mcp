"""Validation utilities and parameter classes."""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Allowlists for cache key stability
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


@dataclass(frozen=True)
class FetchParams:
    """Immutable fetch parameters. Used for cache key + fetch."""

    symbol: str
    period: str
    interval: str
    adjusted: bool
    tz: str = "America/New_York"

    def __post_init__(self) -> None:
        # Normalize symbol: uppercase, strip whitespace
        object.__setattr__(self, "symbol", self.symbol.upper().strip())

        # Normalize period/interval: lowercase, strip whitespace, validate
        period = self.period.lower().strip()
        interval = self.interval.lower().strip()

        if period not in VALID_PERIODS:
            raise ValueError(f"Invalid period '{self.period}'. Must be one of: {VALID_PERIODS}")
        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{self.interval}'. Must be one of: {VALID_INTERVALS}"
            )

        object.__setattr__(self, "period", period)
        object.__setattr__(self, "interval", interval)

    def to_uri(self) -> str:
        """Canonical URI for caching."""
        adj = "adjusted" if self.adjusted else "unadjusted"
        return f"price://{self.symbol}/{self.period}/{self.interval}/{adj}"

    def to_yf_kwargs(self) -> dict[str, Any]:
        """Kwargs for yf.download()."""
        return {
            "tickers": self.symbol,
            "period": self.period,
            "interval": self.interval,
            "auto_adjust": self.adjusted,
            "progress": False,
        }


def check_rule(
    value: float | None,
    threshold: float,
    comparator: Callable[[float, float], bool] = operator.gt,
) -> bool | None:
    """
    Check a rule with nullable boolean semantics.

    If value is None, returns None (not False).

    Args:
        value: The value to check (may be None)
        threshold: The threshold to compare against
        comparator: Comparison function (default: operator.gt)

    Returns:
        True/False if value is not None, None otherwise
    """
    if value is None:
        return None
    return comparator(value, threshold)


def check_rule_expr(
    value1: float | None,
    value2: float | None,
    comparator: Callable[[float, float], bool] = operator.gt,
) -> bool | None:
    """
    Check a rule comparing two values with nullable boolean semantics.

    If either value is None, returns None (not False).

    Args:
        value1: First value (may be None)
        value2: Second value (may be None)
        comparator: Comparison function (default: operator.gt)

    Returns:
        True/False if both values are not None, None otherwise
    """
    if value1 is None or value2 is None:
        return None
    return comparator(value1, value2)
