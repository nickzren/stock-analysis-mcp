"""Sector reference data for contextual analysis."""

from typing import Any

# Sector benchmarks: {sector_name: {metric: (low, median, high)}}
# Values represent approximate ranges for healthy companies in each sector
# Sources: Historical averages, S&P 500 sector data
SECTOR_BENCHMARKS: dict[str, dict[str, tuple[float, float, float]]] = {
    "Technology": {
        "pe": (15.0, 28.0, 45.0),
        "ps": (3.0, 7.0, 15.0),
        "net_margin": (0.08, 0.18, 0.30),
        "revenue_growth": (0.05, 0.12, 0.25),
        "debt_to_equity": (0.0, 0.30, 0.80),
    },
    "Healthcare": {
        "pe": (12.0, 22.0, 40.0),
        "ps": (2.0, 5.0, 12.0),
        "net_margin": (0.05, 0.12, 0.25),
        "revenue_growth": (0.03, 0.08, 0.18),
        "debt_to_equity": (0.10, 0.50, 1.20),
    },
    "Financial Services": {
        "pe": (8.0, 14.0, 22.0),
        "ps": (1.5, 3.0, 6.0),
        "net_margin": (0.10, 0.20, 0.35),
        "revenue_growth": (0.02, 0.06, 0.12),
        "debt_to_equity": (0.50, 2.00, 5.00),
    },
    "Consumer Cyclical": {
        "pe": (12.0, 20.0, 35.0),
        "ps": (0.5, 1.5, 4.0),
        "net_margin": (0.03, 0.07, 0.15),
        "revenue_growth": (0.02, 0.06, 0.15),
        "debt_to_equity": (0.20, 0.60, 1.50),
    },
    "Consumer Defensive": {
        "pe": (15.0, 22.0, 30.0),
        "ps": (1.0, 2.0, 4.0),
        "net_margin": (0.05, 0.10, 0.18),
        "revenue_growth": (0.01, 0.04, 0.08),
        "debt_to_equity": (0.30, 0.80, 1.50),
    },
    "Industrials": {
        "pe": (12.0, 20.0, 30.0),
        "ps": (1.0, 2.5, 5.0),
        "net_margin": (0.04, 0.09, 0.16),
        "revenue_growth": (0.02, 0.06, 0.12),
        "debt_to_equity": (0.30, 0.80, 1.80),
    },
    "Energy": {
        "pe": (6.0, 12.0, 20.0),
        "ps": (0.5, 1.5, 3.0),
        "net_margin": (0.03, 0.10, 0.20),
        "revenue_growth": (-0.05, 0.05, 0.15),
        "debt_to_equity": (0.20, 0.50, 1.20),
    },
    "Utilities": {
        "pe": (14.0, 18.0, 24.0),
        "ps": (1.5, 2.5, 4.0),
        "net_margin": (0.08, 0.14, 0.22),
        "revenue_growth": (0.01, 0.03, 0.06),
        "debt_to_equity": (0.80, 1.30, 2.00),
    },
    "Real Estate": {
        "pe": (20.0, 35.0, 55.0),
        "ps": (3.0, 6.0, 12.0),
        "net_margin": (0.10, 0.25, 0.45),
        "revenue_growth": (0.02, 0.05, 0.10),
        "debt_to_equity": (0.50, 1.00, 2.00),
    },
    "Communication Services": {
        "pe": (12.0, 20.0, 35.0),
        "ps": (2.0, 4.0, 8.0),
        "net_margin": (0.05, 0.15, 0.28),
        "revenue_growth": (0.03, 0.08, 0.18),
        "debt_to_equity": (0.30, 0.80, 1.50),
    },
    "Basic Materials": {
        "pe": (8.0, 15.0, 25.0),
        "ps": (0.8, 1.8, 3.5),
        "net_margin": (0.04, 0.10, 0.18),
        "revenue_growth": (-0.02, 0.05, 0.12),
        "debt_to_equity": (0.20, 0.50, 1.00),
    },
}


def get_sector_benchmarks(sector: str | None) -> dict[str, tuple[float, float, float]] | None:
    """Get benchmark ranges for a sector. Returns None if sector not found."""
    if sector is None:
        return None
    return SECTOR_BENCHMARKS.get(sector)


def calculate_sector_percentile(
    value: float | None,
    low: float,
    median: float,
    high: float,
) -> float | None:
    """
    Calculate where a value falls within sector range as a percentile (0-100).

    0 = at or below low, 50 = at median, 100 = at or above high.
    """
    if value is None:
        return None
    if value <= low:
        return 0.0
    if value >= high:
        return 100.0
    if value <= median:
        # Scale 0-50 between low and median
        return 50.0 * (value - low) / (median - low)
    # Scale 50-100 between median and high
    return 50.0 + 50.0 * (value - median) / (high - median)


def build_sector_comparison(
    sector: str | None,
    pe: float | None = None,
    ps: float | None = None,
    net_margin: float | None = None,
    revenue_growth: float | None = None,
    debt_to_equity: float | None = None,
) -> dict[str, Any] | None:
    """
    Build sector comparison for a stock's metrics.

    Returns dict with percentile rankings vs sector, or None if sector unknown.
    """
    benchmarks = get_sector_benchmarks(sector)
    if benchmarks is None:
        return None

    comparison: dict[str, Any] = {"sector": sector}
    metrics = {
        "pe": pe,
        "ps": ps,
        "net_margin": net_margin,
        "revenue_growth": revenue_growth,
        "debt_to_equity": debt_to_equity,
    }

    for metric_name, value in metrics.items():
        bench = benchmarks.get(metric_name)
        if bench is None or value is None:
            continue
        low, median, high = bench
        percentile = calculate_sector_percentile(value, low, median, high)
        comparison[metric_name] = {
            "value": round(value, 4),
            "sector_low": low,
            "sector_median": median,
            "sector_high": high,
            "percentile": round(percentile, 1) if percentile is not None else None,
        }

    return comparison if len(comparison) > 1 else None  # >1 because "sector" key always present
