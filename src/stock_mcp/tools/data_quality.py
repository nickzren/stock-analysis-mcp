"""Data quality report tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

from stock_mcp.data.yfinance_client import fetch_history, fetch_info
from stock_mcp.utils.provenance import build_meta, build_provenance
from stock_mcp.utils.validators import FetchParams


async def data_quality_report(symbols: list[str]) -> dict[str, Any]:
    """
    Check data availability and quality for a list of symbols.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        Dict with per-symbol data quality and summary
    """
    start_time = perf_counter()

    if not symbols:
        return {
            "meta": build_meta("data_quality_report", 0),
            "data_provenance": {},
            "symbols": [],
            "summary": {
                "valid_count": 0,
                "avg_quality": 0,
                "common_issues": ["no_symbols_provided"],
            },
        }

    symbol_reports: list[dict[str, Any]] = []
    all_issues: list[str] = []

    for symbol in symbols:
        normalized_symbol = symbol.upper().strip()
        report = await _check_symbol(normalized_symbol)
        symbol_reports.append(report)

        if not report["valid"]:
            all_issues.append(f"{normalized_symbol}:invalid_symbol")
        for field in report.get("missing_fields", []):
            all_issues.append(f"missing_{field}")

    # Calculate summary
    valid_count = sum(1 for r in symbol_reports if r["valid"])
    quality_scores = [r["quality_score"] for r in symbol_reports if r["quality_score"] is not None]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    # Find most common issues
    issue_counts: dict[str, int] = {}
    for issue in all_issues:
        # Normalize issue (remove symbol prefix)
        normalized_issue = issue.split(":")[-1] if ":" in issue else issue
        issue_counts[normalized_issue] = issue_counts.get(normalized_issue, 0) + 1

    common_issues = sorted(issue_counts.keys(), key=lambda x: -issue_counts[x])[:5]

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("data_quality_report", duration_ms),
        "data_provenance": {
            "check": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "symbols": symbol_reports,
        "summary": {
            "valid_count": valid_count,
            "avg_quality": round(avg_quality, 2),
            "common_issues": common_issues,
        },
    }


async def _check_symbol(symbol: str) -> dict[str, Any]:
    """Check data quality for a single symbol."""
    report: dict[str, Any] = {
        "symbol": symbol,
        "valid": False,
        "availability": {
            "price": False,
            "fundamentals": False,
            "events": False,
        },
        "staleness": {
            "price_last_bar": None,
            "fundamentals_fiscal_end": None,
            "events_last_update": None,
        },
        "missing_fields": [],
        "quality_score": 0.0,
    }

    # Check price data
    try:
        params = FetchParams(
            symbol=symbol,
            period="5d",
            interval="1d",
            adjusted=True,
        )
        df = await fetch_history(params)
        if len(df) > 0:
            report["availability"]["price"] = True
            report["staleness"]["price_last_bar"] = df["date"].iloc[-1]
    except Exception:
        report["missing_fields"].append("price_data")

    # Check fundamentals
    try:
        info = await fetch_info(symbol)
        if info:
            report["availability"]["fundamentals"] = True
            report["valid"] = True

            # Check for key fields
            key_fields = [
                "regularMarketPrice",
                "marketCap",
                "trailingPE",
                "profitMargins",
                "revenueGrowth",
            ]
            for field in key_fields:
                if info.get(field) is None:
                    report["missing_fields"].append(field)

            # Get fiscal period info
            fiscal_year_end = info.get("lastFiscalYearEnd")
            if fiscal_year_end:
                try:
                    fiscal_date = datetime.fromtimestamp(fiscal_year_end)
                    report["staleness"]["fundamentals_fiscal_end"] = fiscal_date.strftime(
                        "%Y-%m-%d"
                    )
                except (ValueError, TypeError, OSError):
                    pass

            # Check events availability
            earnings_date = info.get("earningsDate")
            if earnings_date:
                report["availability"]["events"] = True
                report["staleness"]["events_last_update"] = datetime.utcnow().strftime(
                    "%Y-%m-%d"
                )
    except Exception:
        report["missing_fields"].append("fundamentals_data")

    # Calculate quality score
    # Weight: price (30%), fundamentals (40%), events (10%), missing fields penalty (20%)
    score = 0.0
    if report["availability"]["price"]:
        score += 0.30
    if report["availability"]["fundamentals"]:
        score += 0.40
    if report["availability"]["events"]:
        score += 0.10

    # Penalty for missing fields (up to 20%)
    missing_penalty = min(len(report["missing_fields"]) * 0.04, 0.20)
    score = max(0, score + 0.20 - missing_penalty)

    report["quality_score"] = round(score, 2)

    return report
