"""Stock news tool."""

from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.data.yfinance_client import fetch_ticker
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.sanitize import sanitize_text


async def stock_news(symbol: str, days: int = 7) -> dict[str, Any]:
    """
    Get recent news and earnings for a stock.

    Args:
        symbol: Stock ticker symbol
        days: Number of days to look back (default: 7)

    Returns:
        Dict with news articles and recent earnings report (if any within period)
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()

    try:
        ticker = await fetch_ticker(symbol)
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch data: {e}",
            symbol=symbol,
        )

    # Get news
    try:
        news_data = ticker.news
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch news: {e}",
            symbol=symbol,
        )

    # Empty news is valid - just return empty list, don't error
    if not news_data:
        news_data = []

    # Filter by date range
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    articles: list[dict[str, Any]] = []

    for item in news_data:
        content = item.get("content", {})
        pub_date_str = content.get("pubDate")

        if not pub_date_str:
            continue

        # Parse date
        try:
            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            pub_date_naive = pub_date.replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue

        # Skip if older than cutoff
        if pub_date_naive < cutoff_date:
            continue

        title = sanitize_text(content.get("title", ""), max_length=200)
        summary = sanitize_text(content.get("summary", ""), max_length=500)
        provider = sanitize_text(content.get("provider", {}).get("displayName", "Unknown"), max_length=50)

        # Get URL
        url = None
        canonical = content.get("canonicalUrl", {})
        if canonical:
            url = canonical.get("url")

        articles.append({
            "date": pub_date_naive.strftime("%Y-%m-%d"),
            "title": title,
            "summary": summary,
            "provider": provider,
            "url": url,
        })

    # Sort by date descending
    articles.sort(key=lambda x: x["date"], reverse=True)

    # Get recent earnings report if within the lookback period
    recent_earnings: dict[str, Any] | None = None
    try:
        earnings_dates = ticker.earnings_dates
        if earnings_dates is not None and len(earnings_dates) > 0:
            for date, row in earnings_dates.iterrows():
                # Convert to datetime for comparison
                if isinstance(date, pd.Timestamp):
                    earnings_date = date.to_pydatetime().replace(tzinfo=None)
                else:
                    earnings_date = datetime.strptime(str(date)[:10], "%Y-%m-%d")

                # Check if this earnings is within our lookback period and in the past
                if cutoff_date <= earnings_date <= datetime.utcnow():
                    estimate = _safe_float(row.get("EPS Estimate"))
                    actual = _safe_float(row.get("Reported EPS"))

                    # Only include if we have actual reported earnings (not future)
                    if actual is not None:
                        surprise = None
                        surprise_pct = None
                        beat_miss = None

                        if estimate is not None and estimate != 0:
                            surprise = actual - estimate
                            surprise_pct = surprise / abs(estimate)
                            beat_miss = "beat" if surprise > 0 else "miss" if surprise < 0 else "inline"

                        recent_earnings = {
                            "date": earnings_date.strftime("%Y-%m-%d"),
                            "eps_estimate": estimate,
                            "eps_actual": actual,
                            "surprise": _safe_round(surprise, 4),
                            "surprise_pct": _safe_round(surprise_pct, 4),
                            "beat_miss": beat_miss,
                        }
                        break  # Only get the most recent one
    except Exception:
        pass

    # Build warnings
    warnings: list[str] = []
    if len(articles) == 0:
        warnings.append(f"No news articles found in the past {days} days")

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("stock_news", duration_ms),
        "data_provenance": {
            "news": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "symbol": normalized_symbol,
        "period_days": days,
        "article_count": len(articles),
        "articles": articles,
        "recent_earnings": recent_earnings,
        "warnings": warnings if warnings else None,
    }


def _safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def _safe_round(value: float | None, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    return round(value, decimals)
