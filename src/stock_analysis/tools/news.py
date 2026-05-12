"""Stock news tool."""

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any, cast

import pandas as pd

from stock_analysis.data.yfinance_client import fetch_ticker
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.provenance import (
    FetchError,
    build_error_response,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)
from stock_analysis.utils.sanitize import sanitize_text

POSITIVE_KEYWORDS = {
    "beat", "beats", "exceeded", "growth", "profit", "surge", "gain",
    "upgrade", "buy", "outperform", "record", "strong", "bullish",
    "raises", "raised", "higher", "boost", "soars", "jumps",
}
NEGATIVE_KEYWORDS = {
    "miss", "missed", "decline", "loss", "cut", "downgrade", "sell",
    "weak", "bearish", "lawsuit", "investigation", "recall", "layoff",
    "warns", "warning", "falls", "drops", "lower", "slump", "plunge",
}

POSITIVE_BIGRAMS = {
    "beat expectations", "raised guidance", "price target raised",
    "strong demand", "market share gains", "record revenue",
    "positive outlook", "margin expansion", "better than expected",
    "earnings beat", "revenue beat", "analyst upgrade",
    "buy rating", "outperform rating", "strong buy",
    "raised dividend", "share buyback", "beat estimates",
    "exceeded expectations", "strong growth",
}

NEGATIVE_BIGRAMS = {
    "missed expectations", "lowered guidance", "price target cut",
    "weak demand", "market share loss", "revenue decline",
    "negative outlook", "margin compression", "worse than expected",
    "earnings miss", "revenue miss", "analyst downgrade",
    "sell rating", "underperform rating", "cut dividend",
    "share dilution", "missed estimates", "below expectations",
    "profit warning", "going concern",
}

_BULLISH_CATALYSTS = {
    "earnings_beat": {"earnings beat", "beat expectations", "beat estimates", "revenue beat"},
    "guidance_raise": {"raised guidance", "raises guidance", "positive outlook", "higher outlook"},
    "analyst_upgrade": {"upgrade", "upgraded", "price target raised", "buy rating", "outperform rating"},
    "product_launch": {"new product", "launch", "rollout", "commercial launch"},
    "partnership_or_contract": {"partnership", "partnered", "contract", "deal", "award", "agreement"},
    "regulatory_approval": {"approval", "approved", "clearance", "authorized"},
    "buyback_or_dividend": {"share buyback", "buyback", "raised dividend", "special dividend"},
    "insider_buying": {"insider buy", "director buy", "ceo buy"},
}

_BEARISH_CATALYSTS = {
    "earnings_miss": {"earnings miss", "missed expectations", "missed estimates", "revenue miss"},
    "guidance_cut": {"lowered guidance", "guidance cut", "cuts guidance", "profit warning"},
    "offering_or_dilution": {"share dilution", "offering", "secondary offering", "atm program", "capital raise"},
    "analyst_downgrade": {"downgrade", "downgraded", "price target cut", "underperform rating", "sell rating"},
    "litigation_or_investigation": {"lawsuit", "investigation", "probe", "sec inquiry", "doj inquiry"},
    "regulatory_setback": {"complete response letter", "rejected", "clinical hold", "warning letter", "recall"},
    "restructuring_or_layoffs": {"layoffs", "restructuring", "job cuts", "cost cuts"},
    "going_concern_or_distress": {"going concern", "bankruptcy", "default", "liquidity crunch"},
    "insider_selling": {"insider sell", "director sell", "ceo sell"},
}

_NEUTRAL_CATALYSTS = {
    "conference_or_presentation": {"conference", "presentation", "fireside chat", "investor day"},
    "split_or_index_change": {"stock split", "reverse split", "index inclusion", "index rebalance"},
}


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
        ticker = await fetch_or_error(fetch_ticker(symbol), symbol)
    except FetchError as fe:
        return fe.response

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

    now = datetime.now(UTC).replace(tzinfo=None)
    cutoff_date = now - timedelta(days=days)
    articles = _parse_articles(news_data, cutoff_date)
    recent_earnings = _build_recent_earnings(ticker, cutoff_date, now)
    sentiment_summary = _build_sentiment_summary(articles, now)
    catalyst_intelligence = _build_catalyst_intelligence(articles)

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
                as_of=utcnow_isoformat_z(),
            ),
        },
        "symbol": normalized_symbol,
        "period_days": days,
        "article_count": len(articles),
        "articles": articles,
        "sentiment": sentiment_summary,
        "catalyst_intelligence": catalyst_intelligence,
        "recent_earnings": recent_earnings,
        "warnings": warnings if warnings else None,
    }


def _parse_articles(news_data: list[dict[str, Any]], cutoff_date: datetime) -> list[dict[str, Any]]:
    """Parse, sanitize, score, and sort news articles."""
    articles: list[dict[str, Any]] = []

    for item in news_data:
        content = item.get("content", {})
        pub_date_str = content.get("pubDate")
        if not pub_date_str:
            continue

        try:
            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            pub_date_naive = pub_date.replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue

        if pub_date_naive < cutoff_date:
            continue

        title = sanitize_text(content.get("title", ""), max_length=200)
        summary = sanitize_text(content.get("summary", ""), max_length=500)
        provider = sanitize_text(
            content.get("provider", {}).get("displayName", "Unknown"),
            max_length=50,
        )

        url = None
        canonical = content.get("canonicalUrl", {})
        if canonical:
            url = canonical.get("url")

        sentiment_result = _score_sentiment(cast(str, title), cast(str, summary))
        articles.append({
            "date": pub_date_naive.strftime("%Y-%m-%d"),
            "title": title,
            "summary": summary,
            "provider": provider,
            "url": url,
            "sentiment": sentiment_result["label"],
            "matched_positive": sentiment_result["matched_positive"] or None,
            "matched_negative": sentiment_result["matched_negative"] or None,
            "catalyst_tags": _extract_catalysts(cast(str, title), cast(str, summary)),
        })

    articles.sort(key=lambda x: x["date"], reverse=True)
    return articles


def _build_recent_earnings(
    ticker: Any,
    cutoff_date: datetime,
    now: datetime,
) -> dict[str, Any] | None:
    """Build the recent earnings report if it falls within the lookback window."""
    with suppress(Exception):
        earnings_dates = ticker.earnings_dates
        if earnings_dates is None or len(earnings_dates) == 0:
            return None

        for date, row in earnings_dates.iterrows():
            if isinstance(date, pd.Timestamp):
                earnings_date = date.to_pydatetime().replace(tzinfo=None)
            else:
                earnings_date = datetime.strptime(str(date)[:10], "%Y-%m-%d")

            if not cutoff_date <= earnings_date <= now:
                continue

            estimate = safe_float(row.get("EPS Estimate"))
            actual = safe_float(row.get("Reported EPS"))
            if actual is None:
                continue

            surprise = None
            surprise_pct = None
            beat_miss = None

            if estimate is not None and estimate != 0:
                surprise = actual - estimate
                surprise_pct = surprise / abs(estimate)
                beat_miss = "beat" if surprise > 0 else "miss" if surprise < 0 else "inline"

            return {
                "date": earnings_date.strftime("%Y-%m-%d"),
                "eps_estimate": estimate,
                "eps_actual": actual,
                "surprise": safe_round(surprise, 4),
                "surprise_pct": safe_round(surprise_pct, 4),
                "beat_miss": beat_miss,
            }

    return None


def _build_sentiment_summary(
    articles: list[dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Build sentiment summary and recency windows."""
    sentiment_windows: dict[str, tuple[datetime | None, dict[str, int]]] = {
        "all": (None, {"positive": 0, "negative": 0, "neutral": 0}),
        "7d": (now - timedelta(days=7), {"positive": 0, "negative": 0, "neutral": 0}),
        "30d": (now - timedelta(days=30), {"positive": 0, "negative": 0, "neutral": 0}),
    }

    for a in articles:
        sentiment = a["sentiment"]
        article_date: datetime | None = None
        with suppress(ValueError, KeyError):
            article_date = datetime.strptime(a["date"], "%Y-%m-%d")
        for cutoff, counts in sentiment_windows.values():
            if cutoff is None or (article_date is not None and article_date >= cutoff):
                counts[sentiment] += 1

    sentiment_counts = sentiment_windows["all"][1]
    sentiment_counts_7d = sentiment_windows["7d"][1]
    sentiment_counts_30d = sentiment_windows["30d"][1]

    sentiment_7d = _derive_sentiment(sentiment_counts_7d)
    sentiment_30d = _derive_sentiment(sentiment_counts_30d)
    sample_size_7d = sum(sentiment_counts_7d.values())
    sample_size_30d = sum(sentiment_counts_30d.values())

    return {
        "overall": _derive_sentiment(sentiment_counts),
        "confidence": _derive_confidence(sample_size_7d),
        "counts": sentiment_counts,
        "method": "keyword_v2",
        "headline_triggers": _build_headline_triggers(articles),
        "sentiment_7d": sentiment_7d,
        "sample_size_7d": sample_size_7d,
        "confidence_7d": _derive_confidence(sample_size_7d),
        "sentiment_30d": sentiment_30d,
        "sample_size_30d": sample_size_30d,
        "confidence_30d": _derive_confidence(sample_size_30d),
    }


def _build_headline_triggers(articles: list[dict[str, Any]]) -> dict[str, list[str]] | None:
    """Aggregate unique headline sentiment triggers."""
    all_positive_triggers: set[str] = set()
    all_negative_triggers: set[str] = set()

    for a in articles:
        if a.get("matched_positive"):
            all_positive_triggers.update(a["matched_positive"])
        if a.get("matched_negative"):
            all_negative_triggers.update(a["matched_negative"])

    if not all_positive_triggers and not all_negative_triggers:
        return None

    return {
        "positive": sorted(all_positive_triggers)[:10],
        "negative": sorted(all_negative_triggers)[:10],
    }


def _build_catalyst_intelligence(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate catalyst tag counts."""
    catalyst_counts: dict[str, dict[str, int]] = {
        "bullish": {},
        "bearish": {},
        "neutral": {},
    }
    for a in articles:
        catalyst_tags = a.get("catalyst_tags") or {}
        for polarity, counts in catalyst_counts.items():
            for tag in catalyst_tags.get(polarity, []) or []:
                counts[tag] = counts.get(tag, 0) + 1

    return {
        "bullish": _sorted_catalyst_counts(catalyst_counts["bullish"]),
        "bearish": _sorted_catalyst_counts(catalyst_counts["bearish"]),
        "neutral": _sorted_catalyst_counts(catalyst_counts["neutral"]),
        "sample_size": len(articles),
        "method": "keyword_catalyst_v1",
    }


def _derive_sentiment(counts: dict[str, int]) -> str | None:
    """Derive overall sentiment from counts."""
    total = sum(counts.values())
    if total == 0:
        return None
    if counts["positive"] > counts["negative"]:
        return "positive"
    if counts["negative"] > counts["positive"]:
        return "negative"
    return "neutral"


def _derive_confidence(count: int) -> str:
    """Derive confidence from sample size."""
    if count >= 10:
        return "high"
    if count >= 5:
        return "moderate"
    if count >= 1:
        return "low"
    return "none"


def _score_sentiment(title: str, summary: str = "") -> dict[str, Any]:
    """
    Upgraded keyword+bigram sentiment scoring (v2).

    Title is weighted 2x. Bigrams checked first (higher confidence).
    Returns dict with sentiment label and matched triggers.
    """
    title_lower = title.lower()
    summary_lower = summary.lower()

    matched_positive: list[str] = []
    matched_negative: list[str] = []

    # Check bigrams first (higher confidence, weight 2)
    for bg in POSITIVE_BIGRAMS:
        if bg in title_lower or bg in summary_lower:
            matched_positive.append(bg)
    for bg in NEGATIVE_BIGRAMS:
        if bg in title_lower or bg in summary_lower:
            matched_negative.append(bg)

    # Check unigrams (title weighted 2x)
    for w in POSITIVE_KEYWORDS:
        if w in title_lower or w in summary_lower:
            matched_positive.append(w)
    for w in NEGATIVE_KEYWORDS:
        if w in title_lower or w in summary_lower:
            matched_negative.append(w)

    # Title match bonus: count title matches double
    title_pos = sum(1 for w in POSITIVE_KEYWORDS if w in title_lower)
    title_neg = sum(1 for w in NEGATIVE_KEYWORDS if w in title_lower)

    pos_score = len(matched_positive) + title_pos  # title counted twice
    neg_score = len(matched_negative) + title_neg

    if pos_score > neg_score:
        sentiment = "positive"
    elif neg_score > pos_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "label": sentiment,
        "matched_positive": sorted(set(matched_positive))[:5],
        "matched_negative": sorted(set(matched_negative))[:5],
    }


def _extract_catalysts(title: str, summary: str = "") -> dict[str, list[str]]:
    """Extract catalyst tags for investor-facing news summaries."""
    text = f"{title} {summary}".lower()
    return {
        "bullish": _match_catalyst_bucket(text, _BULLISH_CATALYSTS),
        "bearish": _match_catalyst_bucket(text, _BEARISH_CATALYSTS),
        "neutral": _match_catalyst_bucket(text, _NEUTRAL_CATALYSTS),
    }


def _match_catalyst_bucket(text: str, patterns: dict[str, set[str]]) -> list[str]:
    matches = []
    for tag, phrases in patterns.items():
        if any(phrase in text for phrase in phrases):
            matches.append(tag)
    return matches


def _sorted_catalyst_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"tag": tag, "count": count}
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
