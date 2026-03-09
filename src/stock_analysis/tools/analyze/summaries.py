"""Section summary text builders for analyze output."""

from typing import Any

from stock_analysis.utils.helpers import format_compact_number, format_pct, format_price


def build_technicals_summary_text(technicals_summary: dict[str, Any]) -> str | None:
    trend = technicals_summary.get("trend", {})
    momentum = technicals_summary.get("momentum", {})
    returns = technicals_summary.get("returns", {})
    position = technicals_summary.get("position_in_52w_range")

    if trend.get("above_sma200") is True:
        trend_clause = "Uptrend (above 200-day MA)"
    elif trend.get("above_sma200") is False:
        trend_clause = "Downtrend (below 200-day MA)"
    else:
        trend_clause = "Trend unclear (200-day MA unavailable)"

    if trend.get("golden_cross") is True:
        trend_clause = f"{trend_clause} with a golden cross"
    elif trend.get("death_cross") is True:
        trend_clause = f"{trend_clause} with a death cross"

    if trend.get("above_sma50") is True and trend.get("above_sma200") is True:
        trend_clause = f"{trend_clause}, above 50-day MA"
    elif trend.get("above_sma50") is False and trend.get("above_sma200") is False:
        trend_clause = f"{trend_clause}, below 50-day MA"

    trend_sentence = f"{trend_clause}."

    momentum_parts: list[str] = []
    rsi_val = momentum.get("rsi")
    if rsi_val is not None:
        if momentum.get("rsi_oversold") is True:
            momentum_parts.append(f"RSI {rsi_val:.1f} (oversold)")
        elif momentum.get("rsi_overbought") is True:
            momentum_parts.append(f"RSI {rsi_val:.1f} (overbought)")
        else:
            momentum_parts.append(f"RSI {rsi_val:.1f}")

    if momentum.get("macd_bullish") is True:
        momentum_parts.append("MACD bullish")

    return_1m = returns.get("return_1m")
    if return_1m is not None and abs(return_1m) >= 0.05:
        momentum_parts.append(f"1M {return_1m * 100:+.1f}%")

    return_3m = returns.get("return_3m")
    if return_3m is not None and abs(return_3m) >= 0.05:
        momentum_parts.append(f"3M {return_3m * 100:+.1f}%")

    if position is not None:
        if position <= 0.2:
            momentum_parts.append("near 52-week lows")
        elif position >= 0.8:
            momentum_parts.append("near 52-week highs")

    if not momentum_parts:
        return trend_sentence

    momentum_sentence = "; ".join(momentum_parts) + "."
    return f"{trend_sentence} {momentum_sentence}"


def build_fundamentals_summary_text(fundamentals_summary: dict[str, Any]) -> str | None:
    valuation = fundamentals_summary.get("valuation", {})
    growth = fundamentals_summary.get("growth", {})
    profitability = fundamentals_summary.get("profitability", {})
    cash_flow = fundamentals_summary.get("cash_flow", {})

    net_margin = profitability.get("net_margin")
    gross_margin = profitability.get("gross_margin")
    is_unprofitable = valuation.get("valuation_note") == "pe_not_meaningful"
    if net_margin is not None and net_margin < 0:
        is_unprofitable = True

    sentence_one_parts: list[str] = []
    if is_unprofitable:
        if net_margin is not None:
            sentence_one_parts.append(f"Unprofitable (net margin {net_margin * 100:.1f}%)")
        else:
            sentence_one_parts.append("Unprofitable")
    elif net_margin is not None:
        sentence_one_parts.append(f"Net margin {net_margin * 100:.1f}%")

    if gross_margin is not None:
        sentence_one_parts.append(f"gross margin {gross_margin * 100:.1f}%")

    fcf_label = cash_flow.get("free_cash_flow_label")
    if fcf_label:
        sentence_one_parts.append(fcf_label)

    sentence_two_parts: list[str] = []
    revenue_yoy = growth.get("revenue_yoy")
    if revenue_yoy is not None:
        sentence_two_parts.append(f"Revenue {revenue_yoy * 100:+.1f}% YoY")

    eps_yoy = growth.get("eps_yoy")
    if eps_yoy is not None:
        sentence_two_parts.append(f"EPS {eps_yoy * 100:+.1f}% YoY")

    pe = valuation.get("pe_trailing")
    ps = valuation.get("ps_trailing")
    if pe is not None:
        sentence_two_parts.append(f"P/E {pe:.1f}x")
    elif ps is not None:
        sentence_two_parts.append(f"P/S {ps:.1f}x")

    if not sentence_one_parts and not sentence_two_parts:
        return None

    sentences: list[str] = []
    if sentence_one_parts:
        sentences.append("; ".join(sentence_one_parts) + ".")
    if sentence_two_parts:
        sentences.append("; ".join(sentence_two_parts) + ".")

    return " ".join(sentences[:2])


def build_risk_summary_text(risk_summary: dict[str, Any]) -> str | None:
    risk_regime = risk_summary.get("risk_regime", {})
    classification = risk_regime.get("classification")
    vol = risk_summary.get("annualized_volatility")
    dd = risk_summary.get("max_drawdown_1y")
    beta = risk_summary.get("beta")

    parts: list[str] = []
    if vol is not None:
        parts.append(f"{vol * 100:.1f}% annualized volatility")
    if dd is not None:
        parts.append(f"{dd * 100:.1f}% max drawdown")
    if beta is not None:
        parts.append(f"beta {beta:.2f}")

    if classification:
        if parts:
            return f"Risk regime {classification}: {', '.join(parts)}."
        return f"Risk regime {classification}."

    if parts:
        return f"Risk profile: {', '.join(parts)}."
    return None


def build_ownership_summary_text(ownership: dict[str, Any]) -> str | None:
    insider_pct = ownership.get("insider_pct")
    institutional_pct = ownership.get("institutional_pct")
    float_shares = ownership.get("float_shares")

    parts: list[str] = []
    inst_label = format_pct(institutional_pct, decimals=1)
    if inst_label:
        parts.append(f"{inst_label} institutional")
    insider_label = format_pct(insider_pct, decimals=2)
    if insider_label:
        parts.append(f"{insider_label} insider")
    float_label = format_compact_number(float_shares)
    if float_label:
        parts.append(f"float {float_label} shares")

    if not parts:
        return None
    return f"Ownership: {', '.join(parts)}."


def build_short_interest_summary_text(short_interest: dict[str, Any]) -> str | None:
    short_pct = short_interest.get("short_pct_of_float")
    days_to_cover = short_interest.get("days_to_cover")
    short_change = short_interest.get("short_change_mom")

    parts: list[str] = []
    pct_label = format_pct(short_pct, decimals=2)
    if pct_label:
        parts.append(f"{pct_label} of float short")
    if days_to_cover is not None:
        parts.append(f"{days_to_cover:.1f} days to cover")
    if short_change is not None and abs(short_change) >= 0.05:
        parts.append(f"{short_change * 100:+.1f}% m/m")

    if not parts:
        return None
    return f"Short interest: {', '.join(parts)}."


def build_analyst_summary_text(
    analyst_coverage: dict[str, Any],
    current_price: float | None,
    currency: str | None = None,
) -> str | None:
    rating = analyst_coverage.get("rating")
    rating_score = analyst_coverage.get("rating_score")
    num_analysts = analyst_coverage.get("num_analysts")
    target_mean = analyst_coverage.get("price_target_mean")
    upside = analyst_coverage.get("upside_to_mean_target")

    sentences: list[str] = []
    summary_parts: list[str] = []

    if rating:
        summary_parts.append(f"Analyst consensus: {str(rating).replace('_', ' ').title()}")
    elif rating_score is not None or num_analysts is not None:
        summary_parts.append("Analyst coverage")
    if rating_score is not None:
        summary_parts.append(f"mean {rating_score:.2f}")
    if num_analysts is not None:
        summary_parts.append(f"{num_analysts} analysts")

    if summary_parts:
        sentence = summary_parts[0]
        if len(summary_parts) > 1:
            sentence = f"{sentence} ({', '.join(summary_parts[1:])})."
        else:
            sentence = f"{sentence}."
        sentences.append(sentence)

    if target_mean is not None:
        target_label = format_price(target_mean, currency)
        if upside is not None:
            sentences.append(f"Mean target {target_label} ({upside * 100:+.1f}% vs current).")
        elif current_price is not None and current_price > 0:
            implied = (target_mean - current_price) / current_price
            sentences.append(f"Mean target {target_label} ({implied * 100:+.1f}% vs current).")
        else:
            sentences.append(f"Mean target {target_label}.")

    if not sentences:
        return None
    return " ".join(sentences[:2])


def build_governance_summary_text(governance: dict[str, Any]) -> str | None:
    overall = governance.get("overall_risk")
    audit = governance.get("audit_risk")
    board = governance.get("board_risk")
    comp = governance.get("compensation_risk")
    rights = governance.get("shareholder_rights_risk")

    if overall is None and audit is None and board is None and comp is None and rights is None:
        return None

    parts: list[str] = []
    if overall is not None:
        parts.append(f"overall {overall}/10")
    if audit is not None:
        parts.append(f"audit {audit}/10")
    if board is not None:
        parts.append(f"board {board}/10")
    if comp is not None:
        parts.append(f"comp {comp}/10")
    if rights is not None:
        parts.append(f"rights {rights}/10")

    return f"Governance risk (1=low, 10=high): {', '.join(parts)}."


def build_valuation_context_summary_text(valuation_context: dict[str, Any]) -> str | None:
    pe_current = valuation_context.get("pe_current")
    ps_current = valuation_context.get("ps_current")
    status_reason = valuation_context.get("status_reason")

    parts: list[str] = []
    if pe_current is not None:
        parts.append(f"current P/E {pe_current:.1f}x")
    if ps_current is not None:
        parts.append(f"current P/S {ps_current:.1f}x")

    if not parts:
        return None
    summary = "; ".join(parts)
    summary = f"{summary}. History unavailable." if status_reason else f"{summary}."
    return summary


def build_news_summary_text(news_summary: dict[str, Any]) -> str | None:
    if not news_summary:
        return None
    article_count = news_summary.get("article_count")
    sentiment = (news_summary.get("sentiment") or {}).get("overall")
    confidence = (news_summary.get("sentiment") or {}).get("confidence")
    catalyst_intelligence = news_summary.get("catalyst_intelligence") or {}
    bullish_catalysts = catalyst_intelligence.get("bullish") or []
    bearish_catalysts = catalyst_intelligence.get("bearish") or []

    parts: list[str] = []
    if article_count is not None:
        parts.append(f"{article_count} recent headlines")
    if bullish_catalysts:
        top = ", ".join(item.get("tag", "").replace("_", " ") for item in bullish_catalysts[:2] if item.get("tag"))
        if top:
            parts.append(f"bullish catalysts {top}")
    if bearish_catalysts:
        top = ", ".join(item.get("tag", "").replace("_", " ") for item in bearish_catalysts[:2] if item.get("tag"))
        if top:
            parts.append(f"risks {top}")
    if sentiment:
        if confidence is not None:
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                parts.append(f"sentiment {sentiment} (conf {confidence:.2f})")
            else:
                parts.append(f"sentiment {sentiment} (conf {confidence})")
        else:
            parts.append(f"sentiment {sentiment}")

    if not parts:
        return None
    return f"News: {', '.join(parts)}."


def build_dip_summary_text(dip_assessment: dict[str, Any]) -> str | None:
    dip_classification = dip_assessment.get("dip_classification") or {}
    dip_depth = dip_assessment.get("dip_depth") or {}
    dip_type = dip_classification.get("type")
    severity = dip_depth.get("severity")
    from_52w_high = dip_depth.get("from_52w_high")

    parts: list[str] = []
    if dip_type:
        parts.append(f"Dip type {dip_type}")
    if severity:
        parts.append(f"severity {severity}")
    if from_52w_high is not None:
        if isinstance(from_52w_high, (int, float)) and not isinstance(from_52w_high, bool):
            pct = format_pct(float(from_52w_high), decimals=1)
            parts.append(f"{pct} from 52W high" if pct else "from 52W high")
        else:
            parts.append(f"{from_52w_high} from 52W high")

    if not parts:
        return None
    return f"{', '.join(parts)}."
