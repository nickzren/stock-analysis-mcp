"""Options-derived signals tool."""

from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

from stock_analysis.data.yfinance_client import fetch_ticker
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.provenance import build_error_response, build_meta, build_provenance


async def options_signals(symbol: str) -> dict[str, Any]:
    """
    Compute options-derived signals for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with implied volatility, put/call ratios, and unusual activity
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()

    try:
        ticker = await fetch_ticker(symbol)
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch ticker: {e}",
            symbol=normalized_symbol,
        )

    # Get current price
    try:
        info = ticker.info
    except Exception:
        info = {}

    current_price = safe_float(info.get("regularMarketPrice")) or safe_float(
        info.get("currentPrice")
    )
    if current_price is None or current_price <= 0:
        return build_error_response(
            error_type="data_unavailable",
            message="Current price unavailable",
            symbol=normalized_symbol,
        )

    # Get expiration dates
    try:
        expirations = ticker.options
    except Exception:
        expirations = ()

    if not expirations:
        return build_error_response(
            error_type="no_options",
            message=f"No options available for {normalized_symbol}",
            symbol=normalized_symbol,
        )

    # Pick nearest monthly expiration: >14 days out, <60 days
    expiration = _pick_expiration(expirations)
    if expiration is None:
        return build_error_response(
            error_type="no_suitable_expiration",
            message="No expiration between 14 and 60 days out",
            symbol=normalized_symbol,
        )

    # Fetch option chain
    try:
        chain = ticker.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch option chain: {e}",
            symbol=normalized_symbol,
        )

    if calls.empty and puts.empty:
        return build_error_response(
            error_type="data_unavailable",
            message="Option chain returned empty",
            symbol=normalized_symbol,
        )

    warnings: list[str] = []

    # Implied volatility
    iv_data = _compute_iv(calls, puts, current_price, info, warnings)

    # Put/call ratio
    pc_data = _compute_put_call_ratio(calls, puts, warnings)

    # Unusual activity
    unusual_data = _compute_unusual_activity(calls, puts)

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("options_signals", duration_ms),
        "data_provenance": {
            "options": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "symbol": normalized_symbol,
        "expiration_used": expiration,
        "implied_volatility": iv_data,
        "put_call_ratio": pc_data,
        "unusual_activity": unusual_data,
        "warnings": warnings or None,
    }


def _pick_expiration(expirations: tuple[str, ...] | list[str]) -> str | None:
    """Pick the nearest expiration between 14 and 60 days out."""
    today = datetime.now().date()
    min_date = today + timedelta(days=14)
    max_date = today + timedelta(days=60)

    best: str | None = None
    best_dt = None

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if min_date <= exp_date <= max_date:
            if best_dt is None or exp_date < best_dt:
                best = exp_str
                best_dt = exp_date

    return best


def _compute_iv(
    calls: Any,
    puts: Any,
    current_price: float,
    info: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Compute ATM implied volatility."""
    lower = current_price * 0.95
    upper = current_price * 1.05

    iv_values: list[float] = []
    for df in (calls, puts):
        if df.empty:
            continue
        atm_rows = df[df["strike"].between(lower, upper)]
        for iv in atm_rows.get("impliedVolatility", []):
            val = safe_float(iv)
            if val is not None and val > 0:
                iv_values.append(val)

    atm_avg_iv = safe_round(sum(iv_values) / len(iv_values), 4) if iv_values else None

    if atm_avg_iv is None:
        iv_rank_note = None
    elif atm_avg_iv > 0.50:
        iv_rank_note = "elevated"
    elif atm_avg_iv < 0.15:
        iv_rank_note = "depressed"
    else:
        iv_rank_note = "normal"

    return {
        "atm_avg_iv": atm_avg_iv,
        "iv_rank_note": iv_rank_note,
    }


def _compute_put_call_ratio(
    calls: Any,
    puts: Any,
    warnings: list[str],
) -> dict[str, Any]:
    """Compute volume-based and OI-based put/call ratios."""
    call_volume = safe_float(calls["volume"].sum()) if "volume" in calls.columns else None
    put_volume = safe_float(puts["volume"].sum()) if "volume" in puts.columns else None
    call_oi = (
        safe_float(calls["openInterest"].sum())
        if "openInterest" in calls.columns
        else None
    )
    put_oi = (
        safe_float(puts["openInterest"].sum())
        if "openInterest" in puts.columns
        else None
    )

    volume_ratio: float | None = None
    if call_volume and put_volume is not None and call_volume > 0:
        volume_ratio = safe_round(put_volume / call_volume, 2)

    oi_ratio: float | None = None
    if call_oi and put_oi is not None and call_oi > 0:
        oi_ratio = safe_round(put_oi / call_oi, 2)

    # Determine signal from whichever ratio is available (prefer volume)
    ratio_for_signal = volume_ratio if volume_ratio is not None else oi_ratio
    signal: str | None = None
    if ratio_for_signal is not None:
        if ratio_for_signal < 0.7:
            signal = "bullish"
        elif ratio_for_signal > 1.0:
            signal = "bearish"
        else:
            signal = "neutral"

    if volume_ratio is None and oi_ratio is None:
        warnings.append("put_call_ratio_unavailable")

    return {
        "volume_based": volume_ratio,
        "oi_based": oi_ratio,
        "signal": signal,
    }


def _compute_unusual_activity(calls: Any, puts: Any) -> dict[str, Any]:
    """Flag strikes where volume > 3x open interest."""
    unusual: list[dict[str, Any]] = []

    for df, option_type in ((calls, "call"), (puts, "put")):
        if df.empty:
            continue
        if "volume" not in df.columns or "openInterest" not in df.columns:
            continue

        for _, row in df.iterrows():
            vol = safe_float(row.get("volume"))
            oi = safe_float(row.get("openInterest"))
            strike = safe_float(row.get("strike"))

            if vol is None or oi is None or strike is None:
                continue
            if oi <= 0:
                continue
            ratio = vol / oi
            if ratio > 3.0:
                unusual.append(
                    {
                        "strike": strike,
                        "type": option_type,
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "ratio": safe_round(ratio, 1),
                    }
                )

    # Sort by ratio descending, take top 5
    unusual.sort(key=lambda x: x["ratio"], reverse=True)
    top = unusual[:5]

    return {
        "count": len(unusual),
        "top_strikes": top,
    }
