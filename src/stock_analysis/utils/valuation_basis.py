"""Valuation basis, domain applicability and instrument/currency support (Step 1 WP4).

Ratios stay scalars. A section that carries ratios also carries ``metric_status``:
``{metric: {"basis_status", "scoring_applicable", "reason"}}``. A ratio is null
when it is undefined or its basis can't be verified, and kept raw when it is
defined but can't be used for scoring (e.g. EV/S with EV <= 0). Consumers read
ratios through ``usable_metric``.
"""

import re
from typing import Any

from stock_analysis.utils.helpers import safe_float

VERIFIED = "verified"
PROVIDER = "provider"  # Yahoo-defined, basis undocumented but accepted (PEG)
UNVERIFIED = "unverified"
UNVERIFIED_CURRENCY = "unverified_currency"
UNVERIFIED_SHARE_UNITS = "unverified_share_units"

# Yahoo quotes some exchanges in minor units (pence, cents, agorot).
MINOR_UNIT_CURRENCIES = frozenset({"GBp", "GBX", "ZAc", "ZAC", "ILA"})
SUPPORTED_QUOTE_TYPE = "EQUITY"
SUPPORTED_CURRENCY = "USD"
_SHARE_UNIT_TOLERANCE = 0.10

# Yahoo preferred tickers: BAC-PL, WFC-PY, ABR-PD. Class shares (BRK-B, BF-B) don't match.
_PREFERRED_SUFFIX = re.compile(r"-P[A-Z]?$")
# Preferred-security phrasing only: a common stock named "Preferred Bank" and an
# ordinary ADR ("American Depositary Shares") must not match.
_PREFERRED_NAME = re.compile(
    r"\b(pfd|preferred\s+(stock|shares?|securities)|non-?cumulative)\b", re.IGNORECASE
)


def metric_entry(basis_status: str, scoring_applicable: bool, reason: str | None = None) -> dict[str, Any]:
    return {"basis_status": basis_status, "scoring_applicable": scoring_applicable, "reason": reason}


def currency_basis(price_currency: Any, financial_currency: Any) -> str:
    """``verified`` when a ratio of price data over statement data has one currency."""
    if (
        isinstance(price_currency, str)
        and price_currency
        and price_currency == financial_currency
        and price_currency not in MINOR_UNIT_CURRENCIES
    ):
        return VERIFIED
    return UNVERIFIED_CURRENCY


def share_units_basis(info: dict[str, Any], price: float | None) -> str:
    """``verified`` when ``marketCap ≈ price × sharesOutstanding``, so per-share
    statement figures and the quoted price count the same shares (ADR ratios and
    multiple share classes break this)."""
    market_cap = safe_float(info.get("marketCap"))
    shares = safe_float(info.get("sharesOutstanding"))
    if not (market_cap and shares and price) or market_cap <= 0:
        return UNVERIFIED_SHARE_UNITS
    implied = price * shares
    return VERIFIED if abs(implied - market_cap) / market_cap <= _SHARE_UNIT_TOLERANCE else UNVERIFIED_SHARE_UNITS


def usable_metric(section: Any, key: str) -> float | None:
    """The metric's value when it may be used for gates, pros, prose or rankings.

    A section without ``metric_status`` (a payload from before WP4) is read as-is.
    """
    if not isinstance(section, dict):
        return None
    value = safe_float(section.get(key))
    status = (section.get("metric_status") or {}).get(key)
    if not isinstance(status, dict):
        return value
    return value if status.get("scoring_applicable") else None


def metric_reason(section: Any, key: str) -> str | None:
    """Why a metric is not usable (None when usable, absent or legacy)."""
    if not isinstance(section, dict):
        return None
    status = (section.get("metric_status") or {}).get(key)
    if not isinstance(status, dict) or status.get("scoring_applicable"):
        return None
    reason = status.get("reason") or status.get("basis_status")
    return reason if isinstance(reason, str) else None


def is_unverified(section: Any, key: str) -> bool:
    """The metric exists but its basis couldn't be verified (not undefined, not
    merely inapplicable): consumers must not read it as a missing value."""
    if not isinstance(section, dict):
        return False
    status = (section.get("metric_status") or {}).get(key)
    if not isinstance(status, dict):
        return False
    reason = str(status.get("reason") or "")
    return status.get("basis_status") not in (VERIFIED, PROVIDER) and not reason.startswith("undefined")


def dividend_yields(info: dict[str, Any], price: float | None) -> dict[str, Any]:
    """Forward and trailing dividend yields as fractions (Q3: forward is primary).

    forward = dividendRate / price; trailing = trailingAnnualDividendRate / price,
    which is a statement figure, so it needs the price and financial currencies
    (and share units) to match. ``basis`` is ``forward``, ``no_dividend`` (Yahoo
    reports zero dividends) or None (no dividend data).
    """
    rate = safe_float(info.get("dividendRate"))
    trailing_rate = safe_float(info.get("trailingAnnualDividendRate"))
    reported_yield = safe_float(info.get("dividendYield"))
    has_price = price is not None and price > 0
    # The quote's dividend and price share a currency unless the price is in a
    # minor unit (pence vs pounds) -- or the currency is unknown (fail closed).
    currency = info.get("currency")
    currency_ok = isinstance(currency, str) and bool(currency) and currency not in MINOR_UNIT_CURRENCIES
    price_ok = has_price and currency_ok

    forward: float | None = None
    basis: str | None = None
    if rate is not None and rate > 0:
        if price_ok:
            forward, basis = rate / price, "forward"  # type: ignore[operator]
        elif has_price:
            basis = UNVERIFIED_CURRENCY
    elif rate == 0 or (rate is None and (trailing_rate == 0 or reported_yield == 0)):
        forward, basis = 0.0, "no_dividend"

    trailing: float | None = None
    if trailing_rate is not None and price_ok:
        if trailing_rate == 0:
            trailing = 0.0
        elif (
            currency_basis(currency, info.get("financialCurrency")) == VERIFIED
            and share_units_basis(info, price) == VERIFIED
        ):
            trailing = trailing_rate / price  # type: ignore[operator]
    return {"yield": forward, "basis": basis, "trailing": trailing}


def classify_support(
    *, quote_type: Any, currency: Any, symbol: Any = None, name: Any = None
) -> dict[str, Any]:
    """Instrument and currency support (D5). Unknown values are ``unverified`` and,
    like ``unsupported``, get no sizing, dollar-liquidity claims or new-money action."""
    if not isinstance(quote_type, str) or not quote_type:
        instrument = {"status": "unverified", "quote_type": None, "reason": "quote_type_unknown"}
    elif quote_type.upper() != SUPPORTED_QUOTE_TYPE:
        instrument = {"status": "unsupported", "quote_type": quote_type, "reason": f"{quote_type.lower()}_not_supported"}
    elif _is_preferred(symbol, name):
        instrument = {"status": "unsupported", "quote_type": quote_type, "reason": "preferred_not_supported"}
    else:
        instrument = {"status": "supported", "quote_type": quote_type, "reason": None}

    if not isinstance(currency, str) or not currency:
        currency_block = {"status": "unverified", "code": None, "reason": "currency_unknown"}
    elif currency == SUPPORTED_CURRENCY:
        currency_block = {"status": "supported", "code": currency, "reason": None}
    else:
        reason = "minor_unit_currency_not_supported" if currency in MINOR_UNIT_CURRENCIES else "non_usd_currency_not_supported"
        currency_block = {"status": "unsupported", "code": currency, "reason": reason}
    return {"instrument": instrument, "currency": currency_block}


def summary_support(summary: dict[str, Any], symbol: Any) -> dict[str, Any]:
    """A stock summary's ``support``, else classified from its fields (unknown fails closed)."""
    return summary.get("support") or classify_support(
        quote_type=summary.get("quote_type"),
        currency=summary.get("currency"),
        symbol=symbol,
        name=summary.get("name"),
    )


def _is_preferred(symbol: Any, name: Any) -> bool:
    if isinstance(symbol, str) and _PREFERRED_SUFFIX.search(symbol.upper()):
        return True
    return isinstance(name, str) and bool(_PREFERRED_NAME.search(name))


def instrument_supported(support: Any) -> bool:
    return bool(_status(support, "instrument") == "supported")


def currency_supported(support: Any) -> bool:
    return bool(_status(support, "currency") == "supported")


def sizing_unavailable_reason(support: Any) -> str | None:
    """Why share/dollar sizing and dollar-liquidity checks don't apply, or None."""
    if not instrument_supported(support):
        return "unsupported_instrument"
    if not currency_supported(support):
        return "unsupported_currency"
    return None


def support_blockers(support: Any) -> list[dict[str, str]]:
    """``{id, reason}`` blockers for an unsupported or unverified instrument/currency."""
    blockers: list[dict[str, str]] = []
    if not instrument_supported(support):
        reason = _reason(support, "instrument") or "quote_type_unknown"
        blockers.append({"id": "unsupported_instrument", "reason": f"instrument not supported ({reason}) — no new-money action"})
    if not currency_supported(support):
        reason = _reason(support, "currency") or "currency_unknown"
        blockers.append({
            "id": "unsupported_currency",
            "reason": f"price currency not supported ({reason}) — no sizing, dollar-liquidity or new-money action",
        })
    return blockers


def _status(support: Any, key: str) -> Any:
    block = support.get(key) if isinstance(support, dict) else None
    return block.get("status") if isinstance(block, dict) else None


def _reason(support: Any, key: str) -> str | None:
    block = support.get(key) if isinstance(support, dict) else None
    reason = block.get("reason") if isinstance(block, dict) else None
    return reason if isinstance(reason, str) else None
