"""Watchlist tools: manage the persisted list and run the two-phase scan."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytz

from stock_analysis.data.cache_manager import classify_session
from stock_analysis.data.yfinance_client import fetch_history
from stock_analysis.tools.trade_setup.orchestrator import (
    analyze_trade_setup,
    validate_sizing_params,
)
from stock_analysis.utils.provenance import build_error_response
from stock_analysis.utils.scan_screen import screen_symbol
from stock_analysis.utils.validators import FetchParams
from stock_analysis.utils.watch_store import (
    MAX_WATCHLIST,
    load_scan_state,
    load_watchlist,
    resolve_data_dir,
    save_scan_state,
    save_watchlist,
)

_ET = pytz.timezone("America/New_York")
_ACTIONABLE = frozenset({"trade_now", "enter_on_trigger"})
_ACTIONS = frozenset({"add", "remove", "list"})


async def manage_watchlist(
    action: str,
    symbols: list[str] | None = None,
    _today: date | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    if action not in _ACTIONS:
        return build_error_response(
            error_type="invalid_parameters",
            message=f"action must be one of {sorted(_ACTIONS)}, got '{action}'",
        )
    directory = data_dir or resolve_data_dir()
    stored, warnings = load_watchlist(directory)
    requested = _normalize(symbols)

    if action == "list":
        return _watchlist_response(stored, warnings)

    if not requested:
        return build_error_response(
            error_type="invalid_parameters",
            message=f"'{action}' requires at least one symbol",
        )

    if action == "add":
        today = (_today or datetime.now(_ET).date()).isoformat()
        merged = dict(stored)
        for symbol in requested:
            merged.setdefault(symbol, {"added": today})
        if len(merged) > MAX_WATCHLIST:
            return build_error_response(
                error_type="invalid_parameters",
                message=(
                    f"watchlist cap is {MAX_WATCHLIST} symbols; "
                    f"add would make {len(merged)}"
                ),
            )
        save_watchlist(directory, merged)
        return _watchlist_response(merged, warnings)

    # remove
    kept = {s: meta for s, meta in stored.items() if s not in requested}
    unknown = [s for s in requested if s not in stored]
    if unknown:
        warnings = [*warnings, {
            "id": "unknown_symbols",
            "reason": f"not on the watchlist: {', '.join(unknown)}",
        }]
    save_watchlist(directory, kept)
    state, _ = load_scan_state(directory)
    state_symbols = state.get("symbols")
    if isinstance(state_symbols, dict) and state_symbols:
        state["symbols"] = {s: v for s, v in state_symbols.items() if s in kept}
        save_scan_state(directory, state)
    return _watchlist_response(kept, warnings)


async def scan_watchlist(
    account_size: float | None = None,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
    _now: datetime | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    param_error = validate_sizing_params(
        account_size, risk_per_trade_pct, max_position_pct
    )
    if param_error is not None:
        return build_error_response(
            error_type="invalid_parameters", message=param_error,
        )

    directory = data_dir or resolve_data_dir()
    now = _now or datetime.now(_ET)
    session = classify_session(now)

    stored, warnings = load_watchlist(directory)
    state, state_warnings = load_scan_state(directory)
    warnings = [*warnings, *state_warnings]
    prior_symbols = state.get("symbols")
    first_scan = "symbols" not in state
    if first_scan:
        prior_symbols = {}
        warnings.append({
            "id": "first_scan",
            "reason": "no prior scan state — transitions start next scan",
        })
    elif not isinstance(prior_symbols, dict):
        # Key present but wrong shape: damaged state, not a clean install.
        prior_symbols = {}
        warnings.append({
            "id": "state_unreadable",
            "reason": "scan state malformed — treating as empty",
        })

    symbols = sorted(stored)
    screens = await asyncio.gather(
        *(_screen_one(symbol) for symbol in symbols)
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    changes: list[dict[str, Any]] = []
    new_state_symbols: dict[str, dict[str, Any]] = {}
    full_cards = 0

    for symbol, screen in zip(symbols, screens, strict=True):
        prior = prior_symbols.get(symbol)
        if not isinstance(prior, dict):
            prior = None
        if screen is None:  # daily fetch failed
            errors.append({"symbol": symbol, "error_type": "data_unavailable"})
            if prior is not None:
                new_state_symbols[symbol] = prior  # untouched on failure
            continue

        needs_card = screen["promote"] or (
            prior is not None and prior.get("action") in _ACTIONABLE
        )
        if needs_card:
            card = await analyze_trade_setup(
                symbol,
                account_size=account_size,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
                _now=now,
            )
            if card.get("error"):
                errors.append({
                    "symbol": symbol,
                    "error_type": str(card.get("error_type")),
                })
                if prior is not None:
                    new_state_symbols[symbol] = prior
                continue
            full_cards += 1
            summary = _summary_from_card(card)
            row = _row(symbol, summary, screen,
                       (card.get("event_risk") or {}).get("earnings_in_days"))
        else:
            summary = _summary_from_screen(screen)
            row = _row(symbol, summary, screen, None)

        changed, notes = _diff(prior, summary)
        row["changed"] = changed
        rows.append(row)
        if changed and prior is not None:
            changes.append({
                "symbol": symbol,
                "from": prior.get("action"),
                "to": summary["action"],
                "notes": notes,
            })
        new_state_symbols[symbol] = summary

    save_scan_state(directory, {
        "scanned_at": now.isoformat(),
        "symbols": new_state_symbols,
    })
    return {
        "scanned_at": now.isoformat(),
        "session": session,
        "symbols_scanned": len(symbols),
        "full_cards": full_cards,
        "changes": changes,
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
    }


async def _screen_one(symbol: str) -> dict[str, Any] | None:
    try:
        df = await fetch_history(FetchParams(symbol, "1y", "1d", True))
    except Exception:
        return None
    return screen_symbol(df)


def _normalize(symbols: list[str] | None) -> list[str]:
    seen: list[str] = []
    for raw in symbols or []:
        symbol = raw.upper().strip()
        if symbol and symbol not in seen:
            seen.append(symbol)
    return seen


def _watchlist_response(
    symbols: dict[str, dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    ordered = sorted(symbols)
    return {"watchlist": ordered, "count": len(ordered), "warnings": warnings}


def _summary_from_card(card: dict[str, Any]) -> dict[str, Any]:
    setup = card.get("setup") or {}
    return {
        "action": card.get("action"),
        "setup_type": setup.get("type"),
        "trigger_satisfied": bool(setup.get("trigger_satisfied", False)),
        "blockers": sorted(b["id"] for b in card.get("blockers", [])),
        "trigger_price": setup.get("trigger_price"),
    }


def _summary_from_screen(screen: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": screen["action_hint"],
        "setup_type": screen["setup_type"],
        "trigger_satisfied": False,
        "blockers": sorted(screen["blocker_ids"]),
        "trigger_price": screen["trigger_price"],
    }


def _row(
    symbol: str,
    summary: dict[str, Any],
    screen: dict[str, Any],
    earnings_in_days: int | None,
) -> dict[str, Any]:
    trigger = summary.get("trigger_price")
    last_close = screen.get("last_close")
    distance = (
        round((trigger - last_close) / last_close, 4)
        if trigger and last_close
        else None
    )
    return {
        "symbol": symbol,
        "action": summary["action"],
        "setup_type": summary["setup_type"],
        "trigger_price": summary["trigger_price"],
        "distance_to_trigger_pct": distance,
        "blockers": summary["blockers"],
        "earnings_in_days": earnings_in_days,
    }


def _diff(
    prior: dict[str, Any] | None,
    summary: dict[str, Any],
) -> tuple[bool, list[str]]:
    if prior is None:
        return False, []
    notes: list[str] = []
    for field in ("action", "setup_type", "trigger_satisfied",
                  "blockers", "trigger_price"):
        if prior.get(field) != summary.get(field):
            notes.append(f"{field}: {prior.get(field)} -> {summary.get(field)}")
    return bool(notes), notes
