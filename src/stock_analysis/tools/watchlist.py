"""Watchlist tools: manage the persisted list and run the two-phase scan."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytz

from stock_analysis.data.cache_manager import classify_session
from stock_analysis.data.yfinance_client import fetch_history
from stock_analysis.tools.trade_setup.card import SWING_HORIZON
from stock_analysis.tools.trade_setup.orchestrator import analyze_trade_setup
from stock_analysis.utils.freshness import FRESHNESS_GATE_IDS, freshness_leg
from stock_analysis.utils.position_sizing import LimitOrigin
from stock_analysis.utils.provenance import build_error_response
from stock_analysis.utils.scan_screen import DAILY_LEG, screen_symbol
from stock_analysis.utils.sizing_inputs import SwingSizingDefault, normalize_sizing_params
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
# Hand-bumped when stage-1, row or diff rules change; a state without it predates WP2.1.
SCAN_RULES_VERSION = 2


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
    account_size: float | str | None = None,
    risk_per_trade_pct: float | str = SwingSizingDefault.RISK,
    max_position_pct: float | str = SwingSizingDefault.POSITION,
    _now: datetime | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    limit_origins: dict[str, LimitOrigin] = {
        "risk_per_trade_pct": "default" if isinstance(risk_per_trade_pct, SwingSizingDefault) else "caller",
        "max_position_pct": "default" if isinstance(max_position_pct, SwingSizingDefault) else "caller",
    }
    try:
        account_size, risk_per_trade_pct, max_position_pct = normalize_sizing_params(
            account_size, risk_per_trade_pct, max_position_pct
        )
    except ValueError as exc:
        return build_error_response(
            error_type="invalid_parameters", message=str(exc),
        )

    directory = data_dir or resolve_data_dir()
    now = _now or datetime.now(_ET)
    session = classify_session(now)

    stored, warnings = load_watchlist(directory)
    state, state_warnings = load_scan_state(directory)
    warnings = [*warnings, *state_warnings]
    prior_symbols = state.get("symbols")
    first_scan = "symbols" not in state
    old_rules: list[Any] = []  # the rules_version behind each comparison made under other rules
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
    elif state.get("rules_version") != SCAN_RULES_VERSION:
        old_rules.append(state.get("rules_version"))

    symbols = sorted(stored)
    screens = await asyncio.gather(
        *(_screen_one(symbol, now, session) for symbol in symbols)
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
                _limit_origins=limit_origins,
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
            event_risk = card.get("event_risk") or {}
            row = _row(symbol, summary, screen,
                       event_risk.get("earnings_in_days"),
                       event_risk.get("expected_move_pct"))
        else:
            summary = _summary_from_screen(screen)
            row = _row(symbol, summary, screen, None)

        if prior is not None and "source" not in prior:  # pre-WP2.1, even when kept past a failed scan
            old_rules.append(None)
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

    if old_rules:
        warnings.append({
            "id": "rules_changed",
            "reason": (
                f"scan rules changed since the last scan (rules_version "
                f"{old_rules[0]!r} -> {SCAN_RULES_VERSION}); "
                "changes may reflect the new rules, not the market"
            ),
        })
    save_scan_state(directory, {
        "rules_version": SCAN_RULES_VERSION,
        "scanned_at": now.isoformat(),
        "symbols": new_state_symbols,
    })
    return {
        "scanned_at": now.isoformat(),
        "session": session,
        "rules_version": SCAN_RULES_VERSION,
        "symbols_scanned": len(symbols),
        "full_cards": full_cards,
        "changes": changes,
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
    }


async def _screen_one(symbol: str, now: datetime, session: str) -> dict[str, Any] | None:
    try:
        df = await fetch_history(FetchParams(symbol, "1y", "1d", True))
    except Exception:
        return None
    return screen_symbol(df, now=now, session=session)


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
    freshness = card.get("freshness") or {}
    leg = freshness_leg(freshness)
    unevaluated = {_pair(u["id"], "daily") for u in card.get("unevaluated") or []}
    if freshness.get("session") == "regular" and freshness.get("basis") == "unverifiable":
        unevaluated |= DAILY_LEG  # the quote leg returned before the daily leg was read
    return {
        "action": card.get("action"),
        "setup_type": setup.get("type"),
        "trigger_satisfied": bool(setup.get("trigger_satisfied", False)),
        "blockers": sorted(_pair(b["id"], leg) for b in card.get("blockers", [])),
        "trigger_price": setup.get("trigger_price"),
        "source": "card",
        "unevaluated": sorted(unevaluated),
    }


def _summary_from_screen(screen: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": screen["action_hint"],
        "setup_type": screen["setup_type"],
        "trigger_satisfied": screen["trigger_satisfied"],
        "blockers": sorted(screen["blocker_ids"]),
        "trigger_price": screen["trigger_price"],
        "source": "screen",
        "unevaluated": screen["unevaluated"],
    }


def _pair(gate_id: str, leg: str | None) -> str:
    """A row's (id, leg) entry: the freshness gates carry the leg they fired or went unevaluated on."""
    return f"{gate_id}@{leg}" if gate_id in FRESHNESS_GATE_IDS else gate_id


def _row(
    symbol: str,
    summary: dict[str, Any],
    screen: dict[str, Any],
    earnings_in_days: int | None,
    expected_move_pct: float | None = None,
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
        "unevaluated": summary["unevaluated"],
        "source": summary["source"],
        "horizon": SWING_HORIZON,
        "earnings_in_days": earnings_in_days,
        "expected_move_pct": expected_move_pct,
    }


def _diff(
    prior: dict[str, Any] | None,
    summary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Blockers compare as the (id, leg) pairs neither side left unevaluated. A pre-WP2.1 entry
    (no source) recorded no legs and no unevaluated list: it compares at id level, skipping the
    ids the new row left unevaluated and fired on no leg."""
    if prior is None:
        return False, []
    after_all = set(summary["blockers"])
    if "source" in prior:
        skip = _strings(prior.get("unevaluated")) | set(summary["unevaluated"])
        before, after = _strings(prior.get("blockers")) - skip, after_all - skip
    else:
        after_ids = {_gate_id(entry) for entry in after_all}
        skip = {_gate_id(entry) for entry in summary["unevaluated"]} - after_ids
        before = {_gate_id(entry) for entry in _strings(prior.get("blockers"))} - skip
        after = after_ids - skip
    notes = [f"{field}: {prior.get(field)} -> {summary.get(field)}"
             for field in ("action", "setup_type", "trigger_satisfied")
             if prior.get(field) != summary.get(field)]
    if before != after:
        notes.append(f"blockers: {sorted(before)} -> {sorted(after)}")
    if prior.get("trigger_price") != summary.get("trigger_price"):
        notes.append(f"trigger_price: {prior.get('trigger_price')} -> {summary.get('trigger_price')}")
    return bool(notes), notes


def _strings(value: Any) -> set[str]:
    return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()


def _gate_id(entry: str) -> str:
    return entry.split("@", 1)[0]
