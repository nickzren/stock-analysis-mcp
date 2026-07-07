"""Local persistence for the watchlist and scan state.

JSON files under a user data dir; atomic writes; corrupt files degrade to
empty state with a {id: "state_unreadable"} warning — a damaged file must
never fail a scan (spec ruling).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DATA_DIR_ENV = "STOCK_ANALYSIS_DATA_DIR"
MAX_WATCHLIST = 25

_WATCHLIST_FILE = "watchlist.json"
_SCAN_STATE_FILE = "scan_state.json"


def resolve_data_dir() -> Path:
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "stock-analysis"
    return Path(os.environ.get("HOME", str(Path.home()))) / ".local" / "share" / "stock-analysis"


def load_watchlist(data_dir: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    payload, warning = _read_json(data_dir / _WATCHLIST_FILE)
    if warning:
        return {}, [warning]
    symbols = payload.get("symbols", {}) if payload else {}
    return (symbols if isinstance(symbols, dict) else {}), []


def save_watchlist(data_dir: Path, symbols: dict[str, dict[str, str]]) -> None:
    _atomic_write(data_dir / _WATCHLIST_FILE, {"symbols": symbols})


def load_scan_state(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    payload, warning = _read_json(data_dir / _SCAN_STATE_FILE)
    if warning:
        return {}, [warning]
    return (payload if isinstance(payload, dict) else {}), []


def save_scan_state(data_dir: Path, state: dict[str, Any]) -> None:
    _atomic_write(data_dir / _SCAN_STATE_FILE, state)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.exists():
        return {}, None
    try:
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("top-level JSON is not an object")
        return loaded, None
    except (ValueError, OSError) as e:
        return None, {
            "id": "state_unreadable",
            "reason": f"{path.name} unreadable ({e}) — treating as empty",
        }


def _atomic_write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
