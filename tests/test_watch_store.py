"""Tests for the local watchlist/scan-state store."""

from pathlib import Path

import pytest

from stock_analysis.utils.watch_store import (
    DATA_DIR_ENV,
    MAX_WATCHLIST,
    load_scan_state,
    load_watchlist,
    resolve_data_dir,
    save_scan_state,
    save_watchlist,
)


class TestResolveDataDir:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "custom"))
        assert resolve_data_dir() == tmp_path / "custom"

    def test_xdg_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv(DATA_DIR_ENV, raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        assert resolve_data_dir() == tmp_path / "xdg" / "stock-analysis"

    def test_home_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv(DATA_DIR_ENV, raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / ".local" / "share" / "stock-analysis"


class TestWatchlistRoundTrip:
    def test_missing_file_is_empty_no_warning(self, tmp_path: Path) -> None:
        symbols, warnings = load_watchlist(tmp_path)
        assert symbols == {} and warnings == []

    def test_save_creates_dir_and_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "nested"
        save_watchlist(target, {"HOOD": {"added": "2026-07-06"}})
        symbols, warnings = load_watchlist(target)
        assert symbols == {"HOOD": {"added": "2026-07-06"}} and warnings == []

    def test_corrupt_file_degrades_with_warning(self, tmp_path: Path) -> None:
        (tmp_path / "watchlist.json").write_text("{not json")
        symbols, warnings = load_watchlist(tmp_path)
        assert symbols == {}
        assert warnings and warnings[0]["id"] == "state_unreadable"

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_path: Path) -> None:
        save_watchlist(tmp_path, {"A": {"added": "2026-07-06"}})
        leftovers = [p for p in tmp_path.iterdir() if p.suffix != ".json"]
        assert leftovers == []


class TestScanStateRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        state = {"scanned_at": "2026-07-06T09:00:00-04:00",
                 "symbols": {"HOOD": {"action": "no_setup", "setup_type": None,
                                      "trigger_satisfied": False, "blockers": [],
                                      "trigger_price": None}}}
        save_scan_state(tmp_path, state)
        loaded, warnings = load_scan_state(tmp_path)
        assert loaded == state and warnings == []

    def test_corrupt_scan_state_degrades(self, tmp_path: Path) -> None:
        (tmp_path / "scan_state.json").write_text("]]")
        loaded, warnings = load_scan_state(tmp_path)
        assert loaded == {}
        assert warnings[0]["id"] == "state_unreadable"


def test_cap_constant() -> None:
    assert MAX_WATCHLIST == 25
