"""Watchlist tool tests: manage validation, scan phases, transitions."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import pytz

import stock_analysis.tools.watchlist as wl_mod
from stock_analysis.utils.watch_store import load_scan_state, save_scan_state

ET = pytz.timezone("America/New_York")
NOW = ET.localize(datetime(2026, 3, 10, 8, 0))  # Tuesday pre_market
TODAY = date(2026, 3, 10)


def daily_df() -> pd.DataFrame:
    closes = [90.0 + i * 0.2 for i in range(60)]
    dates = pd.date_range("2025-12-15", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
    dates[-1] = "2026-03-09"
    return pd.DataFrame({"date": dates, "open": closes,
                         "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
                         "close": closes, "volume": [1_000_000.0] * 60})


class TestManage:
    @pytest.mark.asyncio
    async def test_add_list_remove_round_trip(self, tmp_path: Path) -> None:
        r = await wl_mod.manage_watchlist("add", ["hood", "HOOD", " snow "],
                                          _today=TODAY, data_dir=tmp_path)
        assert r["watchlist"] == ["HOOD", "SNOW"]
        r = await wl_mod.manage_watchlist("list", data_dir=tmp_path)
        assert r["count"] == 2
        r = await wl_mod.manage_watchlist("remove", ["SNOW", "NOPE"], data_dir=tmp_path)
        assert r["watchlist"] == ["HOOD"]
        assert any("NOPE" in w["reason"] for w in r["warnings"])

    @pytest.mark.asyncio
    async def test_bad_action_rejected(self, tmp_path: Path) -> None:
        r = await wl_mod.manage_watchlist("purge", data_dir=tmp_path)
        assert r["error"] is True and r["error_type"] == "invalid_parameters"

    @pytest.mark.asyncio
    async def test_empty_add_rejected(self, tmp_path: Path) -> None:
        r = await wl_mod.manage_watchlist("add", [], data_dir=tmp_path)
        assert r["error_type"] == "invalid_parameters"

    @pytest.mark.asyncio
    async def test_cap_enforced_post_dedupe_nothing_written(self, tmp_path: Path) -> None:
        syms = [f"S{i:02d}" for i in range(24)]
        await wl_mod.manage_watchlist("add", syms, _today=TODAY, data_dir=tmp_path)
        r = await wl_mod.manage_watchlist("add", ["S00", "A1", "A2"],
                                          _today=TODAY, data_dir=tmp_path)
        assert r["error_type"] == "invalid_parameters"  # 24 + 2 new = 26 > 25
        r = await wl_mod.manage_watchlist("list", data_dir=tmp_path)
        assert r["count"] == 24  # unchanged

    @pytest.mark.asyncio
    async def test_remove_drops_scan_state(self, tmp_path: Path) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=tmp_path)
        save_scan_state(tmp_path, {"scanned_at": "x", "symbols": {"HOOD": {"action": "no_setup"}}})
        await wl_mod.manage_watchlist("remove", ["HOOD"], data_dir=tmp_path)
        state, _ = load_scan_state(tmp_path)
        assert "HOOD" not in state.get("symbols", {})

    @pytest.mark.asyncio
    async def test_remove_survives_malformed_scan_state(self, tmp_path: Path) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=tmp_path)
        save_scan_state(tmp_path, {"scanned_at": "x", "symbols": [1, 2]})
        r = await wl_mod.manage_watchlist("remove", ["HOOD"], data_dir=tmp_path)
        assert r.get("error") is None
        assert r["watchlist"] == []


def screen_result(**over: Any) -> dict[str, Any]:
    base = {"promote": False, "action_hint": "no_setup", "setup_type": None,
            "trigger_price": None, "last_close": 100.0, "blocker_ids": []}
    base.update(over)
    return base


def card(action: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "action": action,
        "setup": {"type": "breakout", "trigger_satisfied": False,
                  "trigger_price": 102.0} if action in
                 ("trade_now", "enter_on_trigger", "watch") else None,
        "blockers": [], "event_risk": {"earnings_in_days": 40},
    }
    base.update(over)
    return base


@pytest.fixture
def scan_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def fake_history(params: Any) -> pd.DataFrame:
        return daily_df()

    monkeypatch.setattr(wl_mod, "fetch_history", fake_history)
    return tmp_path


class TestScan:
    @pytest.mark.asyncio
    async def test_first_scan_promote_flow(self, scan_env: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD", "SNOW"], _today=TODAY,
                                      data_dir=scan_env)
        monkeypatch.setattr(wl_mod, "screen_symbol",
                            lambda df: screen_result(promote=True, action_hint="candidate",
                                                     setup_type="breakout",
                                                     trigger_price=102.0))

        async def fake_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return card("enter_on_trigger")

        monkeypatch.setattr(wl_mod, "analyze_trade_setup", fake_card)
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert r["symbols_scanned"] == 2 and r["full_cards"] == 2
        assert r["changes"] == []  # first scan
        assert any(w["id"] == "first_scan" for w in r["warnings"])
        # A clean first scan must NOT look corrupted (Codex P2 regression).
        assert not any(w["id"] == "state_unreadable" for w in r["warnings"])
        state, _ = load_scan_state(scan_env)
        assert state["symbols"]["HOOD"]["action"] == "enter_on_trigger"

    @pytest.mark.asyncio
    async def test_transition_reported_on_upgrade(self, scan_env: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=scan_env)
        save_scan_state(scan_env, {"scanned_at": "x", "symbols": {
            "HOOD": {"action": "no_setup", "setup_type": None,
                     "trigger_satisfied": False, "blockers": [],
                     "trigger_price": None}}})
        monkeypatch.setattr(wl_mod, "screen_symbol",
                            lambda df: screen_result(promote=True, setup_type="breakout",
                                                     trigger_price=102.0,
                                                     action_hint="candidate"))

        async def fake_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return card("enter_on_trigger")

        monkeypatch.setattr(wl_mod, "analyze_trade_setup", fake_card)
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert len(r["changes"]) == 1
        assert r["changes"][0]["from"] == "no_setup"
        assert r["changes"][0]["to"] == "enter_on_trigger"

    @pytest.mark.asyncio
    async def test_previously_actionable_forced_to_phase2(
        self, scan_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=scan_env)
        save_scan_state(scan_env, {"scanned_at": "x", "symbols": {
            "HOOD": {"action": "enter_on_trigger", "setup_type": "breakout",
                     "trigger_satisfied": False, "blockers": [],
                     "trigger_price": 102.0}}})
        monkeypatch.setattr(wl_mod, "screen_symbol",
                            lambda df: screen_result())  # screen says no_setup now
        calls: list[str] = []

        async def fake_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(symbol)
            return card("no_setup")

        monkeypatch.setattr(wl_mod, "analyze_trade_setup", fake_card)
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert calls == ["HOOD"]  # downgrade came from the CARD, not the screen
        assert r["changes"][0]["to"] == "no_setup"

    @pytest.mark.asyncio
    async def test_error_isolates_and_preserves_state(
        self, scan_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await wl_mod.manage_watchlist("add", ["GOOD", "BAD"], _today=TODAY,
                                      data_dir=scan_env)
        prior = {"action": "enter_on_trigger", "setup_type": "breakout",
                 "trigger_satisfied": False, "blockers": [], "trigger_price": 102.0}
        save_scan_state(scan_env, {"scanned_at": "x", "symbols": {"BAD": prior}})

        async def flaky_history(params: Any) -> pd.DataFrame:
            if params.symbol == "BAD":
                raise ValueError("no data")
            return daily_df()

        monkeypatch.setattr(wl_mod, "fetch_history", flaky_history)
        monkeypatch.setattr(wl_mod, "screen_symbol", lambda df: screen_result())
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert [e["symbol"] for e in r["errors"]] == ["BAD"]
        state, _ = load_scan_state(scan_env)
        assert state["symbols"]["BAD"] == prior  # untouched on failure

    @pytest.mark.asyncio
    async def test_sizing_bounds_rejected_before_disk_or_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def explode(params: Any) -> pd.DataFrame:
            raise AssertionError("must not fetch")

        monkeypatch.setattr(wl_mod, "fetch_history", explode)
        r = await wl_mod.scan_watchlist(account_size=-1.0, _now=NOW, data_dir=tmp_path)
        assert r["error_type"] == "invalid_parameters"

    @pytest.mark.asyncio
    async def test_empty_watchlist_is_clean_noop(self, scan_env: Path) -> None:
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert r["symbols_scanned"] == 0 and r["rows"] == [] and r.get("error") is None

    @pytest.mark.asyncio
    async def test_malformed_symbols_list_treated_as_empty(
        self, scan_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=scan_env)
        save_scan_state(scan_env, {"scanned_at": "x", "symbols": [1, 2]})
        monkeypatch.setattr(wl_mod, "screen_symbol", lambda df: screen_result())
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert r.get("error") is None
        assert any(w["id"] == "state_unreadable" for w in r["warnings"])
        assert r["changes"] == []

    @pytest.mark.asyncio
    async def test_malformed_per_symbol_prior_treated_as_no_prior(
        self, scan_env: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=scan_env)
        save_scan_state(scan_env, {"scanned_at": "x", "symbols": {"HOOD": "garbage"}})
        monkeypatch.setattr(wl_mod, "screen_symbol", lambda df: screen_result())
        r = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert r.get("error") is None
        assert r["changes"] == []  # no prior => no transition reported

    @pytest.mark.asyncio
    async def test_phase2_row_carries_expected_move(self, scan_env: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        await wl_mod.manage_watchlist("add", ["HOOD"], _today=TODAY, data_dir=scan_env)
        monkeypatch.setattr(wl_mod, "screen_symbol",
                            lambda df: screen_result(promote=True, action_hint="candidate",
                                                     setup_type="breakout",
                                                     trigger_price=102.0))

        async def fake_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
            c = card("enter_on_trigger")
            c["event_risk"] = {"earnings_in_days": 10, "expected_move_pct": 0.065}
            return c

        monkeypatch.setattr(wl_mod, "analyze_trade_setup", fake_card)
        first = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert first["rows"][0]["expected_move_pct"] == 0.065
        # EM is display data: an EM change alone must NOT create a transition.
        async def fake_card_2(symbol: str, **kwargs: Any) -> dict[str, Any]:
            c = card("enter_on_trigger")
            c["event_risk"] = {"earnings_in_days": 10, "expected_move_pct": 0.09}
            return c

        monkeypatch.setattr(wl_mod, "analyze_trade_setup", fake_card_2)
        second = await wl_mod.scan_watchlist(_now=NOW, data_dir=scan_env)
        assert second["changes"] == []
