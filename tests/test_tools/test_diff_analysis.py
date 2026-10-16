"""Tests for what_changed / detect_changes diff behavior.

Focused on the gap caught in review: the watchlist snapshot exposes
the decision (`new_money_action` since snapshot 3.0.0) and `hard_gate_ids`,
and detect_changes has to surface changes in both.
"""

import pytest

from stock_analysis.tools.diff_analysis import _diff_hard_gates, what_changed
from stock_analysis.utils.normalize import SNAPSHOT_FIELDS


def _snapshot(
    *,
    new_money_action: str | None = "buy",
    hard_gate_ids: list[str] | None = None,
    tilt: str | None = "bullish",
    price: float | None = 100.0,
) -> dict:
    """A complete 3.0.0 watchlist snapshot (every other field a known null)."""
    return {
        **dict.fromkeys(SNAPSHOT_FIELDS["3.0.0"]),
        "snapshot_version": "3.0.0",
        "symbol": "TEST",
        "price": price,
        "tilt": tilt,
        "new_money_action": new_money_action,
        "hard_gate_ids": hard_gate_ids or [],
    }


class TestDiffHardGates:
    """The helper reports new vs cleared gate ids."""

    def test_no_gates_no_change(self) -> None:
        diff = _diff_hard_gates(_snapshot(), _snapshot())
        assert diff["new"] == []
        assert diff["cleared"] == []
        assert diff["changed"] is False

    def test_new_gate_fires(self) -> None:
        prev = _snapshot(hard_gate_ids=[])
        curr = _snapshot(hard_gate_ids=["earnings_blackout"])
        diff = _diff_hard_gates(prev, curr)
        assert diff["new"] == ["earnings_blackout"]
        assert diff["cleared"] == []
        assert diff["changed"] is True

    def test_gate_cleared(self) -> None:
        prev = _snapshot(hard_gate_ids=["falling_knife"])
        curr = _snapshot(hard_gate_ids=[])
        diff = _diff_hard_gates(prev, curr)
        assert diff["new"] == []
        assert diff["cleared"] == ["falling_knife"]

    def test_both_new_and_cleared(self) -> None:
        prev = _snapshot(hard_gate_ids=["earnings_blackout"])
        curr = _snapshot(hard_gate_ids=["falling_knife"])
        diff = _diff_hard_gates(prev, curr)
        assert diff["new"] == ["falling_knife"]
        assert diff["cleared"] == ["earnings_blackout"]


class TestWhatChangedDecisionFields:
    """End-to-end: detect_changes surfaces decision_card field shifts."""

    @pytest.mark.asyncio
    async def test_new_money_action_change_appears_in_material_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub analyze_stock so we don't hit the real pipeline; control the snapshot directly.
        import stock_analysis.tools.diff_analysis as diff_module

        captured: dict = {}

        async def fake_analyze(*args, **kwargs) -> dict:
            return captured["result"]

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)

        # detect_changes uses analyze's own watchlist_snapshot (WP7).
        captured["result"] = {
            "symbol": "TEST",
            "watchlist_snapshot": _snapshot(new_money_action="wait_for_data", hard_gate_ids=["earnings_blackout"]),
        }

        previous = _snapshot(new_money_action="buy", hard_gate_ids=[])

        response = await what_changed("TEST", previous_snapshot=previous)

        assert response["has_previous"] is True
        # new_money_action diff must be present
        assert response["changes"]["new_money_action"]["previous"] == "buy"
        assert response["changes"]["new_money_action"]["current"] == "wait_for_data"
        assert response["changes"]["new_money_action"]["changed"] is True
        # hard_gates diff must show the new gate
        assert "earnings_blackout" in response["changes"]["hard_gates"]["new"]
        # material_changes must include the action and gate lines
        assert "new_money_action: buy -> wait_for_data" in response["material_changes"]
        assert any("earnings_blackout" in line for line in response["material_changes"])
        # summary must mention the action shift
        assert "New money: buy -> wait_for_data." in response["summary"]

    @pytest.mark.asyncio
    async def test_gate_cleared_appears_in_material_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import stock_analysis.tools.diff_analysis as diff_module

        captured: dict = {}

        async def fake_analyze(*args, **kwargs) -> dict:
            return captured["result"]

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)
        captured["result"] = {"symbol": "TEST", "watchlist_snapshot": _snapshot(new_money_action="buy", hard_gate_ids=[])}

        previous = _snapshot(new_money_action="wait_for_data", hard_gate_ids=["falling_knife"])

        response = await what_changed("TEST", previous_snapshot=previous)

        assert "falling_knife" in response["changes"]["hard_gates"]["cleared"]
        assert any(
            "falling_knife" in line and "-hard_gate" in line
            for line in response["material_changes"]
        )
        assert "Cleared hard gate" in response["summary"]
