"""Tests for what_changed / detect_changes diff behavior.

Focused on the gap caught in review: the watchlist snapshot exposes
`decision_action_now` and `hard_gate_ids`, but the diff helpers were
not reading them. detect_changes had to actually surface those.
"""

import pytest

from stock_analysis.tools.diff_analysis import _diff_hard_gates, what_changed


def _snapshot(
    *,
    action_now: str | None = "buy",
    hard_gate_ids: list[str] | None = None,
    tilt: str | None = "bullish",
    price: float | None = 100.0,
) -> dict:
    """Minimal watchlist-snapshot shape used by the diff functions."""
    return {
        "snapshot_version": "1.1.0",
        "symbol": "TEST",
        "price": price,
        "tilt": tilt,
        "decision_action_now": action_now,
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
    async def test_action_now_change_appears_in_material_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub analyze_stock so we don't hit the real pipeline; control the snapshot directly.
        import stock_analysis.tools.diff_analysis as diff_module

        captured: dict = {}

        async def fake_analyze(*args, **kwargs) -> dict:
            return captured["result"]

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)

        # Build a result whose extracted snapshot has action_now=wait_for_data
        def fake_extract(_result: dict) -> dict:
            return _snapshot(
                action_now="wait_for_data",
                hard_gate_ids=["earnings_blackout"],
            )

        monkeypatch.setattr(diff_module, "_extract_snapshot", fake_extract)
        captured["result"] = {"symbol": "TEST"}

        previous = _snapshot(action_now="buy", hard_gate_ids=[])

        response = await what_changed("TEST", previous_snapshot=previous)

        assert response["has_previous"] is True
        # decision_action_now diff must be present
        assert response["changes"]["decision_action_now"]["previous"] == "buy"
        assert response["changes"]["decision_action_now"]["current"] == "wait_for_data"
        assert response["changes"]["decision_action_now"]["changed"] is True
        # hard_gates diff must show the new gate
        assert "earnings_blackout" in response["changes"]["hard_gates"]["new"]
        # material_changes must include the action and gate lines
        assert any("action_now" in line for line in response["material_changes"])
        assert any("earnings_blackout" in line for line in response["material_changes"])
        # summary must mention the action shift
        assert "wait_for_data" in response["summary"]

    @pytest.mark.asyncio
    async def test_gate_cleared_appears_in_material_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import stock_analysis.tools.diff_analysis as diff_module

        captured: dict = {}

        async def fake_analyze(*args, **kwargs) -> dict:
            return captured["result"]

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)
        monkeypatch.setattr(
            diff_module,
            "_extract_snapshot",
            lambda r: _snapshot(action_now="buy", hard_gate_ids=[]),
        )
        captured["result"] = {"symbol": "TEST"}

        previous = _snapshot(action_now="wait_for_data", hard_gate_ids=["falling_knife"])

        response = await what_changed("TEST", previous_snapshot=previous)

        assert "falling_knife" in response["changes"]["hard_gates"]["cleared"]
        assert any(
            "falling_knife" in line and "-hard_gate" in line
            for line in response["material_changes"]
        )
        assert "Cleared hard gate" in response["summary"]
