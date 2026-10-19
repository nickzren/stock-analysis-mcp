"""Current evidence cannot overwrite or impersonate the immutable holder baseline."""

import copy
import json
import sys
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
import requests

from tests.support import holder_matrix as hm


@pytest.mark.parametrize("mode", [[], ["--baseline"]])
def test_baseline_cli_is_disabled_before_any_work(monkeypatch, tmp_path, capsys, mode):
    build = AsyncMock()
    monkeypatch.setattr(hm, "build", build)
    monkeypatch.setattr(sys, "argv", ["holder_matrix", *mode, "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as error:
        hm.main()
    assert error.value.code == 2
    assert "baseline is frozen; use --after" in capsys.readouterr().err
    build.assert_not_awaited()
    assert list(tmp_path.iterdir()) == []


async def test_capture_is_deterministic_and_matches_before_without_claiming_recorded_head(tmp_path):
    recording = await hm.build()
    assert recording == await hm.build()
    before = json.loads((hm.FIXTURE_DIR / "before.json").read_text())
    assert before.pop("recorded_code") == hm.BASE_COMMIT
    previous_fields = copy.deepcopy(recording)
    for row in previous_fields["stocks"].values():
        assert isinstance(row["result"]["decision_card"]["long_term"].pop("if_held"), dict)
    assert previous_fields == {"baseline_code": hm.BASE_COMMIT, **before}
    hm.write_recording(recording, tmp_path)
    assert json.loads((tmp_path / "after.json").read_text()) == recording
    current_table = (tmp_path / "after.md").read_text()
    assert "current output table" in current_table
    assert "Long-term held / basis" in current_table
    for row in recording["stocks"].values():
        holder = row["result"]["decision_card"]["long_term"]["if_held"]
        assert f"{holder['action']} / {holder['basis']}" in current_table
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    hm.write_recording(recording, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_frozen_markdown_renderer_is_unchanged():
    before = json.loads((hm.FIXTURE_DIR / "before.json").read_text())
    before["stocks"] = {name: before["stocks"][name] for name in hm.STOCK_CASES}
    before["positions"] = {name: before["positions"][name] for name in hm.POSITION_CASES}
    assert hm.markdown(before) == (hm.FIXTURE_DIR / "before.md").read_text()


def test_cannot_relabel_before_as_after(tmp_path):
    before = json.loads((hm.FIXTURE_DIR / "before.json").read_text())
    with pytest.raises(ValueError, match="frozen baseline cannot be relabelled"):
        hm.write_recording(before, tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_conflicts_are_preflighted_and_before_files_never_touched(tmp_path):
    recording = await hm.build()
    for name in ("before.json", "before.md"):
        (tmp_path / name).write_bytes((hm.FIXTURE_DIR / name).read_bytes())
    (tmp_path / "after.md").write_text("prior evidence")
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        hm.write_recording(recording, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original
    assert not (tmp_path / "after.json").exists()
    changed = copy.deepcopy(recording)
    changed["positions"]["gain"]["result"]["position"]["current_value"] = float("inf")
    with pytest.raises(ValueError, match="Out of range float"):
        hm.write_recording(changed, tmp_path / "nonfinite")
    assert not (tmp_path / "nonfinite").exists()


async def test_swallowed_network_attempt_still_fails_standalone_generator(monkeypatch):
    real = hm.run_position

    async def accidental_read(name):
        if name == "gain":
            with suppress(RuntimeError):
                requests.get("https://example.invalid/forbidden")
        return await real(name)

    monkeypatch.setattr(hm, "run_position", accidental_read)
    with pytest.raises(AssertionError, match="example.invalid/forbidden"):
        await hm.build()


def test_after_cli_writes_only_current_files(monkeypatch, tmp_path):
    recording = {"baseline_code": hm.BASE_COMMIT}
    monkeypatch.setattr(hm, "build", AsyncMock(return_value=recording))
    monkeypatch.setattr(hm, "markdown", lambda data: "current output table\n")
    monkeypatch.setattr(sys, "argv", ["holder_matrix", "--after", "--output-dir", str(tmp_path)])
    hm.main()
    assert {p.name for p in tmp_path.iterdir()} == {"after.json", "after.md"}


async def test_checked_in_after_matches_current_recorder():
    recording = await hm.build()
    assert recording == json.loads((hm.FIXTURE_DIR / 'after.json').read_text())
    assert hm.markdown(recording) == (hm.FIXTURE_DIR / 'after.md').read_text()
