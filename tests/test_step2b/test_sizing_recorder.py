"""Current sizing evidence cannot overwrite or masquerade as the frozen baseline."""

import json
import sys
from unittest.mock import AsyncMock

import pytest

from tests.support import sizing_matrix as sm


@pytest.mark.parametrize("mode", [[], ["--baseline"]])
def test_cli_refuses_baseline_before_building_or_writing(monkeypatch, tmp_path, capsys, mode):
    output = tmp_path / "recording"
    build = AsyncMock()
    monkeypatch.setattr(sm, "build", build)
    monkeypatch.setattr(sys, "argv", ["sizing_matrix", *mode, "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        sm.main()
    assert error.value.code == 2
    assert "baseline is frozen; use --after" in capsys.readouterr().err
    build.assert_not_awaited()
    assert not output.exists()


async def test_current_recording_has_baseline_reference_without_claiming_recorded_head(tmp_path):
    recording = await sm.build()
    assert recording["baseline_code"] == sm.BASE_COMMIT
    assert "recorded_code" not in recording
    assert "current checkout" in recording["notice"]
    assert recording == await sm.build()
    with pytest.raises(ValueError, match="use after=True"):
        sm.write_recording(recording, tmp_path)
    assert list(tmp_path.iterdir()) == []
    sm.write_recording(recording, tmp_path, after=True)
    assert {path.name for path in tmp_path.iterdir()} == {"after.json", "after.md"}
    assert json.loads((tmp_path / "after.json").read_text()) == recording
    text = (tmp_path / "after.md").read_text()
    assert "current output table" in text
    assert "after.json" in text
    assert "## Availability" in text
    assert "below_one_whole_share" in text
    assert "stop_price_missing" in text
    assert "not_allowed" in text
    assert "sizing falls back to 1%" not in text
    assert "v0.3.0 before table" not in text
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    sm.write_recording(recording, tmp_path, after=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_after_cli_writes_only_after_files(monkeypatch, tmp_path):
    recording = {"baseline_code": sm.BASE_COMMIT, "clock": "synthetic", "notice": "current checkout"}
    monkeypatch.setattr(sm, "build", AsyncMock(return_value=recording))
    monkeypatch.setattr(sm, "markdown", lambda data: "current output table\n")
    monkeypatch.setattr(sys, "argv", ["sizing_matrix", "--after", "--output-dir", str(tmp_path)])
    sm.main()
    assert {path.name for path in tmp_path.iterdir()} == {"after.json", "after.md"}
    assert json.loads((tmp_path / "after.json").read_text()) == recording


def test_after_writer_does_not_relabel_the_frozen_baseline(tmp_path):
    before = json.loads((sm.FIXTURE_DIR / "before.json").read_text())
    with pytest.raises(ValueError, match="baseline cannot be written as after"):
        sm.write_recording(before, tmp_path, after=True)
    assert list(tmp_path.iterdir()) == []


def test_after_writer_preserves_before_files_and_preflights_conflicts(monkeypatch, tmp_path):
    recording = {"baseline_code": sm.BASE_COMMIT}
    monkeypatch.setattr(sm, "markdown", lambda data: "current output table\n")
    for name in ("before.json", "before.md"):
        (tmp_path / name).write_bytes((sm.FIXTURE_DIR / name).read_bytes())
    (tmp_path / "after.md").write_text("different prior evidence")
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="refusing to overwrite after recording"):
        sm.write_recording(recording, tmp_path, after=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
    assert not (tmp_path / "after.json").exists()
