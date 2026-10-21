"""Current response evidence is independent of the installed package release."""

import ast
import copy
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from stock_analysis import SCHEMA_VERSION
from stock_analysis.utils import provenance
from tests.support import holder_matrix as hm

TESTS = Path(__file__).resolve().parent
PACKAGE_VERSION = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")


def _literal_server_assertions(source):
    """Find package literals in server-version comparisons, leaving snapshots alone."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assert):
            continue
        for comparison in ast.walk(node.test):
            if not isinstance(comparison, ast.Compare):
                continue
            parts = list(ast.walk(comparison))
            server_field = any(
                (isinstance(part, ast.Subscript) and isinstance(part.slice, ast.Constant)
                 and part.slice.value == "server_version")
                or (isinstance(part, ast.Name) and part.id == "SERVER_VERSION")
                or (isinstance(part, ast.Attribute) and part.attr in {"SERVER_VERSION", "server_version"})
                or (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                    and part.func.attr == "get" and part.args and isinstance(part.args[0], ast.Constant)
                    and part.args[0].value == "server_version")
                or (isinstance(part, ast.Dict) and any(isinstance(key, ast.Constant)
                    and key.value == "server_version" for key in part.keys))
                for part in parts
            )
            if server_field and any(isinstance(part, ast.Constant) and isinstance(part.value, str)
                                    and PACKAGE_VERSION.fullmatch(part.value) for part in parts):
                lines.append(comparison.lineno)
    return lines


def _recorded_server_versions(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            current = (*path, key)
            if key == "server_version" and isinstance(child, str) and PACKAGE_VERSION.fullmatch(child):
                yield current
            yield from _recorded_server_versions(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _recorded_server_versions(child, (*path, index))


def test_server_assertions_use_the_running_package_version():
    # This exact file intentionally pins release text, rather than response metadata.
    excluded = {TESTS / "test_docs.py"}
    violations = [(str(path.relative_to(TESTS)), line)
                  for path in sorted(TESTS.rglob("*.py")) if path not in excluded
                  for line in _literal_server_assertions(path.read_text())]
    assert violations == []


def test_current_json_recordings_do_not_pin_the_package_version():
    violations = [(str(path.relative_to(TESTS)), location)
                  for path in sorted(TESTS.rglob("*.json")) if path.name != "before.json"
                  for location in _recorded_server_versions(json.loads(path.read_text()))]
    assert violations == []


@pytest.mark.parametrize("source", [
    'assert body["meta"]["server_version"] == "123.456.789"',
    'assert "123.456.789" == body["meta"]["server_version"]',
    'assert SERVER_VERSION == "123.456.789"',
    'assert stock_analysis.SERVER_VERSION == "123.456.789"',
    'assert meta.server_version == "123.456.789"',
    'assert meta.get("server_version") == "123.456.789"',
    'assert meta == {"server_version": "123.456.789"}',
])
def test_literal_assertion_guard_detects_server_version_pins(source):
    assert _literal_server_assertions(source) == [1]


@pytest.mark.parametrize("source", [
    'assert body["meta"]["server_version"] == SERVER_VERSION',
    'assert meta["server_version"] == SERVER_VERSION and snapshot["snapshot_version"] == "4.0.0"',
    'assert snapshot["snapshot_version"] == "4.0.0"',
    'assert SNAPSHOT_VERSION == "4.0.0"',
    'assert body["meta"]["schema_version"] == "8"',
    'assert address == "127.0.0.1"',
    '# server_version was previously 123.456.789\nassert value == "123.456.789"',
])
def test_literal_assertion_guard_preserves_other_version_contracts(source):
    assert _literal_server_assertions(source) == []


def test_recording_guard_distinguishes_package_metadata_from_snapshot_versions():
    version = ".".join(map(str, (123, 456, 789)))
    assert list(_recorded_server_versions({"positions": [{"meta": {"server_version": version}}]})) == [
        ("positions", 0, "meta", "server_version")]
    assert list(_recorded_server_versions({"snapshot_version": version, "schema_version": "8"})) == []


async def test_holder_capture_preserves_metadata_except_execution_version_and_timings(monkeypatch):
    response = {"meta": {"server_version": "release-probe", "schema_version": SCHEMA_VERSION,
                         "tool": "position", "duration_ms": 2.5},
                "nested": {"server_version": "preserved"},
                "data_quality": {"tool_timings": {"technicals": 1.0},
                                 "tool_failures": [{"duration_ms": 1.0, "message": "retained"}]}}
    original = copy.deepcopy(response)
    monkeypatch.setattr(hm.position, "analyze_position", AsyncMock(return_value=response))
    captured = (await hm.run_position("gain"))["result"]
    assert captured == {"meta": {"schema_version": SCHEMA_VERSION, "tool": "position"},
                        "nested": {"server_version": "preserved"},
                        "data_quality": {"tool_failures": [{"message": "retained"}]}}
    assert response == original


async def test_current_holder_recording_ignores_a_changed_reported_version(monkeypatch):
    changed_version = ".".join(map(str, (123, 456, 789)))
    monkeypatch.setattr(provenance, "SERVER_VERSION", changed_version)
    assert provenance.build_meta("probe")["server_version"] == changed_version
    recording = await hm.build()
    for row in recording["positions"].values():
        assert row["result"]["meta"]["schema_version"] == SCHEMA_VERSION
        assert "server_version" not in row["result"]["meta"]
    assert recording == json.loads((hm.FIXTURE_DIR / "after.json").read_text())
