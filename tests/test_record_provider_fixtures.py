"""The manual recording script, exercised offline through the stubbed transport."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.support.transport import FakeResponse, FakeSession, TransportFixture

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record_provider_fixtures.py"


@pytest.fixture
def recorder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("record_provider_fixtures", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_each_response_as_a_transport_fixture(
    recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path
) -> None:
    yf_transport.route("calendar", "valid_one_epoch")
    yf_transport.route("news", "valid_stream")
    yf_transport.route("search", "http_404_json")

    out = recorder.record_symbol("EXMP", tmp_path, sources=("calendar", "news", "search"))

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sources"] == {"calendar": "ok", "news": "ok", "search": "ok"}
    # yfinance re-requests once on status >= 400, so the 404 search is recorded twice.
    assert manifest["files"] == ["01_calendar.json", "02_news.json", "03_search.json", "04_search.json"]
    search = json.loads((out / "03_search.json").read_text())
    assert search["status"] == 404
    assert search["json"]["finance"]["error"]["code"] == "Not Found"
    assert "crumb" not in search["request"]["params"]


def test_reader_errors_are_noted_not_raised(
    recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path
) -> None:
    yf_transport.route("calendar", "http_429")

    out = recorder.record_symbol("EXMP", tmp_path, sources=("calendar",))

    outcome = json.loads((out / "manifest.json").read_text())["sources"]["calendar"]
    assert outcome.startswith("error: YFRateLimitError")


@pytest.mark.parametrize(
    ("url", "final_url"),
    [
        ("https://query1.finance.yahoo.com/v1/test/getcrumb", ""),
        ("https://guce.yahoo.com/copyConsent?sessionId=SID", ""),
        (
            "https://query2.finance.yahoo.com/v10/finance/quoteSummary/EXMP",
            "https://consent.yahoo.com/v2/collectConsent?sessionId=SID",
        ),
    ],
)
def test_session_token_exchanges_are_never_written(
    recorder: ModuleType, tmp_path: Path, url: str, final_url: str
) -> None:
    session = recorder.RecordingSession(FakeSession(), tmp_path)
    response = FakeResponse(TransportFixture(200, "token-bearing body", "text/html"), final_url)

    session._record("GET", url, {}, response)

    assert session.files == []
    assert list(tmp_path.iterdir()) == []


def test_each_run_gets_its_own_directory(
    recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path
) -> None:
    yf_transport.route("calendar", "valid_one_epoch")

    first = recorder.record_symbol("EXMP", tmp_path, sources=("calendar",))
    second = recorder.record_symbol("EXMP", tmp_path, sources=("calendar",))

    assert first != second
    for run in (first, second):
        manifest = json.loads((run / "manifest.json").read_text())
        assert sorted(p.name for p in run.iterdir()) == sorted([*manifest["files"], "manifest.json"])
