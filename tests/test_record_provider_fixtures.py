"""The manual recording script, exercised offline through the stubbed transport."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from yfinance.data import YfData

from tests.support.transport import FakeResponse, FakeSession, TransportFixture, load_fixture

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


def assert_no_session_ids(message: str, out_root: Path) -> None:
    """The synthetic consent fixtures carry ``sessionId=SID``; it must not leak."""
    texts = [message] + [path.read_text() for path in out_root.rglob("*") if path.is_file()]
    assert not [text for text in texts if "sessionId" in text or "SID" in text]


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


class TestConsentGuard:
    """The recorder is read-only: it stops before submitting any cookie consent."""

    @pytest.mark.parametrize(
        ("method", "url"),
        [
            ("GET", "https://fc.yahoo.com"),
            ("GET", "https://query1.finance.yahoo.com/v1/test/getcrumb"),
            ("GET", "https://guce.yahoo.com/consent"),
            ("GET", "https://query2.finance.yahoo.com/v10/finance/quoteSummary/EXMP"),
            ("POST", "https://finance.yahoo.com/xhr/ncp?queryRef=latestNews"),
        ],
    )
    def test_reads_and_cookie_requests_are_allowed(self, recorder: ModuleType, method: str, url: str) -> None:
        recorder._refuse_consent_submission(method, url)

    @pytest.mark.parametrize(
        ("method", "url"),
        [
            ("POST", "https://consent.yahoo.com/v2/collectConsent?sessionId=SID"),
            ("GET", "https://guce.yahoo.com/copyConsent?sessionId=SID"),
            ("POST", "https://consent.yahoo.com/v2/consent?sessionId=SID"),
            ("POST", "https://guce.yahoo.com/anything"),
        ],
    )
    def test_consent_submissions_are_refused(self, recorder: ModuleType, method: str, url: str) -> None:
        with pytest.raises(recorder.ConsentRequiredError):
            recorder._refuse_consent_submission(method, url)

    @pytest.mark.parametrize("fixture", ["redirected_to_consent_form", "redirected_to_consent_no_action"])
    def test_redirect_to_consent_aborts_the_run_before_any_post(
        self, recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path, fixture: str
    ) -> None:
        yf_transport.route("calendar", fixture)
        yf_transport.route("news", "valid_stream")

        with pytest.raises(recorder.ConsentRequiredError, match="consent page") as aborted:
            recorder.record_symbol("EXMP", tmp_path, sources=("calendar", "news"))

        assert_no_session_ids(str(aborted.value), tmp_path)
        assert [call.method for call in yf_transport.calls] == ["GET"]
        assert yf_transport.calls_to("news") == []  # later sources never ran
        (run,) = (tmp_path / "EXMP").iterdir()
        sources = json.loads((run / "manifest.json").read_text())["sources"]
        assert list(sources) == ["calendar"] and sources["calendar"].startswith("aborted:")

    def test_original_handler_and_session_are_restored_after_an_abort(
        self, recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path
    ) -> None:
        yf_transport.route("calendar", "redirected_to_consent_form")

        with pytest.raises(recorder.ConsentRequiredError):
            recorder.record_symbol("EXMP", tmp_path, sources=("calendar",))

        data = YfData()
        assert data._session is yf_transport
        assert "_accept_consent_form" not in data.__dict__

    def test_form_without_action_is_blocked_at_the_session(
        self, recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path
    ) -> None:
        """Defense in depth: yfinance's own handler posts a no-action form back to
        the consent page itself; the session guard refuses it before sending."""
        data = YfData()
        data._session = recorder.RecordingSession(yf_transport, tmp_path)
        consent = FakeResponse(
            load_fixture("calendar", "redirected_to_consent_no_action"),
            "https://consent.yahoo.com/v2/consent?sessionId=SID",
        )

        with pytest.raises(recorder.ConsentRequiredError, match="POST https://consent.yahoo.com/v2/consent") as aborted:
            YfData._accept_consent_form(data, consent, 30)

        assert_no_session_ids(str(aborted.value), tmp_path)
        assert yf_transport.calls == []

    def test_csrf_cookie_path_is_blocked_before_collect_consent(
        self, recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yf_transport.route("consent_page", "csrf_form")
        data = YfData()
        data._session = recorder.RecordingSession(yf_transport, tmp_path)
        data._cookie = None
        monkeypatch.setattr(data, "_load_cookie_curlCffi", lambda: False)

        with pytest.raises(recorder.ConsentRequiredError, match="collectConsent"):
            data._get_cookie_csrf(30)

        assert [(call.method, call.url) for call in yf_transport.calls] == [
            ("GET", "https://guce.yahoo.com/consent")
        ]

    def test_an_abort_stops_the_remaining_symbols(
        self, recorder: ModuleType, yf_transport: FakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yf_transport.route("calendar", "redirected_to_consent_form")
        monkeypatch.setattr(recorder, "READERS", {"calendar": recorder.READERS["calendar"]})

        with pytest.raises(recorder.ConsentRequiredError):
            recorder.record_symbols(["EXMP", "EXMQ"], tmp_path)

        assert [p.name for p in tmp_path.iterdir()] == ["EXMP"]

    def test_main_reports_a_redirect_without_session_ids(
        self,
        recorder: ModuleType,
        yf_transport: FakeSession,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        yf_transport.route("calendar", "redirected_to_consent_form")
        monkeypatch.setattr(recorder, "READERS", {"calendar": recorder.READERS["calendar"]})
        monkeypatch.setattr("sys.argv", ["record", "EXMP", "--out", str(tmp_path)])

        with pytest.raises(SystemExit) as stopped:
            recorder.main()

        captured = capsys.readouterr()
        assert "consent.yahoo.com/v2/collectConsent" in str(stopped.value.code)
        assert_no_session_ids(f"{stopped.value.code}{captured.out}{captured.err}", tmp_path)

    def test_main_exits_with_a_clear_message(
        self, recorder: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def abort(symbols: list[str], out_root: Path) -> list[Path]:
            raise recorder.ConsentRequiredError("consent submission blocked before sending: POST x")

        monkeypatch.setattr(recorder, "record_symbols", abort)
        monkeypatch.setattr("sys.argv", ["record", "EXMP", "--out", str(tmp_path)])

        with pytest.raises(SystemExit, match="Nothing was submitted"):
            recorder.main()
