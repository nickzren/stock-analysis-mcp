"""The offline test harness itself (Step 1 WP0, A0.1)."""

import contextlib
from pathlib import Path

import curl_cffi.requests
import pytest
import requests
import yfinance as yf

from stock_analysis.utils.watch_store import resolve_data_dir
from tests.support.fake_ticker import FakeTicker, http_error
from tests.support.network_guard import NetworkAccessInTestsError, violation_report
from tests.support.transport import FakeSession, UnroutedRequestError, stubbed_yfinance_transport

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/EXMP"


@pytest.mark.parametrize(
    ("make_session", "stack"),
    [(curl_cffi.requests.Session, "curl_cffi"), (requests.Session, "requests")],
)
def test_unstubbed_request_is_blocked_and_recorded(
    _block_network: list[str], make_session: type, stack: str
) -> None:
    with pytest.raises(NetworkAccessInTestsError):
        make_session().get("https://example.invalid/blocked")

    assert _block_network == [f"{stack} GET https://example.invalid/blocked"]
    _block_network.clear()  # recorded as expected; don't fail this test at teardown


def test_swallowed_network_error_is_still_recorded(_block_network: list[str]) -> None:
    """App code catches broad exceptions, so the guard must not rely on raising alone."""
    with contextlib.suppress(Exception):
        _ = yf.Ticker("EXMP").calendar

    assert _block_network, "the real yfinance session reached the network unguarded"
    _block_network.clear()


def test_violation_report_names_each_attempt() -> None:
    report = violation_report(["curl_cffi GET https://a", "requests POST https://b"])

    assert "curl_cffi GET https://a" in report
    assert "requests POST https://b" in report


def test_data_dir_is_isolated_per_test(tmp_path: Path) -> None:
    assert resolve_data_dir() == tmp_path / "stock-analysis-data"


def test_stubbed_transport_rejects_and_records_unrouted_requests(
    yf_transport: FakeSession, _block_network: list[str]
) -> None:
    with pytest.raises(UnroutedRequestError):
        yf_transport.get(CHART_URL)

    assert yf_transport.unrouted == [f"GET {CHART_URL}"]
    assert yf_transport.calls == []
    assert _block_network == [f"stub transport: unrouted GET {CHART_URL}"]
    _block_network.clear()  # recorded as expected; don't fail this test at teardown


def test_r1_13_error_caught_outside_the_stub_still_reports_unrouted(
    _block_network: list[str],
) -> None:
    """R1.13: the test sees its own error unchanged, and the unrouted request is
    still reported to the end-of-test check."""

    class TestError(Exception):
        pass

    with pytest.raises(TestError):  # noqa: SIM117 - the stub must exit via the error
        with stubbed_yfinance_transport() as session:
            with contextlib.suppress(UnroutedRequestError):
                session.get(CHART_URL)
            raise TestError

    assert _block_network == [f"stub transport: unrouted GET {CHART_URL}"]
    _block_network.clear()


def test_stub_does_not_nest(yf_transport: FakeSession) -> None:
    with pytest.raises(RuntimeError, match="does not nest"), stubbed_yfinance_transport():
        pass


class TestFakeTicker:
    def test_serves_values_raises_errors_and_counts_reads(self) -> None:
        ticker = FakeTicker(calendar={"Earnings Date": []}, news=http_error(401))

        assert ticker.calendar == {"Earnings Date": []}
        with pytest.raises(Exception, match="HTTP Error 401"):
            _ = ticker.news
        assert ticker.reads == {"calendar": 1, "news": 1}

    def test_unconfigured_attribute_is_an_error(self) -> None:
        with pytest.raises(AttributeError, match="no configured attribute 'info'"):
            _ = FakeTicker().info


R1_13_INNER = """
import contextlib

import pytest

from tests.conftest import _block_network  # noqa: F401 - the autouse guard under test
from tests.support.transport import UnroutedRequestError, stubbed_yfinance_transport


class InnerError(Exception):
    pass


def test_inner():
    with pytest.raises(InnerError, match="original"):
        with stubbed_yfinance_transport() as session:
            with contextlib.suppress(UnroutedRequestError):
                session.get("https://query2.finance.yahoo.com/v8/finance/chart/EXMP")
            raise InnerError("original")
"""


def test_r1_13_unrouted_request_fails_the_test_at_teardown(pytester: pytest.Pytester) -> None:
    """R1.13 end to end: the inner test sees its own error and passes its body,
    then fails at teardown naming the unrouted request."""
    pytester.makepyfile(R1_13_INNER)

    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:asyncio")

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*stub transport: unrouted GET*chart/EXMP*"])
