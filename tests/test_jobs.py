"""Step 1 WP3: event loop, timeouts, cancellation and singleflight (R3.1-R3.10).

Timing is barrier-controlled: blocking jobs wait on ``threading.Event``s that the
test releases, and deadlines are configured, never asserted as wall-clock times.
"""

import asyncio
import importlib
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from stock_analysis.data import jobs, provider, yfinance_client
from stock_analysis.data.cache_manager import ticker_cache
from tests.support.fake_ticker import FakeTicker, http_error
from tests.support.transport import FakeSession

analyze_orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
options_signals = importlib.import_module("stock_analysis.tools.options_signals")

SYMBOL = "EXMP"


class Barrier:
    """Blocking job bodies plus a detector for overlapping jobs on one symbol."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.started = 0
        self.active = 0
        self.max_active = 0
        self.by_symbol: Counter[str] = Counter()
        self.reentered = False
        self.order: list[str] = []

    def job(self, name: str = "job", symbol: str | None = None, result: Any = None,
            error: Exception | None = None) -> Callable[[], Any]:
        def run() -> Any:
            with self._lock:
                self.started += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.order.append(f"start:{name}")
                if symbol is not None:
                    self.by_symbol[symbol] += 1
                    self.reentered |= self.by_symbol[symbol] > 1
            try:
                self.release.wait(timeout=10)
                if error is not None:
                    raise error
                return result
            finally:
                with self._lock:
                    self.active -= 1
                    self.order.append(f"end:{name}")
                    if symbol is not None:
                        self.by_symbol[symbol] -= 1

        return run

    async def until_started(self, count: int) -> None:
        for _ in range(1000):
            if self.started >= count:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"only {self.started} of {count} jobs started")


async def drain() -> None:
    """Wait until every job thread has finished (the test released its barrier)."""
    for _ in range(1000):
        if jobs.in_flight_jobs() == 0:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"{jobs.in_flight_jobs()} jobs still in flight")


@pytest.fixture
def barrier() -> Iterator[Barrier]:
    b = Barrier()
    yield b
    b.release.set()


async def cancel(task: "asyncio.Task[Any]") -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class TestR31IndependentTools:
    async def test_a_blocked_tool_does_not_time_out_the_others(
        self, monkeypatch: pytest.MonkeyPatch, barrier: Barrier
    ) -> None:
        """Tool A's read blocks a worker past its deadline; B..N finish and report success."""

        class BlockingOptionsTicker:
            ticker = SYMBOL

            @property
            def options(self) -> tuple[str, ...]:
                return barrier.job("options")()  # type: ignore[no-any-return]

        async def fake_fetch_ticker(symbol: str) -> BlockingOptionsTicker:
            return BlockingOptionsTicker()

        async def fake_info(symbol: str) -> dict[str, Any]:
            return {"regularMarketPrice": 50.0}

        async def quick(symbol: str, **kwargs: Any) -> dict[str, Any]:
            await asyncio.to_thread(lambda: None)  # needs the loop after A has blocked
            return {"symbol": symbol, "current_price": 100.0}

        monkeypatch.setattr(options_signals, "fetch_ticker", fake_fetch_ticker)
        monkeypatch.setattr(options_signals, "_fetch_info_or_empty", fake_info)
        for name in ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics",
                     "events_calendar", "stock_news", "ownership_analysis"):
            monkeypatch.setattr(analyze_orchestrator, name, quick)
        monkeypatch.setattr(analyze_orchestrator, "TIMEOUT_SECONDS", 0.5)

        result = await analyze_orchestrator.analyze_stock(SYMBOL)
        barrier.release.set()

        failures = result["data_quality"]["tool_failures"]
        assert [f["tool"] for f in failures] == ["options_signals"]
        assert "timeout" in failures[0]["message"]
        assert result["data_quality"]["in_flight_jobs"] == 1  # A's thread was still running
        await drain()

    async def test_a_tool_timeout_discloses_in_flight_jobs(
        self, monkeypatch: pytest.MonkeyPatch, barrier: Barrier
    ) -> None:
        async def blocked(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return await jobs.run_job(barrier.job("tool"), clock=jobs.ReadClock())  # type: ignore[no-any-return]

        async def quick(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        for name in ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics",
                     "events_calendar", "stock_news", "ownership_analysis"):
            monkeypatch.setattr(analyze_orchestrator, name, quick)
        monkeypatch.setattr(analyze_orchestrator, "options_signals", blocked)
        monkeypatch.setattr(analyze_orchestrator, "TIMEOUT_SECONDS", 0.2)

        result = await analyze_orchestrator.analyze_stock(SYMBOL)
        barrier.release.set()

        (failure,) = result["data_quality"]["tool_failures"]
        assert (failure["tool"], failure["error"], failure["in_flight_jobs"]) == ("options_signals", "TimeoutError", 1)
        await drain()


class TestR32SymbolLock:
    async def test_a_read_queued_behind_the_symbol_lock_times_out_with_its_reason(self, barrier: Barrier) -> None:
        ticker = FakeTicker(SYMBOL)
        first = asyncio.create_task(provider.read_property(ticker, "options", barrier.job("first", SYMBOL, ("x",))))
        await barrier.until_started(1)

        with jobs.deadline_scope(0.05):
            queued = await provider.read_property(ticker, "splits", barrier.job("second", SYMBOL))

        assert (queued.status, queued.error_type) == ("failed", "queued_behind_symbol_lock")
        assert barrier.started == 1  # the queued read never ran
        barrier.release.set()
        assert (await first).status == "ok"
        await drain()

    async def test_another_symbol_is_not_queued(self, barrier: Barrier) -> None:
        blocked = asyncio.create_task(provider.read_property(FakeTicker(SYMBOL), "options", barrier.job("a", SYMBOL)))
        await barrier.until_started(1)

        other = await provider.read_property(FakeTicker("OTHR"), "options", lambda: ("2026-11-20",))

        assert other.status == "ok"
        barrier.release.set()
        await blocked
        await drain()


class TestR33R34Singleflight:
    async def test_cancelled_creator_leaves_the_shared_job_to_its_joiners(self, barrier: Barrier) -> None:
        starts = 0

        async def start() -> str:
            nonlocal starts
            starts += 1
            return await jobs.run_job(barrier.job(result="value"), clock=jobs.ReadClock())  # type: ignore[no-any-return]

        creator = asyncio.create_task(jobs.singleflight("key", start))
        await barrier.until_started(1)
        joiner = asyncio.create_task(jobs.singleflight("key", start))
        await asyncio.sleep(0)

        await cancel(creator)
        late = asyncio.create_task(jobs.singleflight("key", start))  # still in flight: joins
        await asyncio.sleep(0)
        barrier.release.set()

        assert await joiner == ("value", True)
        assert await late == ("value", True)
        assert starts == 1
        await drain()

    async def test_all_waiters_cancelled_the_job_completes_and_the_registry_cleans_up(
        self, barrier: Barrier
    ) -> None:
        loop = asyncio.get_running_loop()
        reported: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: reported.append(context))

        async def start() -> None:
            await jobs.run_job(barrier.job(error=RuntimeError("upstream broke")), clock=jobs.ReadClock())

        waiters = [asyncio.create_task(jobs.singleflight("key", start)) for _ in range(3)]
        await barrier.until_started(1)
        flight = jobs._flights["key"]
        for waiter in waiters:
            await cancel(waiter)

        barrier.release.set()
        await asyncio.wait([flight])

        assert flight.done() and not flight.cancelled()  # don't retrieve its exception here
        assert "key" not in jobs._flights
        del flight, waiters
        import gc

        gc.collect()
        await asyncio.sleep(0)
        assert reported == []
        await drain()

    async def test_a_failed_flight_is_not_reused(self) -> None:
        starts = 0

        async def start() -> None:
            nonlocal starts
            starts += 1
            raise RuntimeError("failed")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await jobs.singleflight("key", start)
        assert starts == 2


class TestR33R34CallerDeadlines:
    async def test_a_joiner_stops_at_its_own_scope(self, barrier: Barrier) -> None:
        async def start() -> str:
            return await jobs.run_job(barrier.job(result="value"), clock=jobs.ReadClock())  # type: ignore[no-any-return]

        creator = asyncio.create_task(jobs.singleflight("key", start))
        await barrier.until_started(1)

        with jobs.deadline_scope(0.05), pytest.raises(jobs.JobTimeoutError):
            await jobs.singleflight("key", start)

        barrier.release.set()
        assert await creator == ("value", False)
        await drain()

    async def test_the_shared_work_outlives_its_creators_scope(self, barrier: Barrier) -> None:
        async def start() -> str:
            return await jobs.run_job(barrier.job(result="value"), clock=jobs.ReadClock())  # type: ignore[no-any-return]

        with jobs.deadline_scope(0.05):
            creator = asyncio.create_task(jobs.singleflight("key", start))
        await barrier.until_started(1)
        joiner = asyncio.create_task(jobs.singleflight("key", start))
        with pytest.raises(jobs.JobTimeoutError):
            await creator

        barrier.release.set()
        assert await joiner == ("value", True)
        await drain()

    async def test_concurrent_fetch_info_calls_share_one_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        release, reads = threading.Event(), []

        class SlowInfoTicker:
            @property
            def info(self) -> dict[str, Any]:
                reads.append(1)
                release.wait(timeout=10)
                return {"quoteType": "ETF", "symbol": SYMBOL}

        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: SlowInfoTicker())
        first = asyncio.create_task(yfinance_client._fetch_info_with_singleflight(SYMBOL))
        second = asyncio.create_task(yfinance_client._fetch_info_with_singleflight(SYMBOL))
        await asyncio.sleep(0)
        release.set()

        (_, _, first_joined), (_, _, second_joined) = await asyncio.gather(first, second)
        assert (len(reads), first_joined, second_joined) == (1, False, True)
        await drain()


class TestRunJobOutcomes:
    async def test_a_finished_job_is_not_reported_late(self) -> None:
        """The loop was busy past the deadline, but the job had already finished."""
        with jobs.deadline_scope(0.1):
            task = asyncio.create_task(jobs.run_job(lambda: "done", clock=jobs.ReadClock()))
        await asyncio.sleep(0)  # the job is submitted
        while jobs.in_flight_jobs():
            time.sleep(0.001)  # the job finishes while the loop is busy ...
        time.sleep(0.15)  # ... and stays busy past the deadline (e.g. pandas work)

        assert await task == "done"

    async def test_an_expired_read_is_a_timeout_not_rate_limited(self) -> None:
        with jobs.deadline_scope(0):
            read = await provider.read_property(FakeTicker(SYMBOL), "splits", lambda: None)
            with pytest.raises(yfinance_client.YFinanceRetryError, match="deadline passed: timeout"):
                await yfinance_client._retry_with_backoff("probe", lambda: None)

        assert (read.status, read.error_type) == ("failed", "timeout")

    async def test_a_submit_that_fails_after_queueing_never_runs_the_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real, queued, ran = jobs._executor, [], []

        class QueueThenFail:
            def submit(self, fn: Callable[[], Any]) -> Any:
                queued.append(real.submit(fn))
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(jobs, "_executor", QueueThenFail())

        with pytest.raises(RuntimeError, match="new thread"):
            await jobs.run_job(lambda: ran.append(1), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        await asyncio.wrap_future(queued[0])

        assert ran == []
        assert (jobs.in_flight_jobs(), SYMBOL in jobs._gate.locked_symbols) == (0, False)


class TestGateOrder:
    async def test_a_freed_symbol_lock_goes_to_the_earliest_waiter(self, barrier: Barrier) -> None:
        order: list[str] = []
        holder = asyncio.create_task(
            jobs.run_job(barrier.job("holder", SYMBOL), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        )
        await barrier.until_started(1)
        first = asyncio.create_task(
            jobs.run_job(lambda: order.append("first"), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        )
        await asyncio.sleep(0)  # queued behind the holder

        barrier.release.set()
        while "end:holder" not in barrier.order:
            time.sleep(0.001)  # the holder's thread finishes while the loop is busy ...
        time.sleep(0.01)  # ... and its done-callback frees the lock; nobody is woken yet
        late = asyncio.create_task(
            jobs.run_job(lambda: order.append("late"), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        )
        await asyncio.gather(holder, first, late)

        assert order == ["first", "late"]

    async def test_a_full_pool_times_out_as_queued_for_admission(self, barrier: Barrier) -> None:
        blocked = [asyncio.create_task(jobs.run_job(barrier.job(f"j{i}"), clock=jobs.ReadClock()))
                   for i in range(jobs._gate.permits)]
        await barrier.until_started(jobs._gate.permits)

        with jobs.deadline_scope(0.05):
            read = await provider.read_property(FakeTicker("OTHR"), "splits", lambda: None)

        assert (read.status, read.error_type) == ("failed", "queued_for_admission")
        barrier.release.set()
        await asyncio.gather(*blocked)
        await drain()

    async def test_time_queued_for_a_slot_does_not_spend_the_read_budget(
        self, monkeypatch: pytest.MonkeyPatch, barrier: Barrier
    ) -> None:
        """A batch tool's reads wait their turn without failing (no enclosing scope)."""
        monkeypatch.setattr(jobs, "READ_DEADLINE_SECONDS", 0.05)
        blocked = [asyncio.create_task(provider.read_property(FakeTicker(f"B{i}"), "splits", barrier.job(f"j{i}")))
                   for i in range(jobs._gate.permits)]
        await barrier.until_started(jobs._gate.permits)
        waiting = asyncio.create_task(provider.read_property(FakeTicker("OTHR"), "splits", lambda: ("ok",)))
        await asyncio.sleep(0.15)  # queued well past the 0.05 s read budget
        barrier.release.set()

        assert (await waiting).status == "ok"
        await asyncio.gather(*blocked)
        await drain()


class TestRequestTimeoutCap:
    async def test_property_requests_are_capped(self, yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "valid_one_epoch")
        ticker = yfinance_client.yf.Ticker(SYMBOL)

        read = await provider.read_property(ticker, "calendar", lambda: ticker.calendar)

        assert read.status == "ok"
        assert {call.timeout for call in yf_transport.calls} == {jobs.REQUEST_TIMEOUT_SECONDS}


class TestR35R36Ownership:
    async def test_admission_holds_while_orphaned_jobs_run(self, barrier: Barrier) -> None:
        permits = jobs._gate.permits
        first = [asyncio.create_task(jobs.run_job(barrier.job(f"a{i}"), clock=jobs.ReadClock()))
                 for i in range(permits)]
        await barrier.until_started(permits)
        for task in first:
            await cancel(task)
        assert jobs.in_flight_jobs() == permits  # orphans still hold their permits

        second = [asyncio.create_task(jobs.run_job(barrier.job(f"b{i}"), clock=jobs.ReadClock()))
                  for i in range(permits)]
        for _ in range(20):
            await asyncio.sleep(0.005)
        assert barrier.started == permits  # nothing new was admitted

        barrier.release.set()
        await asyncio.gather(*second)
        assert barrier.max_active <= permits
        assert barrier.started == 2 * permits
        await drain()

    async def test_symbol_lock_is_held_until_the_thread_finishes(self, barrier: Barrier) -> None:
        first = asyncio.create_task(
            jobs.run_job(barrier.job("first", SYMBOL), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        )
        await barrier.until_started(1)
        await cancel(first)

        second = asyncio.create_task(
            jobs.run_job(barrier.job("second", SYMBOL), clock=jobs.ReadClock(), symbol_lock=SYMBOL)
        )
        for _ in range(20):
            await asyncio.sleep(0.005)
        assert barrier.started == 1

        barrier.release.set()
        await second
        assert not barrier.reentered
        assert barrier.order.index("end:first") < barrier.order.index("start:second")
        await drain()

    async def test_a_waiter_timeout_also_keeps_the_lock(self, barrier: Barrier) -> None:
        with jobs.deadline_scope(0.05), pytest.raises(jobs.JobTimeoutError) as timed_out:
            await jobs.run_job(barrier.job("slow", SYMBOL), clock=jobs.ReadClock(), symbol_lock=SYMBOL)

        assert timed_out.value.reason == "timeout"
        assert SYMBOL in jobs._gate.locked_symbols
        barrier.release.set()
        await drain()
        assert SYMBOL not in jobs._gate.locked_symbols


class TestR37NothingHeldWhileSleeping:
    @pytest.fixture
    def sleeps(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
        seen: list[tuple[int, bool]] = []

        async def record(delay: float) -> None:
            seen.append((jobs.in_flight_jobs(), SYMBOL in jobs._gate.locked_symbols))

        monkeypatch.setattr(provider, "_sleep", record)
        monkeypatch.setattr(yfinance_client, "_sleep", record)
        return seen

    async def test_property_read_backoff(self, sleeps: list[tuple[int, bool]]) -> None:
        ticker = FakeTicker(SYMBOL, splits=http_error(503))

        await provider.read_property(ticker, "splits", lambda: ticker.splits)

        assert sleeps and all(entry == (0, False) for entry in sleeps)

    async def test_transport_read_backoff(self, sleeps: list[tuple[int, bool]], yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "http_503")

        await provider.read_calendar_events(SYMBOL)

        assert sleeps and all(entry == (0, False) for entry in sleeps)

    async def test_client_retry_backoff(self, sleeps: list[tuple[int, bool]]) -> None:
        with pytest.raises(yfinance_client.YFinanceRetryError):
            await yfinance_client._retry_with_backoff("probe", _raiser(http_error(503)))

        assert sleeps and all(entry == (0, False) for entry in sleeps)


def _raiser(error: Exception) -> Callable[[], Any]:
    def call() -> Any:
        raise error

    return call


class TestR38PropertyReadsAreJobs:
    async def test_each_attempt_runs_on_a_worker_thread(self) -> None:
        threads: list[str] = []

        def read() -> tuple[str, ...]:
            threads.append(threading.current_thread().name)
            if len(threads) == 1:
                raise http_error(503)
            return ("2026-11-20",)

        result = await provider.read_property(FakeTicker(SYMBOL), "options", read)

        assert (result.status, result.attempts) == ("ok", 2)
        assert threading.main_thread().name not in threads


class TestR39Deadlines:
    async def test_transport_requests_carry_the_request_timeout(self, yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "valid_one_epoch")

        await provider.read_calendar_events(SYMBOL)

        assert [call.timeout for call in yf_transport.calls] == [jobs.REQUEST_TIMEOUT_SECONDS]

    async def test_a_hung_read_fails_structured_inside_the_tool_deadline(
        self, monkeypatch: pytest.MonkeyPatch, yf_transport: FakeSession
    ) -> None:
        """events_calendar still returns: its hung calendar read fails at the read deadline."""
        entered, release = yf_transport.hold("calendar")
        yf_transport.route("calendar", "valid_one_epoch")
        yf_transport.route("earnings_page", "past_only_table")

        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol, splits=None)

        async def fake_info(symbol: str) -> dict[str, Any]:
            return {}

        events = importlib.import_module("stock_analysis.tools.events")
        monkeypatch.setattr(events, "fetch_ticker", fake_fetch_ticker)
        monkeypatch.setattr(events, "fetch_info", fake_info)
        calendar_reads: list[provider.SourceRead] = []

        async def spy(symbol: str) -> provider.SourceRead:
            calendar_reads.append(await provider.read_calendar_events(symbol))
            return calendar_reads[-1]

        monkeypatch.setattr(events, "read_calendar_events", spy)

        async def quick(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        for name in ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics",
                     "stock_news", "ownership_analysis", "options_signals"):
            monkeypatch.setattr(analyze_orchestrator, name, quick)
        monkeypatch.setattr(analyze_orchestrator, "TIMEOUT_SECONDS", 0.5)

        try:
            result = await analyze_orchestrator.analyze_stock(SYMBOL)
        finally:
            release.set()

        assert entered.is_set()
        assert [(read.status, read.error_type) for read in calendar_reads] == [("failed", "timeout")]
        assert result["data_quality"]["tool_failures"] == []  # events_calendar returned, not timed out
        await drain()

    async def test_client_read_stops_at_its_deadline(self, barrier: Barrier) -> None:
        with jobs.deadline_scope(0.05), pytest.raises(yfinance_client.YFinanceRetryError, match="deadline"):
            await yfinance_client._retry_with_backoff("probe", barrier.job("slow"))
        barrier.release.set()
        await drain()


class TestR310SharedSourceReads:
    async def test_concurrent_reads_share_one_fetch(self, yf_transport: FakeSession) -> None:
        yf_transport.route("earnings_page", "valid_table")

        first, second = await asyncio.gather(
            provider.read_earnings_page(SYMBOL), provider.read_earnings_page(SYMBOL)
        )

        assert first is second
        assert len(yf_transport.calls_to("earnings_page")) == 1

    async def test_events_and_news_share_the_earnings_page(self, yf_transport: FakeSession) -> None:
        events = importlib.import_module("stock_analysis.tools.events")
        yf_transport.route("earnings_page", "valid_table")

        await asyncio.gather(events.read_earnings_dates(SYMBOL), events.read_earnings_dates(SYMBOL))

        assert len(yf_transport.calls_to("earnings_page")) == 1

    async def test_a_failed_read_is_shared_in_flight_but_never_cached(self, yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "http_404_json")

        first, second = await asyncio.gather(
            provider.read_calendar_events(SYMBOL), provider.read_calendar_events(SYMBOL)
        )
        one_fetch = len(yf_transport.calls_to("calendar"))  # yfinance re-requests once on >= 400
        after = await provider.read_calendar_events(SYMBOL)

        assert first is second and first.status == "failed"
        assert after is not first
        assert len(yf_transport.calls_to("calendar")) == 2 * one_fetch


def test_ticker_cache_is_untouched_by_job_tests() -> None:
    assert ticker_cache.get("OTHR") is None
