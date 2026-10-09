"""Executor jobs that own their admission permit and symbol lock (Step 1 WP3, R5).

Every blocking yfinance read runs as one job on a bounded thread pool. A job takes
an admission permit, and optionally a per-symbol lock, when it is submitted and
gives them back only when its thread finishes, in the future's done-callback.
Cancelling or timing out the waiter stops only the waiting: an orphaned job keeps
its permit and lock, so the limit can't be bypassed and no two jobs ever hold one
symbol's lock at once. Slots go to waiters in arrival order.

A read's time limit is a ``ReadClock``: ``READ_DEADLINE_SECONDS`` from its first
admission, capped by any enclosing ``deadline_scope`` (analyze's per-tool
deadline). Time spent queued for a slot counts only against the scope, so a
batch tool's healthy reads aren't dropped just for waiting their turn; without a
scope, that queue wait is unbounded (it ends when running jobs finish). The
per-request timeout bounds each HTTP request, not a job that makes several.

Shared by ``yfinance_client`` and ``provider``; it lives in its own module because
the provider imports the client.
"""

import asyncio
import contextlib
import contextvars
import math
import os
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

MAX_WORKERS = int(os.environ.get("YF_MAX_WORKERS", "4"))
# How long one read (all attempts and backoff) may run once admitted, and how long
# one HTTP request may wait (yfinance_client caps every yfinance request at it).
READ_DEADLINE_SECONDS = float(os.environ.get("YF_READ_DEADLINE_SECONDS", "8"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("YF_REQUEST_TIMEOUT_SECONDS", "5"))

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_scope_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("_scope_deadline", default=None)
# False inside a singleflight's shared work: it runs until its fetch finishes, so a
# source stays owned by one fetch; each caller waits only within its own clock.
_budgeted: contextvars.ContextVar[bool] = contextvars.ContextVar("_budgeted", default=True)


class JobTimeoutError(TimeoutError):
    """The waiter gave up at its deadline. ``reason`` is ``queued_behind_symbol_lock``,
    ``queued_for_admission`` or ``timeout`` (the job ran too long; it keeps running)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ReadClock:
    """One read's time limit; create it when the read starts."""

    scope: float | None = field(default_factory=_scope_deadline.get)
    budgeted: bool = field(default_factory=_budgeted.get)
    budget_end: float | None = None

    def remaining(self) -> float:
        ends = [end for end in (self.scope, self.budget_end) if end is not None]
        return min(ends) - time.monotonic() if ends else math.inf

    def admitted(self) -> None:
        if self.budgeted and self.budget_end is None:
            self.budget_end = time.monotonic() + READ_DEADLINE_SECONDS

    def timeout(self) -> float | None:
        """For ``asyncio.timeout``: None when unlimited."""
        left = self.remaining()
        return None if left == math.inf else max(left, 0)


@contextlib.contextmanager
def deadline_scope(seconds: float) -> Iterator[None]:
    """Cap every read started in this context (and tasks it creates) at now + seconds."""
    token = _scope_deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _scope_deadline.reset(token)


@dataclass(eq=False)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: "asyncio.Future[None]"
    symbol: str | None
    granted: bool = False


class _Gate:
    """Admission permits and per-symbol locks, released only when a job's thread is done.

    State is guarded by a thread lock because releases run on worker threads. A
    release grants the freed slot to the earliest queued waiter that can use it
    before waking that waiter on its own loop, so later arrivals can't barge.
    """

    def __init__(self, permits: int):
        self.permits = permits
        self.running = 0
        self.locked_symbols: set[str] = set()
        self._lock = threading.Lock()
        self._queue: list[_Waiter] = []

    def blocked_reason(self, symbol: str | None) -> str | None:
        with self._lock:
            return self._blocked_reason(symbol)

    def _blocked_reason(self, symbol: str | None) -> str | None:
        if symbol is not None and symbol in self.locked_symbols:
            return "queued_behind_symbol_lock"
        if self.running >= self.permits:
            return "queued_for_admission"
        return None

    def _take(self, symbol: str | None) -> None:
        self.running += 1
        if symbol is not None:
            self.locked_symbols.add(symbol)

    def _give_back(self, symbol: str | None) -> None:
        self.running -= 1
        if symbol is not None:
            self.locked_symbols.discard(symbol)
        for waiter in list(self._queue):
            if self._blocked_reason(waiter.symbol) is not None:
                continue
            self._queue.remove(waiter)
            try:
                waiter.loop.call_soon_threadsafe(_wake, waiter.future)
            except RuntimeError:  # its loop is closed; it will never run
                continue
            self._take(waiter.symbol)
            waiter.granted = True

    async def acquire(self, symbol: str | None) -> str | None:
        """Take a slot, queueing behind earlier waiters; returns why it queued, if it did."""
        loop = asyncio.get_running_loop()
        with self._lock:
            reason = self._blocked_reason(symbol)
            if reason is None:
                self._take(symbol)
                return None
            waiter = _Waiter(loop, loop.create_future(), symbol)
            self._queue.append(waiter)
        try:
            await waiter.future
        except BaseException:
            with self._lock:
                if waiter.granted:
                    self._give_back(symbol)  # granted as we were cancelled: pass it on
                elif waiter in self._queue:
                    self._queue.remove(waiter)
            raise
        return reason

    def release(self, symbol: str | None) -> None:
        with self._lock:
            self._give_back(symbol)

    def start(self, fn: Callable[[], T], symbol: str | None) -> "Future[T]":
        """Run ``fn`` on the slot ``acquire`` took; the slot is released when it finishes.

        Synchronous, so nothing can cancel the caller between taking the slot and
        attaching the release. ``ThreadPoolExecutor.submit`` queues the work before
        it may fail to start a thread, so the job waits for ``go`` and runs ``fn``
        only if it was submitted cleanly and owns the slot.
        """
        go = threading.Event()
        owned: list[bool] = []

        def job() -> T | None:
            go.wait()
            return fn() if owned else None

        try:
            future = _executor.submit(job)
        except BaseException:
            go.set()
            self.release(symbol)
            raise
        owned.append(True)
        go.set()
        future.add_done_callback(lambda _: self.release(symbol))
        return future  # type: ignore[return-value]

    def reset(self) -> None:
        """Test seam: forget all state (between tests, after their jobs are released)."""
        with self._lock:
            self.running = 0
            self.locked_symbols.clear()
            self._queue.clear()


def _wake(future: "asyncio.Future[None]") -> None:
    if not future.done():
        future.set_result(None)


_gate = _Gate(MAX_WORKERS)


def in_flight_jobs() -> int:
    """Jobs whose threads have not finished, including ones nobody waits for anymore."""
    return _gate.running


async def run_job(fn: Callable[[], T], *, clock: ReadClock, symbol_lock: str | None = None) -> T:
    """Run ``fn`` as one executor job, within ``clock``.

    Raises ``JobTimeoutError`` when the job could not be admitted, or did not
    finish, in time. A started job keeps its permit and lock until it finishes. A
    job that finished is never reported as timed out, even if the loop was late.
    """
    if clock.remaining() <= 0:
        raise JobTimeoutError(_gate.blocked_reason(symbol_lock) or "timeout")
    try:
        async with asyncio.timeout(clock.timeout()):
            queued_reason = await _gate.acquire(symbol_lock)
    except TimeoutError:
        raise JobTimeoutError(_gate.blocked_reason(symbol_lock) or "queued_for_admission") from None
    if clock.remaining() <= 0:  # admitted, but woken after the deadline: don't start late
        _gate.release(symbol_lock)
        raise JobTimeoutError(queued_reason or "timeout")
    future = _gate.start(fn, symbol_lock)
    clock.admitted()
    try:
        async with asyncio.timeout(clock.timeout()):
            return await asyncio.wrap_future(future)
    except TimeoutError:
        if future.done() and not future.cancelled():
            return future.result()  # also re-raises the job's own error
        raise JobTimeoutError("timeout") from None


_flights: dict[Any, "asyncio.Task[Any]"] = {}


async def singleflight(key: Any, start: Callable[[], Coroutine[Any, Any, T]]) -> tuple[T, bool]:
    """Share one in-flight ``start()`` among concurrent callers with the same key.

    Returns (result, joined). The work runs in a detached task with no read budget
    and outside its creator's scope, so the key stays owned until the underlying
    fetch actually finishes (each HTTP request is still capped). Every caller, the
    creator included, awaits it through ``asyncio.shield``, so a cancelled caller
    never cancels it, and waits only within its own ``ReadClock`` (budget from
    joining, capped by its scope); on expiry it gets ``JobTimeoutError`` and the
    work goes on for the others. A done-callback removes the entry and retrieves
    any exception, so an outcome nobody awaited is never logged as unretrieved.
    The entry lives only while in flight: nothing, failed results included, is
    cached.
    """
    loop = asyncio.get_running_loop()
    task = _flights.get(key)
    joined = task is not None and task.get_loop() is loop and not task.done()
    if task is None or not joined:
        context = contextvars.copy_context()
        context.run(_scope_deadline.set, None)
        context.run(_budgeted.set, False)
        task = loop.create_task(start(), context=context)
        _flights[key] = task
        task.add_done_callback(lambda t: _land(key, t))
    clock = ReadClock()
    clock.admitted()
    try:
        async with asyncio.timeout(clock.timeout()):
            return await asyncio.shield(task), joined
    except TimeoutError:
        if task.done() and not task.cancelled():
            return task.result(), joined
        raise JobTimeoutError("timeout") from None


def _land(key: Any, task: "asyncio.Task[Any]") -> None:
    if _flights.get(key) is task:
        del _flights[key]
    if not task.cancelled():
        task.exception()


def clear_flights() -> None:
    """Test seam: drop in-flight entries left by a finished test's event loop."""
    _flights.clear()


async def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)
