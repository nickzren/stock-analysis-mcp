"""Executor jobs that own their admission permit and symbol lock (Step 1 WP3, R5).

Every blocking yfinance read runs as one job on a bounded thread pool. A job takes
an admission permit, and optionally a per-symbol lock, when it is submitted and
gives them back only when its thread finishes, in the future's done-callback.
Cancelling or timing out the waiter stops only the waiting: an orphaned job keeps
its permit and lock, so the limit can't be bypassed and no two jobs ever hold one
symbol's lock at once.

Deadlines are absolute ``time.monotonic()`` values. ``deadline_scope`` lets a
caller (analyze's per-tool deadline) cap every read started inside it.

Shared by ``yfinance_client`` and ``provider``; it lives in its own module because
the provider imports the client.
"""

import asyncio
import contextlib
import os
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, TypeVar

T = TypeVar("T")

MAX_WORKERS = int(os.environ.get("YF_MAX_WORKERS", "4"))
# How long one read (all attempts and backoff) may take, and how long one request
# may wait on the network. Request timeouts reach only the transport reads and
# history; other yfinance properties keep yfinance's own 30 s timeout, so their
# waiter stops at the read deadline while the thread finishes in the background.
READ_DEADLINE_SECONDS = float(os.environ.get("YF_READ_DEADLINE_SECONDS", "8"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("YF_REQUEST_TIMEOUT_SECONDS", "5"))

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_scope_deadline: ContextVar[float | None] = ContextVar("_scope_deadline", default=None)


class JobTimeoutError(TimeoutError):
    """The waiter gave up at its deadline. ``reason`` is ``queued_behind_symbol_lock``,
    ``queued_for_admission`` or ``timeout`` (the job ran too long; it keeps running)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Gate:
    """Admission permits and per-symbol locks, released only when a job's thread is done.

    State is guarded by a thread lock because releases run on worker threads;
    waiting coroutines are woken on their own event loop.
    """

    def __init__(self, permits: int):
        self.permits = permits
        self.running = 0
        self.locked_symbols: set[str] = set()
        self._lock = threading.Lock()
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []

    def blocked_reason(self, symbol: str | None) -> str | None:
        if symbol is not None and symbol in self.locked_symbols:
            return "queued_behind_symbol_lock"
        if self.running >= self.permits:
            return "queued_for_admission"
        return None

    async def submit(self, fn: Callable[[], T], symbol: str | None) -> "Future[T]":
        loop = asyncio.get_running_loop()
        while True:
            with self._lock:
                if self.blocked_reason(symbol) is None:
                    self.running += 1
                    if symbol is not None:
                        self.locked_symbols.add(symbol)
                    break
                waiter: asyncio.Future[None] = loop.create_future()
                self._waiters.append((loop, waiter))
            try:
                await waiter
            finally:
                with self._lock, contextlib.suppress(ValueError):
                    self._waiters.remove((loop, waiter))
        # No await between taking the slot and submitting, so a cancelled waiter
        # can't leak a permit.
        try:
            future = _executor.submit(fn)
        except BaseException:
            self._release(symbol)
            raise
        future.add_done_callback(lambda _: self._release(symbol))
        return future

    def _release(self, symbol: str | None) -> None:
        with self._lock:
            self.running -= 1
            if symbol is not None:
                self.locked_symbols.discard(symbol)
            waiters = list(self._waiters)
        for loop, waiter in waiters:
            with contextlib.suppress(RuntimeError):  # that loop is already closed
                loop.call_soon_threadsafe(_wake, waiter)

    def reset(self) -> None:
        """Test seam: forget all state (between tests, after their jobs are released)."""
        with self._lock:
            self.running = 0
            self.locked_symbols.clear()
            self._waiters.clear()


def _wake(waiter: "asyncio.Future[None]") -> None:
    if not waiter.done():
        waiter.set_result(None)


_gate = _Gate(MAX_WORKERS)


def in_flight_jobs() -> int:
    """Jobs whose threads have not finished, including ones nobody waits for anymore."""
    return _gate.running


def read_deadline() -> float:
    """Deadline for a read starting now: the read budget, capped by any enclosing scope."""
    own = time.monotonic() + READ_DEADLINE_SECONDS
    scope = _scope_deadline.get()
    return own if scope is None else min(own, scope)


@contextlib.contextmanager
def deadline_scope(seconds: float) -> Iterator[None]:
    """Cap every read started in this context (and tasks it creates) at now + seconds."""
    token = _scope_deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _scope_deadline.reset(token)


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


async def run_job(fn: Callable[[], T], *, deadline: float, symbol_lock: str | None = None) -> T:
    """Run ``fn`` as one executor job and wait for it until ``deadline``.

    Raises ``JobTimeoutError`` when the job could not be admitted, or did not
    finish, in time. A started job keeps its permit and lock until it finishes.
    """
    if remaining(deadline) <= 0:
        raise JobTimeoutError(_gate.blocked_reason(symbol_lock) or "timeout")
    try:
        async with asyncio.timeout(max(remaining(deadline), 0)):
            future = await _gate.submit(fn, symbol_lock)
    except TimeoutError:
        raise JobTimeoutError(_gate.blocked_reason(symbol_lock) or "queued_for_admission") from None
    try:
        async with asyncio.timeout(max(remaining(deadline), 0)):
            return await asyncio.wrap_future(future)
    except TimeoutError as e:
        if future.done() and not future.cancelled() and future.exception() is e:
            raise  # the job itself raised a TimeoutError
        raise JobTimeoutError("timeout") from None


_flights: dict[Any, "asyncio.Task[Any]"] = {}


async def singleflight(key: Any, start: Callable[[], Coroutine[Any, Any, T]]) -> tuple[T, bool]:
    """Share one in-flight ``start()`` among concurrent callers with the same key.

    Returns (result, joined). The work runs in a detached task that every caller,
    the creator included, awaits through ``asyncio.shield``, so a cancelled caller
    never cancels it. A done-callback removes the entry and retrieves any
    exception, so an outcome nobody awaited is never logged as unretrieved. The
    entry lives only while in flight: nothing, failed results included, is cached.
    """
    loop = asyncio.get_running_loop()
    task = _flights.get(key)
    joined = task is not None and task.get_loop() is loop and not task.done()
    if not joined:
        task = loop.create_task(start())
        _flights[key] = task
        task.add_done_callback(lambda t: _land(key, t))
    assert task is not None
    return await asyncio.shield(task), joined


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

