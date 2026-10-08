"""Block ``curl_cffi`` and ``requests`` session HTTP in tests and fail loudly.

These are yfinance's transport and the only network path in ``src/``. Other
clients (curl_cffi ``AsyncSession``, urllib, raw sockets) are not guarded.
yfinance and our tools catch broad exceptions, so raising alone could be
silently absorbed: every blocked attempt is also recorded, and the autouse
fixture in ``tests/conftest.py`` fails the test at teardown if any occurred.
A test marked ``xfail`` would absorb that teardown failure; none exist today.
"""

from collections.abc import Callable
from typing import Any


class NetworkAccessInTestsError(RuntimeError):
    """A test tried to reach the real network."""


def blocked_request(stack: str, violations: list[str]) -> Callable[..., Any]:
    def request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        message = f"{stack} {method} {url}"
        violations.append(message)
        raise NetworkAccessInTestsError(f"network access in tests: {message}")

    return request


def violation_report(violations: list[str]) -> str:
    return (
        "Network access attempted in tests (tests must run offline; stub the "
        "transport with tests.support.transport):\n  " + "\n  ".join(violations)
    )
