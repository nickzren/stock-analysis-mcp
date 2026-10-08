"""Record raw Yahoo responses for local shape-drift checks (manual, networked).

Tests exercise it only offline, through the stubbed transport; they never run
it against the network. Output goes to the gitignored
``tests/fixtures/_recorded/``, one directory per run; recordings are real Yahoo
data and must not be committed. Committed test fixtures are synthetic
(``tests/fixtures/transport/``).

    uv run python scripts/record_provider_fixtures.py MSFT KO TSM
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf
from yfinance.data import YfData

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "_recorded"
# Cookie, crumb and consent exchanges carry session tokens; never write them,
# whether requested directly or reached by redirect. Matched case-insensitively.
_SKIP_URL_FRAGMENTS = ("getcrumb", "fc.yahoo.com", "consent")

READERS = {
    "calendar": lambda symbol: yf.Ticker(symbol).calendar,
    "earnings_page": lambda symbol: yf.Ticker(symbol).get_earnings_dates(limit=12),
    "news": lambda symbol: yf.Ticker(symbol).get_news(),
    "search": lambda symbol: yf.Search(symbol, news_count=10, raise_errors=True).news,
    "info": lambda symbol: yf.Ticker(symbol).info,
}


class RecordingSession:
    """Delegates to the real session and saves each response as a transport fixture.

    Files are named after the reader running (``source``), since one URL family
    (quoteSummary) serves several readers.
    """

    def __init__(self, inner: Any, out_dir: Path):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_out_dir", out_dir)
        object.__setattr__(self, "files", [])
        object.__setattr__(self, "source", "other")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def get(self, url: str, **kwargs: Any) -> Any:
        return self._record("GET", url, kwargs, self._inner.get(url, **kwargs))

    def post(self, url: str, **kwargs: Any) -> Any:
        return self._record("POST", url, kwargs, self._inner.post(url, **kwargs))

    def _record(self, method: str, url: str, kwargs: dict[str, Any], response: Any) -> Any:
        urls = f"{url} {getattr(response, 'url', '')}".lower()
        if any(fragment in urls for fragment in _SKIP_URL_FRAGMENTS):
            return response
        params = {k: v for k, v in (kwargs.get("params") or {}).items() if k != "crumb"}
        content_type = response.headers.get("content-type", "")
        spec: dict[str, Any] = {
            "status": response.status_code,
            "request": {"method": method, "url": url, "params": params},
        }
        try:
            spec["json"] = json.loads(response.text)
        except ValueError:
            spec["content_type"] = content_type or "text/plain"
            spec["body"] = response.text
        path = self._out_dir / f"{len(self.files) + 1:02d}_{self.source}.json"
        path.write_text(json.dumps(spec, indent=2, default=str) + "\n")
        self.files.append(path.name)
        return response


def record_symbol(
    symbol: str, out_root: Path = DEFAULT_OUT, sources: tuple[str, ...] = tuple(READERS)
) -> Path:
    captured_at = datetime.now(ZoneInfo("America/New_York"))
    out_dir = out_root / symbol.upper() / captured_at.strftime("%Y-%m-%dT%H%M%S.%f")
    out_dir.mkdir(parents=True)

    data = YfData()
    original = data._session
    recorder = RecordingSession(original, out_dir)
    data._session = recorder
    YfData.cache_get.cache_clear()
    outcomes: dict[str, str] = {}
    try:
        for source in sources:
            object.__setattr__(recorder, "source", source)
            try:
                READERS[source](symbol)
                outcomes[source] = "ok"
            except Exception as exc:  # recorded for drift review, not raised
                outcomes[source] = f"error: {type(exc).__name__}: {exc}"
    finally:
        data._session = original
        YfData.cache_get.cache_clear()

    manifest = {
        "symbol": symbol.upper(),
        "captured_at": captured_at.isoformat(),
        "yfinance_version": yf.__version__,
        "sources": outcomes,
        "files": recorder.files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for symbol in args.symbols:
        print(record_symbol(symbol, args.out))


if __name__ == "__main__":
    main()
