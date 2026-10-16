"""WP2.4: schema 6 (plan §10 WP2.4, D-13)."""

from stock_analysis import SCHEMA_VERSION
from stock_analysis.utils.provenance import build_meta


def test_schema_version_is_6() -> None:
    """T1 (schema 6, D-13): SCHEMA_VERSION and every response's meta.schema_version are "6". RF."""
    assert SCHEMA_VERSION == "6"
    assert build_meta("analyze_stock")["schema_version"] == "6"
