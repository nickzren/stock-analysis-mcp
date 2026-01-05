"""Tests for response schemas."""

import pytest

from stock_mcp import SCHEMA_VERSION, SERVER_VERSION
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance


class TestBuildMeta:
    """Tests for build_meta function."""

    def test_meta_has_required_fields(self) -> None:
        """Test meta includes all required fields."""
        meta = build_meta("test_tool")

        assert "server_version" in meta
        assert "schema_version" in meta
        assert "tool" in meta

    def test_meta_versions(self) -> None:
        """Test meta includes correct versions."""
        meta = build_meta("test_tool")

        assert meta["server_version"] == SERVER_VERSION
        assert meta["schema_version"] == SCHEMA_VERSION

    def test_meta_tool_name(self) -> None:
        """Test meta includes tool name."""
        meta = build_meta("my_custom_tool")
        assert meta["tool"] == "my_custom_tool"

    def test_meta_duration(self) -> None:
        """Test meta includes duration when provided."""
        meta = build_meta("test_tool", duration_ms=123.456)
        assert meta["duration_ms"] == 123.5  # Rounded to 1 decimal

    def test_meta_no_duration(self) -> None:
        """Test meta excludes duration when not provided."""
        meta = build_meta("test_tool")
        assert "duration_ms" not in meta


class TestBuildProvenance:
    """Tests for build_provenance function."""

    def test_provenance_has_source(self) -> None:
        """Test provenance includes source."""
        prov = build_provenance(source="yfinance")
        assert prov["source"] == "yfinance"

    def test_provenance_has_warnings_list(self) -> None:
        """Test provenance always has warnings list."""
        prov = build_provenance(source="yfinance")
        assert "warnings" in prov
        assert isinstance(prov["warnings"], list)

    def test_provenance_datetime_as_of(self) -> None:
        """Test provenance handles datetime as_of."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 0)
        prov = build_provenance(source="yfinance", as_of=dt)
        assert prov["as_of"] == "2024-01-15T10:30:00"

    def test_provenance_string_as_of(self) -> None:
        """Test provenance handles string as_of."""
        prov = build_provenance(source="yfinance", as_of="2024-01-15T10:30:00Z")
        assert prov["as_of"] == "2024-01-15T10:30:00Z"

    def test_provenance_extra_fields(self) -> None:
        """Test provenance includes extra fields."""
        prov = build_provenance(
            source="yfinance",
            bar_timezone="America/New_York",
            market_state="closed",
        )
        assert prov["bar_timezone"] == "America/New_York"
        assert prov["market_state"] == "closed"

    def test_provenance_custom_warnings(self) -> None:
        """Test provenance accepts custom warnings."""
        prov = build_provenance(source="yfinance", warnings=["stale_data"])
        assert prov["warnings"] == ["stale_data"]


class TestBuildErrorResponse:
    """Tests for build_error_response function."""

    def test_error_response_has_error_flag(self) -> None:
        """Test error response has error: True."""
        resp = build_error_response("invalid_symbol", "Bad symbol")
        assert resp["error"] is True

    def test_error_response_has_type(self) -> None:
        """Test error response has error_type."""
        resp = build_error_response("data_unavailable", "No data")
        assert resp["error_type"] == "data_unavailable"

    def test_error_response_has_message(self) -> None:
        """Test error response has message."""
        resp = build_error_response("invalid_symbol", "Symbol not found: XYZ")
        assert resp["message"] == "Symbol not found: XYZ"

    def test_error_response_has_meta(self) -> None:
        """Test error response has meta block."""
        resp = build_error_response("invalid_symbol", "Bad")
        assert "meta" in resp
        assert resp["meta"]["tool"] == "error"

    def test_error_response_symbol(self) -> None:
        """Test error response includes symbol when provided."""
        resp = build_error_response("invalid_symbol", "Bad", symbol="XYZ")
        assert resp["symbol"] == "XYZ"

    def test_error_response_no_symbol(self) -> None:
        """Test error response excludes symbol when not provided."""
        resp = build_error_response("invalid_symbol", "Bad")
        assert "symbol" not in resp

    def test_error_response_retry_after(self) -> None:
        """Test error response includes retry_after when provided."""
        resp = build_error_response(
            "rate_limited", "Too many requests", retry_after_seconds=60
        )
        assert resp["retry_after_seconds"] == 60
