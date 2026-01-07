"""Stock Analysis MCP Server."""

import os


def get_server_version() -> str:
    """Get version with fallback for dev mode."""
    if version := os.environ.get("SERVER_VERSION"):
        return version
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("stock-mcp")
    except Exception:
        return "dev"


SERVER_VERSION = get_server_version()
# Bump when output schema changes materially (new fields, renamed fields, structure changes)
# v1: Initial schema
# v2: Added watchlist_snapshot with snapshot_version/hash, NaN sanitization, policy_action enhancements
SCHEMA_VERSION = "2"
