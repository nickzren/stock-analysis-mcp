"""Price data resource handler."""

from stock_mcp.data.cache import price_cache


class ResourceNotFoundError(Exception):
    """Resource not found in cache."""

    pass


def read_price_resource(uri: str) -> tuple[str, str]:
    """
    Serve cached price data only. O(1), no transformation.

    Args:
        uri: Resource URI (e.g., price://NVDA/1y/1d/adjusted)

    Returns:
        Tuple of (csv_text, mime_type)

    Raises:
        ResourceNotFoundError: If resource not in cache
    """
    csv_text = price_cache.get_csv(uri)

    if csv_text is None:
        raise ResourceNotFoundError(f"Resource not cached. Call price_history first: {uri}")

    return csv_text, "text/csv"
