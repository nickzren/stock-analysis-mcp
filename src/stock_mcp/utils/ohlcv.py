"""OHLCV data standardization utilities."""

import pandas as pd


def standardize_ohlcv(df: pd.DataFrame, adjusted: bool) -> pd.DataFrame:
    """
    Standardize OHLCV to consistent schema.

    Output columns (always, in this order): date, open, high, low, close, volume
    All lowercase. No 'Adj Close' column. Missing columns filled with NaN.

    Args:
        df: Raw DataFrame from yfinance
        adjusted: Whether auto_adjust was True (documents in provenance, not schema)

    Returns:
        Standardized DataFrame with consistent schema
    """
    df = df.copy()

    # Handle multi-index from yf.download (when fetching multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # If adjusted=False, yfinance includes 'Adj Close'
    # Drop it - we document adjusted in URI, don't leak extra columns
    # Note: auto_adjust=True adjusts ALL OHLC values, not just Close
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    # Lowercase all column names
    df.columns = df.columns.str.lower()

    # Reset index to make date a column
    df = df.reset_index()

    # Normalize date column name
    date_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "index")]
    if date_cols:
        df = df.rename(columns={date_cols[0]: "date"})

    # Format date as ISO (YYYY-MM-DD for daily, full ISO for intraday)
    # Use is_datetime64_any_dtype to handle both naive and tz-aware datetimes
    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        # Better intraday detection: check if any times differ from midnight
        has_time = (df["date"].dt.normalize() != df["date"]).any()
        if has_time:
            # For tz-aware, convert to string with timezone info
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            else:
                df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # CRITICAL: Ensure ALL 6 columns exist in exact order
    # Fill missing columns with NaN for schema stability
    canonical_cols = ["date", "open", "high", "low", "close", "volume"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Enforce exact column order (no extras, no missing)
    df = df[canonical_cols]

    return df


def df_to_rows(df: pd.DataFrame) -> list[dict]:
    """Convert to list of dicts for inline preview. Lowercase keys."""
    return df.to_dict("records")


def df_to_csv(df: pd.DataFrame) -> str:
    """Convert to CSV string for cache/resource."""
    return df.to_csv(index=False)
