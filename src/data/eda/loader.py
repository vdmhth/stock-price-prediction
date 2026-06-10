from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_wide(path: Path | str, date_col: str = "trading_date") -> pd.DataFrame:
    """Load the wide-format EDA dataset with a DatetimeIndex.

    Notes
    -----
    Does NOT forward-fill or reindex to a business-day calendar. Vietnamese
    holidays must be respected as-is; filling them in artificially injects
    fake zero-returns into ACF/stationarity tests.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).drop_duplicates(date_col).set_index(date_col)
    return df


def load_long_stock(path: Path | str, date_col: str = "trading_date") -> pd.DataFrame:
    """Load the long-format per-stock panel (one row per stock-date)."""
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.sort_values(["stock_code", date_col]).reset_index(drop=True)
