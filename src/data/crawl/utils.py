from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = "VCI"

DATE_COLUMN_CANDIDATES = ("trading_date", "date", "time", "tradingDate")


def _normalize_interval(interval: str) -> str:
    interval_map = {
        "1d": "1D",
        "d": "1D",
        "day": "1D",
        "1w": "1W",
        "w": "1W",
        "week": "1W",
        "1m": "1M",
        "m": "1M",
        "month": "1M",
    }
    return interval_map.get(interval.lower(), interval)


def _find_date_column(df: pd.DataFrame) -> str:
    for column in DATE_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Cannot find a date column in vnstock data. "
        f"Available columns: {list(df.columns)}"
    )


def _filter_by_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    date_column = _find_date_column(df)
    dates = pd.to_datetime(df[date_column])

    filtered_df = df.loc[
        (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    ].copy()

    filtered_df[date_column] = pd.to_datetime(filtered_df[date_column]).dt.strftime(
        "%Y-%m-%d"
    )

    return filtered_df