from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numeric(s: pd.Series | None) -> pd.Series | None:
    if s is None:
        return None
    return pd.to_numeric(s, errors="coerce")


def _to_datetime(s: pd.Series | None) -> pd.Series | None:
    if s is None:
        return None
    return pd.to_datetime(s, errors="coerce")


def _pct(count: int | float, total: int) -> float:
    if total == 0:
        return np.nan
    return float(count / total * 100)


def build_data_quality_row(
    stock_code: str,
    df: pd.DataFrame,
    date_col: str = "trading_date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> dict:
    """
    Build one data-quality row for a single symbol.

    Checks:
    - row count
    - date range
    - missing dates
    - duplicate dates
    - missing OHLCV
    - close <= 0
    - volume < 0
    - volume == 0
    - invalid OHLC logic
    """

    n_rows = len(df)

    row = {
        "stock_code": stock_code,
        "n_rows": n_rows,
    }

    #   Date checks  
    if date_col in df.columns:
        dates = _to_datetime(df[date_col])

        row["start_date"] = dates.min()
        row["end_date"] = dates.max()
        row["missing_date"] = int(dates.isna().sum())
        row["pct_missing_date"] = _pct(row["missing_date"], n_rows)

        # duplicate only among valid dates
        row["duplicate_dates"] = int(dates.dropna().duplicated().sum())
        row["pct_duplicate_dates"] = _pct(row["duplicate_dates"], n_rows)
    else:
        row["start_date"] = pd.NaT
        row["end_date"] = pd.NaT
        row["missing_date"] = np.nan
        row["pct_missing_date"] = np.nan
        row["duplicate_dates"] = np.nan
        row["pct_duplicate_dates"] = np.nan

    #   Missing OHLCV  
    required_cols = {
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
        "volume": volume_col,
    }

    for name, col in required_cols.items():
        if col in df.columns:
            missing_count = int(df[col].isna().sum())
            row[f"missing_{name}"] = missing_count
            row[f"pct_missing_{name}"] = _pct(missing_count, n_rows)
        else:
            row[f"missing_{name}"] = np.nan
            row[f"pct_missing_{name}"] = np.nan

    #   Numeric conversions  
    open_ = _to_numeric(df[open_col]) if open_col in df.columns else None
    high = _to_numeric(df[high_col]) if high_col in df.columns else None
    low = _to_numeric(df[low_col]) if low_col in df.columns else None
    close = _to_numeric(df[close_col]) if close_col in df.columns else None
    volume = _to_numeric(df[volume_col]) if volume_col in df.columns else None

    #   Price checks  
    if close is not None:
        close_le_0 = int((close <= 0).sum())
        row["close_le_0"] = close_le_0
        row["pct_close_le_0"] = _pct(close_le_0, n_rows)
    else:
        row["close_le_0"] = np.nan
        row["pct_close_le_0"] = np.nan

    #   Volume checks  
    if volume is not None:
        volume_lt_0 = int((volume < 0).sum())
        volume_eq_0 = int((volume == 0).sum())

        row["volume_lt_0"] = volume_lt_0
        row["pct_volume_lt_0"] = _pct(volume_lt_0, n_rows)

        row["volume_eq_0"] = volume_eq_0
        row["pct_volume_eq_0"] = _pct(volume_eq_0, n_rows)
    else:
        row["volume_lt_0"] = np.nan
        row["pct_volume_lt_0"] = np.nan
        row["volume_eq_0"] = np.nan
        row["pct_volume_eq_0"] = np.nan

    # OHLC consistency  
    if all(x is not None for x in [open_, high, low, close]):
        valid_ohlc_mask = open_.notna() & high.notna() & low.notna() & close.notna()

        ohlc_invalid_mask = valid_ohlc_mask & (
            (high < open_)
            | (high < close)
            | (high < low)
            | (low > open_)
            | (low > close)
            | (low > high)
        )

        ohlc_invalid = int(ohlc_invalid_mask.sum())
        row["ohlc_invalid"] = ohlc_invalid
        row["pct_ohlc_invalid"] = _pct(ohlc_invalid, int(valid_ohlc_mask.sum()))
    else:
        row["ohlc_invalid"] = np.nan
        row["pct_ohlc_invalid"] = np.nan

    return row


def build_data_quality(
    data: dict[str, pd.DataFrame],
    date_col: str = "trading_date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Build data-quality table for multiple symbols.

    Parameters
     
    data:
        Dict dạng:
        {
            "FPT": df_fpt,
            "VCB": df_vcb,
            ...
        }

    Returns
    -------
    pd.DataFrame
        One-row-per-symbol data quality table.
    """

    rows = []

    for stock, df in data.items():
        rows.append(
            build_data_quality_row(
                stock_code=stock,
                df=df,
                date_col=date_col,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                volume_col=volume_col,
            )
        )

    return pd.DataFrame(rows).set_index("stock_code")
def build_data_quality_from_wide(
    wide: pd.DataFrame,
    symbols: list[str],
    date_col: str = "trading_date",
) -> pd.DataFrame:
    """
    {stock_code}_open
    {stock_code}_high
    {stock_code}_low
    {stock_code}_close
    {stock_code}_volume
    """

    data: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        col_map = {
            "open": f"{sym}_open",
            "high": f"{sym}_high",
            "low": f"{sym}_low",
            "close": f"{sym}_close",
            "volume": f"{sym}_volume",
        }

        available_cols = {
            new_name: old_name
            for new_name, old_name in col_map.items()
            if old_name in wide.columns
        }

        if "close" not in available_cols:
            continue

        df = wide[list(available_cols.values())].rename(
            columns={
                old_name: new_name
                for new_name, old_name in available_cols.items()
            }
        )

        df = df.reset_index()

        if date_col not in df.columns:
            df = df.rename(columns={df.columns[0]: date_col})

        data[sym] = df

    return build_data_quality(
        data,
        date_col=date_col,
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
    )