
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.crawl import config as cfg


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed1"
PROCESSED_STOCK_DIR = PROCESSED_DIR / "stock"
PROCESSED_MARKET_DIR = PROCESSED_DIR / "market"
PROCESSED_EDA_DIR = PROCESSED_DIR / "eda"

REPORT_TABLES_DIR = DATA_DIR / "reports1" / "tables"


# =============================================================================
# COLUMNS
# =============================================================================

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

QUALITY_COLS = [
    "missing_ohlc",
    "non_positive_ohlc",
    "high_invalid",
    "low_invalid",
    "ohlc_invalid",
    "zero_volume",
]

EVENT_COLS = [
    "hose_disruption",
]

DERIVED_COLS = [
    "log_close",
    "return",
    "log_return",
    "range_pct",
    "traded_value",
]

HOSE_DISRUPTION_DATES = pd.to_datetime([
    "2018-01-23",
    "2018-01-24",
])


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLCV columns exist and are numeric.

    If a market index file does not contain volume, volume is set to NaN.
    """
    df = df.copy()

    for col in OHLCV_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_raw_stock(code: str) -> pd.DataFrame:
    """
    Load raw stock CSV and normalize schema.
    """
    df = pd.read_csv(Path(cfg.RAW_STOCK_DIR)/ f"{code}.csv")

    df = df.rename(columns={"time": "trading_date"})
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df = ensure_ohlcv_columns(df)

    df = (
        df.sort_values("trading_date")
        .drop_duplicates("trading_date", keep="first")
        .reset_index(drop=True)
    )

    return df


def load_raw_market(symbol: str) -> pd.DataFrame:
    """
    Load raw market index CSV and normalize schema.
    """
    df = pd.read_csv(Path(cfg.RAW_MARKET_DIR) / f"{symbol}.csv")

    df = df.rename(columns={"time": "trading_date", "index": "index_code"})
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df = ensure_ohlcv_columns(df)

    df = (
        df.sort_values("trading_date")
        .drop_duplicates("trading_date", keep="first")
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# QUALITY FLAGS + EVENT FLAGS
# =============================================================================

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add OHLCV quality flags.

    OHLC invalid rows are:
    - rows with missing OHLC,
    - rows with OHLC <= 0,
    - rows where high is not the highest value,
    - rows where low is not the lowest value.

    Important:
    - Do not drop rows here.
    - Do not interpolate OHLC.
    - Do not overwrite original OHLC values.
    """
    df = df.copy()

    ohlc = ["open", "high", "low", "close"]

    df["missing_ohlc"] = df[ohlc].isna().any(axis=1).astype(int)

    df["non_positive_ohlc"] = (df[ohlc] <= 0).any(axis=1).astype(int)

    df["high_invalid"] = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    ).astype(int)

    df["low_invalid"] = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    ).astype(int)

    df["ohlc_invalid"] = (
        df["missing_ohlc"].eq(1)
        | df["non_positive_ohlc"].eq(1)
        | df["high_invalid"].eq(1)
        | df["low_invalid"].eq(1)
    ).astype(int)

    df["zero_volume"] = df["volume"].eq(0).astype(int)

    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add known event flags.

    HOSE disruption:
    - 2018-01-23
    - 2018-01-24
    """
    df = df.copy()

    df["hose_disruption"] = (
        df["trading_date"].isin(HOSE_DISRUPTION_DATES)
    ).astype(int)

    return df


# =============================================================================
# DERIVED FEATURES
# =============================================================================

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns.

    Main volatility is not computed here.
    Rolling volatility is computed later in EDA from log_return.

    OHLC invalid rows are retained and only masked when calculating range_pct,
    because range_pct depends on high, low, and close.
    """
    df = df.copy()

    df = add_quality_flags(df)
    df = add_event_flags(df)

    # Close-based features
    valid_close = df["close"].gt(0) & df["close"].notna()

    df["log_close"] = np.where(valid_close, np.log(df["close"]), np.nan)
    df["return"] = df["close"].pct_change()
    df["log_return"] = pd.Series(df["log_close"], index=df.index).diff()

    # OHLC-based feature: mask invalid OHLC only for this calculation
    clean_ohlc = df[["open", "high", "low", "close"]].mask(
        df["ohlc_invalid"].eq(1)
    )

    df["range_pct"] = (
        clean_ohlc["high"] - clean_ohlc["low"]
    ) / clean_ohlc["close"]

    # Volume-based feature
    df["traded_value"] = df["close"] * df["volume"]

    return df


# =============================================================================
# PANELS
# =============================================================================

def build_stock_panel(codes: list[str]) -> dict[str, pd.DataFrame]:
    return {
        code: add_derived_columns(load_raw_stock(code))
        for code in codes
    }


def build_market_panel(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {
        symbol: add_derived_columns(load_raw_market(symbol))
        for symbol in symbols
    }


def common_windows(
    panels: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = max(df["trading_date"].min() for df in panels.values())
    end_date = min(df["trading_date"].max() for df in panels.values())

    return start_date, end_date


def filter_window(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    return df[
        (df["trading_date"] >= start_date)
        & (df["trading_date"] <= end_date)
    ].reset_index(drop=True)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def to_long_stock(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [
        df.assign(stock_code=code)
        for code, df in panels.items()
    ]

    out = pd.concat(parts, ignore_index=True)

    cols = (
        ["stock_code", "trading_date"]
        + OHLCV_COLS
        + QUALITY_COLS
        + EVENT_COLS
        + DERIVED_COLS
    )

    return (
        out[cols]
        .sort_values(["stock_code", "trading_date"])
        .reset_index(drop=True)
    )


def to_long_market(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [
        df.assign(index_code=symbol)
        for symbol, df in panels.items()
    ]

    out = pd.concat(parts, ignore_index=True)

    cols = (
        ["index_code", "trading_date"]
        + OHLCV_COLS
        + QUALITY_COLS
        + EVENT_COLS
        + DERIVED_COLS
    )

    return (
        out[cols]
        .sort_values(["index_code", "trading_date"])
        .reset_index(drop=True)
    )


def build_eda_wide(
    stock_panels: dict[str, pd.DataFrame],
    market_panels: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build wide-format common-period dataset for EDA.

    Columns:
    - {symbol}_close
    - {symbol}_log_return
    - {symbol}_range_pct
    - {symbol}_ohlc_invalid
    - {symbol}_zero_volume
    - {symbol}_hose_disruption
    """
    frames = []

    wide_features = [
        "close",
        "log_return",
        "range_pct",
        "ohlc_invalid",
        "zero_volume",
        "hose_disruption",
    ]

    for code, df in stock_panels.items():
        sub = df[["trading_date"] + wide_features].rename(
            columns={col: f"{code}_{col}" for col in wide_features}
        )
        frames.append(sub.set_index("trading_date"))

    for symbol, df in market_panels.items():
        sub = df[["trading_date"] + wide_features].rename(
            columns={col: f"{symbol}_{col}" for col in wide_features}
        )
        frames.append(sub.set_index("trading_date"))

    wide = pd.concat(frames, axis=1, join="inner").sort_index().reset_index()

    return filter_window(wide, start, end)


# =============================================================================
# DATA QUALITY
# =============================================================================

def build_quality_table(
    stock_panels: dict[str, pd.DataFrame],
    market_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build data quality table on the same common-period panels used for EDA.

    The calendar is the native dataset calendar created from the selected
    common-period panels. No pd.bdate_range is used, because inserting
    Vietnamese holidays or Tet as artificial trading dates would create
    fake missing dates.
    """
    rows = []

    all_panels: dict[str, pd.DataFrame] = {}
    all_panels.update(stock_panels)
    all_panels.update(market_panels)

    all_trading_dates = sorted(
        set().union(
            *[
                set(pd.to_datetime(df["trading_date"]).dropna())
                for df in all_panels.values()
            ]
        )
    )
    global_calendar = pd.DatetimeIndex(all_trading_dates)

    for symbol, df in all_panels.items():
        df = df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])

        n = len(df)

        symbol_dates = pd.DatetimeIndex(df["trading_date"].dropna().unique())
        missing_dates = global_calendar.difference(symbol_dates)

        duplicate_dates = df["trading_date"].duplicated().sum()

        zero_return = df["log_return"].eq(0)
        zero_return_no_volume = zero_return & df["volume"].eq(0)
        zero_return_with_volume = zero_return & df["volume"].gt(0)

        def pct(x: int | float) -> float:
            return float(x / n) if n > 0 else np.nan

        rows.append(
            {
                "symbol": symbol,
                "n_rows": n,
                "start_date": df["trading_date"].min(),
                "end_date": df["trading_date"].max(),

                "missing_date": int(len(missing_dates)),
                "pct_missing_date": pct(len(missing_dates)),

                "duplicate_dates": int(duplicate_dates),
                "pct_duplicate_dates": pct(duplicate_dates),

                "ohlc_invalid": int(df["ohlc_invalid"].sum()),
                "pct_ohlc_invalid": df["ohlc_invalid"].mean(),

                "volume_eq_0": int(df["zero_volume"].sum()),
                "pct_volume_eq_0": df["zero_volume"].mean(),

                "zero_return_rows": int(zero_return.sum()),
                "pct_zero_return_rows": zero_return.mean(),

                "zero_return_no_volume_rows": int(zero_return_no_volume.sum()),
                "pct_zero_return_no_volume_rows": zero_return_no_volume.mean(),

                "zero_return_with_volume_rows": int(zero_return_with_volume.sum()),
                "pct_zero_return_with_volume_rows": zero_return_with_volume.mean(),

                "missing_log_return_rows": int(df["log_return"].isna().sum()),
                "pct_missing_log_return_rows": df["log_return"].isna().mean(),

                "hose_disruption_rows": int(df["hose_disruption"].sum()),
                "pct_hose_disruption_rows": df["hose_disruption"].mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    PROCESSED_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_EDA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    stock_panels = build_stock_panel(cfg.STOCK_CODES)
    market_panels = build_market_panel(cfg.MARKET_INDICES)

    stock_start, stock_end = common_windows(stock_panels)
    market_start, market_end = common_windows(market_panels)

    full_start = max(stock_start, market_start)
    full_end = min(stock_end, market_end)

    print(f"Stock data common window: {stock_start.date()} to {stock_end.date()}")
    print(f"Market data common window: {market_start.date()} to {market_end.date()}")
    print(f"Overall common window: {full_start.date()} to {full_end.date()}")

    # 1. Full stock data for audit only
    stock_full = to_long_stock(stock_panels)
    out = PROCESSED_STOCK_DIR / "stock_full_clean.csv"
    stock_full.to_csv(out, index=False, encoding="utf-8")
    print("Done stock_full_clean.csv")

    # 2. Common-period panels used for all downstream EDA/modeling
    stock_common_panels = {
        code: filter_window(df, full_start, full_end)
        for code, df in stock_panels.items()
    }

    market_common_panels = {
        symbol: filter_window(df, full_start, full_end)
        for symbol, df in market_panels.items()
    }

    # 3. Save common-period stock data
    stock_common = to_long_stock(stock_common_panels)
    out = PROCESSED_STOCK_DIR / "stock_common_clean.csv"
    stock_common.to_csv(out, index=False, encoding="utf-8")
    print("Done stock_common_clean.csv")

    # 4. Save common-period market data
    market_common = to_long_market(market_common_panels)
    out = PROCESSED_MARKET_DIR / "market_common_clean.csv"
    market_common.to_csv(out, index=False, encoding="utf-8")
    print("Done market_common_clean.csv")

    # 5. Save common-period wide EDA dataset
    eda_wide = build_eda_wide(
        stock_common_panels,
        market_common_panels,
        full_start,
        full_end,
    )
    out = PROCESSED_EDA_DIR / "eda_processed.csv"
    eda_wide.to_csv(out, index=False, encoding="utf-8")
    print("Done eda_processed.csv")

    # 6. Data quality report on common period
    quality = build_quality_table(stock_common_panels, market_common_panels)
    out = REPORT_TABLES_DIR / "000000_data_quality.csv"
    quality.to_csv(out, index=False, encoding="utf-8")
    print("Done 000000_data_quality.csv")


if __name__ == "__main__":
    main()