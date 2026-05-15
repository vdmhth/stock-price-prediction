"""
This script is to build processed datasets from raw CSVs for EDA and modeling.
Expected output includes those below stuff:
stock/stock_full_clean.csv
stock/stock_common_clean.csv
market/market_full_clean.csv
eda/eda_processed.csv
"""
from __future__ import annotations
from pathlib  import Path

import numpy as np
import pandas as pd
from ..crawl import config as cfg

PROCESSED_DIR = Path("data/processed")
PROCESSED_STOCK_DIR = PROCESSED_DIR / "stock"
PROCESSED_MARKET_DIR = PROCESSED_DIR / "market"
PROCESSED_EDA_DIR = PROCESSED_DIR / "eda"

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
DERIVED_COLS = ["log_close", "return", "log_return", "range_pct", "traded_value", "volatility"]

def load_raw_stock(code:str) ->pd.DataFrame:
    # load raw stock CSV, normalize schema and cutoff
    df = pd.read_csv(Path(cfg.RAW_STOCK_DIR)/f'{code}.csv')
    df = df.rename(columns = {'time':'trading_date'})
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    df = df.sort_values('trading_date').drop_duplicates("trading_date").reset_index(drop=True)
    return df

def load_raw_market(symbol:str) ->pd.DataFrame:
    df = pd.read_csv(Path(cfg.RAW_MARKET_DIR)/f'{symbol}.csv')
    df = df.rename(columns = {'time':'trading_date','index':'index_code'})
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    df = df.sort_values('trading_date').drop_duplicates("trading_date").reset_index(drop=True)
    return df

def add_derived_columns(df:pd.DataFrame) ->pd.DataFrame:
    # add log_close, return, log_return, range_pct, traded_value
    df['log_close'] = np.log(df['close'])
    df['return'] = df['close'].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['traded_value'] = df['close'] * df['volume']
    df['volatility']   = np.sqrt((np.log(df['high'] / df['low']) ** 2) / (4 * np.log(2)))
    return df

def build_stock_panel(codes:list[str])->dict[str,pd.DataFrame]:
    return {c: add_derived_columns(load_raw_stock(c)) for c in codes}

def build_market_panel(symbols:list[str])->dict[str,pd.DataFrame]:
    return {s: add_derived_columns(load_raw_market(s)) for s in symbols}

# common windows
def common_windows(panels:dict[str,pd.DataFrame]) ->tuple[pd.Timestamp,pd.Timestamp]:
    start_date = max(df['trading_date'].min() for df in panels.values())
    end_date = min(df['trading_date'].max() for df in panels.values())
    return start_date, end_date
def filter_window(df:pd.DataFrame, start_date:pd.Timestamp, end_date:pd.Timestamp) ->pd.DataFrame:
    return df[(df['trading_date'] >= start_date) & (df['trading_date'] <= end_date)].reset_index(drop=True)

def to_long_stock(panels:dict[str,pd.DataFrame])->pd.DataFrame:
    parts = [df.assign(stock_code = code) for code, df in panels.items()]
    out = pd.concat(parts,ignore_index = True)
    cols = ['stock_code', 'trading_date'] + OHLCV_COLS + DERIVED_COLS
    return out[cols].sort_values(['stock_code','trading_date']).reset_index(drop=True)
def to_long_market(panels:dict[str,pd.DataFrame]) ->pd.DataFrame:
    parts = [df.assign(index_code = symbol) for symbol, df in panels.items()]
    out = pd.concat(parts, ignore_index=True)
    cols = ['index_code', 'trading_date'] + OHLCV_COLS + DERIVED_COLS
    return out[cols].sort_values(['index_code', 'trading_date']).reset_index(drop=True)

def build_eda_wide(stock_panels: dict[str, pd.DataFrame],market_panels: dict[str, pd.DataFrame],start: pd.Timestamp,end: pd.Timestamp,) -> pd.DataFrame:
    """Wide-format dataset for correlation, ADF/KPSS, ACF/PACF."""

    keep_cols = ["close", "log_return", "volume", "volatility"]

    frames = []
    for code, df in stock_panels.items():
        sub = df[["trading_date"] + keep_cols].rename(
            columns={c: f"{code}_{c}" for c in keep_cols}
        )
        frames.append(sub.set_index("trading_date"))
    for sym, df in market_panels.items():
        sub = df[["trading_date"] + keep_cols].rename(
            columns={c: f"{sym}_{c}" for c in keep_cols}
        )
        frames.append(sub.set_index("trading_date"))

    wide = pd.concat(frames, axis=1, join="inner").sort_index().reset_index()
    return filter_window(wide, start, end)

def main()->None:
    stock_panels = build_stock_panel(cfg.STOCK_CODES)
    market_panels = build_market_panel(cfg.MARKET_INDICES)
    stock_start, stock_end = common_windows(stock_panels)
    market_start, market_end = common_windows(market_panels)

    full_start,full_end = max(stock_start, market_start), min(stock_end, market_end)
    print(f"Stock data common window: {stock_start.date()} to {stock_end.date()}")
    print(f"Market data common window: {market_start.date()} to {market_end.date()}")
    print(f"Overall common window: {full_start.date()} to {full_end.date()}")

    stock_full = to_long_stock(stock_panels)
    out = PROCESSED_STOCK_DIR / "stock_full_clean.csv"
    stock_full.to_csv(out, index=False,encoding = 'utf-8')
    print(f"Done stock_full_clean.csv")
    stock_common_panels = {c: filter_window(df, stock_start, stock_end) for c, df in stock_panels.items()}
    stock_common = to_long_stock(stock_common_panels)
    out = PROCESSED_STOCK_DIR / "stock_common_clean.csv"
    stock_common.to_csv(out, index=False, encoding='utf-8')
    print(f"Done stock_common_clean.csv")

    market_common_panels = {s:filter_window(d,market_start,market_end) for s,d in market_panels.items()}
    market_common = to_long_market(market_common_panels)
    out = PROCESSED_MARKET_DIR / "market_common_clean.csv"
    market_common.to_csv(out, index=False, encoding='utf-8')
    print(f"Done market_common_clean.csv")

    eda_wide = build_eda_wide(stock_panels, market_panels, full_start, full_end)
    out = PROCESSED_EDA_DIR / "eda_processed.csv"
    eda_wide.to_csv(out, index=False, encoding='utf-8')  
    print(f"Done eda_processed.csv")

if __name__ == "__main__":
    main()