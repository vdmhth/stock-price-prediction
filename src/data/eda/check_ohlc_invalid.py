from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_STOCK_DIR = PROJECT_ROOT / "data" / "processed" / "market"
RAW_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "market"

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


SYMBOLS = ['stock_common_clean','market_common_clean']

def find_csv(symbol: str) -> Path | None:
    """Find raw csv file for a symbol in likely raw-data folders."""

    candidates = [
        RAW_STOCK_DIR / f"{symbol}.csv",
        RAW_INDEX_DIR / f"{symbol}.csv",
        PROJECT_ROOT / "data" / "processed" / f"{symbol}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names to trading_date/open/high/low/close/volume."""

    rename_map = {}

    for col in df.columns:
        c = col.lower()

        if c in ["time", "date", "tradingdate", "trading_date"]:
            rename_map[col] = "trading_date"
        elif c == "open":
            rename_map[col] = "open"
        elif c == "high":
            rename_map[col] = "high"
        elif c == "low":
            rename_map[col] = "low"
        elif c in ["close", "matchedprice"]:
            rename_map[col] = "close"
        elif c in ["volume", "nmvolume"]:
            rename_map[col] = "volume"

    return df.rename(columns=rename_map)


def check_ohlc_invalid(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return rows that violate OHLC consistency."""

    df = normalize_columns(df.copy())

    required = ['index_code',"trading_date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[skip] {symbol}: missing columns {missing}")
        return pd.DataFrame()

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[["open", "high", "low", "close"]].notna().all(axis=1)

    invalid_mask = valid & (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    invalid = df.loc[
        invalid_mask,
        ['index_code',"trading_date", "open", "high", "low", "close"]
        + (["volume"] if "volume" in df.columns else []),
    ].copy()

    invalid.insert(0, "symbol", symbol)

    return invalid


def main() -> None:
    all_invalid = []

    for symbol in SYMBOLS:
        path = find_csv(symbol)

        if path is None:
            print(f"[missing file] {symbol}")
            continue

        print(f"[check] {symbol}: {path}")

        df = pd.read_csv(path)
        invalid = check_ohlc_invalid(df, symbol)

        print(f"  invalid rows: {len(invalid)}")

        if len(invalid) > 0:
            print(invalid.to_string(index=False))
            all_invalid.append(invalid)

    if all_invalid:
        out = pd.concat(all_invalid, ignore_index=True)
    else:
        out = pd.DataFrame(
            columns=["index_code" ,"trading_date", "open", "high", "low", "close", "volume"]
        )

    save_path = REPORT_DIR / "00_ohlc_invalid_rows.csv"
    out.to_csv(save_path, index=False)

    print(f"\nSaved invalid OHLC rows to: {save_path}")


if __name__ == "__main__":
    main()