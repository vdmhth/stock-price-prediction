from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

EDA_DATASET = DATA_DIR / "processed" / "eda" / "eda_processed.csv"
OUT_DIR = DATA_DIR / "reports" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SYMBOLS = ["FPT", "VCB", "VIC", "VNM", "HPG", "VNINDEX", "VN30"]


def main() -> None:
    wide = pd.read_csv(EDA_DATASET, parse_dates=["trading_date"])

    rows = []

    for sym in SYMBOLS:
        vol_col = f"{sym}_volume"
        close_col = f"{sym}_close"
        ret_col = f"{sym}_log_return"

        if vol_col not in wide.columns:
            print(f"[skip] {sym}: missing {vol_col}")
            continue

        tmp = wide[wide[vol_col] == 0].copy()

        print(f"{sym}: {len(tmp)} zero-volume days")

        if len(tmp) == 0:
            continue

        keep_cols = ["trading_date", vol_col]

        if close_col in wide.columns:
            keep_cols.append(close_col)

        if ret_col in wide.columns:
            keep_cols.append(ret_col)

        tmp = tmp[keep_cols]

        rename_map = {
            vol_col: "volume",
            close_col: "close",
            ret_col: "log_return",
        }

        tmp = tmp.rename(columns=rename_map)
        tmp.insert(0, "symbol", sym)

        rows.append(tmp)

    if rows:
        zero_volume_days = pd.concat(rows, ignore_index=True)
    else:
        zero_volume_days = pd.DataFrame(
            columns=["symbol", "trading_date", "volume", "close", "log_return"]
        )

    zero_volume_days = zero_volume_days.sort_values(
        ["symbol", "trading_date"]
    )

    print("\nZero-volume rows:")
    print(zero_volume_days.to_string(index=False))

    out_path = OUT_DIR / "00_zero_volume_days.csv"
    zero_volume_days.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()