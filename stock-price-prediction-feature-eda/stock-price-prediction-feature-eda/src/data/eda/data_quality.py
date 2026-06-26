from __future__ import annotations

from pathlib import Path

import pandas as pd


KEY_QUALITY_COLUMNS = [
    "symbol",
    "n_rows",
    "start_date",
    "end_date",

    "pct_missing_date",
    "pct_duplicate_dates",
    "pct_ohlc_invalid",
    "pct_volume_eq_0",
    "pct_zero_return_rows",
    "pct_missing_log_return_rows",
    "pct_hose_disruption_rows",
]


def load_data_quality(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. "
            "Run `python -m src.data.processing.build_processed` first."
        )

    return pd.read_csv(path)


def build_key_data_quality(
    quality: pd.DataFrame,
) -> pd.DataFrame:
    existing_cols = [c for c in KEY_QUALITY_COLUMNS if c in quality.columns]

    key_quality = quality[existing_cols].copy()

    sort_col = "pct_ohlc_invalid"
    if sort_col in key_quality.columns:
        key_quality = key_quality.sort_values(sort_col, ascending=False)

    return key_quality


def save_key_data_quality(
    quality_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    quality = load_data_quality(quality_path)

    key_quality = build_key_data_quality(quality)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_quality.to_csv(output_path, index=False, encoding="utf-8")

    return key_quality


def print_key_data_quality(
    key_quality: pd.DataFrame,
) -> None:
    display_cols = [
        c for c in KEY_QUALITY_COLUMNS
        if c in key_quality.columns
    ]

    print("\n[Key Data Quality Metrics]")
    print(key_quality[display_cols].to_string(index=False))