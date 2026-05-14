from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .config import END_DATE, MARKET_INDICES, RAW_MARKET_DIR, START_DATE
    from .config import INTERVAL as CONFIG_INTERVAL
    from .utils import DEFAULT_SOURCE, _filter_by_date_range, _normalize_interval
    from .vnstock_client import _history_with_current_vnstock
except ImportError:
    from config import END_DATE, MARKET_INDICES, RAW_MARKET_DIR, START_DATE
    from config import INTERVAL as CONFIG_INTERVAL
    from utils import DEFAULT_SOURCE, _filter_by_date_range, _normalize_interval
    from vnstock_client import _history_with_current_vnstock


def crawl_market_index(
    index_symbol: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """Download OHLCV history for one market index from vnstock."""
    index_symbol = index_symbol.upper().strip()
    interval = _normalize_interval(interval)

    df = _history_with_current_vnstock(
        code=index_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        source=source,
    )

    if df is None or df.empty:
        raise ValueError(f"No market index data returned for {index_symbol}")

    df = df.copy()
    df = _filter_by_date_range(
        df=df,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise ValueError(f"No market index data in date range for {index_symbol}")

    df.insert(0, "index", index_symbol)
    return df


def save_market_index(
    index_symbol: str,
    output_dir: str | Path = RAW_MARKET_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
) -> Path:
    """Download one market index and save it as CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = crawl_market_index(
        index_symbol=index_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        source=source,
    )

    file_path = output_path / f"{index_symbol.upper()}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    return file_path


def crawl_all_market_indices(
    indices: Iterable[str] = MARKET_INDICES,
    output_dir: str | Path = RAW_MARKET_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
    sleep_seconds: float = 0.5,
) -> dict[str, Path]:
    """Download and save configured market indices."""
    saved_files: dict[str, Path] = {}

    for index_symbol in indices:
        file_path = save_market_index(
            index_symbol=index_symbol,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source=source,
        )

        saved_files[index_symbol.upper()] = file_path
        print(f"Saved {index_symbol.upper()} -> {file_path}")
        time.sleep(sleep_seconds)

    return saved_files