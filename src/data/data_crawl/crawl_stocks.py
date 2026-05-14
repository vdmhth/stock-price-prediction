from __future__ import annotations

import code
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


try:
    from .config import END_DATE, RAW_STOCK_DIR, START_DATE, CODE
    from .config import INTERVAL as CONFIG_INTERVAL
    from .utils import DEFAULT_SOURCE, _filter_by_date_range, _normalize_interval
    from .vnstock_client import (
        _history_with_current_vnstock,
        _history_with_legacy_vnstock,
    )
except ImportError:
    from config import END_DATE, RAW_STOCK_DIR, START_DATE, CODE
    from config import INTERVAL as CONFIG_INTERVAL
    from utils import DEFAULT_SOURCE, _filter_by_date_range, _normalize_interval
    from vnstock_client import (
        _history_with_current_vnstock,
        _history_with_legacy_vnstock,
    )


def crawl_stock_price(
    code: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """Download OHLCV price history for one code from vnstock."""
    code = code.upper().strip()
    interval = _normalize_interval(interval)

    try:
        df = _history_with_current_vnstock(
            code=code,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source=source,
        )
    except (ImportError, AttributeError, TypeError):
        df = _history_with_legacy_vnstock(
            code=code,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

    if df is None or df.empty:
        raise ValueError(f"No stock price data returned for {code}")

    df = df.copy()
    df = _filter_by_date_range(
        df=df,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise ValueError(f"No stock price data in date range for {code}")

    df.insert(0, "code", code)
    return df


def save_stock_price(
    code: str,
    output_dir: str | Path = RAW_STOCK_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
) -> Path:
    """Download one code and save it as CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = crawl_stock_price(
        code=code,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        source=source,
    )

    file_path = output_path / f"{code.upper()}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    return file_path


def crawl_all_stock_prices(
    codes: Iterable[str] = CODE,
    output_dir: str | Path = RAW_STOCK_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = CONFIG_INTERVAL,
    source: str = DEFAULT_SOURCE,
    sleep_seconds: float = 0.5,
) -> dict[str, Path]:
    """Download and save stock prices for all configured codes."""
    saved_files: dict[str, Path] = {}

    for code in codes:
        file_path = save_stock_price(
            code=code,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source=source,
        )

        saved_files[code.upper()] = file_path
        print(f"Saved {code.upper()} -> {file_path}")
        time.sleep(sleep_seconds)

    return saved_files