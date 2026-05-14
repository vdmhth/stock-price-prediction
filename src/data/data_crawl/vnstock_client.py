from __future__ import annotations

import pandas as pd


def _history_with_current_vnstock(
    code: str,
    start_date: str,
    end_date: str,
    interval: str,
    source: str,
) -> pd.DataFrame:
    from vnstock import Quote

    quote = Quote(source=source, symbol=code)
    return quote.history(start=start_date, end=end_date, interval=interval)


def _history_with_legacy_vnstock(
    code: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    from vnstock import stock_historical_data

    resolution = "1D" if interval == "1D" else interval

    return stock_historical_data(
        symbol=code,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        type="stock",
        beautify=True,
        decor=False,
    )