from __future__ import annotations

import pandas as pd


SOURCE = "VCI"


def fetch_history(
    code: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    from vnstock import Quote

    quote = Quote(source=SOURCE, symbol=code)
    df = quote.history(start=start_date, end=end_date, interval=interval)

    if df is None or df.empty:
        raise ValueError(
            f"VCI returned no data for {code} in [{start_date}, {end_date}]"
        )

    return df