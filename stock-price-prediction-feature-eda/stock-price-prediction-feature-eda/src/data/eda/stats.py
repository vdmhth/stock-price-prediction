from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import TRADING_DAYS_PER_YEAR

def describe_price(s: pd.Series) -> pd.Series:
    """Descriptive stats appropriate for a price level series."""
    s = s.dropna()
    return pd.Series(
        {
            "count": int(s.count()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q25": s.quantile(0.25),
            "q50": s.quantile(0.50),
            "q75": s.quantile(0.75),
            "max": s.max(),
            "skew": s.skew(),
            "kurtosis_excess": sps.kurtosis(s),
        }
    )
def describe_return(s: pd.Series) -> pd.Series:
    """Descriptive stats appropriate for a log-return series.

    Adds annualised mean/std, Sharpe-like ratio and tail quantiles q01/q99
    (1-day historical VaR thresholds).
    """
    s = s.dropna()
    mu = s.mean()
    sd = s.std()
    sharpe_like = (mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else np.nan
    return pd.Series(
        {
            "count": int(s.count()),
            "mean": mu,
            "std": sd,
            "mean_annual": mu * TRADING_DAYS_PER_YEAR,
            "std_annual": sd * np.sqrt(TRADING_DAYS_PER_YEAR),
            "sharpe_like": sharpe_like,
            "min": s.min(),
            "q01": s.quantile(0.01),
            "q05": s.quantile(0.05),
            "q25": s.quantile(0.25),
            "q50": s.quantile(0.50),
            "q75": s.quantile(0.75),
            "q95": s.quantile(0.95),
            "q99": s.quantile(0.99),
            "max": s.max(),
            "skew": s.skew(),
            "kurtosis_excess": sps.kurtosis(s),
        }
    )

def describe_wide(
    df: pd.DataFrame,
    columns: list[str],
    kind: str = "return",
) -> pd.DataFrame:
    """Tabulate descriptive stats for a list of columns.

    Parameters
    ----------
    kind : 'price' or 'return'
    """
    if kind not in {"price", "return"}:
        raise ValueError("kind must be 'price' or 'return'")
    fn = describe_return if kind == "return" else describe_price
    return pd.DataFrame({c: fn(df[c]) for c in columns}).T


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown of a price series (negative number, e.g. -0.4 = -40%)."""
    s = prices.dropna()
    if len(s) == 0:
        return np.nan
    cummax = s.cummax()
    dd = s / cummax - 1.0
    return float(dd.min())
