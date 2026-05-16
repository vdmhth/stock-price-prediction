''' Rooling volatility and liquidity diagnosis'''
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import HOSE_DAILY_LIMIT_LOG, TRADING_DAYS_PER_YEAR, ZERO_RETURN_TOL

def rolling_vol_annualized(returns:pd.Series, window: int =60) ->pd.Series:
    # rolling standard deviation of returns, annualizezd by sqrt(252)
    return returns.rolling(window).std()*np.sqrt(TRADING_DAYS_PER_YEAR)
def rolling_vol_panel(df:pd.DataFrame,return_cols: list[str],window: int =60) ->pd.DataFrame:
    return pd.DataFrame(
        {c: rolling_vol_annualized(df[c],window) for c in return_cols}, index = df.index
    )
def liquidity_stats(returns: pd.Series, volume: pd.Series | None= None):
    """Diagnostics on data quality/liquidity of a return series
    pct_zero -> % of days with log_return ~= 0
    pct_limit_up % of days hitting the +7% HOSE cap (within 1 bp)
    pct _limit_down % of days hitting the -7% HOSE cap 
    pct_zero_no_volume % of zero_return days that also have volume = 0
    """
    r = returns.dropna()
    n = len(r)
    if  n ==0 :
        return {}
    zero = r.abs() < ZERO_RETURN_TOL
    limit_up = r > (HOSE_DAILY_LIMIT_LOG - 1e-4)
    limit_down  = r < - (HOSE_DAILY_LIMIT_LOG - 1e-4)
    out = {
        'n': int(n),
        'pct_zero': float(zero.mean()*100),
        'pct_limit_up': float(limit_up.mean()*100),
        'pct_limit_down': float(limit_down.mean()*100),
    }
    if volume is not None:
        v = volume.reindex(r.index)
        out['pct_zero_no_volume'] = float(
            (zero&(v.fillna(0)==0)).mean()*100
        )
    return out
