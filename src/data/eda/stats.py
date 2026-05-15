'''
Descriptive statistics and EDA utilities for stock and market data.
'''
from __future__ import annotations
import pandas as pd
from scipy import stats

def describe_series(s:pd.Series) ->pd.Series:
    s = s.dropna()
    return pd.Series({
        'count': s.count(),
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
        'skew': s.skew(),
        'kurtosis_excess': stats.kurtosis(s),
        'q01'   : s.quantile(0.01),
        'q99'   : s.quantile(0.99),
    })
def describe_wide(df:pd.DataFrame,columns:list[str]) ->pd.DataFrame:
    return pd.DataFrame({col: describe_series(df[col]) for col in columns}).T