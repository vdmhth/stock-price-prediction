# acf/pacf computation (return raw values and confidence bands)
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def compute_acf(series:pd.Series,nlags: int = 40) ->tuple[np.ndarray,np.ndarray]:
    s = series.dropna()
    values,confint = acf(s,nlags = nlags,alpha = 0.05,fft = True)
    return values, confint
def compute_pacf(series:pd.Series,nlags: int = 40) ->tuple[np.ndarray, np.ndarray]:
    s = series.dropna()
    values, confint = pacf(s,nlags = nlags, alpha =0.05,method = 'ywm')
    return values, confint
def acf_table(series:pd.Series,nlags :int = 20)->pd.DataFrame:
    acf_vals , _ = compute_acf(series,nlags=nlags)
    pacf_vals, _ = compute_pacf(series,nlags=nlags)
    return pd.DataFrame(
        {
            'lag':np.arange(nlags+1),
            'acf':acf_vals,
            'pacf':pacf_vals,
        }
    ).set_index('lag')