"""Distribution diagnosis: normality test and Student-t fit"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps

def jarque_bera(s:pd.Series) ->dict:
    """Jarque-Bera test for normality based on skew+kurtosis.

    H0: data is normal.
    p < 0.05 => reject normality.
    """
    s = s.dropna()
    stat,pval = sps.jarque_bera(s)
    return {
        "jb_stat": float(stat),
        'jb_pval': float(pval),
        'jb_normal': pval >= 0.05,
    }
def fit_student_t(s:pd.Series) ->dict:
    """Fit a Student-t distribution by MLE.

    Returns the degrees of freedom; smaller df ⇒ fatter tails.
    Typical equity returns fit df in [3, 6]."""
    s = s.dropna().values
    df,loc,scale= sps.t.fit(s)
    ll_t = float(np.sum(sps.t.logpdf(s, df, loc=loc, scale=scale)))
    nloc,nscale = float(np.mean(s)), float(np.std(s, ddof=1))
    ll_n = float(np.sum(sps.norm.logpdf(s, loc=nloc, scale=nscale)))
    return {
        't_df':float(df),
        't_loc':float(loc),
        't_scale':float(scale),
        'loglik_t':ll_t,
        'loglik_normal':ll_n,
        'loglik_diff':ll_t - ll_n,  
    }

def distribution_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        s = df[c]
        rows.append({"series": c, **jarque_bera(s), **fit_student_t(s)})
    return pd.DataFrame(rows).set_index("series")
