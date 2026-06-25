"""
Correlation analysis: Pearson, Spearman, partial, rolling.

"""
from __future__ import annotations

import pandas as pd
import numpy as np

def pearson_corr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(method="pearson", min_periods=30)


def spearman_corr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(method="spearman", min_periods=30)
def partial_corr(df:pd.DataFrame, x:str,y:str, given: str)-> float:
    """
            rho_{xy|z} = (rho_xy - rho_xz * rho_yz) /
                     sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    """
    sub = df [[x,y,given]].dropna()
    rho = sub.corr().to_numpy()
    r_xy, r_xz, r_yz = rho[0,1], rho[0,2], rho[1,2]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom==0:
        return np.nan
    return float((r_xy - r_xz*r_yz) / denom)

def partial_corr_matrix(df:pd.DataFrame,columns:list[str],given:str) ->pd.DataFrame:
    out = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            if a == b:
                out.loc[a, b] = 1.0
            elif j > i:
                v = partial_corr(df, a, b, given)
                out.loc[a, b] = v
                out.loc[b, a] = v
    return out


def rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    return s1.rolling(window).corr(s2)


def pairwise_rolling(df: pd.DataFrame, base: str, others: list[str],
                     window: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {c: rolling_corr(df[base], df[c], window) for c in others},
        index=df.index,
    )
