"""Correlation analysis (Pearson, rolling)."""

from __future__ import annotations

import pandas as pd


def pearson_corr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix for given columns (drops rows with NaN)."""
    return df[columns].dropna().corr(method="pearson")


def rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return s1.rolling(window).corr(s2)


def pairwise_rolling(df: pd.DataFrame, base: str, others: list[str],
                     window: int = 60) -> pd.DataFrame:
    """Rolling correlation of `base` vs each column in `others`."""
    return pd.DataFrame(
        {c: rolling_corr(df[base], df[c], window) for c in others},
        index=df.index,
    )
