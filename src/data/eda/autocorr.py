"""Autocorrelation, Ljung-Box, ARCH-LM.

Reading the three together
--------------------------
- ACF/PACF: visual diagnostic. Flat ⇒ no linear predictability.
- Ljung-Box on returns: H0 = white noise. p > 0.05 ⇒ returns are white
  noise (weak-form market efficiency consistent).
- Ljung-Box on |returns| or returns**2: H0 = no autocorrelation in the
  variance proxy. p < 0.05 ⇒ volatility clustering exists ⇒ GARCH justified.
- Engle ARCH-LM: formal test that conditional heteroscedasticity is
  present. p < 0.05 ⇒ ARCH effects ⇒ GARCH justified.

In typical equity data: LB on r fails to reject, LB on |r| and ARCH-LM
both reject decisively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf


def compute_acf(s: pd.Series, nlags: int = 40):
    s = s.dropna()
    return acf(s, nlags=nlags, alpha=0.05, fft=True)


def compute_pacf(s: pd.Series, nlags: int = 40):
    s = s.dropna()
    return pacf(s, nlags=nlags, alpha=0.05, method="ywm")


def acf_table(s: pd.Series, nlags: int = 20) -> pd.DataFrame:
    """Tabulate ACF and PACF values for ARIMA order selection."""
    acf_vals, _ = compute_acf(s, nlags=nlags)
    pacf_vals, _ = compute_pacf(s, nlags=nlags)
    return pd.DataFrame(
        {"lag": np.arange(nlags + 1), "acf": acf_vals, "pacf": pacf_vals}
    ).set_index("lag")


def ljung_box(s: pd.Series, lags: list[int] | None = None) -> dict:
    """Ljung-Box on the raw series. H0: no autocorrelation up to `lag`."""
    if lags is None:
        lags = [5, 10, 20]
    s = s.dropna()
    res = acorr_ljungbox(s, lags=lags, return_df=True)
    out = {}
    for lag in lags:
        out[f"lb_{lag}_stat"] = float(res.loc[lag, "lb_stat"])
        out[f"lb_{lag}_pval"] = float(res.loc[lag, "lb_pvalue"])
    return out


def ljung_box_abs(s: pd.Series, lags: list[int] | None = None) -> dict:
    """Ljung-Box on |returns| — tests volatility clustering.

    H0: no autocorrelation in |r| up to `lag`.
    Reject ⇒ volatility clustering ⇒ GARCH justified.
    """
    if lags is None:
        lags = [5, 10, 20]
    s = s.dropna().abs()
    res = acorr_ljungbox(s, lags=lags, return_df=True)
    out = {}
    for lag in lags:
        out[f"lb_abs_{lag}_stat"] = float(res.loc[lag, "lb_stat"])
        out[f"lb_abs_{lag}_pval"] = float(res.loc[lag, "lb_pvalue"])
    return out


def arch_lm(s: pd.Series, lags: int = 10) -> dict:
    """Engle's ARCH-LM test for heteroscedasticity.

    H0: no ARCH effect at lags 1..lags.
    Reject ⇒ heteroscedasticity ⇒ GARCH justified.
    """
    s = s.dropna()
    stat, pval, _, _ = het_arch(s, nlags=lags)
    return {
        "arch_lags": int(lags),
        "arch_stat": float(stat),
        "arch_pval": float(pval),
        "arch_effect": bool(pval < 0.05),
    }


def autocorr_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Combined LB on r, LB on |r|, and ARCH-LM for each column."""
    rows = []
    for c in columns:
        s = df[c]
        rows.append(
            {
                "series": c,
                **ljung_box(s),
                **ljung_box_abs(s),
                **arch_lm(s),
            }
        )
    return pd.DataFrame(rows).set_index("series")
