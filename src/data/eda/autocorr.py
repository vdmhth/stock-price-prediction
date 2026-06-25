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
        lags = [5, 20]

    s = s.dropna()
    res = acorr_ljungbox(s, lags=lags, return_df=True)

    out = {}
    for lag in lags:
        out[f"lb_{lag}_stat"] = float(res.loc[lag, "lb_stat"])
        out[f"lb_{lag}_pval"] = float(res.loc[lag, "lb_pvalue"])

    return out

def ljung_box_abs(s: pd.Series, lags: list[int] | None = None) -> dict:
    if lags is None:
        lags = [5, 20]

    s = s.dropna().abs()
    res = acorr_ljungbox(s, lags=lags, return_df=True)

    out = {}
    for lag in lags:
        out[f"lb_abs_{lag}_stat"] = float(res.loc[lag, "lb_stat"])
        out[f"lb_abs_{lag}_pval"] = float(res.loc[lag, "lb_pvalue"])

    return out

def arch_lm(s: pd.Series, lags: int = 10) -> dict:
    s = s.dropna()
    stat, pval, _, _ = het_arch(s, nlags=lags)
    return {
        "arch_lags": int(lags),
        "arch_stat": float(stat),
        "arch_pval": float(pval),
        "arch_effect": bool(pval < 0.05),
    }


def arch_lm_multi(s: pd.Series, lags: list[int] | None = None) -> dict:
    if lags is None:
        lags = [5, 20]

    out = {}

    for lag in lags:
        res = arch_lm(s, lags=lag)

        out[f"arch_{lag}_stat"] = res["arch_stat"]
        out[f"arch_{lag}_pval"] = res["arch_pval"]
        out[f"arch_{lag}_effect"] = res["arch_effect"]

    return out
def autocorr_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []

    diagnostic_lags = [5, 20]

    for c in columns:
        s = df[c]
        rows.append(
            {
                "series": c,
                **ljung_box(s, lags=diagnostic_lags),
                **ljung_box_abs(s, lags=diagnostic_lags),
                **arch_lm_multi(s, lags=diagnostic_lags),
            }
        )

    return pd.DataFrame(rows).set_index("series")
