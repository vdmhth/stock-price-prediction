"""Stationarity tests: ADF, KPSS (both 'c' and 'ct'), Zivot-Andrews.

ADF H0: unit root (non-stationary). Reject (p < 0.05) ⇒ stationary.
KPSS H0: stationary. Reject (p < 0.05) ⇒ non-stationary.

Joint logic with KPSS regression='c' (level stationarity):
    ADF stat,   KPSS stat   ⇒  stationary
    ADF nstat,  KPSS nstat  ⇒  non-stationary (likely I(1))
    ADF stat,   KPSS nstat  ⇒  difference-stationary
                                (or structural break — verify with ZA)
    ADF nstat,  KPSS stat   ⇒  trend-stationary
                                (re-run KPSS with regression='ct' to confirm)

KPSS p-values from statsmodels are clamped to [0.01, 0.10] (lookup table).
Treat the boundary values as 'p ≤ 0.01' or 'p ≥ 0.10', not exact.
"""
from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(s: pd.Series) -> dict:
    s = s.dropna()
    stat, pval, lags, _, crit, _ = adfuller(s, autolag="AIC")
    return {
        "adf_stat": float(stat),
        "adf_pval": float(pval),
        "adf_lags": int(lags),
        "adf_crit_5pct": float(crit["5%"]),
        "adf_stationary": bool(pval < 0.05),
    }


def kpss_test(s: pd.Series, regression: str = "c") -> dict:
    """KPSS test. regression='c' for level, 'ct' for trend stationarity."""
    s = s.dropna()
    with warnings.catch_warnings():
        # Suppress InterpolationWarning when p-value is at the table boundary.
        warnings.filterwarnings("ignore")
        stat, pval, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {
        f"kpss_{regression}_stat": float(stat),
        f"kpss_{regression}_pval": float(pval),
        f"kpss_{regression}_lags": int(lags),
        f"kpss_{regression}_crit_5pct": float(crit["5%"]),
        f"kpss_{regression}_stationary": bool(pval > 0.05),
    }


_VERDICT_MAP = {
    (True, True): "stationary",
    (False, False): "non-stationary",
    (True, False): "difference-stationary",
    (False, True): "trend-stationary",
}


def stationarity_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Run ADF + KPSS('c') + KPSS('ct') for each column.

    If the level-KPSS rejects but trend-KPSS does not, the series is more
    naturally described as trend-stationary than as integrated.
    """
    rows = []
    for c in columns:
        adf = adf_test(df[c])
        kp_c = kpss_test(df[c], regression="c")
        kp_ct = kpss_test(df[c], regression="ct")
        verdict = _VERDICT_MAP[
            (adf["adf_stationary"], kp_c["kpss_c_stationary"])
        ]
        # If level-KPSS rejects but trend-KPSS does not, prefer trend label.
        if (
            verdict == "difference-stationary"
            and kp_ct["kpss_ct_stationary"]
        ):
            verdict = "trend-stationary (level-KPSS rejected)"
        rows.append({"series": c, **adf, **kp_c, **kp_ct, "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


def zivot_andrews(s: pd.Series, regression: str = "c") -> dict:
    """ADF allowing one endogenous structural break.

    Useful when ADF and KPSS disagree (the difference-stationary case)
    and you suspect a regime shift rather than non-stationarity.

    H0: unit root with no break. Reject (p < 0.05) ⇒ trend-stationary with
    a break at the estimated break date.
    """
    from statsmodels.tsa.stattools import zivot_andrews as za

    s = s.dropna()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # API: returns (stat, pvalue, crit, bpidx, lags) on modern versions.
        res = za(s.values, regression=regression, autolag="AIC")
    stat, pval = float(res[0]), float(res[1])
    bp_idx = int(res[3])
    return {
        "za_stat": stat,
        "za_pval": pval,
        "za_break_date": str(s.index[bp_idx]) if bp_idx < len(s) else None,
        "za_stationary_with_break": bool(pval < 0.05),
    }
