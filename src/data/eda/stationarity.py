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
        "adf_pval_at_bound": bool(pval <= 0.001 or pval >= 0.99), 
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
        f"kpss_{regression}_pval_at_bound": bool(pval <= 0.01 or pval >= 0.10), 
        f"kpss_{regression}_crit_5pct": float(crit["5%"]),
        f"kpss_{regression}_stationary": bool(pval > 0.05),
    }


_VERDICT_MAP = {
    (True, True): "stationary",
    (False, False): "non-stationary",
    (True, False): "difference-stationary",
    (False, True): "trend-stationary",
}

def _fmt_p(pval: float, at_bound: bool, kind: str) -> str:

    if at_bound:
        if kind == "kpss":
            return "<0.01" if pval <= 0.01 else ">0.10"
        return ">0.99" if pval >= 0.99 else "<0.001"
    return f"{pval:.4f}"
def stationarity_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:

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
def zivot_andrews_summary(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:

    rows = []
    for c in columns:
        za_c = zivot_andrews(df[c], regression="c")
        za_ct = zivot_andrews(df[c], regression="ct")
        rows.append(
            {
                "series": c,
                "za_c_stat": za_c["za_stat"],
                "za_c_pval": za_c["za_pval"],
                "za_c_break_date": za_c["za_break_date"],
                "za_c_break_with_level_shift": za_c["za_stationary_with_break"],
                "za_ct_stat": za_ct["za_stat"],
                "za_ct_pval": za_ct["za_pval"],
                "za_ct_break_date": za_ct["za_break_date"],
                "za_ct_break_with_trend": za_ct["za_stationary_with_break"],
            }
        )
    return pd.DataFrame(rows).set_index("series")
def merged_stationarity_table(
    df: pd.DataFrame, symbols: list[str]
) -> pd.DataFrame:
    sc = stationarity_summary(df, [f"{s}_close" for s in symbols])
    sr = stationarity_summary(df, [f"{s}_log_return" for s in symbols])

    rows = []
    for s in symbols:
        c = sc.loc[f"{s}_close"]
        r = sr.loc[f"{s}_log_return"]
        rows.append({
            "Symbol": s,
            "ADF p (close)":   _fmt_p(c["adf_pval"],   c["adf_pval_at_bound"],   "adf"),
            "KPSS p (close)":  _fmt_p(c["kpss_c_pval"], c["kpss_c_pval_at_bound"], "kpss"),
            "Verdict (close)": c["verdict"],
            "ADF p (return)":   _fmt_p(r["adf_pval"],   r["adf_pval_at_bound"],   "adf"),
            "KPSS p (return)":  _fmt_p(r["kpss_c_pval"], r["kpss_c_pval_at_bound"], "kpss"),
            "Verdict (return)": r["verdict"],
        })
    return pd.DataFrame(rows).set_index("Symbol")