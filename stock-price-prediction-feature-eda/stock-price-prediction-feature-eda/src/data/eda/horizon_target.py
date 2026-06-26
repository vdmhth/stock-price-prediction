from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .constants import STOCK_CODES, TRADING_DAYS_PER_YEAR

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def forward_cum_return(log_return: pd.Series, h: int) -> pd.Series:
    s = pd.to_numeric(log_return, errors="coerce")
    # rolling sum of the *next* h returns, aligned to time t
    fwd = s.shift(-1).rolling(window=h, min_periods=h).sum().shift(-(h - 1))
    fwd.name = f"fwd_cum_ret_{h}"
    return fwd


def _train_slice(s: pd.Series, train_frac: float) -> pd.Series:
    s = s.dropna()
    n_train = int(len(s) * train_frac)
    return s.iloc[:n_train]


def _nonoverlap_blocks(log_return: pd.Series, h: int) -> pd.Series:
    s = pd.to_numeric(log_return, errors="coerce").dropna()
    n = (len(s) // h) * h
    s = s.iloc[:n]
    blocks = s.values.reshape(-1, h).sum(axis=1)
    return pd.Series(blocks)

def target_distribution(log_return: pd.Series, h: int) -> pd.Series:
    """Descriptive stats of the forward cumulative log-return at horizon `h`."""
    t = forward_cum_return(log_return, h).dropna()
    return pd.Series(
        {
            "horizon": h,
            "count": int(t.count()),
            "mean": t.mean(),
            "std": t.std(),
            "skew": t.skew(),
            "kurtosis_excess": sps.kurtosis(t),
            "q01": t.quantile(0.01),
            "q05": t.quantile(0.05),
            "q50": t.quantile(0.50),
            "q95": t.quantile(0.95),
            "q99": t.quantile(0.99),
            "min": t.min(),
            "max": t.max(),
            "pct_positive": (t > 0).mean(),
        }
    )

def variance_ratio(log_return: pd.Series, h: int) -> tuple[float, float]:
    r = pd.to_numeric(log_return, errors="coerce").dropna().values
    n = len(r)
    if n <= h + 1:
        return np.nan, np.nan

    mu = r.mean()
    # 1-step variance (unbiased)
    var_1 = np.sum((r - mu) ** 2) / (n - 1)
    if var_1 == 0:
        return np.nan, np.nan

    rolled = np.convolve(r, np.ones(h), mode="valid")  # overlapping h-sums
    m = h * (n - h + 1) * (1 - h / n)
    var_h = np.sum((rolled - h * mu) ** 2) / m
    vr = var_h / var_1

    # heteroskedasticity-robust standard error (Lo & MacKinlay 1988)
    delta = 0.0
    for j in range(1, h):
        num = np.sum(((r[j:] - mu) ** 2) * ((r[:-j] - mu) ** 2))
        den = (np.sum((r - mu) ** 2)) ** 2
        d_j = num / den
        delta += (2.0 * (h - j) / h) ** 2 * d_j
    se = np.sqrt(delta) if delta > 0 else np.nan
    z = (vr - 1.0) / se if se and se > 0 else np.nan
    return float(vr), float(z)


def target_persistence(log_return: pd.Series, h: int) -> pd.Series:
    blocks = _nonoverlap_blocks(log_return, h)
    ac1 = blocks.autocorr(1) if len(blocks) > 2 else np.nan
    vr, z = variance_ratio(log_return, h)
    return pd.Series(
        {
            "horizon": h,
            "n_nonoverlap_blocks": int(len(blocks)),
            "autocorr_lag1_nonoverlap": ac1,
            "variance_ratio": vr,
            "vr_zstat": z,
            "rw_rejected_5pct": (abs(z) > 1.96) if pd.notna(z) else np.nan,
        }
    )

def naive_baseline_r2(log_return: pd.Series, h: int) -> pd.Series:
    blocks = _nonoverlap_blocks(log_return, h)
    if len(blocks) < 10:
        return pd.Series({"horizon": h, "r2_drift": np.nan, "r2_persistence": np.nan})

    n_train = int(len(blocks) * 0.70)
    train, test = blocks.iloc[:n_train], blocks.iloc[n_train:]
    sst = float(np.sum((test - test.mean()) ** 2))
    if sst == 0:
        return pd.Series({"horizon": h, "r2_drift": np.nan, "r2_persistence": np.nan})

    # drift: predict train mean
    pred_drift = train.mean()
    sse_drift = float(np.sum((test - pred_drift) ** 2))
    r2_drift = 1.0 - sse_drift / sst

    # persistence: predict previous block value
    prev = blocks.shift(1).iloc[n_train:]
    mask = prev.notna()
    sse_pers = float(np.sum((test[mask] - prev[mask]) ** 2))
    sst_pers = float(np.sum((test[mask] - test[mask].mean()) ** 2))
    r2_pers = 1.0 - sse_pers / sst_pers if sst_pers > 0 else np.nan

    return pd.Series(
        {"horizon": h, "r2_drift": r2_drift, "r2_persistence": r2_pers}
    )


def horizon_target_panel(
    wide: pd.DataFrame,
    stock_codes: list[str] = STOCK_CODES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    train_frac: float = 0.70,
) -> pd.DataFrame:
    rows = []
    for sym in stock_codes:
        col = f"{sym}_log_return"
        if col not in wide.columns:
            continue
        r_full = wide[col]
        r_train = _train_slice(r_full, train_frac)
        for h in horizons:
            dist = target_distribution(r_train, h)
            pers = target_persistence(r_train, h)
            base = naive_baseline_r2(r_train, h)
            row = {"symbol": sym}
            row.update(dist.to_dict())
            row.update({k: v for k, v in pers.to_dict().items() if k != "horizon"})
            row.update({k: v for k, v in base.to_dict().items() if k != "horizon"})
            rows.append(row)
    out = pd.DataFrame(rows)
    cols = ["symbol", "horizon"] + [c for c in out.columns if c not in {"symbol", "horizon"}]
    return out[cols]