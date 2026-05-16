"""One-row-per-symbol executive summary table.

n, pct_missing             : sanity
mean, std, std_annual      : level/scale
sharpe_like                : mean/std * sqrt(252)
skew, kurt_excess          : shape
q01, q05, q95, q99         : tail / historical VaR
max_drawdown               : worst peak-to-trough on the underlying price
pct_zero                   : liquidity flag
n_zero                     : number of zero-return observations
pct_zero_no_volume         : % zero-return observations with volume = 0
n_zero_no_volume           : count of zero-return observations with volume = 0
pct_zero_with_volume       : % zero-return observations with volume > 0
n_zero_with_volume         : count of zero-return observations with volume > 0
pct_limit_up, pct_limit_down : HOSE 7%-cap incidence
jb_pval                    : normality (small ⇒ non-normal)
t_df                       : Student-t MLE df (3-5 ⇒ fat tails)
adf_pval, kpss_c_pval      : stationarity (return should be stationary)
lb_10_pval                 : autocorr in r        (large ⇒ white noise)
lb_abs_10_pval             : autocorr in |r|      (small ⇒ vol clustering)
arch_pval                  : ARCH-LM              (small ⇒ heteroscedasticity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .autocorr import arch_lm, ljung_box, ljung_box_abs
from .constants import TRADING_DAYS_PER_YEAR
from .distribution import fit_student_t, jarque_bera
from .stationarity import adf_test, kpss_test
from .stats import max_drawdown
from .volatility import liquidity_stats

def zero_return_counts(
    returns: pd.Series,
    volume: pd.Series | None = None,
    zero_tol: float = 1e-12,
) -> dict:
    """Count zero-return observations, split by volume.

    pct_* values are computed over valid non-missing return observations.
    """

    r = pd.to_numeric(returns, errors="coerce")
    valid_mask = r.notna()
    zero_mask = valid_mask & (r.abs() <= zero_tol)

    n_valid = int(valid_mask.sum())
    n_zero = int(zero_mask.sum())

    out = {
        "n_zero": n_zero,
        "pct_zero": float(n_zero / n_valid * 100) if n_valid > 0 else np.nan,
    }

    if volume is None:
        out.update(
            {
                "n_zero_no_volume": np.nan,
                "pct_zero_no_volume": np.nan,
                "n_zero_with_volume": np.nan,
                "pct_zero_with_volume": np.nan,
            }
        )
        return out

    v = pd.to_numeric(volume, errors="coerce")

    zero_no_volume = zero_mask & (v == 0)
    zero_with_volume = zero_mask & (v > 0)

    n_zero_no_volume = int(zero_no_volume.sum())
    n_zero_with_volume = int(zero_with_volume.sum())

    out.update(
        {
            "n_zero_no_volume": n_zero_no_volume,
            "pct_zero_no_volume": float(n_zero_no_volume / n_valid * 100)
            if n_valid > 0
            else np.nan,
            "n_zero_with_volume": n_zero_with_volume,
            "pct_zero_with_volume": float(n_zero_with_volume / n_valid * 100)
            if n_valid > 0
            else np.nan,
        }
    )

    return out
def build_summary_row(
    stock_code: str,
    price: pd.Series,
    returns: pd.Series,
    volume: pd.Series | None = None,
) -> dict:
    """Build a single summary row for one stock code."""
    r = returns.dropna()
    p = price.dropna()
    mu, sd = r.mean(), r.std()

    row = {
        "stock_code": stock_code,
        "n": int(r.count()),
        "pct_missing": float(returns.isna().mean() * 100),
        "mean": float(mu),
        "std": float(sd),
        "std_annual": float(sd * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_like": float((mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if sd > 0
        else np.nan,
        "skew": float(r.skew()),
        "kurt_excess": float(r.kurt()),
        "q01": float(r.quantile(0.01)),
        "q05": float(r.quantile(0.05)),
        "q95": float(r.quantile(0.95)),
        "q99": float(r.quantile(0.99)),
        "max_drawdown": max_drawdown(p),
    }

    row.update(liquidity_stats(returns, volume=volume))
    row.update(zero_return_counts(returns, volume=volume))
    # --- distribution ---
    jb = jarque_bera(r)
    tfit = fit_student_t(r)
    row["jb_pval"] = jb["jb_pval"]
    row["t_df"] = tfit["t_df"]

    # --- stationarity ---
    adf = adf_test(r)
    kp = kpss_test(r, regression="c")
    row["adf_pval"] = adf["adf_pval"]
    row["kpss_c_pval"] = kp["kpss_c_pval"]

    # --- autocorr / vol clustering ---
    lb = ljung_box(r, lags=[10])
    lb_abs = ljung_box_abs(r, lags=[10])
    arch = arch_lm(r, lags=10)
    row["lb_10_pval"] = lb["lb_10_pval"]
    row["lb_abs_10_pval"] = lb_abs["lb_abs_10_pval"]
    row["arch_pval"] = arch["arch_pval"]

    return row

def build_summary(
    wide: pd.DataFrame,
    stock_codes: list[str],
    has_volume: bool = True,
) -> pd.DataFrame:
    """Build the full summary table for a list of symbols.

    Expects wide to contain `{sym}_close`, `{sym}_log_return`, and optionally
    `{sym}_volume` columns.
    """
    rows = []
    for sym in stock_codes:
        price_col = f"{sym}_close"
        ret_col = f"{sym}_log_return"
        vol_col = f"{sym}_volume"
        if price_col not in wide or ret_col not in wide:
            continue
        volume = wide[vol_col] if has_volume and vol_col in wide else None
        rows.append(
            build_summary_row(
                sym, wide[price_col], wide[ret_col], volume=volume
            )
        )
    return pd.DataFrame(rows).set_index("stock_code")
