from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import STOCK_CODES, TRADING_DAYS_PER_YEAR
from .horizon_target import DEFAULT_HORIZONS, forward_cum_return

try:  # mutual information is optional
    from sklearn.feature_selection import mutual_info_regression

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

def build_candidate_features(
    wide: pd.DataFrame,
    sym: str,
    market_col: str = "VNINDEX_log_return",
) -> pd.DataFrame:
    """Construct backward-looking candidate features for one stock.

    Returns a DataFrame indexed like `wide`. Every column is computed from data
    up to and including time t, so pairing it with a forward target is leak-free.
    """
    r = pd.to_numeric(wide[f"{sym}_log_return"], errors="coerce")
    feats = pd.DataFrame(index=wide.index)

    # --- own return momentum (backward sums) ---
    feats["ret_lag1"] = r.shift(1)
    feats["ret_sum5"] = r.shift(1).rolling(5, min_periods=5).sum()
    feats["ret_sum20"] = r.shift(1).rolling(20, min_periods=20).sum()

    # --- recent realised volatility (the structure the EDA found persistent) ---
    feats["abs_ret_lag1"] = r.shift(1).abs()
    feats["vol_20"] = r.shift(1).rolling(20, min_periods=20).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    feats["vol_60"] = r.shift(1).rolling(60, min_periods=45).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    # vol regime: current 20d vol relative to its own 1y average
    vol20_raw = r.shift(1).rolling(20, min_periods=20).std()
    feats["vol_z"] = (
        vol20_raw - vol20_raw.rolling(252, min_periods=120).mean()
    ) / vol20_raw.rolling(252, min_periods=120).std()

    # --- intraday range proxy, if present ---
    range_col = f"{sym}_range_pct"
    if range_col in wide.columns:
        rng = pd.to_numeric(wide[range_col], errors="coerce")
        feats["range_pct_lag1"] = rng.shift(1)
        feats["range_pct_mean5"] = rng.shift(1).rolling(5, min_periods=5).mean()

    # --- market-level features (per the partial-correlation finding) ---
    if market_col in wide.columns:
        m = pd.to_numeric(wide[market_col], errors="coerce")
        feats["mkt_ret_lag1"] = m.shift(1)
        feats["mkt_ret_sum5"] = m.shift(1).rolling(5, min_periods=5).sum()
        feats["mkt_vol_60"] = m.shift(1).rolling(60, min_periods=45).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    return feats


def screen_features(
    wide: pd.DataFrame,
    sym: str,
    h: int,
    market_col: str = "VNINDEX_log_return",
    train_frac: float = 0.70,
    add_mutual_info: bool = True,
) -> pd.DataFrame:
    """Associate each candidate feature with the h-step forward target.

    Reports Pearson r, Spearman rho and (optionally) mutual information, on the
    training slice only. Rows are sorted by |Spearman|, which is robust to the
    heavy tails documented in the EDA.
    """
    feats = build_candidate_features(wide, sym, market_col=market_col)
    target = forward_cum_return(wide[f"{sym}_log_return"], h)

    data = feats.copy()
    data["__target__"] = target
    data = data.dropna()

    # in-sample slice only (time-ordered, no shuffle)
    n_train = int(len(data) * train_frac)
    data = data.iloc[:n_train]
    y = data["__target__"]
    X = data.drop(columns="__target__")

    if len(data) < 50 or X.shape[1] == 0:
        return pd.DataFrame()

    rows = []
    if add_mutual_info and _HAS_SKLEARN:
        mi = mutual_info_regression(X.values, y.values, random_state=0)
        mi_map = dict(zip(X.columns, mi))
    else:
        mi_map = {}

    for c in X.columns:
        pear = X[c].corr(y, method="pearson")
        spear = X[c].corr(y, method="spearman")
        rows.append(
            {
                "symbol": sym,
                "horizon": h,
                "feature": c,
                "pearson": pear,
                "spearman": spear,
                "abs_spearman": abs(spear) if pd.notna(spear) else np.nan,
                "mutual_info": mi_map.get(c, np.nan),
                "n_obs": int(len(data)),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    return out.reset_index(drop=True)

def feature_target_panel(
    wide: pd.DataFrame,
    stock_codes: list[str] = STOCK_CODES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    market_col: str = "VNINDEX_log_return",
    train_frac: float = 0.70,
    add_mutual_info: bool = True,
) -> pd.DataFrame:
    """Long-format screening table for every (stock, horizon, feature)."""
    parts = []
    for sym in stock_codes:
        if f"{sym}_log_return" not in wide.columns:
            continue
        for h in horizons:
            part = screen_features(
                wide,
                sym,
                h,
                market_col=market_col,
                train_frac=train_frac,
                add_mutual_info=add_mutual_info,
            )
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def best_feature_per_target(panel: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Compact view: top-`k` features (by |Spearman|) for each stock x horizon."""
    if panel.empty:
        return panel
    return (
        panel.sort_values("abs_spearman", ascending=False)
        .groupby(["symbol", "horizon"], sort=False)
        .head(top_k)
        .sort_values(["symbol", "horizon", "abs_spearman"], ascending=[True, True, False])
        .reset_index(drop=True)
    )