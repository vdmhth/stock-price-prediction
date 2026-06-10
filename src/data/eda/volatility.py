from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_vol_panel(
    wide: pd.DataFrame,
    return_cols: list[str],
    window: int = 60,
    min_periods: int | None = None,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Compute close-to-close rolling annualized volatility from log returns.

    Formula:
        rolling_vol_t = std(log_return over rolling window) * sqrt(252)

    This is the main volatility measure used in this project.
    It does not depend on Open, High, or Low.
    """
    if min_periods is None:
        min_periods = window

    out = pd.DataFrame(index=wide.index)

    for col in return_cols:
        if col not in wide.columns:
            continue

        name = col.replace("_log_return", f"_return_vol_{window}")
        out[name] = (
            wide[col]
            .rolling(window=window, min_periods=min_periods)
            .std()
            * np.sqrt(annualization_factor)
        )

    return out


def rolling_vol_multi_window(
    wide: pd.DataFrame,
    return_cols: list[str],
    windows: tuple[int, ...] = (20, 60),
    min_period_ratio: float = 0.75,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling return volatility for multiple windows.

    Example outputs:
    - FPT_return_vol_20
    - FPT_return_vol_60
    """
    out = pd.DataFrame(index=wide.index)

    for window in windows:
        min_periods = int(window * min_period_ratio)

        vol = rolling_vol_panel(
            wide=wide,
            return_cols=return_cols,
            window=window,
            min_periods=min_periods,
            annualization_factor=annualization_factor,
        )

        out = pd.concat([out, vol], axis=1)

    return out


def liquidity_stats(
    returns: pd.Series,
    volume: pd.Series | None = None,
) -> dict:
    """
    Compute liquidity and zero-return diagnostics.

    Compatible with summary.py call:
        row.update(liquidity_stats(returns, volume=volume))

    Parameters
    ----------
    returns:
        Log-return series of one symbol.
    volume:
        Volume series of one symbol. Optional.

    Returns
    -------
    dict
        Summary metrics for zero returns and zero volume.
    """

    out = {}

    returns = pd.to_numeric(returns, errors="coerce")

    zero_return = returns.eq(0)

    out["zero_return_rows"] = int(zero_return.sum())
    out["zero_return_rate"] = zero_return.mean()

    out["missing_return_rows"] = int(returns.isna().sum())
    out["missing_return_rate"] = returns.isna().mean()

    if volume is None:
        out["zero_volume_rows"] = np.nan
        out["zero_volume_rate"] = np.nan

        out["zero_return_no_volume_rows"] = np.nan
        out["zero_return_no_volume_rate"] = np.nan

        out["zero_return_with_volume_rows"] = np.nan
        out["zero_return_with_volume_rate"] = np.nan

        return out

    volume = pd.to_numeric(volume, errors="coerce")

    zero_volume = volume.eq(0)

    out["zero_volume_rows"] = int(zero_volume.sum())
    out["zero_volume_rate"] = zero_volume.mean()

    zero_return_no_volume = zero_return & zero_volume
    zero_return_with_volume = zero_return & volume.gt(0)

    out["zero_return_no_volume_rows"] = int(zero_return_no_volume.sum())
    out["zero_return_no_volume_rate"] = zero_return_no_volume.mean()

    out["zero_return_with_volume_rows"] = int(zero_return_with_volume.sum())
    out["zero_return_with_volume_rate"] = zero_return_with_volume.mean()

    return out