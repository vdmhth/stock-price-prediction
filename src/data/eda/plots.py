from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats as sps
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def _save_plot(fig:plt.Figure, save_path:Path|None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents = True,exist_ok = True)
        fig.savefig(save_path,dpi = 120,bbox_inches = 'tight')
        plt.close(fig)

    else:
        plt.show()
def plot_prices(
        df:pd.DataFrame,
        columns:list[str],
        date_col: str = 'trading_date',
        title: str = 'Close prices',
        log_y:bool=False,
        save_path: Path |None = None,
) -> None:
    """Plot price lines. Set log_y=True for assets with large multi-year growth."""
    fig,ax = plt.subplots(figsize = (12,5))
    x = df[date_col] if date_col is not None else df.index

    for c in columns:
        ax.plot(x,df[c],label =c, linewidth = 1)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price(log scale)' if log_y else "Price")
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc = 'upper left',ncol = 2,fontsize = 9)
    ax.grid(alpha = 0.3)
    _save_plot(fig, save_path)

def plot_return_distribution(
        df:pd.DataFrame, 
        columns:list[str],
        bins: int =60,
        save_path: Path | None = None,
) ->None:
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    axes = axes.flatten()
    for ax, c in zip(axes, columns):
        s = df[c].dropna()
        ax.hist(s, bins=bins, density=True, alpha=0.75, color="steelblue")
        ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_title(c)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, save_path)
def plot_qq(
    series: pd.Series,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """QQ-plot vs Normal and vs Student-t (df fit by MLE).

    Reading:
    - Normal QQ that curves into an S-shape at both ends ⇒ fat tails.
    - t-QQ that is approximately straight ⇒ Student-t is a good model.
    """
    s = series.dropna().values
    df_t, loc, scale = sps.t.fit(s)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sps.probplot(s, dist="norm", plot=axes[0])
    axes[0].set_title(f"QQ vs Normal — {title}")
    sps.probplot(s, dist="t", sparams=(df_t,), plot=axes[1])
    axes[1].set_title(f"QQ vs t(df={df_t:.2f}) — {title}")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, save_path)

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Correlation matrix",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 4, 0.7 * len(corr) + 3))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    _save_plot(fig, save_path)


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s = series.dropna()
    plot_acf(s, lags=lags, ax=axes[0])
    plot_pacf(s, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF — {title}")
    axes[1].set_title(f"PACF — {title}")
    fig.tight_layout()
    _save_plot(fig, save_path)


def plot_rolling_corr(
    rolling_df: pd.DataFrame,
    title: str = "Rolling correlation",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in rolling_df.columns:
        ax.plot(rolling_df.index, rolling_df[c], label=c, linewidth=1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Correlation")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    _save_plot(fig, save_path)
def plot_rolling_vol(
    vol_df: pd.DataFrame,
    title: str = "Rolling annualised volatility",
    save_path: Path | None = None,
) -> None:
    """Companion to plot_rolling_corr: show when volatility spikes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in vol_df.columns:
        ax.plot(vol_df.index, vol_df[c], label=c, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Annualised σ")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    _save_plot(fig, save_path)

def plot_drawdown(
    price: pd.Series,
    title: str = "Drawdown",
    save_path: Path | None = None,
) -> None:
    s = price.dropna()
    dd = s / s.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    _save_plot(fig, save_path)
