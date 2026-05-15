from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
        title: str = 'Cloase prices',
        save_path: Path |None = None,
) -> None:
    fig,ax = plt.subplots(figsize = (12,5))
    for c in columns:
        ax.plot(df[date_col],df[c],label =c, linewidth = 1)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
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
