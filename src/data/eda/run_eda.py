"""End-to-end EDA runner.

Reads:
    data/processed/eda/eda_processed.csv      (wide, returns & prices)
    data/processed/stock/stock_full_clean.csv     (long, full history)

Writes:
    data/reports/tables/*.csv
    data/reports/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.eda.autocorr import acf_table
from src.data.eda.correlation import pairwise_rolling, pearson_corr
from src.data.eda.plots import (
    plot_acf_pacf,
    plot_correlation_heatmap,
    plot_prices,
    plot_return_distribution,
    plot_rolling_corr,
)
from src.data.eda.stationarity import stationarity_summary
from src.data.eda.stats import describe_wide

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EDA_DATASET = DATA_DIR / "processed" / "eda" / "eda_processed.csv"
STOCK_FULL = DATA_DIR / "processed" / "stock" / "stock_full_clean.csv"

TABLES_DIR = DATA_DIR / "reports" / "tables"
FIGURES_DIR = DATA_DIR / "reports" / "figures"

TICKERS = ["FPT", "VCB", "VIC", "VNM", "HPG"]
INDICES = ["VNINDEX", "VN30"]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(EDA_DATASET, parse_dates=["trading_date"]).set_index("trading_date")
    close_cols = [f"{t}_close" for t in TICKERS] + [f"{i}_close" for i in INDICES]
    ret_cols = [f"{t}_log_return" for t in TICKERS] + [f"{i}_log_return" for i in INDICES]

    # ---------- 1. Descriptive statistics ----------
    describe_wide(wide, close_cols).to_csv(TABLES_DIR / "01_describe_close.csv")
    describe_wide(wide, ret_cols).to_csv(TABLES_DIR / "02_describe_log_return.csv")
    print("[1/6] Descriptive stats saved.")

    # ---------- 2. Stationarity ----------
    stationarity_summary(wide, close_cols).to_csv(TABLES_DIR / "03_stationarity_close.csv")
    stationarity_summary(wide, ret_cols).to_csv(TABLES_DIR / "04_stationarity_log_return.csv")
    print("[2/6] Stationarity tests saved.")

    # ---------- 3. Correlation ----------
    pearson_corr(wide, ret_cols).to_csv(TABLES_DIR / "05_correlation_log_return.csv")
    pearson_corr(wide, close_cols).to_csv(TABLES_DIR / "06_correlation_close.csv")

    plot_correlation_heatmap(
        pearson_corr(wide, ret_cols),
        "Log-return correlation",
        FIGURES_DIR / "corr_log_return.png",
    )
    plot_correlation_heatmap(
        pearson_corr(wide, close_cols),
        "Close-price correlation (spurious - trend-driven)",
        FIGURES_DIR / "corr_close.png",
    )
    print("[3/6] Correlation tables + heatmaps saved.")

    # ---------- 4. Price & return plots ----------
    wide_reset = wide.reset_index()
    plot_prices(
        wide_reset, [f"{t}_close" for t in TICKERS],
        title="Stock close prices (common window)",
        save_path=FIGURES_DIR / "prices_stocks.png",
    )
    plot_prices(
        wide_reset, [f"{i}_close" for i in INDICES],
        title="Market index close prices (common window)",
        save_path=FIGURES_DIR / "prices_indices.png",
    )
    plot_return_distribution(
        wide_reset, [f"{t}_log_return" for t in TICKERS],
        save_path=FIGURES_DIR / "return_dist_stocks.png",
    )
    print("[4/6] Price & distribution plots saved.")

    # ---------- 5. ACF / PACF ----------
    for t in TICKERS:
        plot_acf_pacf(
            wide[f"{t}_log_return"], lags=40, title=f"{t} log-return",
            save_path=FIGURES_DIR / f"acf_pacf_{t}_log_return.png",
        )
        # Volatility clustering: ACF/PACF of |return|
        plot_acf_pacf(
            wide[f"{t}_log_return"].abs(), lags=40,
            title=f"{t} |log-return| (volatility)",
            save_path=FIGURES_DIR / f"acf_pacf_{t}_abs_return.png",
        )
        # Tabular ACF/PACF for ARIMA order selection
        acf_table(wide[f"{t}_log_return"], nlags=20).to_csv(
            TABLES_DIR / f"07_acf_pacf_{t}.csv"
        )
    print("[5/6] ACF/PACF figures + tables saved.")

    # ---------- 6. Rolling correlation vs VNINDEX ----------
    rolling = pairwise_rolling(
        wide, base="VNINDEX_log_return",
        others=[f"{t}_log_return" for t in TICKERS],
        window=60,
    )
    rolling.to_csv(TABLES_DIR / "08_rolling_corr_vs_vnindex.csv")
    plot_rolling_corr(
        rolling,
        title="60-day rolling correlation vs VNINDEX (log-return)",
        save_path=FIGURES_DIR / "rolling_corr_vs_vnindex.png",
    )
    print("[6/6] Rolling correlation saved.")

    print("\nAll EDA outputs written to:")
    print(f"  tables  -> {TABLES_DIR}")
    print(f"  figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
