"""End-to-end EDA runner.


  1. Load wide dataset
  2. Descriptive stats     -> 01_describe_close.csv, 02_describe_log_return.csv
  3. Distribution           -> 03_distribution.csv, qq_*.png, return_dist_*.png
  4. Stationarity           -> 04_stationarity_close.csv,
                                05_stationarity_log_return.csv
  5. Autocorr / LB / ARCH   -> 06_acf_pacf_{sym}.csv, 07_autocorr_tests.csv,
                                acf_pacf_*.png
  6. Correlation            -> 08_corr_log_return.csv, 09_corr_close.csv,
                                10_partial_corr_vs_vnindex.csv, corr_*.png
  7. Rolling vol + corr     -> 11_rolling_corr.csv, rolling_corr_*.png,
                                rolling_vol_*.png
  8. Drawdowns              -> drawdown_*.png
  9. Executive summary      -> 00_summary.csv
"""
from __future__ import annotations

from pathlib import Path

from .data_quality import save_key_data_quality, print_key_data_quality
from .volatility import rolling_vol_panel, rolling_vol_multi_window
from .autocorr import acf_table, autocorr_summary
from .constants import (
    MARKET_INDICES,
    STOCK_CODES,
    close_cols,
    return_cols,
)
from .correlation import (
    pairwise_rolling,
    partial_corr_matrix,
    pearson_corr,
    spearman_corr,
)
from .distribution import distribution_summary
from .loader import load_wide
from .plots import (
    plot_acf_pacf,
    plot_correlation_heatmap,
    plot_drawdown,
    plot_prices,
    plot_qq,
    plot_return_distribution,
    plot_rolling_corr,
    plot_rolling_vol,
)
from .stationarity import stationarity_summary
from .stats import describe_wide
from .summary import build_summary
from .volatility import rolling_vol_panel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EDA_DATASET = DATA_DIR / "processed" / "eda" / "eda_processed.csv"
TABLES_DIR = DATA_DIR / "reports" / "tables"
FIGURES_DIR = DATA_DIR / "reports" / "figures"
DATA_QUALITY_TABLE = TABLES_DIR / "00_data_quality.csv"
KEY_DATA_QUALITY_TABLE = TABLES_DIR / "00_key_data_quality.csv"
class StepCounter:
    def __init__(self, total: int) -> None:
        self.total = total
        self.current = 0
    def done(self, mes: str) -> None:
        self.current +=1
        print(f"[{self.current}/{self.total}] {mes}")

def main() -> None:
    steps = StepCounter(10)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_syms = STOCK_CODES + MARKET_INDICES
    cc = close_cols(all_syms)
    rc = return_cols(all_syms)

    # Load  
    wide = load_wide(EDA_DATASET)
    print(
        f"[load] shape={wide.shape}  "
        f"range={wide.index.min().date()} -> {wide.index.max().date()}"
    )
    # ---------- 1b. Data Quality ----------
    key_quality = save_key_data_quality(
        quality_path=DATA_QUALITY_TABLE,
        output_path=KEY_DATA_QUALITY_TABLE,
    )
    print_key_data_quality(key_quality)
    print("[1b] Key data quality table done.")

    #  Descriptive  
    describe_wide(wide, cc, kind="price").to_csv(
        TABLES_DIR / "01_describe_close.csv"
    )
    describe_wide(wide, rc, kind="return").to_csv(
        TABLES_DIR / "02_describe_log_return.csv"
    )
    steps.done("Descriptive stats done.")

    #  Distribution  
    distribution_summary(wide, rc).to_csv(TABLES_DIR / "03_distribution.csv")
    for t in STOCK_CODES:
        plot_qq(
            wide[f"{t}_log_return"],
            title=f"{t} log-return",
            save_path=FIGURES_DIR / f"qq_{t}.png",
        )
    plot_return_distribution(
        wide,
        [f"{t}_log_return" for t in STOCK_CODES],
        save_path=FIGURES_DIR / "return_dist_stocks.png",
    )
    steps.done(" Distribution diagnostics done.")

    #  Stationarity  
    stationarity_summary(wide, cc).to_csv(
        TABLES_DIR / "04_stationarity_close.csv"
    )
    stationarity_summary(wide, rc).to_csv(
        TABLES_DIR / "05_stationarity_log_return.csv"
    )
    steps.done(" Stationarity tests done.")

    #  ACF/PACF, Ljung-Box, ARCH-LM  
    autocorr_summary(wide, rc).to_csv(TABLES_DIR / "07_autocorr_tests.csv")
    for t in STOCK_CODES:
        acf_table(wide[f"{t}_log_return"], nlags=20).to_csv(
            TABLES_DIR / f"06_acf_pacf_{t}.csv"
        )
        plot_acf_pacf(
            wide[f"{t}_log_return"],
            lags=40,
            title=f"{t} log-return",
            save_path=FIGURES_DIR / f"acf_pacf_{t}_log_return.png",
        )
        plot_acf_pacf(
            wide[f"{t}_log_return"].abs(),
            lags=40,
            title=f"{t} |log-return|",
            save_path=FIGURES_DIR / f"acf_pacf_{t}_abs_return.png",
        )
    steps.done("Autocorrelation + Ljung-Box + ARCH-LM done.")

    # Correlation  
    corr_ret = pearson_corr(wide, rc)
    corr_spearman_ret = spearman_corr(wide, rc)
    corr_close = pearson_corr(wide, cc)
    corr_ret.to_csv(TABLES_DIR / "08_corr_log_return.csv")
    corr_spearman_ret.to_csv(TABLES_DIR / "08b_corr_spearman_log_return.csv")
    corr_close.to_csv(TABLES_DIR / "09_corr_close.csv")
    plot_correlation_heatmap(
        corr_ret,
        "Log-return correlation",
        FIGURES_DIR / "corr_log_return.png",
    )
    plot_correlation_heatmap(
    corr_spearman_ret,
    "Spearman log-return correlation",
    FIGURES_DIR / "corr_spearman_log_return.png",
    )
    plot_correlation_heatmap(
        corr_close,
        "Close-price correlation (spurious — trend-driven)",
        FIGURES_DIR / "corr_close.png",
    )
    # Partial correlation between stocks, controlling for VNINDEX
    pcm = partial_corr_matrix(
        wide,
        columns=[f"{t}_log_return" for t in STOCK_CODES],
        given="VNINDEX_log_return",
    )
    pcm.to_csv(TABLES_DIR / "10_partial_corr_vs_vnindex.csv")
    plot_correlation_heatmap(
        pcm,
        "Partial correlation (controlling for VNINDEX)",
        FIGURES_DIR / "corr_partial.png",
    )
    steps.done("Correlation matrices done.")

    # Rolling vol + corr  
    rolling_corr_df = pairwise_rolling(
        wide,
        base="VNINDEX_log_return",
        others=[f"{t}_log_return" for t in STOCK_CODES],
        window=60,
    )
    rolling_corr_df.to_csv(TABLES_DIR / "11_rolling_corr.csv")
    plot_rolling_corr(
        rolling_corr_df,
        title="60-day rolling correlation vs VNINDEX (log-return)",
        save_path=FIGURES_DIR / "rolling_corr_vs_vnindex.png",
    )
    rolling_vol_df = rolling_vol_panel(
        wide=wide,
        return_cols=[f"{t}_log_return" for t in STOCK_CODES],
        window=60,
        min_periods=45,
    )

    rolling_vol_df.to_csv(TABLES_DIR / "11_rolling_return_vol_60.csv")

    plot_rolling_vol(
        rolling_vol_df,
        title="60-day rolling annualized volatility from log returns",
        save_path=FIGURES_DIR / "rolling_return_vol_60.png",
    )

    steps.done("Rolling vol + corr done.")

    # Prices + drawdowns  
    wide_reset = wide.reset_index()
    plot_prices(
        wide_reset,
        close_cols(STOCK_CODES),
        date_col="trading_date",
        title="Stock close prices (log scale)",
        log_y=True,
        save_path=FIGURES_DIR / "prices_stocks_log.png",
    )
    plot_prices(
        wide_reset,
        close_cols(MARKET_INDICES),
        date_col="trading_date",
        title="Market index close prices",
        save_path=FIGURES_DIR / "prices_indices.png",
    )
    for t in STOCK_CODES:
        plot_drawdown(
            wide[f"{t}_close"],
            title=f"{t} drawdown",
            save_path=FIGURES_DIR / f"drawdown_{t}.png",
        )
    steps.done("Price + drawdown plots done.")

    #Executive summary  
    summary = build_summary(wide, stock_codes =all_syms, has_volume=True)
    summary.to_csv(TABLES_DIR / "summary.csv")
    steps.done("Executive summary written: summary.csv")

    print("\nAll EDA outputs written to:")
    print(f"  tables  -> {TABLES_DIR}")
    print(f"  figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
