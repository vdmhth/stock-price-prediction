from __future__ import annotations

from pathlib import Path

try:
    from .crawl_market import crawl_all_market_indices
    from .crawl_stocks import crawl_all_stock_prices
except ImportError:
    from crawl_market import crawl_all_market_indices
    from crawl_stocks import crawl_all_stock_prices


def crawl_all_data() -> dict[str, dict[str, Path]]:
    """Download both configured stocks and market indices."""
    return {
        "stock": crawl_all_stock_prices(),
        "market": crawl_all_market_indices(),
    }

if __name__ == "__main__":
    crawl_all_data()