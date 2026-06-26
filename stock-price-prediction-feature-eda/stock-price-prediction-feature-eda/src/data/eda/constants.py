STOCK_CODES = ['FPT', 'VCB', 'VIC', 'VNM', 'HPG']
MARKET_INDICES = ['VNINDEX', 'VN30']
HOSE_DAILY_LIMIT_LOG = 0.0677
TRADING_DAYS_PER_YEAR = 252
ZERO_RETURN_TOL = 1e-8
def close_cols(symbols:list[str]) -> list[str]:
    return [f"{s}_close" for s in symbols]
def return_cols(symbols:list[str]) -> list[str]:
    return [f"{s}_log_return" for s in symbols]
def volume_cols(symbols:list[str]) -> list[str]:
    return [f"{s}_volume" for s in symbols]
def volatility_cols(symbols:list[str]) -> list[str]:
    return [f"{s}_volatility" for s in symbols]
ALL_SYMBOLS = STOCK_CODES + MARKET_INDICES
