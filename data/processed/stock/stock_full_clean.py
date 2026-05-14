from pathlib import Path
import pandas as pd
RAW_STOCK_DIR = Path(r"E:/stock-price-prediction/data/raw/stock/")
PROCESSED_STOCK_DIR = Path(r"E:/stock-price-prediction/data/processed/stock/")
STOCK_CODES = ["VCB", "FPT", "VIC", "HPG", "VNM"]
OUTPUT_FILE = PROCESSED_STOCK_DIR / "stock_full_clean.csv"

def find_date_col(df:pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in ["time", "date", "trading_date", "tradingDate"]:
            return col
    raise ValueError(f"Cannot find date column. Available ones: {df.columns.tolist()}")

def standardize_stock_data(df:pd.DataFrame,code:str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    date_col = find_date_col(df)
    df = df.rename(columns={date_col: "trading_date"})
    df['stock_code'] = code.upper().strip()
    required_cols= ["stock_code", "trading_date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {code}: {missing_cols}")
    df = df[required_cols].copy()
    df['trading_date'] = pd.to_datetime(df['trading_date'],errors = 'coerce')
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["trading_date"])

    df = df.sort_values(['stock_code','trading_date'])
    df = df.drop_duplicates(subset = ['stock_code','trading_date'], keep='last')
    return df

def build_stock_full_clean() ->pd.DataFrame:
    all_dfs = []
    for code in STOCK_CODES:
        file_path = RAW_STOCK_DIR/f"{code}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found:{file_path}")
        print(f"Processing {file_path}")
        df = pd.read_csv(file_path)
        clean_df = standardize_stock_data(df, code)
        all_dfs.append(clean_df)
    stock_full =pd.concat(all_dfs, ignore_index = True)
    stock_full = stock_full.sort_values(['stock_code','trading_date'],keep = 'last')
    
