import csv, math, openpyxl, os, shutil, tempfile, time

RMSE_COL = {'FPT': 3, 'HPG': 4, 'VCB': 5, 'VIC': 6, 'VNM': 7}
R2_COL   = {'FPT': 11, 'HPG': 12, 'VCB': 13, 'VIC': 14, 'VNM': 15}


def load_and_evaluate(path, test_start='2024-07-22'):
    """
    Đọc file prediction CSV và tính RMSE, R2, Direction Accuracy.

    Hỗ trợ 2 định dạng:
      - (trading_date, actual, prediction)  — XGB/RF
      - (ds, y, yhat)                       — ARIMA/SVR/Prophet

    Nếu có cột date, tự động lọc test set từ test_start (mặc định 2024-07-22).
    """
    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))

    cols = list(rows[0].keys())

    if 'actual' in cols:
        y  = [float(r['actual']) for r in rows]
        yhat = [float(r['prediction']) for r in rows]
        date_key = 'trading_date'
    else:
        y = [float(r['y']) for r in rows]
        yhat = [float(r['yhat']) for r in rows]
        date_key = 'ds'

    if date_key in cols and test_start:
        test = [r for r in rows if r[date_key] >= test_start]
        if test:
            if 'actual' in cols:
                y = [float(r['actual']) for r in test]
                yhat = [float(r['prediction']) for r in test]
            else:
                y = [float(r['y']) for r in test]
                yhat = [float(r['yhat']) for r in test]

    n = len(y)
    if n < 2:
        return {'RMSE': None, 'R2': None, 'DA': None}

    rmse = math.sqrt(sum((yi - yhati)**2 for yi, yhati in zip(y, yhat)) / n)

    mean_y = sum(y) / n
    ss_res = sum((yi - yhati)**2 for yi, yhati in zip(y, yhat))
    ss_tot = sum((yi - mean_y)**2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    correct = sum(1 for i in range(1, n) if (y[i] - y[i-1]) * (yhat[i] - yhat[i-1]) > 0)
    da = correct / (n - 1)

    return {'RMSE': round(rmse, 4), 'R2': round(r2, 4), 'DA': round(da, 4)}


def write_to_table(src, row_rmse_r2_5d, row_rmse_r2_20d, row_da_5d, row_da_20d,
                    values_5d, values_20d):

    lock = os.path.join(os.path.dirname(src), '~$Table.xlsx')
    if os.path.exists(lock):
        os.remove(lock)
    time.sleep(0.5)

    tmp = os.path.join(tempfile.gettempdir(), 'Table_temp.xlsx')
    shutil.copy2(src, tmp)

    wb = openpyxl.load_workbook(tmp)
    ws = wb['Sheet1']

    for stock in RMSE_COL:
        r5, r2_5, da5 = values_5d[stock]
        r20, r2_20, da20 = values_20d[stock]

        ws.cell(row=row_rmse_r2_5d,  column=RMSE_COL[stock], value=r5)
        ws.cell(row=row_rmse_r2_20d, column=RMSE_COL[stock], value=r20)
        ws.cell(row=row_rmse_r2_5d,  column=R2_COL[stock],   value=r2_5)
        ws.cell(row=row_rmse_r2_20d, column=R2_COL[stock],   value=r2_20)
        ws.cell(row=row_da_5d,       column=RMSE_COL[stock], value=da5)
        ws.cell(row=row_da_20d,      column=RMSE_COL[stock], value=da20)

    wb.save(tmp)
    time.sleep(0.5)
    shutil.copy2(tmp, src)
    os.remove(tmp)
