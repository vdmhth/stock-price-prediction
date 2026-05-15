'''
adf and kpss stationary tests
adf h0: non-stationary, reject (p<0.05) => stationary
kpss h0: stationary, reject (p<0.05) => non-stationary

combined verdicts:
adf stationary, kpss stationary => stationary
adf non stat , kpss non stat => non-stat
adf stationary, kpss non stat => difference-stationary
adf non stat, kpss stationary => trend-stationary
'''
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(s:pd.Series)->dict:
    s = s.dropna()
    stat,pval,lags,nobs,crit,_ = adfuller(s,autolag= 'AIC')
    return {
        'adf_stat': stat,
        'adf_pval': pval,
        'adf_lags': lags,
        'adf_crit_5pct': crit['5%'],
        'adf_stationary': pval < 0.05,
    }
def kpss_test(s:pd.Series,regression: str = 'c')->dict:
    # regression: 'c' for level stationary, 'ct' tests trend stationarity
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        stat,pval,lags,crit = kpss(s, regression=regression, nlags='auto')
        return {
            'kpss_stat': stat,
            'kpss_pval': pval,
            'kpss_lags': lags,
            'kpss_crit_5pct': crit['5%'],
            'kpss_stationary': pval > 0.05,
        }

_VERDICT_MAP = {
    (True, True): 'stationary',
    (False,False): 'non-stationary',
    (True,False): 'difference-stationary',
    (False,True): 'trend-stationary',
}
def stationarity_summary(df:pd.DataFrame, columns:list[str]) ->pd.DataFrame:
    rows =[]
    for c in columns:
        adf = adf_test(df[c])
        kp = kpss_test(df[c])
        verdict = _VERDICT_MAP[(adf['adf_stationary'], kp['kpss_stationary'])]
        rows.append({
            'series': c,
            **adf,
            **kp,
            'verdict': verdict,
        })
    return pd.DataFrame(rows).set_index("series")