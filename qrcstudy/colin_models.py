"""Paper feature adapters; inherited normalization, no test-period fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest, write_json
from .models import sequences, reservoir_features, lstm_forecast

MACRO = ['MKT','diff_DP','IP','DEF','EP','SMB','diff_TB','HML','INF','STR']
QR1 = ['RV','MKT','DP','IP','RV_q','STR','DEF']
QR2 = ['RV','MKT','STR','RV_q','EP','INF','DEF']


def inputs(frame, model):
    f=frame.copy()
    f['diff_DP']=f.DP.diff();f['diff_TB']=f.TB.diff()
    # Historical preprocessing pads the first difference only, never missing factors.
    f.loc[f.index[0],['diff_DP','diff_TB']]=0.
    f['har3']=f.RV.rolling(3).mean();f['har12']=f.RV.rolling(12).mean()
    if model=='QR1':cols=QR1
    elif model=='QR2':cols=QR2
    elif model in ['LSTMX','CRLX']:cols=['RV']+MACRO
    elif model=='HAR':cols=['RV','har3','har12']
    elif model=='HARX':cols=['RV','har3','har12']+MACRO
    elif model=='ARMAX':cols=MACRO
    else:cols=['RV']
    return f[cols]


def features(x,model,seed,cache):
    """Compute only the complete contiguous input prefix; missing suffix stays NaN.

    Reservoir states reset every three inputs. No zero imputation or gap-skipping
    occurs. Interior missing rows conservatively block later reservoir features.
    """
    x=np.asarray(x);bad=np.flatnonzero(~np.isfinite(x).all(axis=1))
    n=int(bad[0]) if len(bad) else len(x)
    dims={'QR1':10,'QR2':20,'CRL':50,'CRLX':20}
    out=np.full((len(x)+1,dims[model]),np.nan)
    if n>=3:
        a=reservoir_features(x[:n],model,seed,cache)
        out[:len(a)]=a
    return out


def predict(model,x,y,seq,feature,t,window,seed,threads):
    start=t-window
    if start<15 or not np.isfinite(y[start:t]).all():raise ValueError('Insufficient valid training history')
    if model=='Persistence':return float(y[t-1])
    if model in ['AR1','AR3']:
        p=1 if model=='AR1' else 3
        a=np.array([[y[i-lag] for lag in range(1,p+1)] for i in range(start,t+1)])
    elif model in ['LSTM','LSTMX']:
        if not np.isfinite(seq[start:t+1]).all():raise ValueError('Unavailable required inputs')
        return lstm_forecast(seq,y,start,t,model,seed,threads)
    elif model in ['QR1','QR2','CRL','CRLX']:
        if not np.isfinite(feature[start:t+1]).all():raise ValueError('Unavailable required inputs')
        if model.startswith('QR'):
            a=feature[start:t];w=np.linalg.solve(a.T@a+1e-8*np.eye(a.shape[1]),a.T@y[start:t])
            return float(feature[t]@w)
        from reservoirpy.nodes import Ridge
        ridge=Ridge(ridge=1e-7).fit(feature[start:t],y[start:t,None])
        return float(ridge.run(feature[t:t+1]).flat[0])
    else:
        a=x[start-1:t]
        if not np.isfinite(a).all():raise ValueError('Unavailable required inputs')
        if model=='ARMAX':
            from statsmodels.tsa.arima.model import ARIMA
            fit=ARIMA(y[start:t],exog=a[:-1],order=(3,0,0),trend='n').fit(method_kwargs={'maxiter':1000})
            if not fit.mle_retvals.get('converged',True):raise RuntimeError('ARMAX optimizer did not converge')
            return float(np.asarray(fit.forecast(exog=a[-1:]))[0])
    if not np.isfinite(a).all():raise ValueError('Unavailable required inputs')
    a=np.column_stack([np.ones(len(a)),a]);w=np.linalg.lstsq(a[:-1],y[start:t],rcond=None)[0]
    return float(a[-1]@w)
