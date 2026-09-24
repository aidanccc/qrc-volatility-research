"""Post-benchmark diagnostics. Exploratory, never overwrites primary forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest,write_json,transform,inverse_target
from .models import sequences
from .report import load_validated,losses,stationary_indices,table
from .colin_models import inputs

ALPHAS=np.array([1e-8,1e-6,1e-4,1e-2,1.,100.])


def ridge_grid(train,target,test):
    mean=train.mean(axis=0);sd=train.std(axis=0);sd=np.where(sd<1e-12,1.,sd)
    x=(train-mean)/sd;z=(test-mean)/sd;center=target.mean();gram=x.T@x;rhs=x.T@(target-center)
    return np.array([z@np.linalg.solve(gram+a*np.eye(x.shape[1]),rhs)+center for a in ALPHAS])


def choose_alpha(grid,actual,origin,lookback=24):
    if origin<lookback:raise ValueError('Insufficient validation history')
    err=(grid[origin-lookback:origin]-actual[origin-lookback:origin,None])**2
    if not np.isfinite(err).all():raise ValueError('Invalid validation history')
    return int(err.mean(axis=0).argmin())


def analyze(run,output):
    run=Path(run);out=Path(output)
    if out.exists():raise FileExistsError('Use a new exploratory output directory')
    frame,config=load_validated(run);out.mkdir(parents=True)
    data=Path(config['data']);local=Path(__file__).resolve().parents[1]/'data/snapshots'/data.parent.name/data.name
    if local.exists():data=local
    f=pd.read_csv(data,index_col=0,parse_dates=True).loc[:config['end']];is_colin=config['protocol'].startswith('colin')
    positions=np.flatnonzero(f.index>=pd.Timestamp(config['start']));actual=f.log_rv.iloc[positions].to_numpy()
    if is_colin:
        from quantum_reservoir_qiskit import MIN_RV,DIF
        y=f.RV.to_numpy();inv=lambda z:(z+1)*DIF+MIN_RV
    else:
        x,y,_=transform(f,config['scaler']);y=y.to_numpy();inv=lambda z:inverse_target(z,config['scaler'])
    rows=[];condition=[];hashes={}
    for model in ['CRLX','QR1','QR2']:
        model_x=inputs(f,model).to_numpy() if is_colin else x.to_numpy()
        raw=sequences(model_x).reshape(len(f)+1,-1)
        for seed in config['seeds']:
            path=run/'cache'/f'{model}-{seed}.npy';a=np.load(path);hashes[path.name]=digest(path)
            receipt=json.loads(path.with_suffix('.json').read_text())
            if hashes[path.name]!=receipt['sha256']:raise ValueError('Cache checksum mismatch')
            for window in config['windows']:
                for label,features in [(model+'_matched_ridge',a),(model+'_raw_input_ridge',raw)]:
                    if label.endswith('raw_input_ridge') and seed!=config['seeds'][0]:continue
                    grid=[]
                    for t in positions:
                        train=features[t-window:t];test=features[t]
                        if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('Exploration requires complete scored feature histories')
                        grid.append(inv(ridge_grid(train,y[t-window:t],test)))
                        if label.endswith('matched_ridge'):condition.append(dict(model=model,seed=seed,window=window,date=str(f.index[t].date()),condition_number=float(np.linalg.cond(train))))
                    grid=np.array(grid)
                    for i in range(24,len(positions)):
                        selected=choose_alpha(grid,actual,i);pred=grid[i,selected];t=positions[i]
                        row=dict(model=label,seed=seed,window=window,target_month=str(f.index[t].date()),validation_end=str(f.index[positions[i-1]].date()),alpha=float(ALPHAS[selected]),actual_log_rv=actual[i],predicted_log_rv=pred)
                        row.update({k:float(v) for k,v in losses(actual[i],pred,f.log_rv.iloc[t-1]).items()});rows.append(row)
    pred=pd.DataFrame(rows);pred.to_csv(out/'predictions.csv',index=False);pd.DataFrame(condition).to_csv(out/'conditioning.csv',index=False)
    metrics=pred.groupby(['window','model'])[['mse_log_rv','mae_log_rv','qlike_variance']].mean().reset_index();metrics.to_csv(out/'metrics.csv',index=False)
    # Compare original model losses on precisely the same 80 target dates.
    dates=pd.to_datetime(pred.target_month.unique());primary=frame.loc[frame.target_month.isin(dates)&frame.status.eq('ok')].copy()
    for k,v in losses(primary.actual_log_rv,primary.predicted_log_rv,primary.previous_log_rv).items():primary[k]=v
    primary.groupby(['window','model'])[['mse_log_rv','mae_log_rv','qlike_variance']].mean().to_csv(out/'primary_same_dates.csv')
    intervals=[]
    for window,g in pred.groupby('window'):
        matrix=g.groupby(['target_month','model']).mse_log_rv.mean().unstack()
        for model in ['CRLX','QR1','QR2']:
            diff=(matrix[model+'_matched_ridge']-matrix[model+'_raw_input_ridge']).to_numpy()
            for block in [3,6,12]:
                sample=diff[stationary_indices(len(diff),10000,block,17)].mean(axis=1);lo,hi=np.quantile(sample,[.025,.975]);intervals.append(dict(window=window,model=model,block=block,loss_difference=float(diff.mean()),ci_lower=lo,ci_upper=hi,months=len(diff)))
    pd.DataFrame(intervals).to_csv(out/'matched_intervals.csv',index=False)
    # Fixed training-period threshold labels unusually volatile target months;
    # posthoc slices are descriptive, not a cleaned headline benchmark.
    hist=f.log_rv.loc[:'2017'];med=hist.median();scale=1.4826*(hist-med).abs().median();flags=((f.log_rv-med).abs()>6*scale)
    slices=[]
    primary=frame.loc[frame.actual_log_rv.notna()&frame.status.eq('ok')].copy()
    for k,v in losses(primary.actual_log_rv,primary.predicted_log_rv,primary.previous_log_rv).items():primary[k]=v
    primary['flagged_target']=primary.target_month.map(flags).astype(bool)
    for flag,g in primary.groupby('flagged_target'):
        result=g.groupby(['window','model'])[['mse_log_rv','qlike_variance']].mean().reset_index();result['flagged_target']=flag;result['months']=g.target_month.nunique();slices.append(result)
    pd.concat(slices).to_csv(out/'outlier_sensitivity.csv',index=False)
    write_json(out/'receipt.json',{'protocol':'exploratory-matched-ridge-v1','source_run_identity':json.loads((run/'manifest.json').read_text())['identity'],'source_predictions_sha256':digest(run/'predictions.csv'),'source_sha256':digest(__file__),'cache_hashes':hashes,'alphas':ALPHAS.tolist(),'validation_months':24,'scored_months':len(dates),'flagged_target_months':[str(d.date()) for d in flags.loc[config['start']:].index[flags.loc[config['start']:]]]})
    (out/'report.md').write_text('# Exploratory readout and outlier diagnostics\n\nThese post-benchmark choices are not an untouched confirmation test. Standardization and intercept are fitted only on each training window; alpha minimizes errors on the 24 strictly preceding forecast months. Scores cover January 2020–August 2026 (80 months). Each raw-input ridge uses the same three-step inputs as its paired reservoir. Quantum feature sets differ under the Colin protocol, so QR1/QR2 here are not an isolated virtual-node ablation.\n\n'+table(metrics)+'\n\nSee primary_same_dates.csv for primary models on those same 80 dates, matched_intervals.csv for unadjusted paired bootstrap intervals, and outlier_sensitivity.csv for descriptive target-volatility slices. No primary observation was removed or target clipped.\n')
    print(f'Exploration: {out}')
