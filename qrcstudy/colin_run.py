"""Colin protocol with explicit missing-input records and immutable identities."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from .data import digest,write_json
from .models import MODELS,STOCHASTIC,sequences
from .run import checked_manifest,identity,collect
from .colin_models import inputs,features,predict,QR1,QR2,MACRO


def worker(task):
    out,config,model,seed,limit=task;out=Path(out);run_id=identity(config)
    frame=pd.read_csv(config['data'],index_col=0,parse_dates=True).loc[:config['end']]
    x=inputs(frame,model).to_numpy();y=np.append(frame.RV.to_numpy(),np.nan);seq=sequences(x)
    dates=frame.index.append(pd.DatetimeIndex([frame.index[-1]+pd.offsets.MonthEnd()]))
    feature=None
    if model in ['QR1','QR2','CRL','CRLX']:feature=features(x,model,seed,out/'cache')
    from quantum_reservoir_qiskit import MIN_RV,DIF
    for window in config['windows']:
        folder=out/'checkpoints'/f'{model}-w{window}-s{seed}';folder.mkdir(parents=True,exist_ok=True);count=0
        for t,date in enumerate(dates):
            if date<pd.Timestamp(config['start']):continue
            name=folder/f'{date:%Y-%m}.json'
            if name.exists():
                r=json.loads(name.read_text())
                if r['run_id']!=run_id or r['target_month']!=str(date.date()):raise ValueError('Checkpoint identity/date mismatch')
                continue
            if limit is not None and count>=limit:break
            if t-window<0:raise ValueError('Insufficient calendar history')
            row={'run_id':run_id,'configuration':'colin','model':model,'seed':seed,'window':window,'target_month':str(date.date()),'forecast_origin':str(dates[t-1].date()),'training_start':str(dates[t-window].date()),'training_end':str(dates[t-1].date()),'actual_log_rv':float(frame.log_rv.iloc[t]) if t<len(frame) else None,'previous_log_rv':float(frame.log_rv.iloc[t-1]),'predicted_log_rv':None,'status':'failed'}
            started=time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');s=int(np.random.SeedSequence([seed,date.year,date.month]).generate_state(1)[0])
                    pred=float((predict(model,x,y,seq,feature,t,window,s,config['threads'])+1)*DIF+MIN_RV)
                if not np.isfinite(pred) or not np.isfinite(np.exp(2*pred)):raise FloatingPointError('Nonfinite prediction')
                row.update(predicted_log_rv=pred,status='ok',warnings=sorted(set(str(w.message) for w in caught)))
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
                if isinstance(exc,ValueError) and 'Unavailable required inputs' in str(exc):row['status']='unavailable'
            row['seconds']=time.perf_counter()-started;write_json(name,row);count+=1
            if count%12==0:print(f'{model} seed={seed} window={window} through {date:%Y-%m}',flush=True)
    return f'{model} seed={seed} complete'


def run(data,output,start,end,windows,seeds,models,workers,threads,limit=None):
    data=Path(data).resolve();snapshot=json.loads((data.parent/'manifest.json').read_text())
    for name,sha in snapshot['artifacts'].items():
        if digest(data.parent/name)!=sha:raise ValueError('Snapshot artifact changed: '+name)
    from .colin_data import read_monthly
    frame=read_monthly(data)
    if pd.Timestamp(end)>frame.index[-1]:raise ValueError('Evaluation beyond completed data')
    if pd.Timestamp(start) not in frame.index or pd.Timestamp(end) not in frame.index:raise ValueError('Evaluation boundaries must be available month ends')
    if min(windows)<16 or frame.index.get_loc(pd.Timestamp(start))-max(windows)<15:raise ValueError('Insufficient training history')
    root=Path(__file__).resolve().parents[1]
    source_files=['qrcstudy/colin_data.py','qrcstudy/colin_models.py','qrcstudy/colin_run.py','qrcstudy/models.py','qrcstudy/run.py','qrcstudy/data.py','quantum_reservoir_qiskit.py']
    config={'protocol':'colin-paper-features-v1','data':str(data),'data_sha256':digest(data),'snapshot_sha256':digest(data.parent/'manifest.json'),'start':start,'end':end,'windows':windows,'seeds':seeds,'models':models,'threads':threads,'epochs':100,'features':{m:inputs(frame,m).columns.tolist() for m in models},'target_inverse':snapshot['target_inverse'],'transform_policy':'Inherited normalized inputs; fixed DP/TB differences; no outlier removal; prepared derived identities','versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','torch','statsmodels','reservoirpy','qiskit','arch']},'python':platform.python_version(),'source_hashes':{p:digest(root/p) for p in source_files}}
    checked_manifest(output,config);out=Path(output)
    if not (out/'execution_revision.json').exists():write_json(out/'execution_revision.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_hashes':config['source_hashes']})
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]=str(threads)
    tasks=[(str(out),config,m,s,limit) for m in models for s in (seeds if m in STOCHASTIC else [0])];failures=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(worker,t):(t[2],t[3]) for t in tasks}
        for future in as_completed(pending):
            try:print(future.result(),flush=True)
            except Exception as exc:failures.append({'model_seed':pending[future],'error':repr(exc)});print(f'TASK FAILURE: {exc}',flush=True)
    write_json(out/'execution.json',{'wall_seconds_this_invocation':time.perf_counter()-started,'task_failures':failures,'pilot_limit':limit})
    frame=collect(out);print(frame.groupby(['model','status']).size().to_string())
    if failures:raise RuntimeError('Worker failure; inspect execution.json')
