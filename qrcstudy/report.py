"""Artifact-only validation, volatility losses, uncertainty, and result reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import digest, write_json
from .models import STOCHASTIC


def losses(actual, predicted, previous):
    a,p,b = np.broadcast_arrays(np.asarray(actual,float),np.asarray(predicted,float),np.asarray(previous,float))
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite scored predictions/targets")
    log_ratio=2*(a-p)
    with np.errstate(over="raise",invalid="raise"):
        qlike=np.expm1(log_ratio)-log_ratio
    return {"mse_log_rv":(a-p)**2,"mae_log_rv":np.abs(a-p),"mse_rv":(np.exp(a)-np.exp(p))**2,"mae_rv":np.abs(np.exp(a)-np.exp(p)),"qlike_variance":qlike,"directional_accuracy":(np.sign(p-b)==np.sign(a-b)).astype(float)}


def stationary_indices(n,reps=10000,block_length=6,seed=0):
    rng=np.random.default_rng(seed)
    indices=np.empty((reps,n),dtype=int)
    indices[:,0]=rng.integers(n,size=reps)
    for t in range(1,n):
        fresh=rng.integers(n,size=reps)
        indices[:,t]=np.where(rng.random(reps)<1/block_length,fresh,(indices[:,t-1]+1)%n)
    return indices


def load_validated(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text())
    config=manifest["config"]
    from .run import identity
    if identity(config)!=manifest["identity"]:raise ValueError("Manifest identity invalid")
    data=Path(config["data"])
    # Prefer the local published snapshot in a clone; retain the original path
    # for custom external datasets and execution provenance.
    local_data=Path(__file__).resolve().parents[1]/"data"/"snapshots"/data.parent.name/data.name
    if local_data.exists():data=local_data
    if digest(data)!=config["data_sha256"]:raise ValueError("Dataset changed")
    target=pd.read_csv(data,index_col=0,parse_dates=True).log_rv.loc[:config["end"]]
    dates=pd.date_range(config["start"],config["end"],freq="ME")
    all_dates=dates.append(pd.DatetimeIndex([dates[-1]+pd.offsets.MonthEnd()]))
    rows=[]
    for path in sorted((folder/"checkpoints").glob("*/*.json")):
        r=json.loads(path.read_text())
        if r["run_id"]!=manifest["identity"]:raise ValueError("Prediction belongs to another run")
        rows.append(r)
    if not rows and (folder/"predictions.csv").exists():
        receipt=json.loads((folder/"report_receipt.json").read_text())
        if digest(folder/"predictions.csv")!=receipt["prediction_sha256"]:
            raise ValueError("Published prediction checksum mismatch")
        rows=pd.read_csv(folder/"predictions.csv",float_precision="round_trip").to_dict("records")
        if any(r["run_id"]!=manifest["identity"] for r in rows):
            raise ValueError("Published predictions belong to another run")
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError("No checkpoint predictions")
    frame["target_month"]=pd.to_datetime(frame.target_month)
    keys=["model","seed","window","target_month"]
    if frame.duplicated(keys).any():raise ValueError("Duplicate forecasts")
    expected={(m,s,w,d) for m in config["models"] for s in (config["seeds"] if m in STOCHASTIC else [0]) for w in config["windows"] for d in all_dates}
    observed=set(frame[keys].itertuples(index=False,name=None))
    if expected!=observed:
        raise ValueError(f"Incomplete/mismatched run: missing {len(expected-observed)}, unexpected {len(observed-expected)}")
    for r in frame.itertuples():
        t=r.target_month
        if pd.Timestamp(r.forecast_origin)!=t-pd.offsets.MonthEnd():raise ValueError("Forecast origin mismatch")
        if pd.Timestamp(r.training_end)!=pd.Timestamp(r.forecast_origin):raise ValueError("Training reaches into target")
        if pd.Timestamp(r.training_start)!=t-pd.offsets.MonthEnd(r.window):raise ValueError("Training window mismatch")
        if r.configuration!=("colin" if config.get("protocol", "modern").startswith("colin") else "modern") or r.status not in {"ok","failed","unavailable"}:raise ValueError("Invalid record contract")
        if t in target.index:
            if not np.isclose(r.actual_log_rv,target.loc[t],rtol=0,atol=1e-12):raise ValueError("Actual target mismatch")
        elif pd.notna(r.actual_log_rv):raise ValueError("Unobserved target has an actual value")
        previous=target.loc[t-pd.offsets.MonthEnd()]
        if not np.isclose(previous,r.previous_log_rv,rtol=0,atol=1e-12):raise ValueError("Previous target mismatch")
        if r.status=="ok" and not np.isfinite(r.predicted_log_rv):raise ValueError("Invalid successful forecast")
    frame.to_csv(folder/"predictions.csv",index=False)
    return frame,config


def table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f"{x:.6g}"
        return str(x).replace("|","/")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join("| "+" | ".join(fmt(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))



def report(folder):
    from .study_report import report as render
    return render(folder)
