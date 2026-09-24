"""Immutable, source-audited preparation of the extended monthly dataset."""
from pathlib import Path
import io
import re
import zipfile
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from .data import digest, write_json

BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
SOURCES = {
 'factors.zip': BASE+'ftp/F-F_Research_Data_Factors_CSV.zip',
 'reversal.zip': BASE+'ftp/F-F_ST_Reversal_Factor_CSV.zip',
 'factors-fiz.zip': BASE+'ftp_202412/F-F_Research_Data_Factors_CSV.zip',
 'reversal-fiz.zip': BASE+'ftp_202412/F-F_ST_Reversal_Factor_CSV.zip',
}
FACTORS = ['MKT','SMB','HML','STR']
CUTOFF = '2017-12-31'


def read_monthly(path):
    f = pd.read_csv(path, index_col=0, parse_dates=True)
    if f.index.has_duplicates or not f.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered dates')
    if not f.index.equals(pd.date_range(f.index[0], f.index[-1], freq='ME')):
        raise ValueError('Monthly calendar gap or non-month-end date')
    if np.isinf(f.to_numpy(dtype=float)).any():
        raise ValueError('Infinite input value')
    return f


def french(path):
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode()
    lines = [l for l in text.splitlines() if re.match(r'^\s*\d{6},', l)]
    f = pd.read_csv(io.StringIO('\n'.join(lines)), header=None, index_col=0)
    f.columns = ['MKT','SMB','HML','RF'] if f.shape[1] == 4 else ['STR']
    f.index = pd.to_datetime(f.index.astype(str), format='%Y%m') + pd.offsets.MonthEnd()
    return f.mask(f <= -99)


def affine_map(raw, normalized):
    """Recover an affine scale on pre-1984 data; verify on 1984-2017.

    Require >=90% exact agreement overall and >=85% in each independent half.
    All inverted original values must lie on the provider 0.01 percentage-point
    grid. Residual source revisions are reported separately from scale recovery.
    No post-2017 observations enter scale estimation.
    """
    b = pd.concat([raw.rename('x'),normalized.rename('y')],axis=1).loc[:CUTOFF].dropna()
    train = b.loc[:'1983-12-31']; valid = b.loc['1984-01-01':]
    if min(len(train),len(valid)) < 100: raise ValueError('Insufficient mapping history')
    x,y = train.x.to_numpy(),train.y.to_numpy()
    best=(-1,None); rng=np.random.default_rng(0)
    for _ in range(5000):
        i,j=rng.choice(len(x),2,replace=False)
        if x[i]==x[j]:continue
        a=(y[i]-y[j])/(x[i]-x[j]);offset=y[i]-a*x[i]
        n=int((np.abs(y-a*x-offset)<1e-10).sum())
        if n>best[0]:best=(n,(a,offset))
    if best[1] is None:raise ValueError('No affine mapping')
    a,offset=best[1]
    err=b.y-(a*b.x+offset)
    details={'slope':float(a),'offset':float(offset),'train_exact_fraction':float((err.loc[:'1983'].abs()<1e-10).mean()),'validation_exact_fraction':float((err.loc['1984':].abs()<1e-10).mean()),'max_raw_residual_percentage_points':float((err/a).abs().max()),'calibration_end':CUTOFF}
    details['overall_exact_fraction']=float((err.abs()<1e-10).mean())
    inverted=(b.y-offset)/a
    details['original_grid_max_error']=float(np.max(np.abs(inverted*100-np.round(inverted*100))))
    details['accepted']=bool(a>0 and details['overall_exact_fraction']>=.9 and min(details['train_exact_fraction'],details['validation_exact_fraction'])>=.85 and details['original_grid_max_error']<1e-7)
    return details


def derived(frame):
    """Reconcile lags and separately normalized trailing averages using history."""
    f=frame.copy(); changes=[]; maps={}
    for column,k in [('RV1',1),('RV2',2),('RV_q',3),('RV_a',12)]:
        x=f.RV.shift(k) if column in ['RV1','RV2'] else f.RV.rolling(k).mean().shift(1)
        if column in ['RV_q','RV_a']:
            b=pd.concat([x.rename('x'),f[column].rename('y')],axis=1).loc[:CUTOFF].dropna()
            a=np.column_stack([b.x,np.ones(len(b))]);coef=np.linalg.lstsq(a,b.y,rcond=None)[0]
            if np.max(np.abs(a@coef-b.y))>1e-10:raise ValueError('Unverified derived-feature mapping: '+column)
            x=coef[0]*x+coef[1];maps[column]={'slope':float(coef[0]),'offset':float(coef[1]),'lagged_months':k}
        change=x.notna() & ((f[column]-x).abs()>1e-10)
        for date in f.index[change]:changes.append({'date':str(date.date()),'column':column,'old':float(f.loc[date,column]),'new':float(x.loc[date]),'reason':'Recomputed from preceding RV with verified historical transformation'})
        f.loc[change,column]=x.loc[change]
    return f,changes,maps


def prepare(source, output, as_of='2026-09-24', modern=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError('Snapshot already validated; use a new directory')
    source=Path(source);f=read_monthly(source)
    if f.index[-1]>=pd.Timestamp(as_of).to_period('M').to_timestamp():raise ValueError('Input contains an incomplete month')
    original=f.copy();raw=output/'raw';raw.mkdir(exist_ok=True);sources={}
    for name,url in SOURCES.items():
        path=raw/name
        if not path.exists():
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as r:path.write_bytes(r.read())
        sources[name]={'url':url,'sha256':digest(path),'observed_at':pd.Timestamp.now(tz='UTC').isoformat(),'retrieval_note':'Retrieved for this snapshot; retained local bytes used if present'}
    old=french(raw/'factors-fiz.zip').join(french(raw/'reversal-fiz.zip'))
    new=french(raw/'factors.zip').join(french(raw/'reversal.zip'))
    mappings={};changes=[];reconciliation=[]
    for col in FACTORS:
        mapping=affine_map(old[col],f[col]);mappings[col]=mapping
        a,b=mapping['slope'],mapping['offset']
        for vintage,values in [('FIZ_2024',old[col]),('CIZ_2026',new[col])]:
            check=pd.concat([original[col],(a*values+b).rename('provider')],axis=1).loc[:CUTOFF].dropna()
            for date,row in check.iterrows():reconciliation.append({'date':str(date.date()),'column':col,'vintage':vintage,'original':row[col],'mapped_provider':row.provider,'difference':row.provider-row[col]})
        if not mapping['accepted']:continue
        # FIZ observations through 2024; CIZ only where FIZ no longer exists.
        provider=old[col].combine_first(new[col]).reindex(f.index)
        missing=f[col].isna() & provider.notna()
        for date in f.index[missing]:
            value=a*provider.loc[date]+b;vintage='FIZ_2024' if date in old.index and pd.notna(old.loc[date,col]) else 'CIZ_2026'
            changes.append({'date':str(date.date()),'column':col,'old':None,'new':float(value),'reason':'Official '+vintage+' factor; pre-2018 validated affine scale'})
            f.loc[date,col]=value
    f,derived_changes,derived_maps=derived(f);changes+=derived_changes
    from quantum_reservoir_qiskit import MIN_RV,DIF
    f['log_rv']=(f.RV+1)*DIF+MIN_RV
    f.to_csv(output/'monthly.csv',index_label='Date')
    pd.DataFrame(changes).to_csv(output/'corrections.csv',index=False)
    pd.DataFrame(reconciliation).to_csv(output/'factor_reconciliation.csv',index=False)
    pd.DataFrame({'before':original.isna().sum(),'after':f[original.columns].isna().sum()}).to_csv(output/'missingness.csv',index_label='column')
    flags=[]
    for col in original:
        for kind,values in [('level',f[col]),('change',f[col].diff())]:
            history=values.loc[:CUTOFF];median=history.median();mad=(history-median).abs().median();scale=1.4826*mad
            if scale<=0:continue
            scores=(values-median).abs()/scale
            for date in scores.index[scores>6]:flags.append({'date':str(date.date()),'column':col,'kind':kind,'value':float(values.loc[date]),'robust_z':float(scores.loc[date]),'action':'retained; statistical flag is not proof of error'})
    pd.DataFrame(flags).to_csv(output/'outlier_flags.csv',index=False)
    target_diff=None
    if modern:
        target=pd.read_csv(modern,index_col=0,parse_dates=True).log_rv
        match=pd.concat([f.log_rv.rename('extended'),target.rename('modern')],axis=1).dropna();match['difference']=match.extended-match.modern
        match.to_csv(output/'target_reconciliation.csv',index_label='Date')
        target_diff=float(match.loc['2018':,'difference'].abs().max())
    write_json(output/'sources.json',sources)
    artifacts={p.name:digest(p) for p in sorted(output.glob('*.csv'))}
    artifacts['sources.json']=digest(output/'sources.json')
    manifest={'protocol':'extended-prepared-v1','source_repository':'aidanccc/qrc-volatility-research','source_commit':'ca78df0cae90a658d5b2696e6641bcfa4c3a0356','source_file':str(source),'source_sha256':digest(source),'as_of':as_of,'artifacts':artifacts,'factor_mapping':mappings,'derived_mapping':derived_maps,'target_inverse':{'min':MIN_RV,'span':DIF,'status':'legacy constants applied; Extended construction metadata unavailable'},'rows':len(f),'filled_factor_cells':int(original[FACTORS].isna().sum().sum()-f[FACTORS].isna().sum().sum()),'remaining_missing_factor_cells':int(f[FACTORS].isna().sum().sum()),'derived_corrections':len(derived_changes),'outlier_flags':len(flags),'max_post2017_target_difference':target_diff,'limitations':['Retrospective revised macro data; historical availability not verified','FIZ through December 2024, CIZ from January 2025; provider methodology break','Extended raw construction and exact post-2017 RV normalization unavailable','Original pre-2018 scaling inherited; extension is not a pristine real-time backtest']}
    write_json(output/'manifest.json',manifest)
    print({k:manifest[k] for k in ['rows','filled_factor_cells','remaining_missing_factor_cells','derived_corrections','outlier_flags']})
