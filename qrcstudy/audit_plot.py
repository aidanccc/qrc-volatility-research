"""Render data-quality evidence without changing raw or prepared observations."""
from pathlib import Path
import numpy as np
import pandas as pd


def render(root='.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);snapshot=root/'data/snapshots/colin-2026-09-24-v2';out=root/'results/colin-study-2026-09-24';out.mkdir(parents=True,exist_ok=True)
    missing=pd.read_csv(snapshot/'missingness.csv',index_col=0).loc[['MKT','SMB','HML','STR']]
    targets=pd.read_csv(snapshot/'target_reconciliation.csv',index_col=0,parse_dates=True).loc['2018':]
    flags=pd.read_csv(snapshot/'outlier_flags.csv');flags['date']=pd.to_datetime(flags.date)
    f=pd.read_csv(snapshot/'monthly.csv',index_col=0,parse_dates=True)
    summaries=[]
    for period,part in [('historical',f.loc[:'2017']),('extension',f.loc['2018':])]:
        summary=part.describe(percentiles=[.05,.5,.95]).T
        summary['missing']=part.isna().sum();summary['period']=period;summaries.append(summary.reset_index(names='column'))
    pd.concat(summaries).to_csv(out/'data_summary.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
    x=np.arange(4);axes[0].bar(x-.18,missing.before,width=.36,label='Original missing');axes[0].bar(x+.18,missing.after,width=.36,label='Remaining missing');axes[0].set_xticks(x,missing.index);axes[0].set_ylabel('Missing monthly values');axes[0].set_title('412 of 416 missing factor observations recovered');axes[0].legend()
    axes[1].plot(targets.index,targets.difference*1e6,color='#79529c');axes[1].set_ylabel('Log-RV difference × 1,000,000');axes[1].set_title('Colin target minus independently reconstructed price target')
    axes[2].plot(f.index,f.log_rv,lw=1,color='#234c70',label='All observations retained');hist=f.log_rv.loc[:'2017'];med=hist.median();mad=1.4826*(hist-med).abs().median();dates=pd.DatetimeIndex(flags.date.unique()).intersection(f.index)
    axes[2].scatter(dates,f.loc[dates,'log_rv'],color='#b44b45',s=25,label='Month with any feature flag');axes[2].set_ylabel('Log monthly RV');axes[2].set_title('All months retained; markers identify dates with any feature flag');axes[2].legend()
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'data_audit.png',dpi=160);plt.close(fig)
    return out/'data_audit.png'
