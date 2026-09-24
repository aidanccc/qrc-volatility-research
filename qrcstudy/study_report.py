"""Protocol-neutral reporting; claims computed from validated predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest,write_json
from .models import STOCHASTIC
from .report import load_validated,losses,stationary_indices,table


def report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from arch.bootstrap import MCS
    folder=Path(folder);frame,config=load_validated(folder)
    scored=frame.loc[frame.actual_log_rv.notna() & frame.status.eq('ok')].copy()
    for name,value in losses(scored.actual_log_rv,scored.predicted_log_rv,scored.previous_log_rv).items():scored[name]=value
    measures=['mse_log_rv','mae_log_rv','qlike_variance','directional_accuracy']
    end=pd.Timestamp(config['end']);last=end-pd.offsets.MonthEnd(11);rows=[];seeds=[];coverage=[]
    for period,start in [('post2017',pd.Timestamp(config['start'])),('last12',max(last,pd.Timestamp(config['start'])))]:
        count=len(pd.date_range(start,end,freq='ME'));part=scored.loc[scored.target_month>=start]
        for w in config['windows']:
            for model in config['models']:
                g=part.loc[(part.window==w)&(part.model==model)];ns=len(config['seeds']) if model in STOCHASTIC else 1
                byseed=g.groupby('seed')[measures].mean();seedstd=float(byseed.mse_log_rv.std(ddof=0)) if len(byseed) else np.nan
                row=dict(period=period,window=w,model=model,complete=len(g)==count*ns,successful_forecasts=len(g),expected_forecasts=count*ns,mse_seed_std=seedstd)
                row.update({m:g[m].mean() for m in measures});row['rmse_log_rv']=np.sqrt(row['mse_log_rv']);rows.append(row)
                for seed,r in byseed.iterrows():seeds.append(dict(period=period,window=w,model=model,seed=seed,**r.to_dict(),rmse_log_rv=np.sqrt(r.mse_log_rv)))
    summary=pd.DataFrame(rows).sort_values(['period','window','mse_log_rv']);summary.to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(seeds).to_csv(folder/'metrics_by_seed.csv',index=False);scored.to_csv(folder/'losses.csv',index=False)
    frame.loc[frame.status.ne('ok')].to_csv(folder/'failures.csv',index=False)
    frame.groupby(['model','window','status'],dropna=False).size().rename('records').to_csv(folder/'coverage.csv')
    common=[];stats=[];cis=[]
    for w in config['windows']:
        part=scored.loc[scored.window==w];counts=part.groupby(['target_month','model']).size()
        means=part.groupby(['target_month','model'])[measures].mean()
        valid=[k for k,n in counts.items() if n==(len(config['seeds']) if k[1] in STOCHASTIC else 1)]
        means=means.loc[valid]
        available=sorted(means.index.get_level_values('model').unique())
        full=summary.loc[(summary.period=='post2017')&(summary.window==w)&summary.complete,'model'].tolist()
        for metric in ['mse_log_rv','qlike_variance']:
            matrix=means[metric].unstack('model')
            # All planned models must be present to call this an all-model common-date comparison.
            all_shared=matrix.reindex(columns=config['models']).dropna()
            if metric=='mse_log_rv' and len(all_shared):
                shared=means.loc[means.index.get_level_values(0).isin(all_shared.index)].groupby('model')[measures].mean().reset_index();shared['window']=w;shared['common_months']=len(all_shared);shared['rmse_log_rv']=np.sqrt(shared.mse_log_rv);common.append(shared)
            for scope,mat in [('complete_period',matrix.reindex(columns=full).dropna()),('all_model_common_dates',all_shared)]:
                if mat.shape[1]<2 or len(mat)<12:continue
                mcs=MCS(mat,size=.05,reps=10000,block_size=min(6,len(mat)),method='R',bootstrap='stationary',seed=0);mcs.compute()
                pv=mcs.pvalues.reset_index();pv.columns=['model','mcs_pvalue'];pv['window']=w;pv['loss']=metric;pv['scope']=scope;pv['months']=len(mat);stats.append(pv)
                if 'HAR' in mat:
                    delta=mat.subtract(mat.HAR,axis=0);idx=stationary_indices(len(mat));samples=delta.to_numpy()[idx].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975],axis=0)
                    for i,model in enumerate(mat.columns):cis.append(dict(scope=scope,window=w,model=model,loss=metric,mean_loss_minus_HAR=delta[model].mean(),ci_lower=lo[i],ci_upper=hi[i],months=len(mat)))
    mcs_frame=pd.concat(stats,ignore_index=True) if stats else pd.DataFrame(columns=['model','mcs_pvalue','window','loss','scope','months'])
    mcs_frame.to_csv(folder/'mcs.csv',index=False);pd.DataFrame(cis).to_csv(folder/'loss_difference_ci.csv',index=False)
    (pd.concat(common,ignore_index=True) if common else pd.DataFrame(columns=['model','window','common_months'])).to_csv(folder/'common_dates_metrics.csv',index=False)
    frame.loc[frame.actual_log_rv.isna()].to_csv(folder/'unscored_forecasts.csv',index=False)
    scored.loc[scored.target_month>=last].to_csv(folder/'last12_forecasts.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(config['windows']),1,figsize=(11,4*len(config['windows'])),squeeze=False,constrained_layout=True)
    for ax,w in zip(axes[:,0],config['windows']):
        g=scored.loc[(scored.window==w)&(scored.target_month>=last)];a=g.groupby('target_month').actual_log_rv.first();ax.plot(a.index,np.exp(a)*100,label='Observed',color='black',lw=2)
        for model in ['HAR','HARX','QR1','QR2','LSTMX']:
            p=g.loc[g.model==model].assign(rv=lambda x:np.exp(x.predicted_log_rv)*100).groupby('target_month').rv.mean()
            if len(p):ax.plot(p.index,p,label=model)
        ax.set_title(f"{config['protocol']}: {w}-month window");ax.set_ylabel('Monthly volatility (%)');ax.legend(ncol=3);ax.grid(alpha=.2)
        ax.text(.015,.96,f'{w}-month training window',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
        ax.set_xticks(a.index);ax.set_xticklabels([d.strftime('%b\n%Y') for d in a.index],fontsize=9)
    fig.savefig(folder/'last12_forecasts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(config['windows']),figsize=(12,5),squeeze=False,constrained_layout=True)
    for ax,w in zip(axes[0],config['windows']):
        g=summary.loc[(summary.period=='post2017')&(summary.window==w)&summary.complete].sort_values('mse_log_rv',ascending=False);ax.barh(g.model,g.mse_log_rv);ax.set_title(f'{w}-month window');ax.set_xlabel('Mean squared log-RV error')
    fig.savefig(folder/'model_comparison.png',dpi=160);plt.close(fig)
    text=[f"# {config['protocol']} results",'',f"Evaluation: {config['start']}–{config['end']}. All {len(frame)} expected records validated, including unscored forecasts.",'',f"Successful scored records: {len(scored)}. Failed or unavailable records: {int(frame.status.ne('ok').sum())} (see failures.csv and coverage.csv).",'','## Forecast accuracy','', 'Tables average losses across seeds; seeds are not independent market histories. Rankings below include only complete models. Incomplete models have separate common-date comparisons.']
    for period in ['post2017','last12']:
        text += ['',f'### {period}','',table(summary.loc[summary.period==period,['window','model','complete','successful_forecasts','mse_log_rv','rmse_log_rv','mae_log_rv','qlike_variance','mse_seed_std']])]
        for w in config['windows']:
            g=summary.loc[(summary.period==period)&(summary.window==w)&summary.complete]
            if len(g):r=g.iloc[0];text += ['',f'{w}-month window: {r.model} has the lowest complete-model MSE ({r.mse_log_rv:.6f}).']
    text += ['','![Last twelve months](last12_forecasts.png)','','![Complete-model comparison](model_comparison.png)','','## Statistical evidence','', 'MCS uses 10,000 stationary-bootstrap replications, expected block length six months, seed 0 and familywise size 0.05. HAR loss-difference intervals use paired date resampling of seed-averaged losses. Unadjusted pairwise intervals do not establish familywise superiority. A p-value of one is not proof of a unique winner.','',table(mcs_frame),'','## Protocol and limitations','']
    if config['protocol'].startswith('colin'):
        text += ['Colin paper-feature extension: original QR1/QR2 feature subsets, 11-input LSTMX/CRLX, ten macro predictors for HARX/ARMAX. DP/TB differences are fixed from historical conventions. HAR averages are formed causally. Normalized RV is inverted with inherited legacy constants. Quarterly/annual quantum inputs retain their verified historical transformations.','', 'This is retrospective reconstruction: Colin supplied no construction code or scaling metadata; macro publication availability is unverified. FIZ factors through 2024 transition to CIZ in 2025. Missing August factors block dependent September forecasts. Statistical outliers remain in the primary sample. The target differs slightly from the independently rebuilt modern series, so cross-protocol losses are not pooled.']
    else:
        text += ['Modern price-feature extension: seven causal price-derived inputs; scaling calibrated through 2017 and frozen, input clipping retained and targets never clipped. QR1/QR2 share inputs. This is a separate retrospective price-derived benchmark, not the paper macro feature specification. Raw daily snapshot hashes are verified before training.']
    text += ['', f"This run uses rolling windows {config['windows']} and stochastic seeds {config['seeds']}. The month following {config['end']} is unscored. Quantum results use ideal exact simulation, not hardware or trading returns. Numerical failures are retained without substituting forecasts. No window or model is selected for deployment using these results.",'', '## Reproduce','',f'`python run_study.py report --run {folder}`','', 'See manifest.json for identity, inputs, source hashes and environment. See RUN_ORDER.md and docs-colin/AUDIT.md for preparation and source limitations.']
    (folder/'report.md').write_text('\n'.join(text)+'\n')
    import mistune
    html=mistune.create_markdown(plugins=['table'])((folder/'report.md').read_text())
    (folder/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Volatility benchmark</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#193047}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd}img{max-width:100%}h2{margin-top:40px}</style>'+html+'</html>')
    write_json(folder/'report_receipt.json',{'prediction_sha256':digest(folder/'predictions.csv'),'report_sha256':digest(folder/'report.md'),'report_source_sha256':digest(__file__),'records':len(frame),'failures':int(frame.status.ne('ok').sum()),'scored_months':scored.target_month.nunique(),'mcs_reps':10000,'block_length':6,'seed':0})
    print(f'Report: {folder}/report.md')
