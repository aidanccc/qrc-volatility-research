"""Render diagnostic figures and a linked mentor report from recorded tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import mistune
from compute import ROOT, DATA, MODELS, digest, write_json, history_circuit
from circuit_diagram import draw_gate_sequence

COLORS={'QR1':'#167d9a','QR2':'#d07135','HAR':'#6b648f','Persistence':'#78858c'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'figure.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':160,'pdf.fonttype':42})


def clean_drawing(circuit):
    return "\n".join(line.rstrip() for line in str(circuit.draw(output="text", fold=110)).splitlines()) + "\n"


def acf(values,lag):
    # Keep the original monthly index. Dropping missing rows would change lag meaning.
    return pd.Series(np.asarray(values)).autocorr(lag=lag)


def render(out):
    out=Path(out);tables=out/'tables';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    f=pd.read_csv(DATA,index_col=0,parse_dates=True)
    p=pd.read_csv(tables/'validated_predictions.csv',parse_dates=['target_month'])
    val=json.loads((out/'validation.json').read_text()); manifest=json.loads((out/'manifest.json').read_text())
    catalogue=[]
    def save(fig,name,title,caption,source):
        fig.suptitle(title,fontsize=17,fontweight='bold',x=.02,ha='left')
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(figdir/f'{name}.png');fig.savefig(figdir/f'{name}.pdf');plt.close(fig)
        catalogue.append({'id':name,'title':title,'interpretation':caption,'tables':source,'png':f'figures/{name}.png','pdf':f'figures/{name}.pdf'})
    def table(name,frame,index=False):frame.to_csv(tables/f'{name}.csv',index=index)
    def heat(ax,a,labels,title,vmin=None,vmax=None,cmap='RdBu_r'):
        im=ax.imshow(a,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_xticks(range(len(labels)),labels,rotation=60,ha='right');ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title);plt.colorbar(im,ax=ax,shrink=.8)
    union=list(dict.fromkeys(MODELS['QR1']+MODELS['QR2']))
    historical=f.loc[:'2017'];extension=f.loc['2018':]
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    axs[0,0].plot(f.index,f.log_rv,lw=.8,color=COLORS['QR1']);axs[0,0].axvspan(pd.Timestamp('2018-01-01'),f.index[-1],color='#f4e0c6',alpha=.7);axs[0,0].set(ylabel='Inherited log RV',title='Full monthly history; extension shaded')
    axs[0,1].plot(extension.index,np.exp(extension.log_rv),color=COLORS['QR1']);axs[0,1].set(ylabel='RV (exp of inherited log target)',title='Extended period in RV units')
    for label,group,c in [('1950–2017',historical,COLORS['QR1']),('2018–2026',extension,COLORS['QR2'])]:
        axs[1,0].hist(group.log_rv,bins=25,density=True,histtype='step',lw=2,label=label,color=c)
        axs[1,1].plot(range(1,25),[acf(group.log_rv,l) for l in range(1,25)],'o-',ms=3,label=label,color=c)
    axs[1,0].set(xlabel='Inherited log RV',ylabel='Density',title='Distribution by period');axs[1,0].legend()
    axs[1,1].set(xlabel='Lag (months)',ylabel='Autocorrelation',title='Persistence at monthly lags');axs[1,1].axhline(0,c='gray',lw=.7);axs[1,1].legend()
    table('rv_series',f[['RV','log_rv']].assign(rv=np.exp(f.log_rv)),True)
    ac=pd.DataFrame([{'period':label,'lag':l,'acf':acf(group.log_rv,l)} for label,group in [('historical',historical),('extended',extension)] for l in range(1,25)]);table('rv_autocorrelation',ac)
    save(fig, '01_data_overview', 'Volatility', '920 monthly observations. Volatility has sharp spikes and short-term persistence; the extended period has weaker autocorrelation at longer lags.', 'rv_series.csv; rv_autocorrelation.csv')
    raw=pd.read_csv(ROOT/'1950-2026.csv',index_col=0);raw.index=f.index
    missing=pd.DataFrame({'raw_extension':raw.loc['2018':].isna().sum(),'prepared_extension':extension.isna().sum()}).fillna(0)
    table('missingness',missing,True)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    missing.plot.bar(ax=axs[0],color=[COLORS['QR2'],COLORS['QR1']]);axs[0].set(ylabel='Missing cells',title='Original extension versus prepared snapshot');axs[0].tick_params(axis='x',rotation=70)
    axs[1].imshow(extension.isna().T,aspect='auto',cmap='Greys',vmin=0,vmax=1);axs[1].set_yticks(range(len(f.columns)),f.columns);axs[1].set_xticks([0,24,48,72,103],[str(extension.index[i].date())[:7] for i in [0,24,48,72,103]],rotation=30);axs[1].set_title('Prepared missingness: black = missing')
    save(fig, '02_missingness', 'Missing data', 'Preparation reduced missing values from 416 to 4. The remaining August factors prevent September forecasts that need those inputs.', 'missingness.csv')
    ranges=[]
    for c in union:
        lo,hi=historical[c].min(),historical[c].max();valid=extension[c].dropna()
        ranges.append({'feature':c,'historical_min':lo,'historical_max':hi,'extended_min':valid.min(),'extended_max':valid.max(),'outside_historical_range':int(((valid<lo)|(valid>hi)).sum()),'valid_extended':len(valid),'outside_docstring_minus1_zero':int(((f[c]<-1)|(f[c]>0)).sum())})
    ranges=pd.DataFrame(ranges);table('encoding_ranges',ranges)
    angles=f[union]*np.pi;table('encoded_angles',angles,True)
    fig,axs=plt.subplots(3,3,figsize=(14,10))
    for ax,c in zip(axs.flat,union):
        ax.hist(historical[c]*np.pi,bins=25,density=True,histtype='step',color=COLORS['QR1'],label='1950–2017')
        ax.hist(extension[c].dropna()*np.pi,bins=20,density=True,histtype='step',color=COLORS['QR2'],label='2018–2026')
        ax.axvline(historical[c].min()*np.pi,color='gray',ls=':',lw=.8);ax.axvline(historical[c].max()*np.pi,color='gray',ls=':',lw=.8)
        ax.set(title=c,xlabel='RY angle (radians)',ylabel='Density')
    axs.flat[0].legend(fontsize=8)
    save(fig, '03_encoding_ranges', 'Input angles', 'Each input becomes a rotation angle: π × value. Four DP values and two IP values exceed their historical ranges; inputs were not clipped.', 'encoding_ranges.csv; encoded_angles.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,label,group in [(axs[0],'1950–2017',historical),(axs[1],'2018–2026',extension)]:
        corr=group[union].corr();heat(ax,corr,union,label,-1,1);table(f'input_correlation_{label[:4]}',corr,True)
    save(fig, '04_input_correlations', 'Input correlations', 'DP and EP are strongly correlated. Some relationships change in the extended period, so historical input relationships may not remain stable.', 'input_correlation_1950.csv; input_correlation_2018.csv')
    fig,axs=plt.subplots(1,3,figsize=(14,5))
    for ax,name,title in [(axs[0],'extended_couplings_seed0','Extended: seed 0, both QR models'),(axs[1],'historical_couplings_QR1','Historical QR1: JLD2 matrix 0'),(axs[2],'historical_couplings_QR2','Historical QR2: JLD2 matrix 1')]:
        a=pd.read_csv(tables/f'{name}.csv').to_numpy();heat(ax,a,list(range(10)),title,0,.24,'viridis');ax.axhline(6.5,c='white',lw=1);ax.axvline(6.5,c='white',lw=1)
    save(fig, '05_couplings', 'Qubit couplings', 'All 10 qubits interact: 45 pairs per Trotter step. White lines separate seven input qubits from three memory qubits.', 'extended_couplings_seed0.csv; historical_couplings_QR1.csv; historical_couplings_QR2.csv')
    features={m:pd.read_csv(tables/f'{m}_features_seed0.csv',index_col=0,parse_dates=True) for m in MODELS}
    fig,axs=plt.subplots(2,1,figsize=(14,8))
    for ax,(m,a) in zip(axs,features.items()):
        b=a.loc['2018':];im=ax.imshow(b.T,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set_yticks(range(len(a.columns)),a.columns,fontsize=8);positions=np.linspace(0,len(b)-1,7).round().astype(int);ax.set_xticks(positions,[str(b.index[i].date())[:7] for i in positions]);ax.set_title(f'{m}: seed 0, feature aligned to target month');plt.colorbar(im,ax=ax,label='Expected Z')
    save(fig, '06_feature_traces', 'Quantum outputs', 'Each row is a qubit’s Z expectation at one readout time. Several channels change little across months, suggesting overlapping or weakly varying information.', 'QR1_features_seed0.csv; QR2_features_seed0.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,(m,a) in zip(axs,features.items()):
        ax.bar(a.columns,a.var(ddof=0),color=COLORS[m]);ax.tick_params(axis='x',rotation=75);ax.set(title=m,ylabel='Feature variance (log scale), full valid history',yscale='log')
    save(fig, '07_feature_variance', 'Output variance', 'The least-variable channels have variances of 0.000081 (QR1) and 0.000024 (QR2). Small changes may be difficult to resolve with limited shots.', 'feature_variance.csv')
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for ax,(m,a) in zip(axs,features.items()):
        corr=a.corr();heat(ax,corr,list(a.columns),m,-1,1);ax.tick_params(labelsize=7);table(f'{m}_feature_correlation',corr,True)
    save(fig, '08_feature_correlations', 'Output correlations', 'Several memory-qubit outputs are strongly correlated. This suggests redundancy, but removing outputs or qubits requires testing forecast accuracy.', 'QR1_feature_correlation.csv; QR2_feature_correlation.csv')
    spectrum=[];fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,(m,a) in zip(axs,features.items()):
        train=a.loc[:'2017'];scaled=(train-train.mean())/train.std(ddof=0)
        for label,x,c in [('Raw',train.to_numpy(),COLORS[m]),('Centered, standardized',scaled.to_numpy(),'#414c56')]:
            s=np.linalg.svd(x,compute_uv=False);ax.semilogy(np.arange(1,len(s)+1),s/s[0],'o-',label=label,color=c)
            spectrum.extend({'model':m,'representation':label,'component':i+1,'singular_value':z,'relative_singular_value':z/s[0]} for i,z in enumerate(s))
        ax.set(title=f'{m}: pre-2018 features only',xlabel='Singular-value index',ylabel='Singular value / largest');ax.legend(fontsize=9)
    save(fig, '09_feature_spectra', 'Feature redundancy', 'Small singular values indicate nearly overlapping feature directions. QR2 has more of these directions; extra outputs may add little independent information.', 'feature_spectra.csv')
    cond=pd.read_csv(tables/'rolling_conditioning.csv',parse_dates=['target_month']);fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,w in zip(axs,[120,571]):
        for m in MODELS:
            g=cond.loc[(cond.model==m)&(cond.window==w)];ax.semilogy(g.target_month,g.condition,label=m,color=COLORS[m])
        ax.set(title=f'{w}-month training window',ylabel='Condition number of feature matrix');ax.legend()
    summary=cond.groupby(['model','window']).condition.agg(['min','median','max']).reset_index();table('conditioning_summary',summary)
    save(fig, '10_conditioning', 'Readout conditioning', 'Higher values mean greater sensitivity to small input errors. Median condition numbers are about 967 vs 5,982 for QR1 vs QR2 at 120 months—roughly a sixfold difference.', 'rolling_conditioning.csv; conditioning_summary.csv')
    states=pd.read_csv(tables/'hidden_states.csv');fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,m in zip(axs,MODELS):
        for date,g in states.loc[states.model==m].groupby('target_month'):ax.plot([0]+g.stage.tolist(),[1]+g.hidden_purity.tolist(),'-o',alpha=.45,ms=3,color=COLORS[m])
        ax.axhline(1/8,color='gray',ls=':',label='Maximally mixed 3-qubit state');ax.set(title=m,xticks=[0,1,2,3],xlabel='Inputs processed (0 = fresh memory)',ylabel='Hidden-state purity',ylim=(.1,1.03));ax.legend(fontsize=8)
    save(fig, '11_hidden_purity', 'Memory-state purity', 'Across 12 histories, average purity falls from 1 to 0.509 (QR1) and 0.484 (QR2). Memory becomes mixed even without hardware noise; purity alone does not measure usefulness.', 'hidden_states.csv; selected_histories.csv')
    scored=p.loc[p.status.eq('ok')&p.actual_log_rv.notna()].copy();scored['residual']=scored.predicted_log_rv-scored.actual_log_rv;scored['squared_error']=scored.residual**2
    q=scored.loc[scored.model.isin(MODELS)]
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,m in enumerate(MODELS):
        for col,w in enumerate([120,571]):
            ax=axs[row,col];g=q.loc[(q.model==m)&(q.window==w)];wide=g.pivot(index='target_month',columns='seed',values='predicted_log_rv');actual=g.groupby('target_month').actual_log_rv.first();ax.plot(actual.index,actual,color='#27323b',lw=1,label='Observed')
            for seed in wide:ax.plot(wide.index,wide[seed],color=COLORS[m],alpha=.2,lw=.8)
            ax.plot(wide.index,wide[0],color=COLORS[m],lw=1.2,label='Seed 0; others faint');ax.set(title=f'{m} · {w} months',ylabel='Inherited log RV');ax.legend(fontsize=8)
    save(fig, '12_forecasts', 'Forecasts', 'Black shows observed volatility; colored lines show five reservoir seeds. Both models miss some large spikes. Faint lines show seed variation, not confidence intervals.', 'validated_predictions.csv')
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,m in enumerate(MODELS):
        for col,w in enumerate([120,571]):
            ax=axs[row,col];g=q.loc[(q.model==m)&(q.window==w)];wide=g.pivot(index='target_month',columns='seed',values='residual')
            for seed in wide:ax.plot(wide.index,wide[seed],color=COLORS[m],alpha=.3,lw=.8)
            ax.axhline(0,c='#27323b',lw=.8);ax.set(title=f'{m} · {w} months',ylabel='Predicted − observed log RV')
    save(fig, '13_residual_timeline', 'Forecast errors', 'Error = predicted − observed log volatility. The largest errors cluster around the 2020 volatility shock; positive errors mean overprediction.', 'validated_predictions.csv')
    residual_acf=[];fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,w in enumerate([120,571]):
        for m in MODELS:
            g=q.loc[(q.model==m)&(q.window==w)];axs[row,0].hist(g.residual,bins=30,density=True,histtype='step',color=COLORS[m],label=m,lw=1.5)
            curves=[]
            for seed,s in g.groupby('seed'):
                s=s.set_index('target_month').residual.reindex(pd.date_range('2018-01-31','2026-08-31',freq='ME'));a=[acf(s,l) for l in range(1,13)];curves.append(a)
                residual_acf.extend({'model':m,'window':w,'seed':seed,'lag':l,'acf':z} for l,z in enumerate(a,1))
            arr=np.array(curves);axs[row,1].plot(range(1,13),arr.mean(0),'-o',ms=3,color=COLORS[m],label=m);axs[row,1].fill_between(range(1,13),arr.min(0),arr.max(0),color=COLORS[m],alpha=.12)
        axs[row,0].set(title=f'{w} months: pooled seed errors',xlabel='Residual (log RV)',ylabel='Density');axs[row,0].legend()
        axs[row,1].set(title=f'{w} months: mean / range across seeds',xlabel='Lag (months)',ylabel='Residual autocorrelation');axs[row,1].axhline(0,c='gray',lw=.7);axs[row,1].legend()
    save(fig, '14_residual_distribution', 'Error distribution and autocorrelation', 'Errors have tails and some repeated lag patterns. Shading shows variation across seeds, not confidence intervals.', 'validated_predictions.csv; residual_autocorrelation.csv')
    rolling=[];fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,w in zip(axs,[120,571]):
        for m in ['QR1','QR2','HAR','Persistence']:
            g=scored.loc[(scored.model==m)&(scored.window==w)].groupby('target_month').squared_error.mean().reindex(pd.date_range('2018-01-31','2026-08-31',freq='ME'));r=g.rolling(12,min_periods=12).mean();ax.plot(r.index,r,label=m,color=COLORS[m]);rolling.extend({'model':m,'window':w,'target_month':str(d.date()),'rolling_mse':z} for d,z in r.dropna().items())
        ax.set(title=f'{w}-month training window',ylabel='Trailing 12-month MSE (log RV)');ax.legend(fontsize=8)
    save(fig, '15_rolling_error', 'Rolling error', 'Trailing 12-month squared error peaks around 2020–2021. Model rankings change over time; neither quantum model consistently has the lowest error.', 'rolling_error.csv')
    thresholds=historical.log_rv.quantile([1/3,2/3]).to_numpy();scored['regime']=pd.cut(scored.actual_log_rv,[-np.inf,*thresholds,np.inf],labels=['Low','Middle','High'])
    regime=scored.groupby(['model','window','regime'],observed=True).agg(mse=('squared_error','mean'),months=('target_month','nunique'),records=('squared_error','size')).reset_index();table('error_by_regime',regime)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,w in zip(axs,[120,571]):
        g=regime.loc[(regime.window==w)&regime.model.isin(COLORS)].pivot(index='regime',columns='model',values='mse').reindex(['Low','Middle','High']);g.plot.bar(ax=ax,color=[COLORS[c] for c in g.columns]);ax.set(title=f'{w}-month window',ylabel='Mean squared log-RV error',xlabel='Realized target regime (descriptive)');ax.tick_params(axis='x',rotation=0)
    save(fig, '16_regime_errors', 'Error by volatility level', 'Errors are lowest in the middle-volatility group. Low and high volatility are harder to forecast; groups use historical thresholds and observed outcomes.', 'error_by_regime.csv')
    coverage=p.groupby(['model','window','status']).size().unstack(fill_value=0).reset_index();table('forecast_coverage',coverage)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,w in zip(axs,[120,571]):
        g=coverage.loc[coverage.window==w].set_index('model');cols=[c for c in ['ok','failed','unavailable'] if c in g];g[cols].plot.bar(ax=ax,stacked=True,color=['#167d9a','#bc4b51','#dbb47b']);ax.set(title=f'{w}-month window (includes September)',ylabel='Forecast records, including seeds');ax.tick_params(axis='x',rotation=55)
    save(fig, '17_coverage', 'Forecast availability', 'Two ARMAX fits failed and 44 records were unavailable. Counts include unscored September forecasts; models with five seeds have more records.', 'forecast_coverage.csv; validated_predictions.csv')
    errors=pd.read_csv(tables/'trotter_observable_errors.csv');summary_e=errors.groupby(['model','steps']).absolute_error.agg(['mean','max']).reset_index();table('trotter_error_summary',summary_e)
    resources=pd.read_csv(tables/'circuit_resources.csv');cost=resources.groupby(['model','steps']).agg(total_rxx=('rxx','sum'),max_circuit_depth=('depth','max'),executions_per_shot=('endpoint','size')).reset_index();table('circuit_total_cost',cost)
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for m in MODELS:
        g=summary_e.loc[summary_e.model==m];axs[0].plot(g.steps,g['mean'],'o-',label=f'{m} mean',color=COLORS[m]);axs[0].plot(g.steps,g['max'],'o--',label=f'{m} max',color=COLORS[m]);r=cost.loc[cost.model==m];axs[1].plot(r.steps,r.total_rxx,'o-',label=m,color=COLORS[m]);axs[2].plot(r.total_rxx,g['mean'],'o-',label=m,color=COLORS[m])
    axs[0].set(xlabel='Trotter steps',ylabel='Absolute Z-expectation error',title='12 histories × every readout');axs[1].set(xlabel='Trotter steps',ylabel='RXX gates, summed over endpoints',title='Complete 3-month forecast');axs[2].set(xlabel='RXX gates per shot across endpoints',ylabel='Mean absolute Z error',title='Accuracy versus logical cost')
    for ax in axs:ax.legend(fontsize=8)
    save(fig, '18_accuracy_cost', 'Trotter error and gate count', 'From 1 to 8 steps, mean Z error falls from 0.0704 to 0.0050 (QR1) and 0.0236 to 0.0054 (QR2), while gate counts rise eightfold. These are logical gates, not compiled IonQ counts.', 'trotter_error_summary.csv; circuit_resources.csv; circuit_total_cost.csv')
    shot=pd.read_csv(tables/'shot_uncertainty.csv');shot_summary=shot.groupby(['model','shots']).standard_error.agg(['median','max']).reset_index();table('shot_summary',shot_summary)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,m in zip(axs,MODELS):
        s=shot_summary.loc[shot_summary.model==m];ax.loglog(s.shots,s['median'],'o-',color=COLORS[m],label='Median marginal SE');ax.loglog(s.shots,s['max'],'o--',color=COLORS[m],label='Largest marginal SE');ax.loglog(s.shots,1/np.sqrt(s.shots),':',c='gray',label='Worst-case 1 / sqrt(shots)');ax.set(title=m,xlabel='Shots per endpoint',ylabel='Standard error of mean Z');ax.legend(fontsize=8)
    save(fig, '19_shot_uncertainty', 'Sampling uncertainty', 'Worst-case Z standard error is 0.10, 0.032, and 0.01 at 100, 1,000, and 10,000 shots. This estimates sampling uncertainty only; it excludes hardware noise.', 'shot_uncertainty.csv; shot_summary.csv')
    # A compressed, annotated circuit view, paired with exact Qiskit text drawings.
    fig,ax=plt.subplots(figsize=(15,7));ax.set_xlim(-1.7,15.5);ax.set_ylim(-2,11);ax.axis('off')
    for q in range(10):
        y=9-q;ax.plot([0,14.3],[y,y],c='#b8c2c8',lw=1);ax.text(-.2,y,f'q{q}  '+('input' if q<7 else 'memory'),ha='right',va='center',fontsize=9)
    def block(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#41515d'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,rotation=90 if label=='reset' else 0)
    for x,label in [(0,'RY(πx)\nt−3'),(4.2,'RY(πx)\nt−2'),(8.4,'RY(πx)\nt−1')]:block(x,2.7,1.2,6.6,label,'#d9edf1')
    for x in [1.6,5.8]:block(x,-.3,1.5,9.6,'U(τ)\nXX + Z','#e3e2ef')
    for x in [3.4,7.6]:block(x,2.7,.4,6.6,'reset','#f4e0c6')
    block(10,-.3,1.8,9.6,'dU(τ/V)\nrepeat to\nendpoint v','#e3e2ef');block(12.3,-.3,1.6,9.6,'Measure\nall Z','#d9edf1')
    ax.text(0,10.1,'Memory starts at |000⟩ for each forecast.',fontsize=11)
    ax.text(0,-1.25,'Reset inputs between months. QR2: one complete run per readout time.',fontsize=10)
    save(fig, '20_circuit_protocol', 'Circuit', 'Seven inputs pass through three months of evolution with three memory qubits. Inputs reset between months. QR2 needs separate runs for its two readout times; backend support remains unverified.', 'example_three_month_inputs.csv; example_three_month_angles.csv; circuit_resources.csv')
    circuits=out/'circuits';circuits.mkdir(exist_ok=True)
    values=pd.read_csv(tables/'example_three_month_inputs.csv',index_col=0).to_numpy();j=pd.read_csv(tables/'extended_couplings_seed0.csv').to_numpy()
    fig,gate_table=draw_gate_sequence(j)
    table('gate_sequence',gate_table)
    save(fig, '21_gate_sequence', 'Gate sequence', 'RY encodes each input. One Trotter step applies 45 pairwise RXX gates, then RZ on every qubit. Only this evolution block repeats; measurements occur at the final readout.', 'gate_sequence.csv')
    from quantum_reservoir_trotter import build_trotter_step
    (circuits/'one_trotter_step.txt').write_text(clean_drawing(build_trotter_step(10,j,1)))
    for v in [1,2]:
        # QR2 drawing uses its own feature set at the same recorded target history.
        selected=pd.read_csv(tables/'selected_histories.csv');t=int(selected.iloc[0].position);values=f[MODELS[f'QR{v}']].iloc[t-3:t].to_numpy()
        for endpoint in range(1,v+1):(circuits/f'QR{v}_endpoint{endpoint}.txt').write_text(clean_drawing(history_circuit(values,j,1,v,endpoint)))
    write_json(out/'figure_index.json',catalogue)
    report(out,catalogue,ranges,summary,summary_e,cost,shot_summary,val,manifest)
    print(f'Rendered {len(catalogue)} figures in PNG and PDF; report.html and report.md',flush=True)


def report(out, catalogue, *unused):
    text = '# Plots\n\n'
    ordered = sorted(catalogue, key=lambda item: (0 if item['id'].startswith(('20_', '21_')) else 1, item['id']))
    for item in ordered:
        text += f"![{item['title']}]({item['png']})\n\n{item['interpretation']}\n\n"
        links = [f"[PDF]({item['pdf']})"]
        sources = item['tables'].split('; ')
        links += [f"[Data{f' {i+1}' if len(sources)>1 else ''}](tables/{name})"
                  for i, name in enumerate(sources)]
        text += ' · '.join(links) + '\n\n'
    (out/'FUTURE_CODEX_PROMPTS.md').write_text((ROOT/'analysis/diagnostics/FUTURE_CODEX_PROMPTS.md').read_text())
    (out/'report.md').write_text(text.rstrip() + '\n')
    body = mistune.create_markdown()(text)
    css = ('body{font:16px/1.5 system-ui,sans-serif;color:#222;margin:0;background:white}'
           'main{max-width:1080px;margin:auto;padding:24px}'
           'h1{font-size:24px;font-weight:600;margin:0 0 24px}'
           'img{display:block;width:100%;height:auto;margin-top:36px}'
           'p{margin:12px 0}a{color:#356379;font-size:13px}'
           '@media print{img{break-inside:avoid}main{padding:0}}')
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Plots</title><style>'+css+'</style><main>'+body+'</main></html>')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.output)
