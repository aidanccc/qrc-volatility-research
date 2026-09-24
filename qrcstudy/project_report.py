"""Build the concise project-level report from verified completed run artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest,write_json
from .report import table


def summarize(root='.'):
    root=Path(root);out=root/'results/extended-study-2026-09-24';out.mkdir(parents=True,exist_ok=True)
    from .audit_plot import render
    render(root)
    snapshot=root/'data/snapshots/extended-2026-09-24-v2';audit=json.loads((snapshot/'manifest.json').read_text());receipts={};headline=[];coverage=[]
    for label,folder in [('Extended paper features','extended-2026-09-24'),('Modern price features','modern-2026-09-24')]:
        run=root/'results'/folder;receipt=json.loads((run/'report_receipt.json').read_text())
        if digest(run/'predictions.csv')!=receipt['prediction_sha256'] or digest(run/'report.md')!=receipt['report_sha256']:raise ValueError('Report/prediction receipt mismatch')
        receipts[folder]=receipt
        metrics=pd.read_csv(run/'metrics.csv');p=pd.read_csv(run/'predictions.csv')
        for (period,w),g in metrics.loc[metrics.complete].groupby(['period','window']):
            best=g.sort_values('mse_log_rv').iloc[0];r={'protocol':label,'period':period,'window':w,'lowest_MSE_model':best.model,'MSE':best.mse_log_rv,'RMSE':best.rmse_log_rv}
            for model in ['HAR','QR1','QR2']:
                q=g.loc[g.model==model];r[model+'_MSE']=q.mse_log_rv.iloc[0] if len(q) else np.nan
            headline.append(r)
        coverage.append({'protocol':label,'expected_records':receipt['records'],'successful_scored':int((p.actual_log_rv.notna()&p.status.eq('ok')).sum()),'failed_scored':int((p.actual_log_rv.notna()&p.status.eq('failed')).sum()),'unavailable_scored':int((p.actual_log_rv.notna()&p.status.eq('unavailable')).sum()),'successful_unscored':int((p.actual_log_rv.isna()&p.status.eq('ok')).sum()),'unavailable_unscored':int((p.actual_log_rv.isna()&p.status.eq('unavailable')).sum())})
    pd.DataFrame(headline).to_csv(out/'headline_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    legacy=json.loads((root/'results/legacy-validation-2026-09-24/complete.json').read_text())
    text=['# Extended dataset testing — September 24, 2026','','The new data are integrated and tested on the **Vikas** branch of aidanccc/qrc-volatility-research. Original inputs and historical outputs are preserved. This report separates the paper-feature reconstruction from the modern price-feature experiment.','','## Data findings','',f"The input contains {audit['rows']} monthly observations through August 2026. Of 416 missing factor cells, **{audit['filled_factor_cells']} were recovered**; four August values remain unavailable. The prepared copy corrects **{audit['derived_corrections']} derived-feature inconsistencies**. **{audit['outlier_flags']} statistical flags** were retained for review without deleting observations.",'','The 1950–2017 overlap matches the original dataset to floating-point precision. Official FIZ archives verified factor scales before the extension was filled. FIZ observations through 2024 transition to CIZ in 2025. The source includes no generation script or scaling metadata; retrospective macro availability and exact extension normalization remain unverified.','',f"Extended and independently rebuilt modern log-RV targets differ by up to {audit['max_post2017_target_difference']:.8f} after 2017. Results below are evaluated against each protocol's own target; their MSE levels must not be treated as a controlled head-to-head test of feature sets.",'','![Data audit](data_audit.png)','','## Forecast results','',table(pd.DataFrame(headline)),'','Both windows are reported without selecting one after seeing outcomes. Full-period results cover January 2018–August 2026; last12 covers September 2025–August 2026. Lower MSE is a ranking, not proof of superiority.','',table(pd.DataFrame(coverage)),'','August can be scored using July factors. Missing August inputs make dependent September forecasts unavailable; September has no observed target. Numerical fit failures are retained and excluded from complete-model rankings, with additional all-model common-date comparisons.','','## Statistical evidence','']
    for label,folder in [('Extended','extended-2026-09-24'),('Modern','modern-2026-09-24')]:
        m=pd.read_csv(root/'results'/folder/'mcs.csv');main=m.loc[m.scope.eq('complete_period')]
        for (w,loss),g in main.groupby(['window','loss']):
            included=', '.join(g.loc[g.mcs_pvalue>=.05,'model']);text.append(f'- {label}, {w}-month window, {loss}: 95% Model Confidence Set retains {included}.')
    text += ['','MCS uses 10,000 stationary-bootstrap replications with six-month expected blocks. Each report also contains paired HAR loss-difference intervals, seed-level metrics and common-date comparisons. Seeds are repeated model realizations, not additional market histories.','','## Plots','','### Extended paper features','','![Extended forecasts](../extended-2026-09-24/last12_forecasts.png)','','### Modern price features','','![Modern forecasts](../modern-2026-09-24/last12_forecasts.png)','','## Additional tests and interpretation','','The original legacy notebook pipeline was rerun independently. Quantum maximum absolute discrepancies against the supplied author CSV: '+', '.join(f'{k}={v:.3g}' for k,v in legacy['quantum_max_absolute_error'].items())+'. This verifies faithful reconstruction to the previously observed numerical tolerance, not access to unpublished code.','','Exploratory matched-ridge diagnostics use each reservoir’s same three-step raw inputs, training-only standardization, and a penalty selected using 24 strictly preceding forecast errors. They score the same 80 months beginning January 2020. Conditioning, paired block-bootstrap intervals and descriptive outlier slices are recorded separately. They do not replace the primary benchmark.','','## Read and reproduce','','- [Extended report](../extended-2026-09-24/report.md)','- [Modern report](../modern-2026-09-24/report.md)','- [Extended exploration](../extended-exploratory-2026-09-24/report.md)','- [Modern exploration](../modern-exploratory-2026-09-24/report.md)','- [Data audit](../../docs-dataset/AUDIT.md)','- [Run order](../../RUN_ORDER.md)','','Data-source hashes, correction ledgers, complete commit inventories, test evidence, run identities, per-origin forecast records and reproduction commands accompany the results. Quantum results are ideal local simulations and make no hardware-speedup or trading-profit claim.']
    recent=[r for r in headline if r['protocol']=='Extended paper features' and r['period']=='last12']
    full=[r for r in headline if r['protocol']=='Extended paper features' and r['period']=='post2017']
    lead='On the extended dataset, the lowest final-year MSE is '+ '; '.join(f"{r['lowest_MSE_model']} with the {r['window']}-month window ({r['MSE']:.5f})" for r in recent)+'. Over the full 104-month evaluation: '+ '; '.join(f"{r['lowest_MSE_model']} with {r['window']} months ({r['MSE']:.5f})" for r in full)+'. The confidence sets retain multiple classical and quantum models; these results do not establish unique quantum superiority.'
    text[3:3]=['',lead]
    text += ['', '### Exploratory finding', '', 'On the same 80 months, the extended-data 571-month QR2 input-matched raw ridge has MSE 0.13262 versus 0.13925 for the quantum features with matched ridge. The six-month-block confidence interval for quantum minus raw loss is approximately [-0.01385, 0.03010]. This does not establish a reservoir advantage; it motivates keeping simple input-matched baselines in future studies. These are exploratory results, separate from the 104-month headline evaluation.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    import mistune
    html=mistune.create_markdown(plugins=['table'])((out/'report.md').read_text())
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Extended dataset testing</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#193047}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd}img{max-width:100%}h2{margin-top:40px}</style>'+html+'</html>')
    write_json(out/'receipt.json',{'report_sha256':digest(out/'report.md'),'source_sha256':digest(__file__),'run_receipts':receipts,'legacy_complete_sha256':digest(root/'results/legacy-validation-2026-09-24/complete.json')})
    print(out/'report.md')
