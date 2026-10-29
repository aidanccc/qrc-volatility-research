"""Tables, figures and report.md for the IonQ noise study, from cached local and cloud results.

    .venv/bin/python -m analysis.ionq_cloud.analyze --output results/ionq-noise-YYYY-MM-DD

Offline. Reads <output>/cache/local/ (local.py), <output>/cache/cloud/ (submit.py)
and <output>/cache/reference.npz (noiseless exact and Trotter signals; computed here
if missing).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hardware.paper_circuits import PAPER_MODELS, TOTAL, L_OOS, NQUBIT, ROOT, paper_inputs, ideal_features, \
    paper_forecasts
from quantum_reservoir_qiskit import MSE, compute_qlike
from analysis.noise.noise_model import error_parameters, z_expectations, apply_readout
from analysis.ionq_cloud.circuits import z_from_counts
from analysis.ionq_cloud.local import DEPTHS, RATES, SHOTS, rates, readout_flip, simulate, to_signal
from analysis.ionq_cloud.submit import CHECK_MONTHS

NOISES = ['ideal', 'aria-1', 'forte-1']
BENCHMARKS = ['AR1', 'HAR', 'HARX', 'ARMAX']
COLORS = {'exact': '#333333', 'ideal': '#167d9a', 'aria-1': '#d07135', 'forte-1': '#6b648f'}
LABEL = {'exact': 'Exact (authors)', 'ideal': 'Trotter, noiseless', 'aria-1': 'Aria-1 noise', 'forte-1': 'Forte-1 noise'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160,
                     'pdf.fonttype': 42, 'svg.hashsalt': 'ionq-noise', 'pdf.compression': 6})


def _reference_task(task):
    model, steps = task
    data, _, couplings = paper_inputs()
    return task, ideal_features(data, model, couplings[model], steps=steps, order=None if steps is None else 2)


def references(cache):
    """Noiseless signals: exact (authors' matrix exponential) and order-2 Trotter at each depth."""
    path = cache / 'reference.npz'
    if not path.exists():
        tasks = [(m, None) for m in PAPER_MODELS] + sorted({(m, d[m]) for d in DEPTHS.values() for m in PAPER_MODELS})
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            ref = {f'{m}_{"exact" if s is None else f"s{s}"}': sig for (m, s), sig in pool.map(_reference_task, tasks)}
        np.savez(path, **ref)
    return dict(np.load(path))


def cloud_check(cache, tables):
    """Cloud <Z> (1,000 shots) next to the local model of the same short-history circuit."""
    path = tables / 'cloud_check.csv'
    if path.exists():
        return pd.read_csv(path)
    cloud = {}
    for f in sorted((cache / 'cloud').glob('*.json')):
        record = json.loads(f.read_text())
        for name, p in zip(record['names'], record['probabilities']):
            cloud[(record['noise'], name)] = z_from_counts(p)
    rows = []
    for noise, depth, months in sorted({(n, f.stem.split('_')[1], int(f.stem[-1]))
                                        for f in (cache / 'cloud').glob('*.json') for n in [f.stem.split('_')[0]] if n != 'ideal'}):
        for model in PAPER_MODELS:
            steps = DEPTHS[depth][model]
            local = {n: simulate(n, model, depth, CHECK_MONTHS, history=months, workers=14) for n in {'ideal', noise}}
            for i, l in enumerate(CHECK_MONTHS):
                for e in range(PAPER_MODELS[model]['virtual_nodes']):
                    z_cloud = cloud[(noise, f'{model}_s{steps}_k{months}_l{l}_e{e + 1}')]
                    z_local = z_expectations(apply_readout(local[noise][i, e], readout_flip(noise)))
                    z_clean = z_expectations(local['ideal'][i, e])
                    for q in range(NQUBIT):
                        rows.append({'noise': noise, 'model': model, 'depth': depth, 'history_months': months,
                                     'qubits': 7 * months + 3, 'month': l, 'node': e + 1, 'qubit': q,
                                     'cloud': z_cloud[q], 'local': z_local[q], 'noiseless': z_clean[q]})
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, float_format='%.5f')
    return table


def summarize_check(check):
    def stats(g):
        sigma = np.sqrt(np.clip(1 - g.local ** 2, 1e-6, None) / SHOTS)    # 1,000-shot standard error
        through_origin = lambda y: float((g.noiseless * y).sum() / (g.noiseless ** 2).sum())
        return pd.Series({'values': len(g),
                          'mean_abs_z_noiseless': g.noiseless.abs().mean(),
                          'mean_abs_z_local': g.local.abs().mean(), 'mean_abs_z_cloud': g.cloud.abs().mean(),
                          'shrink_local': through_origin(g.local), 'shrink_cloud': through_origin(g.cloud),
                          'mean_abs_cloud_minus_local': (g.cloud - g.local).abs().mean(),
                          'rms_z_score': float(np.sqrt((((g.cloud - g.local) / sigma) ** 2).mean()))})
    return check.groupby(['noise', 'model', 'depth', 'history_months', 'qubits']).apply(stats, include_groups=False).reset_index()


def local_configs(cache):
    """Local noise configurations present in the cache, e.g. aria-1, aria-1-x0.81, forte-1, forte-1-x2.40."""
    names = {f.name.rsplit('_', 2)[0] for f in (cache / 'local').glob('*.npz')}
    return sorted(names, key=lambda n: (n.split('-x')[0], '-x' in n))


def base(noise):
    return noise.split('-x')[0]


def label(noise):
    if '-x' in noise:
        return f"{LABEL[base(noise)]}, cloud-matched (×{noise.split('-x')[1]})"
    return LABEL[noise]


def ideal_probability_check(cache):
    """Cloud 'ideal' returns exact probabilities but drops small outcomes; compare outcome by outcome."""
    rows = []
    for f in sorted((cache / 'cloud').glob('ideal_*.json')):
        record = json.loads(f.read_text())
        depth, months = f.stem.split('_')[1], int(f.stem[-1])
        cloud = dict(zip(record['names'], record['probabilities']))
        for model in PAPER_MODELS:
            steps = DEPTHS[depth][model]
            prob = simulate('ideal', model, depth, CHECK_MONTHS, history=months, workers=5)
            for i, l in enumerate(CHECK_MONTHS):
                for e in range(PAPER_MODELS[model]['virtual_nodes']):
                    c = cloud[f'{model}_s{steps}_k{months}_l{l}_e{e + 1}']
                    diff = [abs(v - prob[i, e][int(b, 2)]) for b, v in c.items()]
                    rows.append({'model': model, 'depth': depth, 'history_months': months, 'month': l, 'node': e + 1,
                                 'outcomes_returned': len(c), 'mass_returned': sum(c.values()),
                                 'max_abs_prob_diff': max(diff), 'mean_abs_prob_diff': float(np.mean(diff))})
    return pd.DataFrame(rows)


def forecast_table(cache, ref, target, y):
    authors = pd.read_csv(ROOT / 'data/predict_result.csv')
    rows, paths = [], {}
    for model in PAPER_MODELS:
        exact_sig = ref[f'{model}_exact']
        exact_pred, exact_logrv = paper_forecasts(exact_sig, target)
        assert np.abs(exact_logrv - authors[model].to_numpy()).max() < 1e-5   # reproduces the paper
        configs = [('exact', 'exact', exact_sig, None)]
        configs += [('ideal', d, ref[f'{model}_s{DEPTHS[d][model]}'], None) for d in DEPTHS]
        for noise in local_configs(cache):
            for depth in DEPTHS:
                z = np.load(cache / 'local' / f'{noise}_{model}_{depth}.npz')
                configs.append((noise, depth, to_signal(z['z_inf'], model), [to_signal(s, model) for s in z['z_shots']]))
        for noise, depth, sig, shot_sigs in configs:
            pred, logrv = paper_forecasts(sig, target)
            paths[f'{model}_{noise}_{depth}'] = pred
            shot_mse = [MSE(paper_forecasts(s, target)[0], y) for s in shot_sigs] if shot_sigs else [np.nan]
            gap = np.abs(logrv - exact_logrv)
            rows.append({'model': model, 'noise': noise, 'depth': depth,
                         'trotter_steps': 0 if depth == 'exact' else DEPTHS[depth][model],
                         # 45 RXX per Trotter step per evolution block; QR1 has 3 blocks, QR2 3 + 4.
                         'rxx_per_month': 0 if depth == 'exact' else DEPTHS[depth][model] * 45 * (3 if model == 'QR1' else 7),
                         'mse': MSE(pred, y), 'qlike': compute_qlike(pred, y),
                         'mse_change_vs_exact': MSE(pred, y) - MSE(exact_pred, y),
                         'mse_1000_shots': float(np.mean(shot_mse)), 'mse_1000_shots_sd': float(np.std(shot_mse)),
                         'max_forecast_gap_vs_exact': gap.max(), 'mean_forecast_gap_vs_exact': gap.mean(),
                         'mean_abs_z': np.abs(sig[:, 3:]).mean(),
                         'mean_feature_error_vs_exact': np.abs(sig - exact_sig)[:, 3:].mean()})
    return pd.DataFrame(rows), paths


SWEEP_SHOTS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]


def shot_sweep(cache, ref, target, y):
    """MSE against shots per circuit at the chosen depth, noiseless and with each noise model.

    Each <Z> is drawn as an independent binomial estimate from its exact value (5 seeds). At
    1,000 shots this is checked against the full multinomial sampling stored by local.py.
    """
    rows = []
    for model in PAPER_MODELS:
        configs = [('ideal', ref[f'{model}_s{DEPTHS["chosen"][model]}'][:, 3:].T)]
        configs += [(n, np.load(cache / 'local' / f'{n}_{model}_chosen.npz')['z_inf']) for n in local_configs(cache)]
        for k, (noise, z) in enumerate(configs):
            for shots in SWEEP_SHOTS:
                rng = np.random.default_rng([k, list(PAPER_MODELS).index(model), shots])
                mse = [MSE(paper_forecasts(to_signal(2 * rng.binomial(shots, (1 + z) / 2) / shots - 1, model),
                                           target)[0], y) for _ in range(5)]
                rows.append({'model': model, 'noise': noise, 'shots': shots,
                             'mse': float(np.mean(mse)), 'mse_sd': float(np.std(mse))})
    return pd.DataFrame(rows)


def save(fig, figdir, name, title):
    fig.suptitle(title, fontsize=13, fontweight='bold', x=.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(figdir / f'{name}.png'); fig.savefig(figdir / f'{name}.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def figures(figdir, summary, paths, dates, y, bench, check, sweep):
    noises = [n for n in summary.noise.unique() if n != 'exact']
    order = [('exact', 'exact')] + [(n, d) for d in DEPTHS for n in noises]

    def bars(ax, s, column, shots=False):
        s = s.set_index(['noise', 'depth']).loc[order].reset_index()
        x = np.arange(len(s))
        drawn = ax.bar(x, s[column], color=[COLORS[base(n)] for n in s.noise], edgecolor='white')
        if shots:
            ax.errorbar(x, s.mse_1000_shots, s.mse_1000_shots_sd, fmt='D', ms=4, color='black', capsize=2,
                        label='1,000 shots (mean of 5 seeds ± s.d.)')
        for patch, noise, depth in zip(drawn, s.noise, s.depth):
            patch.set_alpha(.45 if depth == 'shallow' else 1)
            if '-x' in noise:
                patch.set_hatch('///')
        ax.set_xticks(x, ['exact' if n == 'exact' else ('noiseless' if n == 'ideal' else n.replace('-x', '\ncloud ×'))
                          for n in s.noise], fontsize=7, rotation=90)
        mid = 1 + len(noises)
        ax.axvline(mid - .5, color='#ddd', lw=.8)
        ax.text(1 + (len(noises) - 1) / 2, ax.get_ylim()[1], 'chosen depth', ha='center', va='bottom', fontsize=8)
        ax.text(mid + (len(noises) - 1) / 2, ax.get_ylim()[1], 'shallow (1 step)', ha='center', va='bottom', fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, PAPER_MODELS):
        ax.set_ylim(.095, .13)
        for name, ls in zip(['HAR', 'HARX'], [':', '--']):
            ax.axhline(bench[name], color='#888', lw=1, ls=ls, label=f'{name} benchmark')
        bars(ax, summary[summary.model == model], 'mse', shots=True)
        ax.set_title(model, loc='left'); ax.set_ylabel('MSE (log RV), 245 forecasts')
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    save(fig, figdir, '01_forecast_mse', 'Forecast error under IonQ noise (bars: infinite shots; diamonds: 1,000 shots)')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, PAPER_MODELS):
        ax.set_ylim(0, .8)
        bars(ax, summary[summary.model == model], 'mean_abs_z')
        ax.set_title(model, loc='left'); ax.set_ylabel('mean |⟨Z⟩| over all features')
    save(fig, figdir, '02_feature_contraction', 'Noise shrinks the reservoir features toward zero')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, model in zip(axes, PAPER_MODELS):
        ax.plot(dates, y, color='#c4c4c4', lw=1, label='actual')
        ax.plot(dates, paths[f'{model}_exact_exact'], color=COLORS['exact'], lw=1.1, label=LABEL['exact'])
        for noise in [n for n in noises if '-x' in n]:
            ax.plot(dates, paths[f'{model}_{noise}_chosen'], color=COLORS[base(noise)], lw=1.1, label=label(noise))
        ax.set_title(model, loc='left'); ax.set_ylabel('normalized log RV')
        ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    save(fig, figdir, '03_forecast_paths', 'Out-of-sample forecasts, 1997-08 to 2017-12 (chosen circuits, no shot noise)')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, noise in zip(axes, ['aria-1', 'forte-1']):
        c = check[check.noise == noise]
        ax.plot([-1, 1], [-1, 1], color='#bbb', lw=.8)
        ax.scatter(c.noiseless, c.local, s=5, color='#999', marker='x', label='local model, 1× calibration')
        ax.scatter(c.noiseless, c.cloud, s=5, color=COLORS[noise], alpha=.6, label='IonQ cloud, 1,000 shots')
        ax.set(title=f'noise model "{noise}"', xlabel='noiseless ⟨Z⟩', ylabel='⟨Z⟩ with noise')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    save(fig, figdir, '04_cloud_vs_local', 'IonQ cloud simulator vs local noise model (10- and 17-qubit check circuits)')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, PAPER_MODELS):
        for name, ls in zip(['HAR', 'HARX'], [':', '--']):
            ax.axhline(bench[name], color='#888', lw=1, ls=ls, label=f'{name} benchmark')
        exact = summary[(summary.model == model) & (summary.noise == 'exact')].mse.item()
        ax.axhline(exact, color=COLORS['exact'], lw=1, label=f'{LABEL["exact"]}, infinite shots')
        for noise, g in sweep[sweep.model == model].groupby('noise', sort=False):
            ax.errorbar(g.shots, g.mse, g.mse_sd, color=COLORS[base(noise)], marker='o', ms=3, capsize=2,
                        ls='--' if '-x' in noise else '-', label=label(noise))
        ax.set_xscale('log'); ax.set_title(f'{model} (chosen depth)', loc='left')
        ax.set(xlabel='shots per circuit', ylabel='MSE (log RV), 245 forecasts')
        ax.legend(frameon=False, fontsize=7, loc='upper right')
    save(fig, figdir, '05_mse_vs_shots', 'Shot noise, not gate noise, sets the forecast error (mean of 5 seeds ± s.d.)')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    cache, tables, figdir = out / 'cache', out / 'tables', out / 'figures'
    tables.mkdir(exist_ok=True); figdir.mkdir(exist_ok=True)
    data, target, _ = paper_inputs()
    y = target[TOTAL - L_OOS:]
    dates = pd.to_datetime(data.index[TOTAL - L_OOS:])
    ref = references(cache)
    bench = pd.read_csv(ROOT / 'results/legacy-validation-2026-09-24/legacy_metrics.csv').set_index('model').mse_log_rv

    check = cloud_check(cache, tables)
    summarize_check(check).to_csv(tables / 'cloud_check_summary.csv', index=False, float_format='%.4f')
    if not (tables / 'cloud_ideal_check.csv').exists():
        ideal_probability_check(cache).to_csv(tables / 'cloud_ideal_check.csv', index=False, float_format='%.6g')
    summary, paths = forecast_table(cache, ref, target, y)
    summary.to_csv(tables / 'forecast_summary.csv', index=False, float_format='%.5f')
    pd.DataFrame({'target_month': dates, 'actual': y, **paths}).to_csv(tables / 'forecasts.csv', index=False, float_format='%.6f')
    scales = json.loads((tables / 'cloud_matched_scales.json').read_text())['scale'] if \
        (tables / 'cloud_matched_scales.json').exists() else {}
    params = [{'noise_model': n, **rates(n), **error_parameters(1.0, rates(n))} for n in RATES]
    params += [{'noise_model': f'{n}-x{s:.2f}', **rates(n), **error_parameters(s, rates(n))} for n, s in scales.items()]
    pd.DataFrame(params).to_csv(tables / 'noise_parameters.csv', index=False)
    for f in sorted((cache / 'local').glob('*.npz')):
        model = f.stem.rsplit('_', 2)[1]
        z = np.load(f)['z_inf']
        cols = [f'z{i % NQUBIT}_node{i // NQUBIT + 1}' for i in range(z.shape[1])]
        pd.DataFrame(z, columns=cols, index=pd.Index(range(3, TOTAL), name='column')) \
            .to_csv(tables / f'features_{f.stem}.csv', float_format='%.5f')
    sweep = shot_sweep(cache, ref, target, y)
    sweep.to_csv(tables / 'shot_sweep.csv', index=False, float_format='%.5f')
    figures(figdir, summary, paths, dates, y, bench, check, sweep)
    manifest = {'inputs': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                           ['data/Data.CSV', 'data/coeff_10.jld2', 'data/predict_result.csv',
                            'analysis/noise/aria1_characterizations.json', 'analysis/ionq_cloud/forte1_characterizations.json']},
                'versions': {p: importlib.metadata.version(p) for p in ['qiskit', 'qiskit-aer', 'qiskit-ionq', 'numpy']},
                'cloud_jobs': sorted(json.loads(f.read_text())['job_id'] for f in (cache / 'cloud').glob('*.json')),
                'cloud_matched_scales': scales,
                'benchmarks_mse': {b: float(bench[b]) for b in BENCHMARKS}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pd.set_option('display.width', 250)
    print(summary.round(4).to_string())
    print(sweep.round(4).to_string())


if __name__ == '__main__':
    main()
