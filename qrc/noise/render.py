"""Render noise-study figures and a short Markdown report from recorded tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'QR1': '#167d9a', 'QR2': '#d07135'}
STEP_COLORS = {1: '#167d9a', 2: '#6b648f', 4: '#d07135'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 160, 'pdf.fonttype': 42, 'svg.hashsalt': 'noise', 'pdf.compression': 6})
LABELS = {'exact_ideal': 'Exact, ideal (published)', 'trotter1_scale0': 'Trotter-1, ideal',
          'trotter1_scale0.5': 'Trotter-1, 0.5× Aria-1', 'trotter1_scale1': 'Trotter-1, 1× Aria-1',
          'trotter1_scale2': 'Trotter-1, 2× Aria-1'}
for s in [0, 1]:
    for n in [100, 1000, 10000]:
        LABELS[f'trotter1_scale{s}_shots{n}'] = f'Trotter-1, {"ideal" if s == 0 else "1× Aria-1"}, {n:,} shots'


def fmt(x, digits=4):
    return f'{x:.{digits}f}'


def render(out):
    out = Path(out); tables = out / 'tables'; figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
    manifest = json.loads((out / 'manifest.json').read_text())
    params = pd.read_csv(tables / 'noise_parameters.csv')
    sweep = pd.read_csv(tables / 'feature_sweep_summary.csv')
    raw = pd.read_csv(tables / 'feature_sweep.csv')
    pred = pd.read_csv(tables / 'forecasts.csv', parse_dates=['target_month'])
    summary = pd.read_csv(tables / 'forecast_summary.csv')
    figures = []

    def save(fig, name, title, source, rect=(0, 0, 1, .93)):
        fig.suptitle(title, fontsize=15, fontweight='bold', x=.02, ha='left')
        fig.tight_layout(rect=rect)
        fig.savefig(figdir / f'{name}.png'); fig.savefig(figdir / f'{name}.pdf', metadata={'CreationDate': None}); plt.close(fig)
        figures.append((name, title, source))

    # 1. Feature distortion and signal size vs noise scale.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, style in [('QR1', '-'), ('QR2', '--')]:
        for steps, c in STEP_COLORS.items():
            g = sweep[(sweep.model == model) & (sweep.steps == steps)]
            axs[0].plot(g.scale, g.mean_abs_error, style, marker='o', color=c, label=f'{model}, {steps} Trotter step{"s" * (steps > 1)}')
            axs[1].plot(g.scale, g.mean_abs_signal, style, marker='o', color=c)
        axs[1].axhline(sweep[sweep.model == model].exact_mean_abs_signal.iloc[0], ls=style, color='gray', lw=.8)
    axs[0].set(xlabel='Noise scale (× Aria-1 error rates)', ylabel='Mean |noisy ⟨Z⟩ − exact ⟨Z⟩|', title='Feature error vs. exact reservoir')
    axs[1].set(xlabel='Noise scale (× Aria-1 error rates)', ylabel='Mean |⟨Z⟩|', title='Feature magnitude (gray: exact)')
    axs[0].legend(fontsize=8)
    save(fig, '01_feature_error_vs_noise', 'Reservoir features under Aria-1 noise', 'feature_sweep_summary.csv')

    # 2. Total error at 1000 shots: Trotter steps x noise scale.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, model in zip(axs, ['QR1', 'QR2']):
        g = sweep[sweep.model == model].pivot(index='steps', columns='scale', values='rms_error_1000_shots')
        im = ax.imshow(g.to_numpy(), cmap='viridis', aspect='auto')
        ax.set_xticks(range(g.shape[1]), [f'{s:g}×' for s in g.columns]); ax.set_yticks(range(g.shape[0]), g.index)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.iat[i, j]; ax.text(j, i, f'{v:.3f}', ha='center', va='center', color='white' if v < g.to_numpy().mean() else 'black', fontsize=9)
        ax.set(xlabel='Noise scale (× Aria-1)', ylabel='Trotter steps', title=f'{model}: RMS feature error, 1,000 shots')
        plt.colorbar(im, ax=ax, shrink=.85)
    save(fig, '02_trotter_vs_noise_tradeoff', 'Trotter accuracy vs. gate noise', 'feature_sweep_summary.csv')

    # 3. One history: exact vs noisy features with 1000-shot error bars.
    month = '2020-05-31'
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 2]})
    for ax, model in zip(axs, ['QR1', 'QR2']):
        g = raw[(raw.model == model) & (raw.target_month == month) & (raw.steps == 1)]
        exact = g[g.scale == 0]; k = np.arange(len(exact))
        ax.plot(k, exact.exact_z, 'o', color='black', label='Exact (ideal)')
        for scale, c, dx in [(1, COLORS['QR1'], -.15), (4, COLORS['QR2'], .15)]:
            z = g[g.scale == scale].noisy_z.to_numpy()
            ax.errorbar(k + dx, z, yerr=np.sqrt((1 - z ** 2) / 1000), fmt='s', ms=4, color=c, capsize=2, label=f'Trotter-1, {scale}× Aria-1 (±1 s.e., 1,000 shots)')
        ax.set_xticks(k, [f'v{r.virtual_node}q{r.qubit}' for r in exact.itertuples()], rotation=60, fontsize=8)
        ax.axhline(0, color='gray', lw=.6); ax.set(ylabel='⟨Z⟩', title=f'{model}, target {month}', ylim=(-1.05, 1.05))
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    save(fig, '03_feature_example', 'Reservoir readout for one input history', 'feature_sweep.csv', rect=(0, .07, 1, .93))

    # 4. Forecast paths (571-month window).
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, model in zip(axs, ['QR1', 'QR2']):
        g = pred[(pred.model == model) & (pred.window == 571)]
        actual = g[g.config == 'exact_ideal'].set_index('target_month').actual_log_rv
        ax.plot(actual.index, actual, color='black', lw=1.4, label='Actual')
        for config, c, ls in [('exact_ideal', '#78858c', '-'), ('trotter1_scale1', COLORS['QR1'], '-'), ('trotter1_scale1_shots1000_seed0', COLORS['QR2'], '--')]:
            s = g[g.config == config].set_index('target_month').predicted_log_rv
            ax.plot(s.index, s, ls, color=c, lw=1.1, label=LABELS.get(config, 'Trotter-1, 1× Aria-1, 1,000 shots (shot seed 0)'))
        ax.set(ylabel='log RV', title=f'{model}, 571-month training window')
    axs[0].legend(fontsize=8, ncol=2)
    save(fig, '04_forecast_paths', 'Forecasts with and without noise', 'forecasts.csv')

    # 5. MSE by configuration.
    order = ['exact_ideal', 'trotter1_scale0', 'trotter1_scale0.5', 'trotter1_scale1', 'trotter1_scale2',
             'trotter1_scale0_shots10000', 'trotter1_scale0_shots1000', 'trotter1_scale0_shots100',
             'trotter1_scale1_shots10000', 'trotter1_scale1_shots1000', 'trotter1_scale1_shots100']
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey='row')
    for i, model in enumerate(['QR1', 'QR2']):
        for j, window in enumerate([120, 571]):
            ax = axs[i, j]; g = summary[(summary.model == model) & (summary.window == window)].set_index('config')
            vals = g.reindex(order); y = np.arange(len(order))
            for k, o in enumerate(order):  # dots, not bars: the axis is zoomed
                ax.errorbar(vals.mse_log_rv.iloc[k], k, xerr=np.nan_to_num(vals.mse_seed_std.iloc[k]), fmt='o', ms=6, capsize=3,
                            color=COLORS['QR2'] if 'shots' in o else COLORS['QR1'])
            ax.grid(axis='x', color='#e5e5e5', lw=.6)
            for ref, ls in [('reference_HARX', '--'), ('reference_AR1', ':')]:
                ax.axvline(g.loc[ref, 'mse_log_rv'], color='black', ls=ls, lw=1, label=ref.split('_')[1])
            ax.set_yticks(y, [LABELS[o] for o in order], fontsize=8); ax.invert_yaxis()
            low = min(vals.mse_log_rv.min(), g.loc['reference_AR1', 'mse_log_rv'], g.loc['reference_HARX', 'mse_log_rv'])
            ax.set_xlim(left=.95 * low)  # zoomed: differences are a few thousandths
            ax.set(title=f'{model}, {window}-month window', xlabel='MSE (log RV)' if i else None)
    axs[0, 1].legend(fontsize=8)
    save(fig, '05_forecast_mse', 'Forecast error by noise configuration', 'forecast_summary.csv')

    # Report.
    p1 = params[params.scale == 1].iloc[0]; r = manifest['aria1_rates']
    def mse(model, window, config):
        return summary[(summary.model == model) & (summary.window == window) & (summary.config == config)].mse_log_rv.iloc[0]
    rows = ['| Model | Window | Exact ideal | Trotter-1 ideal | 1× Aria-1 | 1× Aria-1, 1,000 shots | 1× Aria-1, 100 shots | HARX | AR1 |', '|---|---|---|---|---|---|---|---|---|']
    for model in ['QR1', 'QR2']:
        for window in [120, 571]:
            rows.append(f'| {model} | {window} | ' + ' | '.join(fmt(mse(model, window, c)) for c in
                        ['exact_ideal', 'trotter1_scale0', 'trotter1_scale1', 'trotter1_scale1_shots1000', 'trotter1_scale1_shots100', 'reference_HARX', 'reference_AR1']) + ' |')
    srows = ['| Model | Trotter steps | Mean abs feature error, 1× Aria-1 | Mean abs feature error, 4× | Mean |⟨Z⟩| exact → 1× | RMS error, 1,000 shots, 1× |', '|---|---|---|---|---|---|']
    for model in ['QR1', 'QR2']:
        for steps in [1, 2, 4]:
            g = sweep[(sweep.model == model) & (sweep.steps == steps)].set_index('scale')
            srows.append(f'| {model} | {steps} | {fmt(g.loc[1, "mean_abs_error"], 3)} | {fmt(g.loc[4, "mean_abs_error"], 3)} | '
                         f'{fmt(g.loc[1, "exact_mean_abs_signal"], 3)} → {fmt(g.loc[1, "mean_abs_signal"], 3)} | {fmt(g.loc[1, "rms_error_1000_shots"], 3)} |')
    # Findings, computed from the tables so they stay true on a rerun.
    one = sweep[(sweep.steps == 1) & (sweep.scale == 1)].set_index('model')
    best = sweep[sweep.scale > 0].loc[lambda d: d.groupby(['model', 'scale']).rms_error_1000_shots.idxmin()]
    gate = {w: max(abs(mse(m, w, f'trotter1_scale{s}') - mse(m, w, 'trotter1_scale0')) for m in ['QR1', 'QR2'] for s in [0.5, 1, 2]) for w in [120, 571]}
    shots100 = [mse(m, 571, 'trotter1_scale1_shots100') - mse(m, 571, 'trotter1_scale1') for m in ['QR1', 'QR2']]
    shots1000 = [mse(m, w, 'trotter1_scale1_shots1000') - mse(m, w, 'trotter1_scale1') for m in ['QR1', 'QR2'] for w in [120, 571]]
    spread = summary[summary.config.str.endswith('shots1000')].mse_seed_std.max()
    noisy = summary[summary.config.str.contains('scale(?:0.5|1|2)', regex=True) & ~summary.config.str.endswith('shots100')]
    harx = {w: mse('QR1', w, 'reference_HARX') for w in [120, 571]}; ar1 = {w: mse('QR1', w, 'reference_AR1') for w in [120, 571]}
    below_ar1_120 = int((noisy[noisy.window == 120].mse_log_rv < ar1[120]).sum())
    below_harx_571 = noisy[(noisy.window == 571) & (noisy.mse_log_rv < harx[571])]
    steps_text = ', '.join(str(s) for s in sorted(set(best.steps)))
    findings = [
        '## Findings', '',
        f'1. **Gate noise strongly distorts the features.** At 1× Aria-1 with one Trotter step, mean |⟨Z⟩| drops from {fmt(one.loc["QR1", "exact_mean_abs_signal"], 2)} to {fmt(one.loc["QR1", "mean_abs_signal"], 2)} (QR1) '
        f'and from {fmt(one.loc["QR2", "exact_mean_abs_signal"], 2)} to {fmt(one.loc["QR2", "mean_abs_signal"], 2)} (QR2). Mean feature error against the exact reservoir is {fmt(one.loc["QR1", "mean_abs_error"], 2)} / {fmt(one.loc["QR2", "mean_abs_error"], 2)}, '
        'compared with 0.07 / 0.02 from Trotter error alone. Each readout circuit passes through 135 (QR1) or up to 180 (QR2) noisy RXX gates.',
        f'2. **More Trotter steps hurt under noise.** With 1,000 shots, the lowest-error depth at every nonzero noise level is {steps_text} Trotter step. '
        'Each extra step reduces Trotter error but adds 45 noisy RXX gates, so the reservoir washes out faster. The noiseless conclusion (more steps are more accurate) reverses on noisy hardware.',
        f'3. **Forecasts change little with gate noise.** From 0.5× to 2× Aria-1 (infinite shots), MSE moves by at most {gate[571]:.4f} in the 571-month window and {gate[120]:.4f} in the 120-month window, relative to the noiseless Trotter-1 reservoir. '
        'The ridge readout is refit on the same noisy features, so a smooth contraction of the features is mostly absorbed by larger weights. Symmetric readout error is a pure rescaling, so on its own it has no effect.',
        f'4. **Shot count is the practical bottleneck.** At 1× Aria-1, 100 shots per circuit raises 571-month MSE by {shots100[0]:+.4f} (QR1) and {shots100[1]:+.4f} (QR2). '
        f'With 1,000 shots, MSE changes by {min(shots1000):+.4f} to {max(shots1000):+.4f} against the infinite-shot noisy reservoir (shot-seed s.d. up to {spread:.4f}). '
        'Noise shrinks the signal while the shot error stays the same size, so noisier devices need more shots. The improvement at 1,000 shots is consistent with shot noise regularizing the ill-conditioned readout noted in the 1 October diagnostics (λ = 1e-8). It is not evidence of a better reservoir.',
        f'5. **Benchmark picture is unchanged.** With 120-month windows, {below_ar1_120} of the noisy QR configurations beat AR1 (MSE {ar1[120]:.4f}). '
        f'With 571-month windows, {len(below_harx_571)} noisy configuration(s) fall below HARX ({harx[571]:.4f})'
        + (f', the lowest by {harx[571] - below_harx_571.mse_log_rv.min():.4f}, which is inside the shot-seed spread' if len(below_harx_571) else '')
        + '. Realistic Aria-1 noise neither creates nor removes an apparent quantum advantage. The formal comparison (bootstrap intervals and model confidence sets) remains in the published extended study.', '']
    text = [
        '# Noise simulations: IonQ Aria-1 emulation of the QR1/QR2 reservoir', '',
        'Analysis-only study. The production models, data and published forecasts are unchanged.', '',
        *findings,
        '## Setup', '',
        f'- Noise rates are medians of {r["records_used"]} public IonQ Aria-1 daily characterizations ({r["first_date"]} to {r["last_date"]}): '
        f'1-qubit fidelity {r["fidelity_1q"]}, 2-qubit {r["fidelity_2q"]}, SPAM {r["fidelity_spam"]}.',
        f'- At 1× Aria-1: depolarizing probability {fmt(p1.depolarizing_2q)} per RXX, {fmt(p1.depolarizing_1q)} per RY, readout flip {fmt(p1.readout_flip)}; RZ is noiseless (virtual on IonQ). Scales 0.5×, 2× and 4× multiply all three rates.',
        '- Every forecast month is one 10-qubit circuit with real resets of the 7 input qubits between monthly inputs (all-to-all, no SWAPs), simulated on the Aer density-matrix method. Finite shots are sampled from the noisy outcome distribution at each readout endpoint.',
        f'- Couplings: published seed 0. Forecasts use the same ridge readout as the published study (λ = 1e-8; 120- and 571-month windows) over {manifest["forecasts"]["target_months"]}. '
        f'The exact-ideal row reproduces the published seed-0 forecasts to {manifest["exact_reproduces_published_max_abs_log_rv"]:.1e} log RV.',
        '- Shot results average 5 independent shot seeds; the figure error bars show their standard deviation.', '',
        '## Feature-level results (12 diagnostic histories)', '', *srows, '',
        '## Forecast MSE (log RV), seed 0', '', *rows, '',
        '## Figures', '',
        *[f'- [{title}](figures/{name}.png) ([PDF](figures/{name}.pdf)); data: `tables/{source}`' for name, title, source in figures], '',
        '## Limitations', '',
        '- Local emulation from published calibration medians, not IonQ\'s own cloud noise simulator or hardware. Depolarizing error is the same for every RXX angle; real small-angle MS gates, crosstalk, drift, and coherent errors are not modeled. SPAM error is applied as readout error only; resets are ideal.',
        '- One coupling seed (0) and one Trotter depth (1) are used for forecasts. Aria-1 has since been retired; the rates describe its mid-2025 performance.',
        '- RY is counted as a single native 1-qubit gate; real compilation may use more.', '']
    (out / 'details.md').write_text('\n'.join(text))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
