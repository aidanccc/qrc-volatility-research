"""Noisy (IonQ Aria-1 emulated) reservoir features and forecasts.

Analysis-only. Each forecast month is one 10-qubit circuit that mirrors the
production three-input protocol with real resets of the 7 input qubits:
    RY encode -> U -> reset -> RY encode -> U -> reset -> RY encode -> dU [-> dU]
Aer's density-matrix method returns the exact noisy outcome distribution at
each virtual-node endpoint (equivalent to separate circuits per endpoint).
Readout error and finite shots are applied to those distributions afterwards.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from quantum_reservoir_qiskit import generate_coupling_matrix, MIN_RV, DIF
from quantum_reservoir_trotter import build_trotter_evolution
from qrcstudy.extended_models import QR1, QR2
from qrcstudy.report import losses
from noise_model import aria1_rates, error_parameters, build_noise_model, apply_readout, z_expectations, sample_z

DATA = ROOT / 'data/snapshots/extended-2026-09-24-v2/monthly.csv'
RUN = ROOT / 'results/extended-2026-09-24'
DIAGNOSTICS = ROOT / 'results/diagnostics-2026-10-01/tables'
MODELS = {'QR1': (QR1, 1), 'QR2': (QR2, 2)}
WINDOWS = [120, 571]
SWEEP_SCALES = [0, 0.5, 1, 2, 4]
SWEEP_STEPS = [1, 2, 4]
FORECAST_SCALES = [0, 0.5, 1, 2]
SHOTS = [100, 1000, 10000]
SHOT_SEEDS = range(5)
BATCH = 56


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def reservoir_circuit(values, u, du, virtual_nodes):
    """Three monthly inputs with input-qubit resets; saves probabilities per endpoint."""
    qc = QuantumCircuit(10)
    for lag in range(3):
        if lag:
            qc.reset(range(7))
        for q, value in enumerate(values[lag]):
            qc.ry(float(np.pi * value), q)
        if lag < 2:
            qc.compose(u, inplace=True)
        else:
            for node in range(virtual_nodes):
                qc.compose(du, inplace=True)
                qc.save_probabilities(label=f'node{node}')
    return qc


def probabilities(x, positions, model, steps, scale):
    """(len(positions), virtual_nodes, 1024) noisy distributions before readout."""
    v = MODELS[model][1]
    u, du = build_trotter_evolution(10, generate_coupling_matrix(10, seed=0), 1., steps, v)
    sim = AerSimulator(method='density_matrix', noise_model=build_noise_model(scale))
    out = np.empty((len(positions), v, 1024))
    for start in range(0, len(positions), BATCH):
        chunk = positions[start:start + BATCH]
        result = sim.run([reservoir_circuit(x[t - 3:t], u, du, v) for t in chunk], shots=1).result()
        for i in range(len(chunk)):
            out[start + i] = [result.data(i)[f'node{n}'] for n in range(v)]
    assert np.allclose(out.sum(axis=2), 1, atol=1e-9)
    return out


def features(prob, flip):
    """Exact noisy <Z> features after readout error, shape (months, 10 * v)."""
    return np.array([np.concatenate([z_expectations(apply_readout(p, flip)) for p in row]) for row in prob])


def shot_features(prob, flip, shots, seed):
    """Finite-shot features; each endpoint is an independent run of `shots` shots."""
    rng = np.random.default_rng(seed)
    return np.array([np.concatenate([sample_z(apply_readout(p, flip), shots, rng) for p in row]) for row in prob])


def exact_features(model, dates):
    """Published ideal (matrix-exponential) seed-0 features from the diagnostic study."""
    table = pd.read_csv(DIAGNOSTICS / f'{model}_features_seed0.csv', index_col=0, parse_dates=True)
    return table.reindex(dates).to_numpy()


def forecast(feature, y, t, window):
    """Same ridge readout as qrcstudy.extended_models.predict for QR models."""
    a = feature[t - window:t]
    w = np.linalg.solve(a.T @ a + 1e-8 * np.eye(a.shape[1]), a.T @ y[t - window:t])
    return float(feature[t] @ w)


def compute(out):
    out = Path(out)
    tables = out / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'; cache.mkdir(exist_ok=True)
    started = time.time()
    frame = pd.read_csv(DATA, index_col=0, parse_dates=True)
    y = frame.RV.to_numpy()
    rates = aria1_rates()
    pd.DataFrame([error_parameters(s, rates) for s in sorted(set(SWEEP_SCALES + FORECAST_SCALES))]).to_csv(tables / 'noise_parameters.csv', index=False)
    selected = pd.read_csv(DIAGNOSTICS / 'selected_histories.csv').position.to_numpy()
    published = pd.read_csv(RUN / 'predictions.csv', parse_dates=['target_month'])
    scored = published[published.status.eq('ok') & published.actual_log_rv.notna()]

    # A. Feature sweep on the 12 diagnostic histories: noise scale x Trotter steps.
    sweep = []
    for model, (cols, v) in MODELS.items():
        x = frame[cols].to_numpy()
        exact = exact_features(model, frame.index[selected])
        for steps in SWEEP_STEPS:
            for scale in SWEEP_SCALES:
                print(f'sweep {model} steps={steps} scale={scale}', flush=True)
                flip = error_parameters(scale, rates)['readout_flip']
                z = features(probabilities(x, selected, model, steps, scale), flip)
                for i, t in enumerate(selected):
                    for k in range(10 * v):
                        sweep.append({'model': model, 'target_month': str(frame.index[t].date()), 'steps': steps, 'scale': scale,
                                      'virtual_node': k // 10 + 1, 'qubit': k % 10, 'exact_z': exact[i, k], 'noisy_z': z[i, k]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(tables / 'feature_sweep.csv', index=False)
    rows = []
    for (model, steps, scale), g in sweep.groupby(['model', 'steps', 'scale']):
        bias2 = float(np.mean((g.noisy_z - g.exact_z) ** 2))
        row = {'model': model, 'steps': steps, 'scale': scale, 'mean_abs_error': float(np.mean(abs(g.noisy_z - g.exact_z))),
               'mean_abs_signal': float(np.mean(abs(g.noisy_z))), 'exact_mean_abs_signal': float(np.mean(abs(g.exact_z)))}
        for shots in SHOTS:  # expected RMS error including binomial shot noise
            row[f'rms_error_{shots}_shots'] = float(np.sqrt(bias2 + np.mean((1 - g.noisy_z ** 2) / shots)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables / 'feature_sweep_summary.csv', index=False)

    # B. Full forecasts at Trotter steps = 1 for each noise scale, plus finite shots.
    records = []
    for model, (cols, v) in MODELS.items():
        mine = scored[(scored.model == model) & (scored.seed == 0)]
        targets = sorted({frame.index.get_loc(d) for d in mine.target_month})
        positions = np.arange(min(targets) - max(WINDOWS), max(targets) + 1)
        x = frame[cols].to_numpy()
        configs = {'exact_ideal': exact_features(model, frame.index[positions])}
        for scale in FORECAST_SCALES:
            file = cache / f'{model}-scale{scale}.npy'
            if file.exists():
                prob = np.load(file)
            else:
                print(f'forecast {model} scale={scale}: {len(positions)} circuits', flush=True)
                t0 = time.time(); prob = probabilities(x, positions, model, 1, scale); np.save(file, prob)
                print(f'  {time.time() - t0:.0f}s', flush=True)
            flip = error_parameters(scale, rates)['readout_flip']
            configs[f'trotter1_scale{scale}'] = features(prob, flip)
            if scale in (0, 1):
                for shots in SHOTS:
                    for seed in SHOT_SEEDS:
                        configs[f'trotter1_scale{scale}_shots{shots}_seed{seed}'] = shot_features(prob, flip, shots, seed)
        offset = positions[0]
        for name, feat in configs.items():
            assert np.isfinite(feat).all(), name
            for window in WINDOWS:
                for row in mine[mine.window == window].itertuples():
                    t = frame.index.get_loc(row.target_month)
                    pred = (forecast(feat, y[offset:], t - offset, window) + 1) * DIF + MIN_RV
                    records.append({'model': model, 'window': window, 'config': name, 'target_month': str(row.target_month.date()),
                                    'actual_log_rv': row.actual_log_rv, 'previous_log_rv': row.previous_log_rv,
                                    'predicted_log_rv': pred, 'published_log_rv': row.predicted_log_rv})
    pred = pd.DataFrame(records)
    exact = pred[pred.config == 'exact_ideal']
    reproduction = float(np.max(abs(exact.predicted_log_rv - exact.published_log_rv)))
    assert reproduction < 1e-6, reproduction
    pred.drop(columns='published_log_rv').to_csv(tables / 'forecasts.csv', index=False)
    loss = losses(pred.actual_log_rv, pred.predicted_log_rv, pred.previous_log_rv)
    pred = pred.assign(mse_log_rv=loss['mse_log_rv'], mae_log_rv=loss['mae_log_rv'], qlike_variance=loss['qlike_variance'])
    pred['scale'] = pred.config.str.extract(r'scale([\d.]+)')[0].astype(float)
    pred['shots'] = pred.config.str.extract(r'shots(\d+)')[0].astype(float)
    metrics = pred.groupby(['model', 'window', 'config', 'scale', 'shots'], dropna=False)[['mse_log_rv', 'mae_log_rv', 'qlike_variance']].mean().reset_index()
    metrics.to_csv(tables / 'forecast_metrics.csv', index=False)
    # Shot seeds summarized as mean and spread of MSE.
    summary = metrics.assign(base=metrics.config.str.replace(r'_seed\d+$', '', regex=True)).groupby(['model', 'window', 'base'], sort=False).agg(
        mse_log_rv=('mse_log_rv', 'mean'), mse_seed_std=('mse_log_rv', 'std'), mae_log_rv=('mae_log_rv', 'mean'),
        qlike_variance=('qlike_variance', 'mean'), runs=('mse_log_rv', 'size')).reset_index().rename(columns={'base': 'config'})
    months = pred[['model', 'window', 'target_month']].drop_duplicates()
    for ref in ['HARX', 'AR1', 'Persistence']:
        r = scored[(scored.model == ref) & (scored.seed == 0)]
        for (model, window), g in months.groupby(['model', 'window']):
            m = r[(r.window == window) & r.target_month.dt.strftime('%Y-%m-%d').isin(g.target_month)]
            l = losses(m.actual_log_rv, m.predicted_log_rv, m.previous_log_rv)
            summary.loc[len(summary)] = {'model': model, 'window': window, 'config': f'reference_{ref}', 'mse_log_rv': l['mse_log_rv'].mean(),
                                         'mse_seed_std': np.nan, 'mae_log_rv': l['mae_log_rv'].mean(), 'qlike_variance': l['qlike_variance'].mean(), 'runs': 1}
    summary.to_csv(tables / 'forecast_summary.csv', index=False)

    write_json(out / 'manifest.json', {
        'purpose': 'Local emulation of IonQ Aria-1 noise on the QR1/QR2 reservoir (analysis-only).',
        'aria1_rates': rates, 'noise_source': 'analysis/noise/aria1_characterizations.json',
        'noise_model': 'depolarizing on ry and rxx; rz noiseless; symmetric readout flip = 1 - SPAM fidelity; ideal resets; all-to-all, untranspiled',
        'coupling': 'generate_coupling_matrix(10, seed=0), shared by QR1/QR2 (published seed 0)',
        'sweep': {'histories': len(selected), 'scales': SWEEP_SCALES, 'trotter_steps': SWEEP_STEPS},
        'forecasts': {'trotter_steps': 1, 'scales': FORECAST_SCALES, 'shots': SHOTS, 'shot_seeds': list(SHOT_SEEDS), 'windows': WINDOWS,
                      'target_months': f"{pred.target_month.min()} to {pred.target_month.max()}"},
        'exact_reproduces_published_max_abs_log_rv': reproduction,
        'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [DATA, RUN / 'predictions.csv', ROOT / 'analysis/noise/aria1_characterizations.json',
                                                                         DIAGNOSTICS / 'QR1_features_seed0.csv', DIAGNOSTICS / 'QR2_features_seed0.csv']},
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'analysis/noise').glob('*.py'))},
        'versions': {p: importlib.metadata.version(p) for p in ['qiskit', 'qiskit-aer', 'numpy', 'pandas']},
        'elapsed_seconds': round(time.time() - started)})
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    compute(parser.parse_args().output)
