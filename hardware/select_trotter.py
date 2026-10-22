"""Choose the Trotter order and step count for the paper-faithful IonQ circuits.

Every candidate is scored only on fidelity to the authors' exact model: how far
its noiseless forecasts and reservoir features move from the exact ones. Forecast
accuracy against realized volatility is reported but never used to choose, so the
choice cannot be tuned to the test period. The thresholds are fixed here, before
any candidate is run.

Writes a new results folder and exports the chosen circuits as OpenQASM 3.
Nothing is sent to IonQ or any other service.

    python -m hardware.select_trotter --output results/hardware-paper-YYYY-MM-DD
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from qiskit import qasm3, transpile
from qiskit.quantum_info import DensityMatrix
from threadpoolctl import threadpool_limits

from hardware.paper_circuits import (ROOT, PAPER_MODELS, IONQ_QIS_BASIS, TOTAL, L_OOS, paper_inputs,
                                     ideal_features, paper_forecasts, month_circuits, trotter_block)
from quantum_reservoir_qiskit import MSE, compute_qlike, build_z_observables

ORDERS = [1, 2]
STEPS = [1, 2, 3, 4, 6, 8]
# Selection rule: among candidates whose noiseless forecasts stay within these
# log-RV gaps of the exact model over all 245 forecasts, take the one with the
# fewest RXX gates per forecast month; ties go to the smaller mean gap.
MAX_FORECAST_GAP = 0.05
MEAN_FORECAST_GAP = 0.01
SHOTS = [100, 1000, 10000]
EXPORT_MONTH = TOTAL - L_OOS  # first out-of-sample forecast (August 1997)
CHECK_MONTHS = [3, 300, TOTAL - L_OOS, TOTAL - 1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def run_candidate(task):
    model, order, steps = task
    data, target, couplings = paper_inputs()
    with threadpool_limits(limits=2), redirect_stdout(io.StringIO()):
        signal = ideal_features(data, model, couplings[model], steps, order)
    pred, logrv = paper_forecasts(signal, target)
    return task, signal, pred, logrv


def circuit_cost(model, steps, order):
    circuits = month_circuits(model, EXPORT_MONTH, steps, order)
    compiled = [transpile(c, basis_gates=IONQ_QIS_BASIS, optimization_level=0) for c in circuits]
    ops = [c.count_ops() for c in compiled]
    return {'circuits_per_month': len(circuits),
            'rxx_per_month': int(sum(o.get('rxx', 0) for o in ops)),
            'max_rxx_per_circuit': int(max(o.get('rxx', 0) for o in ops)),
            'max_depth': int(max(c.depth() for c in compiled)),
            'resets_per_circuit': int(ops[0].get('reset', 0))}


def check_exported(model, steps, order, signal, data, couplings):
    """Simulate the exported reset circuits directly and compare with the features used for selection."""
    worst = 0.
    obs = build_z_observables(10)
    for l in CHECK_MONTHS:
        for e, qc in enumerate(month_circuits(model, l, steps, order, data, couplings)):
            state = DensityMatrix.from_label('0' * 10).evolve(qc.remove_final_measurements(inplace=False))
            z = np.array([state.expectation_value(o).real for o in obs])
            worst = max(worst, float(np.abs(z - signal[e * 10:(e + 1) * 10, l]).max()))
    return worst


def main(output):
    out = Path(output)
    if out.exists():
        raise SystemExit(f'{out} exists; use a new output directory')
    started = time.time()
    data, target, couplings = paper_inputs()
    authors = pd.read_csv(ROOT / 'data/predict_result.csv')
    y = target[TOTAL - L_OOS:TOTAL]
    tasks = [(m, None, None) for m in PAPER_MODELS] + [(m, o, s) for m in PAPER_MODELS for o in ORDERS for s in STEPS]
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for (model, order, steps), signal, pred, logrv in pool.map(run_candidate, tasks):
            results[(model, order, steps)] = (signal, pred, logrv)
            print(f'{model} order={order} steps={steps} done', flush=True)

    rows, checks = [], {}
    for model in PAPER_MODELS:
        exact_signal, exact_pred, exact_logrv = results[(model, None, None)]
        checks[model] = {'exact_vs_authors_max_abs': float(np.abs(exact_logrv - authors[model].to_numpy()).max())}
        for order in ORDERS:
            for steps in STEPS:
                signal, pred, logrv = results[(model, order, steps)]
                gap = np.abs(logrv - exact_logrv)
                rows.append({'model': model, 'order': order, 'steps': steps,
                             'max_forecast_gap': float(gap.max()), 'mean_forecast_gap': float(gap.mean()),
                             'max_feature_gap': float(np.abs(signal - exact_signal)[:, 3:].max()),
                             'mean_feature_gap': float(np.abs(signal - exact_signal)[:, 3:].mean()),
                             'max_gap_vs_authors': float(np.abs(logrv - authors[model].to_numpy()).max()),
                             **circuit_cost(model, steps, order),
                             'mse_reported_only': float(MSE(pred, y)), 'qlike_reported_only': float(compute_qlike(pred, y))})
        checks[model]['exact_mse'] = float(MSE(exact_pred, y))
        checks[model]['exact_qlike'] = float(compute_qlike(exact_pred, y))
    table = pd.DataFrame(rows)
    table['passes'] = (table.max_forecast_gap <= MAX_FORECAST_GAP) & (table.mean_forecast_gap <= MEAN_FORECAST_GAP)

    out.mkdir(parents=True)
    (out / 'circuits').mkdir()
    table.to_csv(out / 'trotter_sweep.csv', index=False)
    chosen = {}
    for model in PAPER_MODELS:
        ok = table[(table.model == model) & table.passes].sort_values(['rxx_per_month', 'mean_forecast_gap'])
        if ok.empty:
            chosen[model] = None
            continue
        best = ok.iloc[0]
        order, steps = int(best.order), int(best.steps)
        signal = results[(model, order, steps)][0]
        checks[model]['exported_circuit_vs_selection_features_max_abs'] = check_exported(model, steps, order, signal, data, couplings)
        months = TOTAL - 3
        v = PAPER_MODELS[model]['virtual_nodes']
        chosen[model] = {'order': order, 'steps': steps, **{k: (v_.item() if hasattr(v_, 'item') else v_) for k, v_ in best.items() if k not in ('model', 'order', 'steps')},
                         'job_plan': {'all_feature_months': {'months': months, 'circuits': months * v,
                                                              **{f'shots_at_{s}': months * v * s for s in SHOTS}},
                                      'out_of_sample_only': {'months': L_OOS, 'circuits': L_OOS * v,
                                                             **{f'shots_at_{s}': L_OOS * v * s for s in SHOTS}}}}
        for e, qc in enumerate(month_circuits(model, EXPORT_MONTH, steps, order, data, couplings), start=1):
            (out / 'circuits' / f'{model}_order{order}_steps{steps}_month{EXPORT_MONTH}_endpoint{e}.qasm').write_text(qasm3.dumps(qc))
    selection = {'rule': {'max_forecast_gap': MAX_FORECAST_GAP, 'mean_forecast_gap': MEAN_FORECAST_GAP,
                          'cost': 'fewest RXX gates per forecast month, ties to smaller mean gap',
                          'units': 'log realized volatility, versus the exact (authors) model, all 245 forecasts',
                          'not_used': 'forecast accuracy against realized volatility'},
                 'chosen': chosen, 'checks': checks}
    write_json(out / 'selection.json', selection)
    write_json(out / 'manifest.json', {
        'purpose': 'Trotter order/steps selection for paper-faithful IonQ circuits; nothing submitted',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'uncommitted_changes': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'source_hashes': {p: digest(ROOT / p) for p in ['hardware/paper_circuits.py', 'hardware/select_trotter.py', 'quantum_reservoir_qiskit.py']},
        'input_hashes': {p: digest(ROOT / p) for p in ['data/Data.CSV', 'data/coeff_10.jld2', 'data/predict_result.csv']},
        'grid': {'orders': ORDERS, 'steps': STEPS}, 'export_month_index': EXPORT_MONTH, 'check_months': CHECK_MONTHS,
        'basis': IONQ_QIS_BASIS, 'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ['qiskit', 'numpy', 'scipy']},
        'elapsed_seconds': round(time.time() - started)})
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    main(p.parse_args().output)
