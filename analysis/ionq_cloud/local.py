"""Local Aer simulation of the paper-faithful circuits under IonQ Aria-1 and Forte-1 noise.

    .venv/bin/python -m analysis.ionq_cloud.local --output results/ionq-noise-YYYY-MM-DD

Noise models follow analysis/noise/noise_model.py: depolarizing error on RXX and
RY, noiseless virtual RZ, symmetric readout error. Rates are medians of IonQ's
daily calibration records:
  - Aria-1: analysis/noise/aria1_characterizations.json (Aug-Sep 2025; retired system)
  - Forte-1: analysis/ionq_cloud/forte1_characterizations.json (Sep-Oct 2026)
Every one of the 813 feature months runs as a 10-qubit reset circuit on the
density-matrix method (exact noisy distribution); readout error and 1,000-shot
sampling (5 seeds) are applied afterwards. Results are cached per configuration
in <output>/cache/local/ (gitignored).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import time

import numpy as np
from qiskit_aer import AerSimulator

from hardware.paper_circuits import PAPER_MODELS, TOTAL, NQUBIT, paper_inputs
from analysis.noise.noise_model import SNAPSHOT as ARIA1, aria1_rates, error_parameters, build_noise_model, \
    apply_readout, z_expectations, sample_z
from analysis.ionq_cloud.circuits import history_values, reset_history_circuit

FORTE1 = Path(__file__).with_name('forte1_characterizations.json')
RATES = {'aria-1': ARIA1, 'forte-1': FORTE1}
DEPTHS = {'chosen': {'QR1': 2, 'QR2': 3}, 'shallow': {'QR1': 1, 'QR2': 1}}   # order-2 Trotter steps
MONTHS = list(range(3, TOTAL))                                                # 813 feature columns
SHOTS, SHOT_SEEDS, CHUNK = 1000, range(5), 29


def rates(noise):
    return aria1_rates(RATES[noise])


def probabilities(task):
    """(len(months), virtual_nodes, 1024) noisy outcome distributions before readout."""
    noise, model, steps, months, history, scale = task
    data, _, couplings = paper_inputs()
    spec = PAPER_MODELS[model]
    model_ = None if noise == 'ideal' else build_noise_model(scale, rates(noise))
    sim = AerSimulator(method='density_matrix', noise_model=model_, max_parallel_threads=1)
    circuits = [reset_history_circuit(history_values(data, spec['features'], l, history), couplings[model],
                                      steps, spec['virtual_nodes']) for l in months]
    result = sim.run(circuits, shots=1).result()
    v = spec['virtual_nodes']
    return np.array([[result.data(i)[f'node{n}'] for n in range(v)] for i in range(len(months))])


def features(prob, flip, shots=None, rng=None):
    """<Z> features (months, 10 * v): exact after readout error, or `shots`-shot estimates."""
    z = (lambda p: z_expectations(apply_readout(p, flip))) if shots is None else \
        (lambda p: sample_z(apply_readout(p, flip), shots, rng))
    return np.array([np.concatenate([z(p) for p in row]) for row in prob])


def readout_flip(noise, scale=1.0):
    return 0.0 if noise == 'ideal' else error_parameters(scale, rates(noise))['readout_flip']


def simulate(noise, model, depth, months, history=3, workers=14, scale=1.0):
    """Exact outcome distributions at `scale` x calibration noise; 'ideal' is the noiseless circuit."""
    steps = DEPTHS[depth][model]
    chunks = [months[i:i + CHUNK] for i in range(0, len(months), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [(noise, model, steps, c, history, scale) for c in chunks]
        prob = np.concatenate(list(pool.map(probabilities, tasks)))
    assert np.allclose(prob.sum(axis=2), 1, atol=1e-9)
    return prob


def config_name(noise, scale):
    """'aria-1' at calibration rates; 'aria-1-x1.23' when rescaled to match the cloud model."""
    return noise if scale == 1.0 else f'{noise}-x{scale:.2f}'


def run(out, scales=None):
    """Simulate every month for each noise model at 1x calibration rates and at any `scales`."""
    cache = Path(out) / 'cache' / 'local'
    cache.mkdir(parents=True, exist_ok=True)
    for noise, scale in [(n, 1.0) for n in RATES] + sorted((scales or {}).items()):
        flip = readout_flip(noise, scale)
        for depth in DEPTHS:
            for model in PAPER_MODELS:
                path = cache / f'{config_name(noise, scale)}_{model}_{depth}.npz'
                if path.exists():
                    continue
                started = time.time()
                prob = simulate(noise, model, depth, MONTHS, scale=scale)
                seed = [list(RATES).index(noise), list(DEPTHS).index(depth), list(PAPER_MODELS).index(model), round(100 * scale)]
                shots = np.array([features(prob, flip, SHOTS, np.random.default_rng(seed + [s])) for s in SHOT_SEEDS])
                np.savez(path, months=MONTHS, z_inf=features(prob, flip), z_shots=shots)
                print(f'{config_name(noise, scale)} {model} {depth}: {len(MONTHS)} months in {time.time() - started:.0f} s', flush=True)


def to_signal(z, model):
    """(months, 10v) features for columns 3..815 -> (10v, TOTAL) signal with zero first columns."""
    signal = np.zeros((NQUBIT * PAPER_MODELS[model]['virtual_nodes'], TOTAL))
    signal[:, 3:] = z.T
    return signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloud-matched', action='store_true',
                        help='also run at the noise scales fitted to the cloud by calibrate.py')
    args = parser.parse_args()
    scales = None
    if args.cloud_matched:
        import json
        scales = json.loads((args.output / 'tables' / 'cloud_matched_scales.json').read_text())['scale']
    run(args.output, scales)
