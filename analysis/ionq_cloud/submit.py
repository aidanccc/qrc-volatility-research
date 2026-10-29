"""Cloud check: run short reservoir histories on IonQ's cloud simulator, as in the group's IonQ demo.

    IONQ_API_KEY=... .venv/bin/python -m analysis.ionq_cloud.submit --output results/ionq-noise-YYYY-MM-DD

The full 3-month circuits need 24 qubits. For this project IonQ's simulator ran
no job above 19 qubits (ideal, aria-1) or above 11 qubits (forte-1); larger jobs
stay queued indefinitely (see README). The check therefore uses 1- and 2-month
histories (10 and 17 qubits; forte-1 only 10) with the same gates. These are
compared against the local noise models in analyze.py.

Only `ionq_simulator` is used; there is no QPU path. Job ids and returned
probabilities are cached in <output>/cache/cloud/ (gitignored), so the run
resumes if interrupted. The API key is read from the environment only.
"""
from pathlib import Path
import argparse
import json
import os
import time
import warnings

import numpy as np
from qiskit_ionq import IonQProvider

from hardware.paper_circuits import TOTAL, paper_inputs
from analysis.ionq_cloud.circuits import cloud_circuits
from analysis.ionq_cloud.local import DEPTHS

BACKEND = 'ionq_simulator'
SHOTS = 1000
SEED = 20261028                      # fixed noise/sampling seed, so the cloud runs are reproducible
NOISE_MODELS = ['ideal', 'aria-1', 'forte-1']
HISTORIES = {'ideal': [1, 2], 'aria-1': [1, 2], 'forte-1': [1]}   # months of history: 10 / 17 qubits
CHECK_MONTHS = [int(l) for l in np.linspace(3, TOTAL - 1, 10)]


def simulator(noise_model):
    backend = IonQProvider(os.environ['IONQ_API_KEY']).get_backend(BACKEND)
    if backend.name != BACKEND:      # hard guard: never a QPU
        raise RuntimeError(f'refusing backend {backend.name}')
    backend.set_options(noise_model=noise_model, sampler_seed=SEED)
    return backend


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    warnings.filterwarnings('ignore', category=UserWarning)
    cache = out / 'cache' / 'cloud'; cache.mkdir(parents=True, exist_ok=True)
    logfile = (out / 'cloud.log').open('a')

    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True); logfile.write(line + '\n'); logfile.flush()

    data, _, couplings = paper_inputs()
    pending = []
    for noise in NOISE_MODELS:
        backend = simulator(noise)
        for depth, steps in DEPTHS.items():
            for months in HISTORIES[noise]:
                path = cache / f'{noise}_{depth}_k{months}.json'
                record = json.loads(path.read_text()) if path.exists() else {}
                if 'probabilities' in record:
                    continue
                batch = [qc for model in steps for l in CHECK_MONTHS
                         for qc in cloud_circuits(model, l, steps[model], months, data, couplings)]
                if 'job_id' not in record:
                    job = backend.run(batch, shots=SHOTS)
                    record = {'job_id': job.job_id(), 'noise': noise, 'shots': SHOTS, 'seed': SEED,
                              'names': [qc.name for qc in batch], 'qubits': batch[0].num_qubits,
                              'submitted': time.time()}
                    path.write_text(json.dumps(record))
                    log(f'{path.stem}: submitted {record["job_id"]}, {len(batch)} circuits of {batch[0].num_qubits} qubits')
                pending.append((backend, path, batch))
    for backend, path, batch in pending:
        record = json.loads(path.read_text())
        result = backend.retrieve_job(record['job_id']).result(wait=5, timeout=3600)
        record['probabilities'] = [{k: float(v) for k, v in result.get_probabilities(qc).items()} for qc in batch]
        record['completed'] = time.time()
        path.write_text(json.dumps(record))
        log(f'{path.stem}: done, {record["completed"] - record["submitted"]:.0f} s after submission')
    log('complete')


if __name__ == '__main__':
    main()
