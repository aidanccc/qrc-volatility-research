# IonQ noise simulations of the paper-faithful QR1/QR2 circuits

Results: [report](../../results/ionq-noise-2026-10-28/report.md).

This study asks what happens to the paper's QR1/QR2 forecasts when the circuits we would actually send to IonQ hardware run with IonQ-level noise. Those circuits come from `hardware/`: the authors' `coeff_10.jld2` couplings and the paper's features, with second-order Trotter evolution.

It follows the group's demo `qml_demo_noisy_and_qpu/test_ionq_sim.ipynb`, which runs a toy circuit on IonQ's cloud simulator:

```python
backend = IonQProvider().get_backend("ionq_simulator")
backend.set_options(noise_model="aria-1")
backend.run(qc).result().get_counts()
```

It extends `analysis/noise/` (22 Oct). That study emulated Aria-1 locally on the *extended* seed-0 reservoir; this one uses the paper's own model.

## Why there are two parts

The plan was to run every circuit on IonQ's cloud simulator. Two limits made that impossible (checked 28–29 Oct 2026 with the group's API key, project "FIRE QML class"):

1. **No resets.** The simulator's gate list has no `reset`, and reset circuits (sent as OpenQASM 3 by `qiskit-ionq` 1.1.1) fail with `SimulationError`.
   - Without resets, each monthly input needs its own 7 fresh qubits: 3×7 + 3 memory = **24 qubits**.
2. **Qubit cutoff.** For this project, cloud jobs above **19 qubits** (`ideal`, `aria-1`) or above **11 qubits** (`forte-1`) stay queued ("ready") and never start, even a 2-gate circuit sent through the raw REST API.
   - IonQ's docs list limits of 25 and 29 qubits, and none of the project's 1,500 earlier jobs used 20 or more qubits.
   - All stuck test jobs were cancelled.

So the study has two parts:

- **Local simulation of every forecast month** (`local.py`). Each month is one 10-qubit reset circuit, run on Aer's density-matrix method.
  - The noise follows `analysis/noise/noise_model.py`: depolarizing error on RXX and RY, a noiseless virtual RZ, and readout error.
  - Rates are medians of IonQ's daily calibration records:
    - Aria-1: 25 days, Aug–Sep 2025 (`analysis/noise/aria1_characterizations.json`)
    - Forte-1: 29 days, 29 Sep–27 Oct 2026 (`forte1_characterizations.json`)
  - Readout error and 1,000-shot sampling (5 seeds) are applied to the exact noisy distribution.
- **Cloud check** (`submit.py`). The same gates run on IonQ's cloud simulator with its own `ideal`, `aria-1` and `forte-1` noise models, at 1,000 shots and a fixed seed.
  - They use 1- and 2-month input histories, which take 10 and 17 qubits. `forte-1` gets only the 1-month, 10-qubit version.
  - The runs cover 10 months spread over the sample, both models and both depths.
  - These results are compared with the local model of the same circuits. This shows whether the local noise model behaves like IonQ's.

## Configurations

- **Depth (order-2 Trotter):**
  - `chosen` is the noiseless pick from `hardware/select_trotter.py`: QR1 uses 2 steps (270 RXX per month), QR2 uses 3 steps (945).
  - `shallow` uses 1 step (135 / 315 RXX).
- **Months:** all 813 feature months, so the readout is trained and tested on noisy features, as a real hardware run would be.
- **Readout:** the authors' rolling ridge regression (571-month window, λ = 1e-8, 245 forecasts), via `hardware.paper_circuits.paper_forecasts`.

## Files

- `circuits.py`: the reset circuit (local) and the unrolled reset-free circuit (cloud). Both give the same result.
- `local.py`: the full local noisy simulation, with results cached in `<run>/cache/local/`.
- `submit.py`: the cloud check.
  - It only ever uses `ionq_simulator`; there is no QPU path.
  - The key comes from `IONQ_API_KEY`, and results are cached in `<run>/cache/cloud/`.
- `calibrate.py`: fits one noise scale per IonQ model, so the local model shrinks the check-circuit features as much as the cloud does (Aria-1 0.73×, Forte-1 2.84×). `local.py --cloud-matched` then reruns every month at those scales.
- `analyze.py`: produces the tables, figures (including a sweep of MSE against shots), `manifest.json` and the numbers in the report.
- `test_ionq_cloud.py`: offline tests. They check that:
  - both circuit forms reproduce the authors' partial-trace features to 1e-10, including the 10- and 17-qubit short histories
  - the gate counts match
  - noise shrinks the features
  - the bit order is right
- `forte1_characterizations.json`: snapshot of IonQ's public Forte-1 calibration records.
- `requirements.txt`: pins `qiskit-ionq`. It is kept apart from the root requirements, so the environment of the published runs is unchanged.

## Run

```bash
uv pip install --python .venv/bin/python -r analysis/ionq_cloud/requirements.txt
PYTHONPATH=. .venv/bin/python -m unittest analysis.ionq_cloud.test_ionq_cloud -v
PYTHONPATH=. .venv/bin/python -m analysis.ionq_cloud.local --output results/ionq-noise-YYYY-MM-DD    # ~1 h, 14 cores
export IONQ_API_KEY=...                                                                               # never commit it
PYTHONPATH=. .venv/bin/python -m analysis.ionq_cloud.submit --output results/ionq-noise-YYYY-MM-DD   # ~5 min
PYTHONPATH=. .venv/bin/python -m analysis.ionq_cloud.calibrate --output results/ionq-noise-YYYY-MM-DD  # needs tables/cloud_check.csv (written by analyze)
PYTHONPATH=. .venv/bin/python -m analysis.ionq_cloud.local --output results/ionq-noise-YYYY-MM-DD --cloud-matched  # ~1 h more
PYTHONPATH=. .venv/bin/python -m analysis.ionq_cloud.analyze --output results/ionq-noise-YYYY-MM-DD
```

Caches are gitignored. Per-month noisy features (`tables/features_*.csv`) and the per-value cloud-check table (`tables/cloud_check.csv`) are committed, so the forecasts and the comparison can be checked without the cache.

If IonQ lifts the qubit cutoff, the full 24-qubit circuits are `circuits.cloud_circuits(model, l, steps, 3, data, couplings)`. Submit them the same way as in `submit.py`.
