# Noise simulations (IonQ Aria-1 emulation), 22 October 2026

Results: [report](../../results/noise-2026-10-22/report.md).

This study runs the existing QR1/QR2 reservoir circuits under realistic trapped-ion noise and asks two questions: how much does the noise change the reservoir features, and how much does it change the forecasts? It follows the IonQ demo from the group's `qml_demo_noisy_and_qpu` materials, which ran a toy circuit on IonQ's cloud simulator with `noise_model="aria-1"`. Here the same idea runs locally, with no API key and no cloud jobs:

- `aria1_characterizations.json` is a snapshot of IonQ's public Aria-1 daily calibration records, retrieved 22 Oct 2026. Aria-1 is now retired, so the healthy mid-2025 records are used.
- `noise_model.py` builds a qiskit-aer noise model from those records:
  - depolarizing error on RXX (2-qubit) and RY gates
  - RZ is noiseless (it is virtual on IonQ)
  - SPAM is applied as readout error
  - a `scale` factor gives 0.5×, 2× and 4× sweeps.
- `compute.py` builds one 10-qubit circuit per forecast month, with real input-qubit resets and no SWAPs (IonQ is all-to-all). It simulates the exact noisy distribution with the Aer density-matrix method, then adds readout error and finite shots (100 / 1,000 / 10,000). Its outputs:
  - **Sweep**: 12 diagnostic histories × noise scale {0, 0.5, 1, 2, 4} × Trotter steps {1, 2, 4}.
  - **Forecasts**: 2018–2026 at Trotter=1 with the published seed-0 couplings and the published ridge readout, at noise scales {0, 0.5, 1, 2}, plus shots at 0× and 1× (5 shot seeds).
- `render.py` writes 5 figures (PNG + PDF), `report.md`, and the tables.
- `test_noise.py` checks four things:
  - at zero noise, the reset circuit reproduces the production Trotter features
  - the readout map is correct
  - noise shrinks the features
  - the noise model and circuit structure are as intended.

## Run

```bash
.venv/bin/python analysis/noise/run.py --output results/noise-new-run
```

This needs `qiskit-aer==0.18.0` (listed in `requirements.txt`). It takes roughly an hour on a 14-core laptop, and the full-series noisy circuits dominate that time. The cached distributions in `<run>/cache/` are gitignored. Re-render only:

```bash
PYTHONPATH=.:analysis/noise .venv/bin/python analysis/noise/render.py --output results/noise-2026-10-22
```

Limitations are listed at the end of the report.
