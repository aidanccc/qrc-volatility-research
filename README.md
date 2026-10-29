# Quantum reservoir computing for volatility forecasting

This repo reproduces and tests the QR1/QR2 quantum reservoir models from [arXiv:2505.13933](https://arxiv.org/abs/2505.13933), which forecast monthly realized volatility of the S&P 500.

## Results

| Study | Main finding |
|---|---|
| [1. Paper reproduction](results/1-paper-reproduction/report.md) | Our QR1/QR2 match the authors' predictions to 1.2e-6. QR2 has the lowest error of 11 models. |
| [2. Benchmark on new data](results/2-benchmark/report.md) | On 2018–2026 data, no model is clearly best, and the quantum models have no clear edge. |
| [3. Diagnostics](results/3-diagnostics/report.md) | Many reservoir outputs are redundant, and the readout is very sensitive to small errors. |
| [4. Noise, extended model](results/4-noise-extended/report.md) | Aria-1 gate noise barely changes the forecasts. 100 shots is too few. |
| [5. Hardware circuits](results/5-hardware-circuits/report.md) | The paper's models as IonQ gate circuits score almost the same as the paper. |
| [6. IonQ noise](results/6-ionq-noise/report.md) | At 1,000 shots, the forecasts drop to HAR level. Matching the paper needs about 10⁵–10⁶ shots per circuit. |

## Layout

```
data/        inputs: the paper's data (Data.CSV, coeff_10.jld2, predict_result.csv), 1950-2026.csv, prepared snapshots
original/    the authors' notebooks and Julia code, unchanged
qrc/         our code
  reservoir.py, trotter.py   quantum reservoir (exact and Trotter)
  benchmark/                 data preparation, all 12 models, rolling forecasts, reports
  diagnostics/  noise/  hardware/  ionq/   one folder per study
results/     one folder per study, each with report.md
tests/       all tests
```

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
```

The IonQ study also needs `pip install -r qrc/ionq/requirements.txt`.

## Run

Always write to a new output folder.

```bash
# 1. Paper reproduction (reruns the original notebooks)
.venv/bin/python -m qrc.benchmark run --configuration legacy --output results/NEW

# 2. Benchmark (several hours); --configuration modern for the price features
.venv/bin/python -m qrc.benchmark run --configuration extended --output results/NEW
.venv/bin/python -m qrc.benchmark report --run results/NEW
.venv/bin/python -m qrc.benchmark explore --run results/NEW --output results/NEW-exploratory

# 3. Diagnostics and 4. noise (extended model)
.venv/bin/python -m qrc.diagnostics.run --output results/NEW
.venv/bin/python -m qrc.noise.run --output results/NEW

# 5. Hardware circuits (~1 h)
.venv/bin/python -m qrc.hardware.select_trotter --output results/NEW

# 6. IonQ noise (~2 h locally; submit needs IONQ_API_KEY and uses only IonQ's simulator)
.venv/bin/python -m qrc.ionq.local --output results/NEW
.venv/bin/python -m qrc.ionq.submit --output results/NEW
.venv/bin/python -m qrc.ionq.analyze --output results/NEW
.venv/bin/python -m qrc.ionq.calibrate --output results/NEW
.venv/bin/python -m qrc.ionq.local --output results/NEW --cloud-matched
.venv/bin/python -m qrc.ionq.analyze --output results/NEW
```

To rebuild the dataset from `data/1950-2026.csv`:

```bash
.venv/bin/python -m qrc.benchmark prepare-extended --output data/snapshots/NEW --as-of YYYY-MM-DD --modern-data data/snapshots/2026-09-14-v2/monthly.csv
```

Scripts that write a long generated report save it as `details.md`. The short `report.md` in each results folder is the summary.
