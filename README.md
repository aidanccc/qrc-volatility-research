# Updated monthly dataset

`1950-2026.csv` contains 920 monthly observations through August 2026. The prepared snapshot fills 412 verified factor values and corrects 211 derived values. Four August factor values remain missing. The original data and flagged outliers are preserved.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_study.py run --configuration extended --output results/new-run
python run_study.py report --run results/new-run
```

The prepared snapshot is included. Use a new output directory for each changed dataset or configuration. The default run tests all models with 120- and 571-month windows and five stochastic seeds; it can take several hours.

See [results](results/extended-study-2026-09-24/report.md), [data checks](docs-dataset/AUDIT.md), and [run instructions](RUN_ORDER.md) for preparation, a quick pilot, and the separate modern benchmark. Targets differ between the two protocols, so their scores are reported separately.

## Circuit and data diagnostics

The [1 October diagnostic report](results/diagnostics-2026-10-01/report.html) explains the 10-qubit circuit and includes 21 data, reservoir, forecast, and resource figures. [Markdown report](results/diagnostics-2026-10-01/report.md) · [Reproduction instructions](analysis/diagnostics/README.md) · [Deferred optimization prompts](analysis/diagnostics/FUTURE_CODEX_PROMPTS.md). Model implementations and published benchmarks are unchanged.

## Noise simulations

The [22 October noise report](results/noise-2026-10-22/report.md) reruns the QR1/QR2 reservoir circuits under a local emulation of IonQ Aria-1 noise, built from IonQ's public calibration data. It includes noise and shot sweeps and their effect on forecasts. See [instructions](analysis/noise/README.md).

The [28 October IonQ noise report](results/ionq-noise-2026-10-28/report.md) does the same for the paper-faithful circuits in `hardware/`, using Aria-1 and Forte-1 noise and a check against IonQ's cloud simulator. It finds that shot noise, not gate noise, limits the forecasts. Run commands are in [RUN_ORDER.md](RUN_ORDER.md).
