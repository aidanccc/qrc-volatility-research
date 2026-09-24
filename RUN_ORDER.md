# Run instructions

Install `requirements.txt` in a Python virtual environment. The prepared monthly snapshot is included; the original notebooks remain available for the historical workflow.

## Validate and run

```bash
python -m unittest discover -s tests -v
# Quick integration pilot; use a separate directory from the full run.
python run_study.py run --configuration extended --models HAR Persistence --seeds 0 --limit 1 --output results/new-pilot
# Full benchmark, followed by reporting.
python run_study.py run --configuration extended --output results/new-run
python run_study.py report --run results/new-run
```

Defaults: January 2018–August 2026, all 12 models, 120- and 571-month rolling windows, and seeds 0–4 for stochastic models. September predictions are unscored. Missing inputs and failed fits remain explicit. Resume with the same command and configuration; changed source or data requires a new output directory. A partial pilot cannot produce a complete benchmark report.

## Prepare a new snapshot

```bash
python run_study.py prepare-extended --data 1950-2026.csv --output data/snapshots/new-data --as-of 2026-09-24 --modern-data data/snapshots/2026-09-14-v2/monthly.csv
python run_study.py run --configuration extended --data data/snapshots/new-data/monthly.csv --output results/new-data-run
```

Preparation downloads official factor sources and records hashes, scaling checks, missingness, and corrections. Use the actual preparation date for `--as-of`. Review [the audit](docs-dataset/AUDIT.md) and [assumptions](ASSUMPTIONS.md) before changing inputs. Do not fill unverified factors or score incomplete months.

## Other workflows

```bash
# Published reports regenerate without training caches.
python run_study.py report --run results/extended-2026-09-24
python run_study.py report --run results/modern-2026-09-24
python run_study.py summarize
# Original notebook workflow.
python run_study.py run --configuration legacy --output results/new-legacy-run
# Exploratory diagnostics require the run's local reservoir feature caches.
python run_study.py explore --run results/new-run --output results/new-exploration
```

Fresh modern training requires a daily-price snapshot: run `python run_study.py download --as-of YYYY-MM-DD --output data/snapshots/new-prices`, then `python run_study.py run --configuration modern --data data/snapshots/new-prices/monthly.csv --output results/new-modern-run`. Historical raw downloads are not redistributed. Keep modern and paper-feature target scores separate.

## Code map

- `qrcstudy/extended_data.py`: prepare and audit monthly data.
- `qrcstudy/extended_models.py`: paper-specific model inputs.
- `qrcstudy/extended_run.py`: rolling forecasts and checkpoints.
- `qrcstudy/data.py`, `models.py`, `run.py`: modern price-feature benchmark.
- `qrcstudy/report.py`, `study_report.py`, `project_report.py`: validation and reports.
- `qrcstudy/exploration.py`: exploratory readout and outlier diagnostics.

Published manifests, prediction records, and source inventories retain their original identifiers for verification. Directory and module names now use `extended`; new experiments also use that configuration. Renamed source files produce new run identities, so use fresh output directories for training. Historical source code remains available at revision `7497cc6`.
