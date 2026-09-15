# Quantum reservoir volatility forecasting

This project studies monthly S&P 500 realized-volatility forecasts using quantum
reservoir simulation, classical reservoirs, LSTMs, and statistical baselines.
It contains a reconstruction of the partial academic release and a separately
identified current-market extension through **August 2026**.

The modern experiment evaluates January 2018–August 2026 and highlights the
last twelve months. It uses current public daily prices, seven causal market
features, five seeds, and both 571-month and 120-month rolling training windows.
The quantum models run locally with ideal simulated dynamics.

## Read the results

- [Detailed project and modernization audit](MODERNIZATION_AUDIT.md)
- [Modern results and plots](results/modern-2026-09-14/report.md)
- [Date-indexed modern predictions](results/modern-2026-09-14/predictions.csv)
- [Per-seed metrics](results/modern-2026-09-14/metrics_by_seed.csv)
- [Model Confidence Sets](results/modern-2026-09-14/mcs.csv)
- [Legacy rerun metrics](results/legacy-2026-09-14/legacy_metrics.csv)
- [Legacy quantum reference verification](results/legacy-2026-09-14/quantum_reference_comparison.json)
- `Current_Market_Results.ipynb` is a readable entry point to a selected modern run.

A successful reproduction does not require quantum superiority. The previous
README overstated the saved statistical evidence; the original text is archived
locally in `docs/README_BEFORE_MODERNIZATION.md`. A Model Confidence Set p-value
of one does not establish a uniquely superior model.

## Run locally

Use the existing `.venv`, or install the pinned direct dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements-lock.txt` records the complete local environment used for the run.
To use the validated included snapshot:

```bash
python run_study.py run --configuration modern   --data data/snapshots/2026-09-14-v2/monthly.csv   --output results/modern-2026-09-14 --workers 4 --threads 2
python run_study.py report --run results/modern-2026-09-14
```

The default is all 12 models, seeds 0–4, both training windows, 104 scored
months and one unscored forecast. Repeating the same command resumes completed
checkpoints. A changed dataset, model source, environment, or configuration
requires a new output directory. `--limit 1` runs one pending origin per
model/seed/window; remove the limit to continue. `--models`, `--seeds`,
`--windows`, `--start`, and `--end` configure a separate experiment explicitly.

To obtain another immutable snapshot, supply a new output directory and date:

```bash
python run_study.py download --as-of 2026-09-14 --output data/snapshots/new-download
```

Downloads retain raw Yahoo and FRED payloads, validate daily granularity and
exchange sessions, reconcile provider differences, and derive complete months.
The existing snapshot is never overwritten. Change `--data` and `--end` when
running a newer experiment; the endpoint is not silently extended on resume.

To reproduce the original notebook workflow in an isolated directory:

```bash
python run_study.py run --configuration legacy --output results/legacy-2026-09-14
```

This executes preprocessing, the original quantum driver, LSTM, classical
reservoir, and the comparison notebook. Original data and results remain
preserved. The comparison consumes freshly regenerated quantum predictions.
The legacy run retains historical loss formulas and records numerical agreement
with the author CSV; it is not a claim of access to unpublished author code.

## Architecture

- `qrcstudy/data.py`: raw snapshots, session validation, volatility construction,
  feature scaling, and historical reconciliation.
- `qrcstudy/models.py`: reusable statistical, reservoir, and LSTM model adapters.
- `qrcstudy/run.py`: frozen run identities, worker scheduling, and checkpoints.
- `qrcstudy/report.py`: validated result loading, modern losses, uncertainty,
  tables, and plots.
- `qrcstudy/legacy.py`: isolated execution of the academic notebook pipeline.
- `quantum_reservoir_qiskit.py`: original dense quantum simulation shared by
  legacy and modern runners. `quantum_reservoir_trotter.py` is a separate
  approximation implementation; no Trotter sweep is part of this experiment.

The original notebooks remain available. Their legacy normalized data and
historical metrics must not be mixed with modern run artifacts.

## Interpretation

Forecasts target next month's log realized volatility using information through
the previous month. Modern QLIKE uses positive variances. Both training windows
are reported, with no selection on the final year. Model failures are explicit;
full-period comparisons use complete models, with separate common-date results
for models having failed fits. Five seeds are not five independent histories.
September 2026 is an unscored forecast, not a completed realized target.

Sources: [paper](https://arxiv.org/html/2505.13933v2),
[FRED S&P 500](https://fred.stlouisfed.org/series/SP500),
[statistical comparison documentation](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html).
