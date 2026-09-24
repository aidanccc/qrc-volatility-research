# Quantum reservoir volatility forecasting

This shared research repository reproduces and extends [Quantum Reservoir Computing for Realized Volatility Forecasting](https://arxiv.org/html/2505.13933v2). Vikas's current contribution continues on the **Vikas** branch.

## Colin's September 24 dataset

The new `1950-2026.csv` contains 920 monthly observations through August 2026. The original file is preserved. A separately prepared snapshot fills 412 missing factor cells using official sources and validated historical scaling, corrects 211 derived-feature inconsistencies, and retains statistical outliers for review. Four August factor cells remain unavailable.

- [Dataset audit and provenance](docs-colin/AUDIT.md)
- [Execution status](REPRO_STATUS.md)
- [Reproduction commands](RUN_ORDER.md)
- [Assumptions and limitations](ASSUMPTIONS.md)
- [Overall results](results/colin-study-2026-09-24/report.md)
- [Colin paper-feature results](results/colin-2026-09-24/report.md)
- [Modern price-feature results](results/modern-2026-09-24/report.md)

The two protocols have slightly different realized-volatility targets and must be interpreted separately. Both evaluate January 2018–August 2026 with 571- and 120-month rolling windows, five stochastic seeds, eleven model families and persistence. September forecasts are unscored; missing required inputs and failed fits are explicit. A low error rank or MCS p-value of one does not establish unique quantum superiority.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_study.py run --configuration colin --output results/NEW_COLIN_RUN
python run_study.py report --run results/NEW_COLIN_RUN
```

Colin's prepared snapshot is included. Preparing a new snapshot downloads official factors and records source hashes and corrections. Modern training requires retained daily raw inputs or a fresh immutable download; see RUN_ORDER.md. Reports regenerate from published aggregate predictions and checksum receipts. A different source, dataset or configuration requires a new output directory.

## Architecture

- `qrcstudy/colin_data.py`: source verification, missing factors, derived identities and audit artifacts.
- `qrcstudy/colin_models.py` and `colin_run.py`: original paper feature adapters, chronological windows and resumable checkpoints.
- `qrcstudy/data.py`, `models.py`, `run.py`: separately identified modern price-feature study, ported with attribution from Vikas's personal repository.
- `qrcstudy/report.py`, `study_report.py`, `project_report.py`: record validation and data-driven reports.
- `qrcstudy/exploration.py`: separately labeled past-only ridge tuning, matched raw-input baselines and outlier diagnostics.
- Original notebooks, `preprocess.py`, `run_qrc_simulation.py` and quantum simulators retain the historical workflow. `qrcstudy/legacy.py` reruns it in an isolated workspace.

The original data, coupling matrices and results are preserved. [Commit inventories](docs-colin/shared-history.json) and the [historical README](docs-colin/README_BEFORE_EXTENSION.md) document project evolution. The modern infrastructure originated in personal commit `6e7ddf0`; this shared-repository integration preserves its attribution.

Quantum results use ideal local simulation. This is a faithful reconstruction and retrospective extension, not a claim of unpublished author-code access, historical live trading performance, or quantum hardware advantage.
