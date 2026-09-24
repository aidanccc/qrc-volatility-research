# Colin study execution

Use the existing Python environment or install requirements.txt. The original notebooks retain their historical workflow. New experiments use separate immutable snapshots and output directories.

1. Inspect docs-colin/AUDIT.md and source manifests.
2. Prepare with `python run_study.py prepare-colin --data 1950-2026.csv --output data/snapshots/NEW --as-of 2026-09-24 --modern-data data/snapshots/2026-09-14-v2/monthly.csv` once the integration milestone is available.
3. Run unit tests, then a separate integration pilot, then full colin and modern configurations. Never resume across differing source/data identities.
4. Regenerate reports from completed forecasts. Unavailable inputs and numerical failures remain explicit.

## Canonical full runs

```bash
python -m unittest discover -s tests -v
python run_study.py run --configuration colin --data data/snapshots/colin-2026-09-24-v2/monthly.csv --output results/colin-2026-09-24 --workers 4 --threads 2
python run_study.py run --configuration modern --data data/snapshots/2026-09-14-v2/monthly.csv --output results/modern-2026-09-24 --workers 4 --threads 2
python run_study.py report --run results/colin-2026-09-24
python run_study.py report --run results/modern-2026-09-24
```

Full runs use all 12 models, both windows and five stochastic seeds. An immutable snapshot manifest is mandatory. A fresh clone may regenerate reports from aggregate forecasts and receipts without private caches. Fresh modern training requires downloading a new raw snapshot with `python run_study.py download --as-of 2026-09-24 --output data/snapshots/NEW`, then specifying that monthly file and a new results directory; the published historical raw download is intentionally not redistributed.

`colin-pilot-2026-09-24` is a diagnostic first-origin run, not a final comparison. Its source hashes precede the final integration revision. Existing historical results and original notebooks are preserved.

## Exploratory diagnostics and overall report

```bash
python run_study.py run --configuration legacy --output results/legacy-validation-2026-09-24
python run_study.py explore --run results/colin-2026-09-24 --output results/colin-exploratory-2026-09-24
python run_study.py explore --run results/modern-2026-09-24 --output results/modern-exploratory-2026-09-24
python run_study.py summarize
```

Exploration requires local reservoir feature caches. Its 24-month past-error tuning is excluded from headline results. The primary manifests identify source revision 7497cc6; subsequent report/exploration additions do not change primary model source hashes. Vikas is the publication branch, explicitly selected by the contributor.
