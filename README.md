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
