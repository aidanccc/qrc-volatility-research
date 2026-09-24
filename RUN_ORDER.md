# Colin study execution

Use the existing Python environment or install requirements.txt. The original notebooks retain their historical workflow. New experiments use separate immutable snapshots and output directories.

1. Inspect docs-colin/AUDIT.md and source manifests.
2. Prepare with `python run_study.py prepare-colin --data 1950-2026.csv --output data/snapshots/NEW --as-of 2026-09-24 --modern-data data/snapshots/2026-09-14-v2/monthly.csv` once the integration milestone is available.
3. Run unit tests, then a separate integration pilot, then full colin and modern configurations. Never resume across differing source/data identities.
4. Regenerate reports from completed forecasts. Unavailable inputs and numerical failures remain explicit.
