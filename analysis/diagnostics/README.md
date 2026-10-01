# Diagnostic study, 1 October 2026

Start with [the mentor report](../../results/diagnostics-2026-10-01/report.html) or its [Markdown version](../../results/diagnostics-2026-10-01/report.md). Each of the 21 figures has PNG/PDF exports, source tables, an interpretation, and limitations. [Future prompts](FUTURE_CODEX_PROMPTS.md) describe deferred fixes and optimization experiments.

## Architecture

Prepared monthly snapshot → model-specific seven-feature inputs → RY input encoding → exact or Trotter reservoir evolution → Z expectation features → past-only rolling ridge readout → forecasts → artifact validation and diagnostic plots.

`compute.py` calls the unmodified production exact simulator for full seed-0 features. Instrumentation retains intermediate hidden states for 12 selected histories and compares existing Trotter circuits against exact evolution. `render.py` renders the recorded tables. `test_diagnostics.py` checks reset-channel equivalence, complete circuit resource counts, and correlated shot statistics. `verify.py` checks preservation, artifact integrity, and deterministic re-rendering.

## Reproduce locally

Use the existing `.venv` from the repository root. The pinned scientific environment passed `pip check`; no packages were changed. No provider token, cloud account, Julia install, or neural-network retraining is required.

```bash
# Pick a new output directory. An existing directory is rejected by the runner.
.venv/bin/python analysis/diagnostics/run.py --output results/diagnostics-new-run
```

The observed initial computation took about seven minutes on this machine. It regenerates exact QR1/QR2 features for seed 0 and uses all existing published forecast seeds. A complete run produces 21 PNGs, 21 PDFs, source tables, manifests, test logs, and a linked HTML/Markdown report. Timings and PDF timestamps need not match; numeric tables and PNGs should reproduce in the same environment.

For a report-only rebuild using existing computed tables:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/qrc-diagnostics-mpl XDG_CACHE_HOME=/tmp/qrc-diagnostics-cache OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/diagnostics/render.py --output results/diagnostics-2026-10-01
PYTHONPATH=.:analysis/diagnostics .venv/bin/python analysis/diagnostics/verify.py --output results/diagnostics-2026-10-01 --check-render
```

The cache contains only compact seed-0 feature arrays with checksum receipts; no dense quantum operators are saved. The existing `.gitignore` excludes these recomputable caches. Features also appear as dated CSV tables. Published forecast validation happens on temporary copies because the inherited validator rewrites its input CSV.

## Boundaries

All new code is analysis-only. The model, data snapshot, published forecasts, and original notebook sources remain unchanged. Conditioning and feature plots describe seed 0; forecast plots use seeds 0–4 and both 120/571-month windows. Historical saved reference comparisons are distinct from the newly executed historical three-input instrumentation check. The study does not claim a fresh full historical rerun, quantum advantage, calibrated IonQ noise results, or demonstrated device executability.

See [migration and preservation](MIGRATION.md), the run manifest, validation receipt, and final verification receipt for provenance.
