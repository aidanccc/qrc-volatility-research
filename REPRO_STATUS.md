# Colin extension: completed on Vikas

September 24, 2026. [Overall report](results/colin-study-2026-09-24/report.md).

## Verified completion

- Data: 920 rows; 412 missing factor cells recovered; four August factor cells remain unavailable; 211 derived identities corrected; 138 statistical flags retained without deleting rows. Original CSV and notebooks unchanged.
- Colin run: all 7,560 expected records; 7,486 successful scored forecasts, two scored ARMAX convergence failures, 28 successful unscored forecasts, and 44 explicitly unavailable September forecasts. No worker failures.
- Modern run: all 7,560 expected records; 7,486 successful scored forecasts, two scored ARMAX convergence failures, and 72 unscored forecasts. All 7,558 successful predictions match the previous modern publication exactly; failure statuses also match.
- Legacy rerun: preprocessing, quantum simulation, LSTM, classical reservoir and comparison notebooks completed. QR1/QR2 reference discrepancies are 1.20e-6/1.08e-6.
- Validation: 22 unit/integration tests pass; immutable snapshot/source identity checks pass; pilot resume preserves checkpoints byte-for-byte. Forecast plots inspected with explicit month-end labels.
- Full comparison: both training windows, all models, five stochastic seeds, seed-level errors, positive-variance QLIKE, RMSE/MAE, stationary-bootstrap intervals, MCS and common-date results published.
- Exploratory ridge/conditioning/outlier diagnostics completed separately on matching 80-month evaluation dates with strictly past-only 24-month penalty selection.

Colin final-year MSE is lowest for HARX under both windows (120: 0.063840; 571: 0.063851). Full-period complete-model MSE is lowest for AR1/120 (0.182363) and HARX/571 (0.171100). Confidence sets retain several models; no unique quantum superiority is established.

Primary runs used model source revision 7497cc6; manifests and execution_revision.json verify the model hashes. Subsequent commits add diagnostics and reporting without changing those primary model sources. Colin's raw construction/scaling metadata and historical macro availability remain unresolved, so this is a retrospective reconstruction rather than exact author-code or live historical forecasting evidence. Separate protocol targets are not pooled.

## Milestone history


## Data audit completed

Original file preserved; 920 rows audited; 412 factor cells recovered; four remain missing. Prepared v2 corrects 211 derived cells. 138 outlier flags retained without deletion. Historical data agree with the original. Exact new RV normalization and macro vintages remain unresolved. See docs-colin/AUDIT.md.

Integration and full benchmark execution pending. No new-data model results claimed at this milestone.

## Integration validation

Paper-feature adapters, explicit missing-input records, dynamic record validation and protocol-neutral reports added. Ten-model first-origin pilot completed all 20 fits across both windows without failures. Quantum adapter validation compares the real simulator with a cached adapter on a small input, including an unavailable suffix. No model result is interpreted from the pilot. The separate modern interface retains its original default configuration.

## Reporting and branch validation

Publication moved to the existing Vikas branch by explicit user instruction; the temporary branch was deleted after verifying identical published history. The 21-test suite passes. A two-model, 14-month report integration run completed, produced tables/plots/MCS, and was visually checked. Its checkpoint resume preserves all 30 records byte-for-byte. Full Colin, modern and original legacy workflows are running; final metrics remain pending.
