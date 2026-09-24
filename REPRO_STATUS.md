# Dataset testing status

September 24, 2026. [Overall report](results/extended-study-2026-09-24/report.md).

## Verified completion

- Data: 920 rows; 412 missing factor cells recovered; four August factor cells remain unavailable; 211 derived identities corrected; 138 statistical flags retained without deleting rows. Original CSV and notebooks unchanged.
- Extended run: all 7,560 expected records; 7,486 successful scored forecasts, two scored ARMAX convergence failures, 28 successful unscored forecasts, and 44 explicitly unavailable September forecasts. No worker failures.
- Modern run: all 7,560 expected records; 7,486 successful scored forecasts, two scored ARMAX convergence failures, and 72 unscored forecasts. All 7,558 successful predictions match the previous modern publication exactly; failure statuses also match.
- Legacy rerun: preprocessing, quantum simulation, LSTM, classical reservoir and comparison notebooks completed. QR1/QR2 reference discrepancies are 1.20e-6/1.08e-6.
- Fresh clone: all 22 tests pass without local caches or checkpoints; regenerating both primary reports preserves prediction, metric, MCS and Markdown report hashes exactly. Evidence: docs-dataset/fresh_clone_validation.json and docs-dataset/tests-fresh-clone.txt.
- Validation: 22 unit/integration tests pass; immutable snapshot/source identity checks pass; pilot resume preserves checkpoints byte-for-byte. Forecast plots inspected with explicit month-end labels.
- Full comparison: both training windows, all models, five stochastic seeds, seed-level errors, positive-variance QLIKE, RMSE/MAE, stationary-bootstrap intervals, MCS and common-date results published.
- Exploratory ridge/conditioning/outlier diagnostics completed separately on matching 80-month evaluation dates with strictly past-only 24-month penalty selection.

Extended final-year MSE is lowest for HARX under both windows (120: 0.063840; 571: 0.063851). Full-period complete-model MSE is lowest for AR1/120 (0.182363) and HARX/571 (0.171100). Confidence sets retain several models; no unique quantum superiority is established.

Primary runs used model source revision 7497cc6; manifests and execution_revision.json verify the model hashes. Subsequent commits add diagnostics and reporting without changing those primary model sources. the extended dataset’s raw construction/scaling metadata and historical macro availability remain unresolved, so this is a retrospective reconstruction rather than exact author-code or live historical forecasting evidence. Separate protocol targets are not pooled.

## Maintenance

Dataset modules, commands, and artifact directories now use neutral `extended` names. Historical manifests and prediction records are unchanged. New training runs require new output directories because source identities changed.

Cleanup verification: all 22 tests pass; the renamed CLI completed a four-fit HAR/persistence pilot. Published report regeneration is checked separately from retraining.
