# Extended paper-feature results

Evaluation: 2025-07-31–2026-08-31. All 30 expected records validated, including unscored forecasts.

Successful scored records: 28. Failed or unavailable records: 0 (see failures.csv and coverage.csv).

## Forecast accuracy

Tables average losses across seeds; seeds are not independent market histories. Rankings below include only complete models. Incomplete models have separate common-date comparisons.

### post2017

| window | model | complete | successful_forecasts | mse_log_rv | rmse_log_rv | mae_log_rv | qlike_variance | mse_seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | HAR | True | 14 | 0.106763 | 0.326745 | 0.283207 | 0.20485 | 0 |
| 120 | Persistence | True | 14 | 0.151665 | 0.389442 | 0.348755 | 0.356422 | 0 |

120-month window: HAR has the lowest complete-model MSE (0.106763).

### last12

| window | model | complete | successful_forecasts | mse_log_rv | rmse_log_rv | mae_log_rv | qlike_variance | mse_seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | HAR | True | 12 | 0.0901862 | 0.30031 | 0.26002 | 0.185485 | 0 |
| 120 | Persistence | True | 12 | 0.135863 | 0.368595 | 0.324983 | 0.31363 | 0 |

120-month window: HAR has the lowest complete-model MSE (0.090186).

![Last twelve months](last12_forecasts.png)

![Complete-model comparison](model_comparison.png)

## Statistical evidence

MCS uses 10,000 stationary-bootstrap replications, expected block length six months, seed 0 and familywise size 0.05. HAR loss-difference intervals use paired date resampling of seed-averaged losses. Unadjusted pairwise intervals do not establish familywise superiority. A p-value of one is not proof of a unique winner.

| model | mcs_pvalue | window | loss | scope | months |
| --- | --- | --- | --- | --- | --- |
| Persistence | 0.0048 | 120 | mse_log_rv | complete_period | 14 |
| HAR | 1 | 120 | mse_log_rv | complete_period | 14 |
| Persistence | 0.0048 | 120 | mse_log_rv | all_model_common_dates | 14 |
| HAR | 1 | 120 | mse_log_rv | all_model_common_dates | 14 |
| Persistence | 0.0054 | 120 | qlike_variance | complete_period | 14 |
| HAR | 1 | 120 | qlike_variance | complete_period | 14 |
| Persistence | 0.0054 | 120 | qlike_variance | all_model_common_dates | 14 |
| HAR | 1 | 120 | qlike_variance | all_model_common_dates | 14 |

## Protocol and limitations

Paper-feature extension: original QR1/QR2 feature subsets, 11-input LSTMX/CRLX, ten macro predictors for HARX/ARMAX. DP/TB differences are fixed from historical conventions. HAR averages are formed causally. Normalized RV is inverted with inherited legacy constants. Quarterly/annual quantum inputs retain their verified historical transformations.

This is retrospective reconstruction: the source includes no construction code or scaling metadata; macro publication availability is unverified. FIZ factors through 2024 transition to CIZ in 2025. Missing August factors block dependent September forecasts. Statistical outliers remain in the primary sample. The target differs slightly from the independently rebuilt modern series, so cross-protocol losses are not pooled.

This run uses rolling windows [120] and stochastic seeds [0]. The month following 2026-08-31 is unscored. Quantum results use ideal exact simulation, not hardware or trading returns. Numerical failures are retained without substituting forecasts. No window or model is selected for deployment using these results.

## Reproduce

`python run_study.py report --run results/report-validation-2026-09-24`

See manifest.json for identity, inputs, source hashes and environment. See RUN_ORDER.md and docs-dataset/AUDIT.md for preparation and source limitations.
