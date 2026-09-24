# Exploratory readout and outlier diagnostics

These post-benchmark choices are not an untouched confirmation test. Standardization and intercept are fitted only on each training window; alpha minimizes errors on the 24 strictly preceding forecast months. Scores cover January 2020–August 2026 (80 months). Each raw-input ridge uses the same three-step inputs as its paired reservoir. Quantum feature sets differ under the Extended protocol, so QR1/QR2 here are not an isolated virtual-node ablation.

| window | model | mse_log_rv | mae_log_rv | qlike_variance |
| --- | --- | --- | --- | --- |
| 120 | CRLX_matched_ridge | 0.297864 | 0.324323 | 0.789857 |
| 120 | CRLX_raw_input_ridge | 0.153052 | 0.304162 | 0.531214 |
| 120 | QR1_matched_ridge | 0.151464 | 0.291777 | 0.591572 |
| 120 | QR1_raw_input_ridge | 0.153052 | 0.304162 | 0.531214 |
| 120 | QR2_matched_ridge | 0.154092 | 0.293192 | 0.58619 |
| 120 | QR2_raw_input_ridge | 0.153052 | 0.304162 | 0.531214 |
| 571 | CRLX_matched_ridge | 0.142754 | 0.287197 | 0.531637 |
| 571 | CRLX_raw_input_ridge | 0.139499 | 0.284251 | 0.510066 |
| 571 | QR1_matched_ridge | 0.138462 | 0.281401 | 0.506477 |
| 571 | QR1_raw_input_ridge | 0.139499 | 0.284251 | 0.510066 |
| 571 | QR2_matched_ridge | 0.139144 | 0.280283 | 0.519394 |
| 571 | QR2_raw_input_ridge | 0.139499 | 0.284251 | 0.510066 |

See primary_same_dates.csv for primary models on those same 80 dates, matched_intervals.csv for unadjusted paired bootstrap intervals, and outlier_sensitivity.csv for descriptive slices by flagged preceding three-month input history. No primary observation was removed or target clipped.
