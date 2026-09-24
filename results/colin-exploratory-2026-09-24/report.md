# Exploratory readout and outlier diagnostics

These post-benchmark choices are not an untouched confirmation test. Standardization and intercept are fitted only on each training window; alpha minimizes errors on the 24 strictly preceding forecast months. Scores cover January 2020–August 2026 (80 months). Each raw-input ridge uses the same three-step inputs as its paired reservoir. Quantum feature sets differ under the Colin protocol, so QR1/QR2 here are not an isolated virtual-node ablation.

| window | model | mse_log_rv | mae_log_rv | qlike_variance |
| --- | --- | --- | --- | --- |
| 120 | CRLX_matched_ridge | 0.234086 | 0.357313 | 1.40532 |
| 120 | CRLX_raw_input_ridge | 0.243681 | 0.347039 | 1.46742 |
| 120 | QR1_matched_ridge | 0.263465 | 0.324776 | 0.611497 |
| 120 | QR1_raw_input_ridge | 0.16446 | 0.319037 | 0.403947 |
| 120 | QR2_matched_ridge | 0.187991 | 0.314439 | 0.910868 |
| 120 | QR2_raw_input_ridge | 0.173212 | 0.318908 | 0.638405 |
| 571 | CRLX_matched_ridge | 0.148125 | 0.291124 | 0.583098 |
| 571 | CRLX_raw_input_ridge | 0.144593 | 0.283494 | 0.542752 |
| 571 | QR1_matched_ridge | 0.148238 | 0.282712 | 0.603162 |
| 571 | QR1_raw_input_ridge | 0.143847 | 0.290578 | 0.497815 |
| 571 | QR2_matched_ridge | 0.139247 | 0.282085 | 0.528877 |
| 571 | QR2_raw_input_ridge | 0.132616 | 0.268565 | 0.441181 |

See primary_same_dates.csv for primary models on those same 80 dates, matched_intervals.csv for unadjusted paired bootstrap intervals, and outlier_sensitivity.csv for descriptive slices by flagged preceding three-month input history. No primary observation was removed or target clipped.
