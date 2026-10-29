# Paper reproduction

The authors' original notebooks were rerun unchanged, with fixed seeds.

## Findings

1. Our QR1 and QR2 simulations match the authors' published predictions to within 1.2e-6.
2. QR2 has the lowest error of all 11 models. QR1 is fourth, behind ARMAX and HARX.

## Forecast error (MSE, 245 forecasts, 1997-08 to 2017-12)

| Model | MSE |
|---|---|
| QR2 | 0.1038 |
| ARMAX | 0.1039 |
| HARX | 0.1047 |
| QR1 | 0.1051 |
| LSTMX | 0.1088 |
| CRLX | 0.1098 |
| HAR | 0.1156 |
| AR3 | 0.1179 |
| CRL | 0.1181 |
| LSTM | 0.1292 |
| AR1 | 0.1301 |

## Files

- `predictions.csv`: every forecast
- `legacy_metrics.csv`: the table above
- `quantum_reference_comparison.json`: largest gap to the authors' predictions
- `original-outputs/`: plots, Diebold-Mariano tests and predictions from the first notebook run (April 2026)
