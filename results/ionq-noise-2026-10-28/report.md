# IonQ noise simulations of QR1 and QR2

## Findings

1. Gate noise alone barely changes the forecasts.
2. At 1,000 shots, the forecasts drop to the level of the HAR benchmark and lose to HARX.
3. Matching the paper takes about 100,000 to 1,000,000 shots per circuit.
4. The 1-step circuit is not better than the chosen circuit.
5. IonQ's cloud `aria-1` noise is milder than its calibration data, and cloud `forte-1` noise is harsher.
6. IonQ's cloud simulator could not run the full 24-qubit circuits, so all months were simulated locally.

## Forecast error (MSE, 245 forecasts)

| Model | Noise | Circuit | Infinite shots | 1,000 shots |
|---|---|---|---|---|
| QR1 | none (paper) | exact | 0.1051 | – |
| QR1 | none | chosen | 0.1052 | – |
| QR1 | Aria-1 | chosen | 0.1053 | 0.1158 |
| QR1 | Forte-1 | chosen | 0.1051 | 0.1148 |
| QR1 | Aria-1 | 1-step | 0.1077 | 0.1174 |
| QR1 | Forte-1 | 1-step | 0.1088 | 0.1168 |
| QR2 | none (paper) | exact | 0.1038 | – |
| QR2 | none | chosen | 0.1042 | – |
| QR2 | Aria-1 | chosen | 0.1076 | 0.1214 |
| QR2 | Forte-1 | chosen | 0.1050 | 0.1190 |
| QR2 | Aria-1 | 1-step | 0.1113 | 0.1172 |
| QR2 | Forte-1 | 1-step | 0.1112 | 0.1185 |

Benchmarks: HAR 0.1156, HARX 0.1047.

## Forecast error against shots (chosen circuit)

| Model | Noise | 1,000 | 10,000 | 100,000 | 1,000,000 |
|---|---|---|---|---|---|
| QR1 | none | 0.1158 | 0.1134 | 0.1073 | 0.1056 |
| QR1 | Aria-1 | 0.1186 | 0.1128 | 0.1090 | 0.1061 |
| QR1 | Forte-1 | 0.1155 | 0.1122 | 0.1093 | 0.1053 |
| QR2 | none | 0.1169 | 0.1123 | 0.1098 | 0.1076 |
| QR2 | Aria-1 | 0.1219 | 0.1184 | 0.1154 | 0.1103 |
| QR2 | Forte-1 | 0.1192 | 0.1152 | 0.1125 | 0.1110 |

Each value is the mean of 5 random shot draws, so it can differ from the table above by up to 0.004.

## IonQ cloud check

| Cloud noise model | Matches local noise at | Effect on forecasts |
|---|---|---|
| `aria-1` | 0.73× Aria-1 calibration | Infinite-shot MSE changes by at most 0.001 |
| `forte-1` | 2.84× Forte-1 calibration | Infinite-shot MSE changes by at most 0.003 |

Cloud `ideal` matches the local noiseless simulation to within 0.0014 per outcome probability.

## Figures

- [Forecast error](figures/01_forecast_mse.png)
- [Feature shrinkage](figures/02_feature_contraction.png)
- [Forecast paths](figures/03_forecast_paths.png)
- [Cloud vs local](figures/04_cloud_vs_local.png)
- [Error against shots](figures/05_mse_vs_shots.png)

All numbers are in `tables/`.
