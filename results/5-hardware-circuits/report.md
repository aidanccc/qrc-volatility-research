# Hardware circuits for the paper's models

The authors' QR1/QR2 models were turned into gate circuits that IonQ hardware can run. The exact time evolution is replaced by Trotter steps, and each month's input qubits are reset before the next month is loaded.

## Findings

1. The reset circuits give the same features as the authors' method to 1e-10.
2. Second-order Trotter is needed. No first-order setting was close enough to the exact model, even with 8 steps.
3. The chosen circuits score almost the same as the paper: 0.1052 against 0.1051 for QR1, and 0.1042 against 0.1038 for QR2.

## Chosen circuits

A setting is accepted if its forecasts stay within 0.05 (max) and 0.01 (mean) of the exact model's. The cheapest accepted setting is chosen.

| Model | Trotter | Max / mean gap to exact | RXX gates per month | Circuits per month | MSE (paper) |
|---|---|---|---|---|---|
| QR1 | order 2, 2 steps | 0.035 / 0.006 | 270 | 1 | 0.1052 (0.1051) |
| QR2 | order 2, 3 steps | 0.039 / 0.005 | 945 | 2 | 0.1042 (0.1038) |

## Job size

| Model | Months | Circuits | Shots at 1,000 per circuit |
|---|---|---|---|
| QR1, all months | 813 | 813 | 813,000 |
| QR2, all months | 813 | 1,626 | 1,626,000 |
| QR1, forecast months only | 245 | 245 | 245,000 |
| QR2, forecast months only | 245 | 490 | 490,000 |

## Files

- `trotter_sweep.csv`: every setting tried (order 1–2, steps 1–8)
- `selection.json`: the chosen settings and checks
- `circuits/`: OpenQASM 3 files for the chosen circuits, for one month
