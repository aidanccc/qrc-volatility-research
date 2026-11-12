# Locked protocol: every setting chosen before any test month is seen

Inputs fixed in advance: ['RV', 'DP', 'IP', 'RV_q', 'DEF', 'EP', 'INF']. Validation = one-month-ahead forecasts 1987-08..1997-07 (120 months, 451-month window), entirely inside the paper's training period. Grid: 5 coupling matrices x K [1, 2, 3, 4, 5] x V [1, 2] x 6 readouts = 300 quantum configurations.

## 1. What validation chose

| matrix | K | V | readout | lambda | val MSE |
|---|---|---|---|---|---|
| 2 | 5 | 1 | ridge | 100 | 0.112 |
| 3 | 5 | 1 | ridge | 100 | 0.1124 |
| 2 | 5 | 2 | ridge | 100 | 0.1128 |
| 2 | 5 | 1 | ridge | 10 | 0.1131 |
| 4 | 5 | 1 | ridge | 100 | 0.1135 |
| 3 | 5 | 2 | ridge | 100 | 0.1135 |
| 4 | 5 | 1 | paper | 1e-08 | 0.1136 |
| 3 | 5 | 1 | ridge | 10 | 0.1139 |

The paper's QR2 recipe (matrix 1, K=3, V=2, paper readout) ranks 297 of 300 on validation (val MSE 0.1307 vs best 0.1120).
- Top-20 configurations by K: 5: 13, 4: 4, 3: 2, 2: 1
- Top-20 configurations by V: 1: 14, 2: 6
- Top-20 configurations by readout: ridge: 19, paper: 1

Linear baseline chosen on validation: lags 1..3, ridge lambda 100 (val MSE 0.1087).

## 2. Scored once on each test window

Reference for dMSE / Diebold-Mariano: Locked linear (same inputs). 'sign' uses a 12-month block-bootstrap 95% CI (tie = interval includes 0).


**1997-2017**

| model | MSE | QLIKE | dMSE vs Locked linear (same inputs) | DM p | sign |
|---|---|---|---|---|---|
| HAR | 0.1156 | 1.5685 | -0.0054 | 0.1473 | tie |
| Locked QRC (top-5 average) | 0.1177 | 1.6097 | -0.0034 | 0.1540 | tie |
| Locked QRC (best on validation) | 0.1184 | 1.6279 | -0.0026 | 0.3004 | tie |
| Locked QRC, inputs clipped to 1950-1997 range | 0.1188 | 1.6304 | -0.0022 | 0.3722 | tie |
| Paper QR2 recipe (not locked, for reference) | 0.1196 | 1.7626 | -0.0015 | 0.7097 | tie |
| Locked linear, inputs clipped | 0.1210 | 1.6605 | -0.0001 | 0.8924 | tie |
| Locked linear (same inputs) | 0.1211 | 1.6722 | nan | nan | nan |
| AR(1) | 0.1302 | 1.7184 | 0.0091 | 0.1795 | tie |
| Random walk | 0.1428 | 1.8783 | 0.0217 | 0.0821 | worse |

**2018-2026**

| model | MSE | QLIKE | dMSE vs Locked linear (same inputs) | DM p | sign |
|---|---|---|---|---|---|
| AR(1) | 0.1757 | 1.0257 | -0.0089 | 0.5106 | tie |
| Locked linear, inputs clipped | 0.1811 | 1.0595 | -0.0035 | 0.2201 | tie |
| Locked QRC, inputs clipped to 1950-1997 range | 0.1823 | 1.0803 | -0.0023 | 0.5585 | tie |
| Locked QRC (best on validation) | 0.1834 | 1.0938 | -0.0012 | 0.7548 | tie |
| Locked linear (same inputs) | 0.1846 | 1.0902 | nan | nan | nan |
| Locked QRC (top-5 average) | 0.1857 | 1.1461 | 0.0011 | 0.8553 | tie |
| HAR | 0.1871 | 1.0858 | 0.0025 | 0.6537 | tie |
| Random walk | 0.2287 | 1.4230 | 0.0441 | 0.0193 | worse |
| Paper QR2 recipe (not locked, for reference) | 0.2812 | 5.4155 | 0.0966 | 0.2912 | tie |
