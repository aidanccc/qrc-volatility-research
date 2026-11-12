# Is the quantum part doing anything, and how accurate is it?

Same 245 test months, same rolling ridge readout for every model. 'K=1' = one month of memory (the hardware candidate).

| model (same ridge readout, lambda=1, standardized) | MSE | OOS R2 vs rolling mean | typical error factor exp(RMSE) | median abs % error in RV | months within +-25% |
|---|---|---|---|---|---|
| Linear, same inputs, lags 1-3 (21) | 0.1024 | 0.6718 | 1.3772 | 19.7174 | 60.8163 |
| Classical: cos/sin of inputs (14) | 0.1050 | 0.6635 | 1.3827 | 19.8173 | 60.4082 |
| Linear, same inputs, lag 1 (7) | 0.1068 | 0.6577 | 1.3865 | 19.9166 | 61.2245 |
| QR1 K=1 quantum (10 Z features) | 0.1077 | 0.6548 | 1.3885 | 20.8646 | 59.5918 |
| HAR (3) | 0.1156 | 0.6294 | 1.4050 | 21.5631 | 55.9184 |
| QR1 K=1 quantum + ZZ correlators (55) | 0.1169 | 0.6254 | 1.4076 | 23.3185 | 54.6939 |
| J=0, no entanglement (= cos(pi x), 10) | 0.1236 | 0.6038 | 1.4214 | 22.6196 | 53.4694 |
| Classical: + pairwise products (105) | 0.1555 | 0.5016 | 1.4834 | 25.0822 | 49.7959 |

| classical basis explaining each quantum <Z> | median R2 | min R2 |
|---|---|---|
| cos/sin of inputs (1-body) | 0.9945 | 0.4677 |
| + pairwise products (2-body) | 1.0000 | 0.9982 |

Note: with RY encoding, every quantum feature is exactly a polynomial in cos(pi x_i), sin(pi x_i) (up to 7-body products), and a 10-qubit circuit is simulated exactly on a laptop. So the question is only whether the circuit picks a *useful* fixed feature map, not whether it is classically hard.

# Circuit reduction: fewer couplings, 1 Trotter step

Readout trained on exact simulated features of the same circuit. s = assumed shrink 0.794^(RXX/90).

| couplings kept | Trotter steps | RXX gates | est. noise shrink s | exact MSE |
|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.1236 |
| 5 | 1.0000 | 5.0000 | 0.9873 | 0.1222 |
| 10 | 1.0000 | 10.0000 | 0.9747 | 0.1189 |
| 15 | 1.0000 | 15.0000 | 0.9623 | 0.1134 |
| 20 | 1.0000 | 20.0000 | 0.9500 | 0.1163 |
| 30 | 1.0000 | 30.0000 | 0.9260 | 0.1084 |
| 45 | 1.0000 | 45.0000 | 0.8911 | 0.1079 |
| 0 | 2.0000 | 0.0000 | 1.0000 | 0.1236 |
| 5 | 2.0000 | 10.0000 | 0.9747 | 0.1226 |
| 10 | 2.0000 | 20.0000 | 0.9500 | 0.1195 |
| 15 | 2.0000 | 30.0000 | 0.9260 | 0.1108 |
| 20 | 2.0000 | 40.0000 | 0.9026 | 0.1111 |
| 30 | 2.0000 | 60.0000 | 0.8575 | 0.1083 |
| 45 | 2.0000 | 90.0000 | 0.7940 | 0.1078 |

# Shot reduction (noise-aware ridge, local readout-noise model)

Best of {Z, Z+ZZ} x {lambda 1, 10} per cell (mean of 5 shot draws). HAR = 0.1156, linear (lags 1-3) on QR1 inputs in Q2 table.

| design | 50 | 100 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|
| 10 couplings, 1 step | 0.1305 | 0.1241 | 0.1216 | 0.1211 | 0.1207 | 0.1186 |
| 15 couplings, 1 step | 0.1288 | 0.1244 | 0.1201 | 0.1158 | 0.1150 | 0.1148 |
| 45 couplings, 1 step | 0.1328 | 0.1216 | 0.1179 | 0.1156 | 0.1129 | 0.1124 |
| 5 couplings, 1 step | 0.1321 | 0.1268 | 0.1237 | 0.1245 | 0.1224 | 0.1210 |
| paper-like: 45 couplings, 2 steps | 0.1376 | 0.1267 | 0.1153 | 0.1169 | 0.1127 | 0.1118 |

Which readout won (features, lambda):

| design | 50 | 100 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|
| 10 couplings, 1 step | Z, 10.0 | Z, 10.0 | Z, 1.0 | Z, 1.0 | Z, 10.0 | Z, 10.0 |
| 15 couplings, 1 step | Z, 1.0 | Z, 1.0 | Z, 1.0 | Z, 10.0 | Z, 10.0 | Z, 1.0 |
| 45 couplings, 1 step | Z, 10.0 | Z, 1.0 | Z, 10.0 | Z, 10.0 | Z, 10.0 | Z, 10.0 |
| 5 couplings, 1 step | Z, 10.0 | Z, 10.0 | Z, 1.0 | Z, 1.0 | Z, 1.0 | Z, 10.0 |
| paper-like: 45 couplings, 2 steps | Z, 10.0 | Z, 1.0 | Z, 1.0 | Z, 1.0 | Z, 1.0 | Z, 1.0 |

# Noise-model check against the IonQ cloud (aria-1) counts

- 20 circuits, 1000 shots: Z shrink s = 0.802, ZZ shrink = 0.653 (model assumes s^2 = 0.643); corr(hardware Z, exact Z) = 0.998

# Cost of a hardware run (USD)

12 test months; calibration circuits dropped (s is estimated from the test circuits themselves against their simulated features: no target data is used). Gate counts from the circuits above (+7 RY +RZ layers +10 measurements).

| design | RXX | shots | Forte commercial (no error mitigation) | Forte commercial (error mitigation on, default) | IonQ research-credit estimator |
|---|---|---|---|---|---|
| paper-like: 45 couplings, 2 steps | 90 | 100 | 309 | 2,018 | 32 |
| paper-like: 45 couplings, 2 steps | 90 | 300 | 393 | 2,018 | 97 |
| paper-like: 45 couplings, 2 steps | 90 | 1,000 | 1,308 | 2,018 | 324 |
| 45 couplings, 1 step | 45 | 100 | 309 | 2,018 | 16 |
| 45 couplings, 1 step | 45 | 300 | 309 | 2,018 | 49 |
| 45 couplings, 1 step | 45 | 1,000 | 677 | 2,018 | 162 |
| 15 couplings, 1 step | 15 | 100 | 309 | 2,018 | 12 |
| 15 couplings, 1 step | 15 | 300 | 309 | 2,018 | 16 |
| 15 couplings, 1 step | 15 | 1,000 | 309 | 2,018 | 54 |
| 10 couplings, 1 step | 10 | 100 | 309 | 2,018 | 12 |
| 10 couplings, 1 step | 10 | 300 | 309 | 2,018 | 12 |
| 10 couplings, 1 step | 10 | 1,000 | 309 | 2,018 | 36 |
| 5 couplings, 1 step | 5 | 100 | 309 | 2,018 | 12 |
| 5 couplings, 1 step | 5 | 300 | 309 | 2,018 | 12 |
| 5 couplings, 1 step | 5 | 1,000 | 309 | 2,018 | 18 |

Prices: Azure Quantum / Google listing for IonQ Forte (per 2-qubit gate-shot $0.001121, 1-qubit $0.00022, minimum per circuit $25.79 without / $168.20 with error mitigation) and the IonQ research-credit resource estimator ($0.0003 per 2-qubit gate-shot, $1 minimum). Confirm with IonQ's dry-run estimate before submitting.
