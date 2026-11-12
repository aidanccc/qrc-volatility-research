# Overfitting diagnostics and classical-replacement test

## D1 + D5 — Train/test gap, and replacing the quantum circuit (paper test window 1997-08..2017-12)

Every model: same 245 test months, same 571-month rolling window, ridge readout. Non-QRC models use QR2's 7 inputs. dMSE > 0 => worse than QR2; CI = 12-month block bootstrap.

| Model | MSE | QLIKE | params | train MSE | test/train | dMSE vs QR2 | 95% CI lo | 95% CI hi | DM p |
|---|---|---|---|---|---|---|---|---|---|
| Linear, same inputs, lags 1-3 | 0.1016 | 1.3735 | 22.0000 | 0.0785 | 1.2939 | -0.0002 | -0.0010 | 0.0006 | 0.6634 |
| QR2 + intercept | 0.1036 | 1.3994 | 21.0000 | 0.0818 | 1.2662 | -0.0000 | -0.0001 | 0.0001 | 0.7491 |
| QR2 | 0.1038 | 1.4004 | 20.0000 | 0.0822 | 1.2615 | nan | nan | nan | nan |
| QR1 | 0.1051 | 1.4428 | 10.0000 | 0.0849 | 1.2372 | 0.0001 | -0.0006 | 0.0008 | 0.7801 |
| QR1 + intercept | 0.1052 | 1.4335 | 11.0000 | 0.0841 | 1.2502 | 0.0001 | -0.0006 | 0.0008 | 0.7566 |
| Echo-state network (20 nodes, median seed) | 0.1077 | 1.5065 | 21.0000 | 0.0807 | 1.3350 | 0.0003 | -0.0008 | 0.0013 | 0.5269 |
| Linear, same inputs, lag 1 | 0.1090 | 1.4633 | 8.0000 | 0.0844 | 1.2909 | 0.0004 | -0.0006 | 0.0012 | 0.3515 |
| HAR | 0.1156 | 1.5685 | 4.0000 | 0.0907 | 1.2755 | 0.0010 | -0.0000 | 0.0020 | 0.0649 |
| QR2 with J=0 (no entanglement) | 0.1229 | 1.7078 | 20.0000 | 0.0913 | 1.3452 | 0.0015 | 0.0006 | 0.0024 | 0.0013 |
| QR1 with J=0 (no entanglement) | 0.1234 | 1.6993 | 10.0000 | 0.0916 | 1.3471 | 0.0016 | 0.0006 | 0.0026 | 0.0028 |
| Cosine random features (20 nodes, median seed) | 0.1276 | 1.7141 | 21.0000 | 0.0868 | 1.4706 | 0.0019 | 0.0001 | 0.0044 | 0.0555 |
| AR(1) | 0.1302 | 1.7184 | 2.0000 | 0.0985 | 1.3226 | 0.0021 | 0.0009 | 0.0035 | 0.0023 |
| Random walk | 0.1428 | 1.8783 | 0.0000 | nan | nan | 0.0032 | 0.0014 | 0.0056 | 0.0030 |
- Cosine random features: 30 seeds, MSE median 0.1266, best 0.1081, share of seeds beating QR2 0%
- Echo-state network: 30 seeds, MSE median 0.1073, best 0.1023, share of seeds beating QR2 7%
- Expected test/train ratio from parameter count alone (OLS, n=571, p=20): (n+p)/(n-p) = 1.073. Ratios far above this come from regime change (the test window contains 2000-02 and 2008-09), not from the readout memorising noise.

How much of each QR2 reservoir feature is a simple classical function of the inputs (in-sample R^2):

| classical basis | median R2 | min R2 | features |
|---|---|---|---|
| linear in 3 lags | 0.675 | 0.198 | 20.000 |
| + cos/sin(pi x) of 3 lags | 0.996 | 0.761 | 20.000 |

## D2 — Regularization path (ridge lambda = capacity knob)

(Features standardized inside each training window, unpenalized intercept, so lambda is comparable.)

- QR2 (20 quantum features): test MSE at lambda=1e-8 0.1036; best 0.1035 at lambda=0.032 (-0.1%)
- Linear, QR2 inputs, lags 1-3 (21): test MSE at lambda=1e-8 0.1016; best 0.1016 at lambda=1e-08 (+0.0%)
- Cosine random features (20): test MSE at lambda=1e-8 0.1081; best 0.1059 at lambda=10 (-2.0%)
If heavy regularization helped a lot, the readout would be overfitting. (Picking lambda on the test window is itself selection; this is a diagnostic only.)

## D3 — Learning curve (training-window length)

Test MSE (same 245 test months) vs training-window length:

| train window (months) | Cosine random features (20) | Linear, QR2 inputs, lags 1-3 (21) | QR2 (20 quantum features) |
|---|---|---|---|
| 36 | 0.4762 | 0.3145 | 0.5171 |
| 60 | 0.2655 | 0.1957 | 0.2497 |
| 96 | 0.1680 | 0.1438 | 0.1726 |
| 144 | 0.1298 | 0.1168 | 0.1293 |
| 240 | 0.1196 | 0.1113 | 0.1148 |
| 360 | 0.1113 | 0.1027 | 0.1031 |
| 480 | 0.1086 | 0.1018 | 0.1055 |
| 571 | 0.1081 | 0.1016 | 0.1036 |

## D4 — Selection bias: couplings, memory depth, feature set

- QR1 over 20 coupling matrices: paper's ms[0] ranks 4/20 (MSE 0.1051); median 0.1094; share beating same-input linear (0.1016) 0%; share beating HAR (0.1156) 100%
- QR2 over 20 coupling matrices: paper's ms[1] ranks 2/20 (MSE 0.1038); median 0.1111; share beating same-input linear (0.1016) 0%; share beating HAR (0.1156) 95%
- QR2 memory depth K=1..5 MSE: 0.1141, 0.1118, 0.1038, 0.1104, 0.1099 (paper K=3 is the single best point)
- 40 random 7-input sets (RV + 6 of 11), QR2 recipe vs linear on the SAME inputs: QRC better in 0%; median QRC-linear MSE +0.0129; QRC MSE median 0.1225 vs linear 0.1102
- Same with an intercept added to the QRC readout (the paper's has none): QRC better in 0%; median QRC+int MSE 0.1223

## D6 — True holdout 2018-01..2026-08 (data never used for any modelling choice)

Inputs (pre-specified, MKT/STR unavailable after 2017): ['RV', 'DP', 'IP', 'RV_q', 'DEF', 'EP', 'INF']. Coupling ms[0]/ms[1], V, K, lambda exactly as in the paper. 104 test months.

| Model | MSE | QLIKE | train MSE | test/train | dMSE vs QR2 recipe (holdout inputs) | 95% CI lo | 95% CI hi | DM p |
|---|---|---|---|---|---|---|---|---|
| AR(1) | 0.1757 | 1.0257 | 0.1151 | 1.5265 | -0.0085 | -0.0256 | 0.0013 | 0.2920 |
| Linear, same inputs, lags 1-3 | 0.1772 | 1.0436 | 0.0990 | 1.7902 | -0.0084 | -0.0254 | 0.0010 | 0.2852 |
| Echo-state network (20 nodes, median seed) | 0.1823 | 1.0771 | 0.1029 | 1.7714 | -0.0080 | -0.0245 | 0.0010 | 0.3047 |
| Linear, same inputs, lag 1 | 0.1830 | 1.0787 | 0.1081 | 1.6932 | -0.0079 | -0.0239 | 0.0010 | 0.2940 |
| HAR | 0.1871 | 1.0858 | 0.1080 | 1.7334 | -0.0076 | -0.0239 | 0.0012 | 0.3224 |
| Cosine random features (20 nodes, median seed) | 0.1886 | 1.0973 | 0.1057 | 1.7847 | -0.0075 | -0.0242 | 0.0015 | 0.3387 |
| QR1 recipe (holdout inputs) | 0.2165 | 2.3154 | 0.1066 | 2.0304 | -0.0052 | -0.0153 | 0.0003 | 0.2731 |
| QR1 recipe (holdout inputs) + intercept | 0.2202 | 2.6172 | 0.1065 | 2.0681 | -0.0049 | -0.0144 | 0.0002 | 0.2731 |
| Random walk | 0.2287 | 1.4230 | nan | nan | -0.0042 | -0.0212 | 0.0061 | 0.5873 |
| QR2 recipe (holdout inputs) | 0.2812 | 5.4155 | 0.1035 | 2.7171 | nan | nan | nan | nan |
| QR2 recipe (holdout inputs) + intercept | 0.2815 | 5.4708 | 0.1035 | 2.7198 | 0.0000 | -0.0000 | 0.0001 | 0.5327 |
| QR1 recipe (holdout inputs) with J=0 (no entanglement) | 0.3352 | 1.3817 | 0.1084 | 3.0932 | 0.0044 | -0.0007 | 0.0151 | 0.3499 |
| QR2 recipe (holdout inputs) with J=0 (no entanglement) | 0.3352 | 1.3816 | 0.1084 | 3.0932 | 0.0044 | -0.0007 | 0.0151 | 0.3499 |

Same inputs, paper test window vs true holdout:

| Model (holdout inputs) | MSE 1997-2017 | MSE 2018-2026 |
|---|---|---|
| QR1 recipe (holdout inputs) | 0.1179 | 0.2165 |
| QR2 recipe (holdout inputs) | 0.1196 | 0.2812 |
| Linear, same inputs, lags 1-3 | 0.1143 | 0.1772 |
| HAR | 0.1156 | 0.1871 |
| AR(1) | 0.1302 | 0.1757 |
| Random walk | 0.1428 | 0.2287 |

- QR2 recipe over 40 coupling matrices on the holdout: median 0.2468, best 0.1946; share beating same-input linear (0.1772) 0%
