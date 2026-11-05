# D7 — Repeated trials: train vs test error with confidence bands

QR2 refit with 20 independent coupling-matrix draws (the paper reports one); echo-state network refit with 20 seeds. Model band = 5-95% across runs. Data band = 95% CI from a 12-month block bootstrap of the test months.


### 1997-08..2017-12 (paper window)

| model | runs | train MSE | test MSE mean | test MSE sd | band lo | band hi | band type | test/train |
|---|---|---|---|---|---|---|---|---|
| QR2 (coupling draw) | 20 | 0.0830 | 0.1104 | 0.0034 | 0.1038 | 0.1152 | 5-95% across runs | 1.3289 |
| QR2 (paper draw only) | 1 | 0.0822 | 0.1038 | nan | 0.0783 | 0.1298 | block bootstrap 95% | 1.2615 |
| Echo-state network (seed) | 20 | 0.0809 | 0.1073 | 0.0032 | 0.1033 | 0.1131 | 5-95% across runs | 1.3265 |
| Linear, same inputs, lags 1-3 | 1 | 0.0785 | 0.1016 | nan | 0.0773 | 0.1270 | block bootstrap 95% | 1.2939 |
| HAR | 1 | 0.0907 | 0.1156 | nan | 0.0904 | 0.1446 | block bootstrap 95% | 1.2755 |

### 2018-01..2026-08 (true holdout)

| model | runs | train MSE | test MSE mean | test MSE sd | band lo | band hi | band type | test/train |
|---|---|---|---|---|---|---|---|---|
| QR2 (coupling draw) | 20 | 0.1038 | 0.2532 | 0.0423 | 0.2030 | 0.3162 | 5-95% across runs | 2.4397 |
| QR2 (paper draw only) | 1 | 0.1035 | 0.2812 | nan | 0.0981 | 0.5786 | block bootstrap 95% | 2.7171 |
| Echo-state network (seed) | 20 | 0.1014 | 0.1831 | 0.0034 | 0.1787 | 0.1877 | 5-95% across runs | 1.8054 |
| Linear, same inputs, lags 1-3 | 1 | 0.0990 | 0.1772 | nan | 0.1018 | 0.2642 | block bootstrap 95% | 1.7902 |
| HAR | 1 | 0.1080 | 0.1871 | nan | 0.1023 | 0.2893 | block bootstrap 95% | 1.7334 |

- 1997-08..2017-12 (paper window): QR2 draws beating linear 0%, beating HAR 95%; paper draw ranks 2/20

- 2018-01..2026-08 (true holdout): QR2 draws beating linear 0%, beating HAR 0%; paper draw ranks 15/20

# D8 — Year-by-year forecast error, 1997-08..2026-08 (continuous run, 7 inputs available throughout)

Mean squared error per calendar year (lower is better). 1997 = Aug-Dec only, 2026 = Jan-Aug only.

| year | QR2 (paper draw) | QR1 (paper draw) | Linear, same inputs, lags 1-3 | HAR | AR(1) | QR2 band 5% | QR2 band 95% | best model |
|---|---|---|---|---|---|---|---|---|
| 1997 | 0.1569 | 0.1147 | 0.1529 | 0.1628 | 0.1725 | 0.1269 | 0.1676 | QR1 (paper draw) |
| 1998 | 0.1280 | 0.1139 | 0.1294 | 0.1131 | 0.1260 | 0.1198 | 0.1408 | HAR |
| 1999 | 0.0473 | 0.0417 | 0.0420 | 0.0416 | 0.0586 | 0.0412 | 0.0746 | HAR |
| 2000 | 0.1152 | 0.1035 | 0.1217 | 0.1214 | 0.1592 | 0.1019 | 0.1256 | QR1 (paper draw) |
| 2001 | 0.0767 | 0.0732 | 0.0835 | 0.0821 | 0.1099 | 0.0685 | 0.0842 | QR1 (paper draw) |
| 2002 | 0.0940 | 0.1032 | 0.0846 | 0.0950 | 0.1191 | 0.0899 | 0.1126 | Linear, same inputs, lags 1-3 |
| 2003 | 0.0555 | 0.0591 | 0.0421 | 0.0601 | 0.0647 | 0.0554 | 0.0759 | Linear, same inputs, lags 1-3 |
| 2004 | 0.0460 | 0.0501 | 0.0512 | 0.0442 | 0.0524 | 0.0458 | 0.0646 | HAR |
| 2005 | 0.0637 | 0.0691 | 0.0708 | 0.0580 | 0.0680 | 0.0586 | 0.0732 | HAR |
| 2006 | 0.0743 | 0.0733 | 0.0720 | 0.0644 | 0.0647 | 0.0702 | 0.0765 | HAR |
| 2007 | 0.1281 | 0.1238 | 0.1080 | 0.1285 | 0.1178 | 0.1148 | 0.1305 | Linear, same inputs, lags 1-3 |
| 2008 | 0.2062 | 0.2826 | 0.1887 | 0.2333 | 0.2437 | 0.2099 | 0.3290 | Linear, same inputs, lags 1-3 |
| 2009 | 0.1781 | 0.0737 | 0.0694 | 0.0789 | 0.0692 | 0.0770 | 0.1416 | AR(1) |
| 2010 | 0.1437 | 0.1343 | 0.1776 | 0.1204 | 0.1317 | 0.1319 | 0.1479 | HAR |
| 2011 | 0.1903 | 0.1958 | 0.1807 | 0.1936 | 0.1994 | 0.1884 | 0.2009 | Linear, same inputs, lags 1-3 |
| 2012 | 0.1142 | 0.1125 | 0.1287 | 0.1111 | 0.0872 | 0.0970 | 0.1229 | AR(1) |
| 2013 | 0.0897 | 0.0910 | 0.1054 | 0.1018 | 0.1249 | 0.0861 | 0.0969 | QR2 (paper draw) |
| 2014 | 0.2373 | 0.2520 | 0.2379 | 0.2589 | 0.3335 | 0.2347 | 0.2543 | QR2 (paper draw) |
| 2015 | 0.1285 | 0.1256 | 0.1007 | 0.1164 | 0.1234 | 0.1182 | 0.1354 | Linear, same inputs, lags 1-3 |
| 2016 | 0.1635 | 0.1688 | 0.1742 | 0.1815 | 0.2033 | 0.1573 | 0.1691 | QR2 (paper draw) |
| 2017 | 0.0951 | 0.1128 | 0.1015 | 0.0887 | 0.1302 | 0.0951 | 0.1285 | HAR |
| 2018 | 0.3471 | 0.3117 | 0.3031 | 0.3339 | 0.2716 | 0.3124 | 0.3541 | AR(1) |
| 2019 | 0.2207 | 0.2431 | 0.1990 | 0.2570 | 0.2983 | 0.2193 | 0.2444 | Linear, same inputs, lags 1-3 |
| 2020 | 1.3224 | 0.7822 | 0.4336 | 0.4657 | 0.3919 | 0.6364 | 1.6056 | AR(1) |
| 2021 | 0.0660 | 0.0711 | 0.0798 | 0.0641 | 0.0760 | 0.0645 | 0.0899 | HAR |
| 2022 | 0.0516 | 0.0570 | 0.0448 | 0.0740 | 0.0710 | 0.0514 | 0.0687 | Linear, same inputs, lags 1-3 |
| 2023 | 0.0604 | 0.0574 | 0.0697 | 0.0600 | 0.0544 | 0.0501 | 0.0608 | AR(1) |
| 2024 | 0.0869 | 0.0830 | 0.0971 | 0.0796 | 0.0849 | 0.0711 | 0.0903 | HAR |
| 2025 | 0.2310 | 0.2251 | 0.2588 | 0.2481 | 0.2315 | 0.2245 | 0.2434 | QR1 (paper draw) |
| 2026 | 0.0771 | 0.0683 | 0.0745 | 0.0591 | 0.0650 | 0.0629 | 0.0857 | HAR |

Years won (lowest MSE): HAR 10, Linear, same inputs, lags 1-3 8, AR(1) 5, QR1 (paper draw) 4, QR2 (paper draw) 3
QR2 beats same-input linear in 15/30 years. Largest QR2 losses: 2020 (+0.889), 2009 (+0.109), 2018 (+0.044)
Share of each model's total 1997-2026 squared error that comes from 2020 alone: QR2 (paper draw) 27%, QR1 (paper draw) 18%, Linear, same inputs, lags 1-3 11%, HAR 12%, AR(1) 9%
