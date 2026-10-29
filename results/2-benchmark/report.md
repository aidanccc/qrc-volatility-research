# Benchmark on new data (2018–2026)

All 12 models forecast 104 new months, January 2018 to August 2026. Each uses rolling windows of 120 and 571 months, and the stochastic models run with 5 seeds.

- **Extended**: the paper's features, extended to 2026 (`extended/`).
- **Modern**: 7 features rebuilt from daily prices (`modern/`).

## Findings

1. No model is clearly best. The 95% model confidence sets keep most classical and quantum models.
2. With the 571-month window, HARX has the lowest error on the extended features, and QR2 is second.
3. With the 120-month window, AR1 has the lowest error, and both quantum models do worse.
4. The data were repaired: 412 of 416 missing factor values were recovered, 211 derived values were corrected, and 4 August values are still missing.
5. In an exploratory check, a plain ridge on the same raw inputs did as well as or better than the quantum features.

## Forecast error (MSE, 104 months)

| Model | Extended, 571 | Extended, 120 | Modern, 571 | Modern, 120 |
|---|---|---|---|---|
| HARX | 0.1711 | 0.2319 | 0.1737 | 0.1877 |
| QR2 | 0.1737 | 0.2350 | 0.1760 | 0.2165 |
| AR1 | 0.1757 | 0.1824 | 0.1757 | 0.1824 |
| CRLX | 0.1771 | 0.2618 | 0.1710 | 0.2029 |
| QR1 | 0.1802 | 0.2172 | 0.1714 | 0.1859 |
| LSTMX | 0.1869 | 0.2345 | 0.1782 | 0.2303 |
| HAR | 0.1871 | 0.1925 | 0.1871 | 0.1925 |
| AR3 | 0.1878 | 0.1933 | 0.1878 | 0.1933 |
| CRL | 0.1879 | 0.1932 | 0.1879 | 0.1932 |
| ARMAX | 0.1975 | 0.2519 | 0.1774 | 0.1965 |
| LSTM | 0.2026 | 0.2417 | 0.2026 | 0.2417 |
| Persistence | 0.2287 | 0.2287 | 0.2287 | 0.2287 |

Extended and modern use slightly different targets, so compare models within a column, not across columns.

## Last 12 months (September 2025 to August 2026)

| Features | Window | Best model | Best MSE | QR1 | QR2 |
|---|---|---|---|---|---|
| Extended | 571 | HARX | 0.0639 | 0.0828 | 0.0727 |
| Extended | 120 | HARX | 0.0638 | 0.1045 | 0.1234 |
| Modern | 571 | HAR | 0.0855 | 0.0942 | 0.0889 |
| Modern | 120 | LSTMX | 0.0677 | 0.1037 | 0.1115 |

## Figures

- [Data check](data_audit.png)
- [Extended: model comparison](extended/model_comparison.png), [last 12 months](extended/last12_forecasts.png)
- [Modern: model comparison](modern/model_comparison.png), [last 12 months](modern/last12_forecasts.png)

Full tables (per-seed errors, confidence sets, loss intervals, failures) are in `extended/`, `modern/` and their `exploratory/` folders.
