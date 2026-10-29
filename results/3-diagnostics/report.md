# Circuit and data diagnostics

These diagnostics cover the extended QR1/QR2 reservoir (seed 0), which ran on the 2018–2026 benchmark.

## Findings

1. Many reservoir outputs barely change from month to month, and several are strongly correlated with each other, so some outputs carry little extra information.
2. The readout is badly conditioned: the median condition number is about 1,000 for QR1 and 6,000 for QR2. Small feature errors can therefore move the forecasts a lot.
3. Even without noise, the memory qubits become mixed: purity falls to about 0.5.
4. Both models miss the largest volatility spikes. Errors are largest around 2020.
5. Going from 1 to 8 Trotter steps cuts feature error by 4–14×, but the gate count goes up 8×.
6. At 1,000 shots, each feature has a standard error of up to 0.03.

## Numbers

| | QR1 | QR2 |
|---|---|---|
| Median readout condition number (120-month window) | 967 | 5,982 |
| Memory purity after 3 months | 0.509 | 0.484 |
| Mean feature error, 1 → 8 Trotter steps | 0.070 → 0.005 | 0.024 → 0.005 |
| Smallest feature variance | 0.000081 | 0.000024 |

| Shots | Worst-case feature standard error |
|---|---|
| 100 | 0.10 |
| 1,000 | 0.032 |
| 10,000 | 0.010 |

## Figures

- Circuit: [protocol](figures/20_circuit_protocol.png), [gate sequence](figures/21_gate_sequence.png), [couplings](figures/05_couplings.png)
- Data: [volatility](figures/01_data_overview.png), [missing data](figures/02_missingness.png), [input ranges](figures/03_encoding_ranges.png), [input correlations](figures/04_input_correlations.png)
- Reservoir: [outputs](figures/06_feature_traces.png), [variance](figures/07_feature_variance.png), [correlations](figures/08_feature_correlations.png), [redundancy](figures/09_feature_spectra.png), [conditioning](figures/10_conditioning.png), [purity](figures/11_hidden_purity.png)
- Forecasts: [paths](figures/12_forecasts.png), [errors over time](figures/13_residual_timeline.png), [error distribution](figures/14_residual_distribution.png), [rolling error](figures/15_rolling_error.png), [error by volatility level](figures/16_regime_errors.png), [availability](figures/17_coverage.png)
- Cost: [Trotter error vs gates](figures/18_accuracy_cost.png), [shot uncertainty](figures/19_shot_uncertainty.png)

All numbers are in `tables/`.
