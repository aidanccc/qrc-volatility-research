# Noise simulations: IonQ Aria-1 emulation of the QR1/QR2 reservoir

Analysis-only study. The production models, data and published forecasts are unchanged.

## Findings

1. **Gate noise strongly distorts the features.** At 1× Aria-1 with one Trotter step, mean |⟨Z⟩| drops from 0.66 to 0.48 (QR1) and from 0.67 to 0.49 (QR2). Mean feature error against the exact reservoir is 0.20 / 0.19, compared with 0.07 / 0.02 from Trotter error alone. Each readout circuit passes through 135 (QR1) or up to 180 (QR2) noisy RXX gates.
2. **More Trotter steps hurt under noise.** With 1,000 shots, the lowest-error depth at every nonzero noise level is 1 Trotter step. Each extra step reduces Trotter error but adds 45 noisy RXX gates, so the reservoir washes out faster. The noiseless conclusion (more steps are more accurate) reverses on noisy hardware.
3. **Forecasts change little with gate noise.** From 0.5× to 2× Aria-1 (infinite shots), MSE moves by at most 0.0021 in the 571-month window and 0.0128 in the 120-month window, relative to the noiseless Trotter-1 reservoir. The ridge readout is refit on the same noisy features, so a smooth contraction of the features is mostly absorbed by larger weights. Symmetric readout error is a pure rescaling, so on its own it has no effect.
4. **Shot count is the practical bottleneck.** At 1× Aria-1, 100 shots per circuit raises 571-month MSE by +0.0215 (QR1) and +0.0128 (QR2). With 1,000 shots, MSE changes by -0.0236 to -0.0012 against the infinite-shot noisy reservoir (shot-seed s.d. up to 0.0078). Noise shrinks the signal while the shot error stays the same size, so noisier devices need more shots. The improvement at 1,000 shots is consistent with shot noise regularizing the ill-conditioned readout noted in the 1 October diagnostics (λ = 1e-8). It is not evidence of a better reservoir.
5. **Benchmark picture is unchanged.** With 120-month windows, 0 of the noisy QR configurations beat AR1 (MSE 0.1824). With 571-month windows, 1 noisy configuration(s) fall below HARX (0.1711), the lowest by 0.0014, which is inside the shot-seed spread. Realistic Aria-1 noise neither creates nor removes an apparent quantum advantage. The formal comparison (bootstrap intervals and model confidence sets) remains in the published extended study.

## Setup

- Noise rates are medians of 25 public IonQ Aria-1 daily characterizations (2025-08-07 to 2025-09-02): 1-qubit fidelity 0.9998, 2-qubit 0.9867, SPAM 0.9958.
- At 1× Aria-1: depolarizing probability 0.0177 per RXX, 0.0004 per RY, readout flip 0.0042; RZ is noiseless (virtual on IonQ). Scales 0.5×, 2× and 4× multiply all three rates.
- Every forecast month is one 10-qubit circuit with real resets of the 7 input qubits between monthly inputs (all-to-all, no SWAPs), simulated on the Aer density-matrix method. Finite shots are sampled from the noisy outcome distribution at each readout endpoint.
- Couplings: published seed 0. Forecasts use the same ridge readout as the published study (λ = 1e-8; 120- and 571-month windows) over 2018-01-31 to 2026-08-31. The exact-ideal row reproduces the published seed-0 forecasts to 5.5e-10 log RV.
- Shot results average 5 independent shot seeds; the figure error bars show their standard deviation.

## Feature-level results (12 diagnostic histories)

| Model | Trotter steps | Mean abs feature error, 1× Aria-1 | Mean abs feature error, 4× | Mean |⟨Z⟩| exact → 1× | RMS error, 1,000 shots, 1× |
|---|---|---|---|---|---|
| QR1 | 1 | 0.195 | 0.425 | 0.664 → 0.476 | 0.233 |
| QR1 | 2 | 0.245 | 0.536 | 0.664 → 0.420 | 0.283 |
| QR1 | 4 | 0.382 | 0.630 | 0.664 → 0.283 | 0.422 |
| QR2 | 1 | 0.188 | 0.473 | 0.674 → 0.487 | 0.216 |
| QR2 | 2 | 0.302 | 0.590 | 0.674 → 0.372 | 0.334 |
| QR2 | 4 | 0.454 | 0.655 | 0.674 → 0.220 | 0.488 |

## Forecast MSE (log RV), seed 0

| Model | Window | Exact ideal | Trotter-1 ideal | 1× Aria-1 | 1× Aria-1, 1,000 shots | 1× Aria-1, 100 shots | HARX | AR1 |
|---|---|---|---|---|---|---|---|---|
| QR1 | 120 | 0.2113 | 0.2171 | 0.2235 | 0.2045 | 0.2237 | 0.2319 | 0.1824 |
| QR1 | 571 | 0.1848 | 0.1857 | 0.1844 | 0.1827 | 0.2059 | 0.1711 | 0.1757 |
| QR2 | 120 | 0.2318 | 0.2416 | 0.2451 | 0.2214 | 0.2267 | 0.2319 | 0.1824 |
| QR2 | 571 | 0.1729 | 0.1745 | 0.1739 | 0.1727 | 0.1867 | 0.1711 | 0.1757 |

## Figures

- [Reservoir features under Aria-1 noise](figures/01_feature_error_vs_noise.png) ([PDF](figures/01_feature_error_vs_noise.pdf)); data: `tables/feature_sweep_summary.csv`
- [Trotter accuracy vs. gate noise](figures/02_trotter_vs_noise_tradeoff.png) ([PDF](figures/02_trotter_vs_noise_tradeoff.pdf)); data: `tables/feature_sweep_summary.csv`
- [Reservoir readout for one input history](figures/03_feature_example.png) ([PDF](figures/03_feature_example.pdf)); data: `tables/feature_sweep.csv`
- [Forecasts with and without noise](figures/04_forecast_paths.png) ([PDF](figures/04_forecast_paths.pdf)); data: `tables/forecasts.csv`
- [Forecast error by noise configuration](figures/05_forecast_mse.png) ([PDF](figures/05_forecast_mse.pdf)); data: `tables/forecast_summary.csv`

## Limitations

- Local emulation from published calibration medians, not IonQ's own cloud noise simulator or hardware. Depolarizing error is the same for every RXX angle; real small-angle MS gates, crosstalk, drift, and coherent errors are not modeled. SPAM error is applied as readout error only; resets are ideal.
- One coupling seed (0) and one Trotter depth (1) are used for forecasts. Aria-1 has since been retired; the rates describe its mid-2025 performance.
- RY is counted as a single native 1-qubit gate; real compilation may use more.
