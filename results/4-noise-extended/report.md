# Noise simulations of the extended reservoir (IonQ Aria-1)

The extended QR1/QR2 reservoir (seed 0, first-order Trotter) was simulated under Aria-1 noise. The error rates come from IonQ's published calibration data.

## Findings

1. At 1× Aria-1 noise, the features shrink by about 28% and are off from the exact values by about 0.19 on average.
2. With noise, 1 Trotter step is always best. Each extra step adds 45 noisy gates.
3. Gate noise barely changes the forecasts: MSE moves by at most 0.002 with the 571-month window, from 0.5× to 2× noise.
4. 100 shots raises MSE by 0.013–0.022. At 1,000 shots or more, the change is within the random spread between shot seeds.
5. Noise does not change how the quantum models compare with the benchmarks.

## Feature error (12 test histories)

| Model | Trotter steps | Mean error, 1× noise | Mean error, 4× noise | Mean \|⟨Z⟩\|, exact → 1× |
|---|---|---|---|---|
| QR1 | 1 | 0.195 | 0.425 | 0.664 → 0.476 |
| QR1 | 2 | 0.245 | 0.536 | 0.664 → 0.420 |
| QR1 | 4 | 0.382 | 0.630 | 0.664 → 0.283 |
| QR2 | 1 | 0.188 | 0.473 | 0.674 → 0.487 |
| QR2 | 2 | 0.302 | 0.590 | 0.674 → 0.372 |
| QR2 | 4 | 0.454 | 0.655 | 0.674 → 0.220 |

## Forecast error (MSE, 2018–2026)

| Model | Window | Exact | 1 Trotter step | 1× noise | 1× noise, 1,000 shots | 1× noise, 100 shots | HARX | AR1 |
|---|---|---|---|---|---|---|---|---|
| QR1 | 120 | 0.2113 | 0.2171 | 0.2235 | 0.2045 | 0.2237 | 0.2319 | 0.1824 |
| QR1 | 571 | 0.1848 | 0.1857 | 0.1844 | 0.1827 | 0.2059 | 0.1711 | 0.1757 |
| QR2 | 120 | 0.2318 | 0.2416 | 0.2451 | 0.2214 | 0.2267 | 0.2319 | 0.1824 |
| QR2 | 571 | 0.1729 | 0.1745 | 0.1739 | 0.1727 | 0.1867 | 0.1711 | 0.1757 |

## Figures

- [Feature error vs noise](figures/01_feature_error_vs_noise.png)
- [Trotter steps vs noise](figures/02_trotter_vs_noise_tradeoff.png)
- [One input history](figures/03_feature_example.png)
- [Forecast paths](figures/04_forecast_paths.png)
- [Forecast error](figures/05_forecast_mse.png)

All numbers are in `tables/`.
