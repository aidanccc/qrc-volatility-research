# IonQ noise simulations of the paper-faithful QR1/QR2 circuits

Analysis-only study. The production models, data and published forecasts are unchanged. Code and instructions: [`analysis/ionq_cloud/`](../../analysis/ionq_cloud/README.md).

The question: what happens to the paper's QR1/QR2 forecasts (arXiv:2505.13933) when the circuits we would send to IonQ hardware are run with IonQ-level noise? The circuits are the ones in `hardware/`: the authors' couplings and features, second-order Trotter evolution, and resets between monthly inputs.

## Findings

1. **Gate noise alone barely changes the forecasts.** With exact (infinite-shot) noisy features at the chosen depth, MSE is 0.1051–0.1053 for QR1 and 0.1050–0.1078 for QR2, across all four noise models. The authors' values are 0.1051 and 0.1038. This holds even though noise shrinks mean |⟨Z⟩| from 0.64 to 0.38–0.53 (QR1) and from 0.67 to 0.26–0.47 (QR2). The ridge readout is refit on the noisy features, so a smooth shrinkage is mostly absorbed by larger weights. QR2 loses more because it runs 945 noisy RXX gates per month, against QR1's 270.
2. **Shot noise is what hurts.** At 1,000 shots per circuit, MSE rises to 0.114–0.116 (QR1) and 0.119–0.122 (QR2) at the chosen depth. That is about HAR (0.1156) and clearly worse than HARX (0.1047). Noise makes this worse because it shrinks the signal while the shot error stays the same size.
3. **Reaching the paper's accuracy would take roughly 10⁵–10⁶ shots per circuit.** Even with no gate noise, QR1 needs about 3×10⁵ shots to come within 0.001 of its exact MSE. QR2 is still at 0.1076 at 10⁶ shots ([figure 5](figures/05_mse_vs_shots.png)). The readout uses λ = 1e-8, which makes it very sensitive to small feature errors (see the 1 October diagnostics). For scale, one full QR1 run is 813 circuits, and QR2 is 1,626.
4. **The shallow (1-step) circuit is not a better choice here.** At infinite shots it is worse than the chosen depth for both models (QR1 0.108–0.110, QR2 0.111), because its Trotter error is larger. At 1,000 shots, shot noise dominates either way: QR1 is slightly worse at shallow depth, and QR2 is 0.0005–0.004 better, about one to two seed s.d. This differs from the 22 October study, where one step was best under noise. That study used the extended reservoir with first-order Trotter, where more steps added gate noise without fixing a large Trotter error.
5. **IonQ's cloud noise models do not match the published calibration rates, and they disagree in opposite directions.** On the same 10- and 17-qubit check circuits:
   - Cloud `aria-1` shrinks the features *less* than a depolarizing model at Aria-1 calibration rates. The best single match is 0.73× those rates.
   - Cloud `forte-1` shrinks them much *more* than Forte-1 calibration rates. The best match is 2.84×. It also shows a separate branch of strongly shrunk values ([figure 4](figures/04_cloud_vs_local.png)), which no single uniform depolarizing rate can reproduce.
   - Rerunning every month at these matched scales changes no conclusion above: infinite-shot MSE moves by at most 0.003, and 1,000-shot MSE stays at 0.114–0.122.
6. **The full circuits could not run on IonQ's cloud simulator.** It has no reset. The reset-free version needs 24 qubits (3 months × 7 inputs + 3 memory). For this project, cloud jobs above 19 qubits (`ideal`, `aria-1`) or above 11 qubits (`forte-1`) stay queued and never start. That is below IonQ's documented limits of 25 and 29 qubits, and none of the project's roughly 1,500 earlier jobs used 20 or more qubits. So the full study ran locally, and the cloud was used only to check the local noise model on shorter circuits.

## Setup

- **Circuits.** Built from `hardware/paper_circuits.py`: the authors' `coeff_10.jld2` couplings, the paper's features and τ = 1. QR1 has 1 virtual node and QR2 has 2.
  - Chosen depth (from `hardware/select_trotter.py`): order 2 with QR1 at 2 steps (270 RXX per month) and QR2 at 3 steps (945). Shallow depth: 1 step (135 / 315).
  - The noiseless Trotter circuits score 0.1052 / 0.1042, and the exact model reproduces the authors' `predict_result.csv` to under 1e-5 (asserted in `analyze.py`).
- **Local simulation (all 813 feature months).** Each month is one 10-qubit reset circuit on the qiskit-aer density-matrix method.
  - Noise: depolarizing error on each RXX and RY, noiseless virtual RZ, and symmetric readout error.
  - Rates are medians of IonQ's public daily calibrations:

    | Noise model | Calibrations used | 2-qubit fidelity | 1-qubit fidelity | SPAM fidelity | RXX depolarizing | Readout flip |
    |---|---|---|---|---|---|---|
    | Aria-1 | 25 days, Aug–Sep 2025 | 0.9867 | 0.9998 | 0.9958 | 0.0177 | 0.0042 |
    | Forte-1 | 29 days, 29 Sep–27 Oct 2026 | 0.9951 | 0.9999 | 0.9963 | 0.0065 | 0.0037 |

  - The cloud-matched variants multiply all rates by 0.73 (Aria-1) and 2.84 (Forte-1). See `tables/noise_parameters.csv`.
  - Shots are sampled from the exact noisy distribution, with 5 seeds.
  - The shot sweep (figure 5) samples each ⟨Z⟩ binomially instead. At 1,000 shots it agrees with the full sampling to within 0.0036 MSE, about the seed spread.
- **Readout.** The authors' rolling ridge regression: 571-month window, λ = 1e-8, 245 forecasts from 1997-08 to 2017-12. It is trained and tested on noisy features, as a hardware run would be.
  - Benchmarks are on the same sample: AR1 0.1301, HAR 0.1156, HARX 0.1047, ARMAX 0.1039.
- **Cloud check.** 10 jobs on `ionq_simulator` (never a QPU), 1,000 shots, fixed seed.
  - Circuits: noise models `ideal`, `aria-1` and `forte-1`; both models and both depths; 10 months spread over the sample.
  - The circuits use 1-month histories (10 qubits) for all three noise models, and 2-month histories (17 qubits) for `ideal` and `aria-1`.
  - The cloud's `ideal` model returns exact probabilities but drops small outcomes: 84–96% of the probability mass comes back. On the outcomes it does return, it matches our noiseless simulation to within 1.4e-3, consistent with IonQ's 1e-3 compilation precision. This confirms that qubit and bit ordering are right (`tables/cloud_ideal_check.csv`).
  - The cloud-matched scale is the one at which the local model's average shrinkage (the through-origin slope of noisy against noiseless ⟨Z⟩) equals the cloud's (`tables/cloud_matched_scales.json`).

## Forecast MSE (log RV), 245 forecasts

| Model | Noise | Depth | RXX per month | MSE, infinite shots | MSE, 1,000 shots (± s.d.) | Mean \|⟨Z⟩\| | Mean feature error vs exact |
|---|---|---|---|---|---|---|---|
| QR1 | exact (authors) | – | – | 0.1051 | – | 0.642 | 0 |
| QR1 | noiseless Trotter | chosen | 270 | 0.1052 | – | 0.633 | 0.013 |
| QR1 | Aria-1 | chosen | 270 | 0.1053 | 0.1158 ± 0.0026 | 0.393 | 0.250 |
| QR1 | Aria-1 ×0.73 (cloud-matched) | chosen | 270 | 0.1052 | 0.1149 ± 0.0037 | 0.445 | 0.198 |
| QR1 | Forte-1 | chosen | 270 | 0.1051 | 0.1148 ± 0.0018 | 0.526 | 0.117 |
| QR1 | Forte-1 ×2.84 (cloud-matched) | chosen | 270 | 0.1053 | 0.1143 ± 0.0014 | 0.380 | 0.263 |
| QR1 | noiseless Trotter | shallow | 135 | 0.1098 | – | 0.567 | 0.089 |
| QR1 | Aria-1 | shallow | 135 | 0.1077 | 0.1174 ± 0.0011 | 0.440 | 0.205 |
| QR1 | Forte-1 | shallow | 135 | 0.1088 | 0.1168 ± 0.0026 | 0.513 | 0.135 |
| QR2 | exact (authors) | – | – | 0.1038 | – | 0.670 | 0 |
| QR2 | noiseless Trotter | chosen | 945 | 0.1042 | – | 0.668 | 0.003 |
| QR2 | Aria-1 | chosen | 945 | 0.1076 | 0.1214 ± 0.0040 | 0.271 | 0.399 |
| QR2 | Aria-1 ×0.73 (cloud-matched) | chosen | 945 | 0.1066 | 0.1205 ± 0.0002 | 0.340 | 0.330 |
| QR2 | Forte-1 | chosen | 945 | 0.1050 | 0.1190 ± 0.0013 | 0.468 | 0.203 |
| QR2 | Forte-1 ×2.84 (cloud-matched) | chosen | 945 | 0.1078 | 0.1224 ± 0.0026 | 0.258 | 0.412 |
| QR2 | noiseless Trotter | shallow | 315 | 0.1111 | – | 0.633 | 0.041 |
| QR2 | Aria-1 | shallow | 315 | 0.1113 | 0.1172 ± 0.0015 | 0.456 | 0.214 |
| QR2 | Forte-1 | shallow | 315 | 0.1112 | 0.1185 ± 0.0015 | 0.558 | 0.113 |

Shallow rows at the cloud-matched scales are in `tables/forecast_summary.csv`. Each is within 0.0012 (infinite shots) and 0.0017 (1,000 shots) of the matching 1× row.

## Cloud check (noisy models, 1,000 shots)

| Noise model | Model | Depth | Qubits | Shrinkage, local 1× | Shrinkage, cloud | Mean \|cloud − local\| | RMS z-score |
|---|---|---|---|---|---|---|---|
| aria-1 | QR1 | chosen | 10 / 17 | 0.726 / 0.654 | 0.789 / 0.731 | 0.046 / 0.054 | 2.0 / 2.2 |
| aria-1 | QR2 | chosen | 10 / 17 | 0.513 / 0.441 | 0.615 / 0.543 | 0.078 / 0.075 | 3.0 / 2.8 |
| aria-1 | QR1 | shallow | 10 / 17 | 0.842 / 0.796 | 0.884 / 0.848 | 0.030 / 0.039 | 1.4 / 1.8 |
| aria-1 | QR2 | shallow | 10 / 17 | 0.791 / 0.748 | 0.846 / 0.802 | 0.043 / 0.042 | 2.1 / 1.9 |
| forte-1 | QR1 | chosen | 10 | 0.886 | 0.706 | 0.127 | 6.3 |
| forte-1 | QR2 | chosen | 10 | 0.776 | 0.492 | 0.211 | 9.8 |
| forte-1 | QR1 | shallow | 10 | 0.935 | 0.831 | 0.070 | 3.4 |
| forte-1 | QR2 | shallow | 10 | 0.913 | 0.771 | 0.107 | 5.9 |

Shrinkage is the slope of noisy against noiseless ⟨Z⟩ (1 = no noise). The z-score uses the 1,000-shot standard error, so a value near 1 would mean agreement within shot noise. Per-value data: `tables/cloud_check.csv`.

## Figures

- [Forecast error by configuration](figures/01_forecast_mse.png) ([PDF](figures/01_forecast_mse.pdf)); data: `tables/forecast_summary.csv`
- [Feature shrinkage](figures/02_feature_contraction.png) ([PDF](figures/02_feature_contraction.pdf)); data: `tables/forecast_summary.csv`
- [Forecast paths, cloud-matched noise](figures/03_forecast_paths.png) ([PDF](figures/03_forecast_paths.pdf)); data: `tables/forecasts.csv`
- [IonQ cloud vs local noise model](figures/04_cloud_vs_local.png) ([PDF](figures/04_cloud_vs_local.pdf)); data: `tables/cloud_check.csv`
- [MSE against shots per circuit](figures/05_mse_vs_shots.png) ([PDF](figures/05_mse_vs_shots.pdf)); data: `tables/shot_sweep.csv`

## Limitations

- This is simulation only, nothing ran on hardware. IonQ's cloud noise models are black boxes, and the comparison above shows they differ from the published calibration medians. Real hardware may differ from both.
- The local model treats every RXX angle the same (uniform depolarizing error). It does not model crosstalk, drift, coherent errors or noisy resets, and SPAM error is applied as readout error only. The cloud `forte-1` results show qubit-dependent structure that this model cannot reproduce.
- The cloud check only reaches 1- and 2-month histories (10 and 17 qubits), and `forte-1` only the 1-month one, because of the qubit cutoff described above. The full 3-month circuits were checked only locally.
- Aria-1 is retired. Its rates describe mid-2025 performance.
