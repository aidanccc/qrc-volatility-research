# Paper-faithful IonQ circuits

This folder turns the authors' QR1/QR2 models (arXiv:2505.13933, [original code](https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting)) into gate circuits that IonQ hardware can run. Nothing here submits jobs to IonQ or any other service.

## What is the authors' model, and what is not

Taken unchanged from `Time_series.jl` and `Time_serial_Finance_regression.ipynb`:

- couplings from `data/coeff_10.jld2`: `ms[1]` for QR1 and `ms[2]` for QR2 (Julia indexing)
- features: QR1 `RV, MKT, DP, IP, RV_q, STR, DEF`; QR2 `RV, MKT, STR, RV_q, EP, INF, DEF`
- H = Σ J_ij X_i X_j + Σ Z_i on 10 qubits: inputs on 0–6, memory on 7–9
- RY(π·x) encoding, three monthly inputs, τ = 1, Z readout on all qubits, virtual nodes 1 (QR1) and 2 (QR2)
- rolling ridge readout: 571-month window, λ = 1e-8, 245 forecasts

Our exact simulator reproduces the authors' published predictions to about 1.2e-6.

Changed for hardware, and nothing else:

1. **Evolution.** The authors used exact matrix exponentials. Here each evolution is a Trotter circuit of RXX and RZ gates.
2. **Memory hand-off.** The authors' partial trace becomes a reset of input qubits 0–6, followed by re-encoding the next month. `tests/test_paper_circuits.py` checks that the reset circuits give the same Z values as the partial-trace protocol, to 1e-10.
3. **Readout.** Measurement ends a run. So each QR2 virtual node is its own circuit: one evolution block for endpoint 1, two for endpoint 2. On hardware, Z values become finite-shot estimates.

## Trotter formulas

- **First order:** XX layer, then Z layer, repeated. This matches `quantum_reservoir_trotter.py`.
- **Second order (Strang):** half Z layer, XX layer, half Z layer. The half Z layers between steps merge into one, so second order uses **the same RXX gates** as first order. It adds only one extra RZ layer, and RZ is virtual (error-free) on IonQ. The error falls about 4× per doubling of steps, instead of 2×.

## How the configuration is chosen

`select_trotter.py` runs every combination of order {1, 2} and steps {1, 2, 3, 4, 6, 8} for both models, all noiseless. It compares each against the exact model over all 245 forecasts. The thresholds were fixed in code before the sweep ran:

- a candidate passes if its forecasts stay within **0.05 log-RV (max)** and **0.01 log-RV (mean)** of the exact model
- among passing candidates, take the one with the **fewest RXX gates per forecast month**; ties go to the smaller mean gap.

Forecast accuracy against realized volatility is reported but never used to choose, so the test period cannot influence the choice. The full table is in `trotter_sweep.csv`, so a different threshold can be applied without rerunning.

## Result (22 October 2026, `results/hardware-paper-2026-10-22`)

| Model | Chosen | Max / mean forecast gap vs exact | RXX per forecast month | Max depth | Circuits per month |
|---|---|---|---|---|---|
| QR1 | order 2, 2 steps | 0.035 / 0.0057 | 270 | 82 | 1 |
| QR2 | order 2, 3 steps | 0.039 / 0.0047 | 945 | 149 | 2 (135 + 2×405 RXX) |

No first-order candidate passed, even at 8 steps, which costs 4× (QR1) and 2.7× (QR2) more RXX gates than the chosen second-order configurations. For comparison, the paper's MSE is 0.1051 (QR1) and 0.1038 (QR2); the chosen circuits score 0.1052 and 0.1042 (reported only, not used to choose). Each exported circuit was simulated directly with resets and matched the selection features to 3e-15.

To get closer to the paper, at a higher gate cost: second order at 4 steps gives max gaps of 0.008 (QR1) and 0.022 (QR2).

## Run

```bash
.venv/bin/python -m hardware.select_trotter --output results/hardware-paper-YYYY-MM-DD
.venv/bin/python -m unittest tests.test_paper_circuits -v
```

The sweep takes about an hour on a 14-core laptop. Most of that time goes to the 26 full-series simulations and the direct check of the exported reset circuits. Outputs:

- `trotter_sweep.csv`: every candidate's fidelity gaps, gate counts, depth and reported-only accuracy
- `selection.json`: the rule, the chosen configuration, end-to-end checks and the job size
- `circuits/*.qasm`: OpenQASM 3 for the first out-of-sample month (August 1997) of each chosen configuration
- `manifest.json`: source and input hashes, versions

`hardware.paper_circuits.month_circuits(model, l, steps, order)` builds the circuits for any forecast column `l`.

## Not yet established

- **Reset support.** IonQ must support mid-circuit reset of qubits 0–6 on the target system; this has not been confirmed. Without reset, the protocol would need 24 qubits (7 per month plus 3 memory), which changes the hardware cost. That would need an explicit decision; nothing here does it silently.
- **Native gates.** No IonQ native-gate compilation, timing or backend query has been done; `qiskit-ionq` is not installed. Gate counts and depth are logical counts in the QIS basis {rxx, rz, ry, reset, measure}.
- **Noise and shots.** Hardware noise and shot noise come on top of the Trotter gap. Reservoir features at 1,000 shots have a standard error of up to about 0.03 per Z value, and the ridge readout is ill-conditioned (see the 1 October diagnostics). Measure the effect of shots and noise on forecasts with these circuits before paying for hardware time.
- **Training data.** The readout is trained on features from all earlier months. A full hardware reproduction needs circuits for all 813 feature months, not only the 245 forecast months. Training on simulated features and running only the forecast months on hardware is a methodology choice the team has to make explicitly. `selection.json` gives the job size for both options.
