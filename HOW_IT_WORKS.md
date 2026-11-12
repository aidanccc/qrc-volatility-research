# How this project works (plain-English guide)

This page explains the whole project from scratch: what is being predicted, what the
"quantum reservoir" actually does, how the forecasts are made, what the diagnostics found,
what the real quantum-hardware run was, and why we do **not** need a million shots.

If you only read one section, read **[7. The short version of the results](#7-the-short-version-of-the-results)**.

---

## 1. The task: forecast next month's stock-market volatility

- **Realized volatility (RV)** = how much the US stock market bounced around in a month,
  computed from daily returns. We work with its logarithm and normalise it.
- **Goal:** using information available at the end of month *t*, predict RV for month *t+1*.
- **Data:** `data/Data.CSV`, monthly, Jan 1950 – Dec 2017 (816 months). Columns are RV plus
  macro-financial predictors (market return MKT, short-term reversal STR, dividend-price
  ratio DP, inflation INF, default spread DEF, industrial production IP, earnings-price EP,
  quarterly RV `RV_q`, …). All inputs are scaled to the range [-1, 0].
- **Extra data:** `data/1950-2026.csv` extends the series to Aug 2026. We use 2018–2026 as a
  **true holdout**: data that no model choice was ever tuned on.

**How a forecast is scored.** We compare predictions with what actually happened over the
**test window** Aug 1997 – Dec 2017 (245 months), using:
- **MSE**: mean squared error. Lower is better, and it's the main number everywhere.
- **QLIKE**: a loss used for volatility that punishes under-predicting risk.
- **Diebold–Mariano (DM) test / Model Confidence Set (MCS)**: statistical tests that ask
  whether one model's errors are *significantly* smaller than another's, or just luckier.

**Rolling window.** Every model is re-fitted for each test month using only the previous
571 months, then makes one forecast. No model ever sees the future.

---

## 2. What is "reservoir computing"?

A reservoir computer has two parts:

1. **The reservoir**: a fixed, *untrained*, messy dynamical system. You feed your inputs in,
   it mixes them up non-linearly, and you read out a list of numbers ("features").
2. **The readout**: a plain **linear regression** (ridge regression) from those features to
   the target. *This is the only part that is trained.*

So the reservoir is really just a **fixed feature generator**. The classical version (an
"echo-state network") uses a random recurrent neural network as the reservoir. The quantum
version uses a small quantum system.

---

## 3. What the quantum reservoir does, step by step

Code: `qrc/quantum_reservoir_qiskit.py` (exact simulator), `qrc/fast_qrc.py` (same maths, ~100× faster).

The reservoir is **10 qubits**: 7 *input* qubits (one per input feature) and 3 *memory* qubits.

For **each month we want to forecast**:

1. **Start fresh.** Put the 3 memory qubits in the state |000⟩.
2. **Feed in the last K = 3 months, oldest first.** For each of those months:
   - **Encode:** rotate input qubit *i* by an angle π × (feature *i*) using an `RY` gate.
     So the data is literally stored as rotation angles.
   - **Mix:** let all 10 qubits evolve under a fixed **Ising Hamiltonian**
     H = Σ J<sub>ij</sub> X<sub>i</sub>X<sub>j</sub> + Σ Z<sub>i</sub>.
     The couplings J<sub>ij</sub> are random numbers drawn once (`data/coeff_10.jld2`) and never trained.
     This step entangles the qubits and mixes the inputs together.
   - **Forget the inputs, keep the memory:** discard (reset) the 7 input qubits and keep
     the 3 memory qubits, which now carry a trace of that month.
3. **Read out.** After the newest month, measure ⟨Z⟩ (the average spin, a number between
   −1 and +1) on all 10 qubits. That gives **10 features**.
   - **QR1** reads once → 10 features.
   - **QR2** uses 2 "virtual nodes" (reads at two moments of the last evolution) → 20 features.
4. **Predict.** Ridge regression maps those 10 or 20 numbers to next month's RV.

On a laptop all of this is simulated **exactly** (10 qubits = a 1024×1024 matrix). On a real
quantum computer, step 2's "mix" becomes a circuit of `RXX` and `RZ` gates
(`qrc/quantum_reservoir_trotter.py`), and step 3 becomes *measuring the qubits many times*.
That's where shots come in.

---

## 4. Shots, and why a million shots is pointless

**What a shot is.** When you measure a qubit you don't get ⟨Z⟩. You get a single random
answer: +1 or −1. ⟨Z⟩ is the *average* of that answer. A real quantum computer has to run the same
circuit many times (each run = one **shot**) and average the results. It's like estimating
how biased a coin is by flipping it many times.

**How accurate is the average?** Standard probability: the error of an average of N
coin flips is about **1/√N**:

| shots per circuit | typical error in each ⟨Z⟩ |
|---|---|
| 100 | ± 0.10 |
| 400 (what we used on hardware) | ± 0.05 |
| 1,000 | ± 0.03 |
| 10,000 | ± 0.01 |
| 1,000,000 | ± 0.001 |

So the teacher is right in practice. **More shots only give a more precise estimate of a probability,
and the gains shrink fast.** You need 4× the shots to halve the error and 100× the
shots to get 10× more precision. Beyond a few hundred to ~1,000 shots that extra
precision no longer matters, for three reasons:

1. **The forecast itself is far noisier.** Even a perfect model misses next month's
   volatility by ~38% on a typical month (MSE ≈ 0.10). Shaving ±0.03 off a feature to
   ±0.001 doesn't change that.
2. **Hardware noise is a bias, not randomness.** On real hardware the ⟨Z⟩ values come out
   *shrunk* by a factor s (we measured s ≈ 0.80 on IonQ Aria-1 and s ≈ 0.96 on Forte-1).
   Averaging more shots gives you a very precise estimate of the **wrong (shrunk)** number.
   Shots can't fix that. It's handled by making the readout "noise-aware".
3. **Our own results show the plateau.** With the noise-aware readout
   (`results/research/quantum_value_and_cost.md`, "Shot reduction"), test MSE goes
   0.138 → 0.127 → 0.115 → 0.113 → 0.112 for 50 → 100 → 200 → 500 → 1,000 shots, against
   0.108 with *infinite* shots. Almost all of the benefit is gone by ~300–500 shots.

**One nuance, to be precise.** It isn't literally true that more shots *never* help. Each extra shot does
reduce random noise a little; the returns just diminish very quickly. An early experiment
(`results/research/e4_shots.csv`, `experiments/run_research_suite.py`, E4) did keep
improving up to 100,000 shots per circuit. That happened because it used the paper's readout, which is almost
un-regularised (ridge λ = 10⁻⁸) and over-reacts to tiny noise in the features. Some reservoir features
only vary by about ±0.005 from month to month, which is smaller than the ±0.03 shot noise at 1,000 shots, and an
un-regularised regression still leans heavily on them. Vikas's IonQ noise study (`Vikas` branch,
`results/6-ionq-noise`) hit the same effect and concluded that 100,000–1,000,000 shots were needed. That's where
the "million shots" figure comes from. The right
fix was a better readout (noise-aware ridge, λ = 1–10), not more shots. The "millions"
printed in that experiment's log were **total shots summed over all ~813 circuits of a full
backtest**, not shots per circuit.

**Bottom line:** 300–1,000 shots per circuit is plenty. We used 400 on real hardware.

---

## 5. What every folder is for

```
data/          inputs: Data.CSV (1950–2017), 1950-2026.csv (holdout), coupling matrices, paper's predictions
qrc/           the quantum reservoir itself (the simulator library everything else imports)
pipeline/      reproduce the paper: preprocess data, run QR1/QR2 simulation
notebooks/     baselines + the paper's main comparison (LSTM, echo-state network, HAR/AR/ARMAX, MCS/DM tests)
experiments/   our own research: is QRC really better, is it overfitted, does the quantum part matter?
hardware/      planning and running the real IonQ quantum-computer experiment
julia/         the original authors' Julia code (reference only; needs their private packages + GPU)
results/       every output table, plot and report (see results/README.md for an index)
```

### The order to read/run things

| step | file | what it answers |
|---|---|---|
| 1 | `pipeline/preprocess.py` | Prepares `data/Data_raw.csv` and `data/dff.csv` for the notebooks |
| 2 | `pipeline/run_qrc_simulation.py` | Re-creates the paper's QR1/QR2 forecasts in Python; matches the authors' Julia output to 1e-6 |
| 3 | `notebooks/LSTM.ipynb`, `notebooks/classical_reservoir.ipynb` | Neural-network and echo-state-network baselines |
| 4 | `notebooks/Reservoir_Learning.ipynb` | Linear baselines + the paper's MCS/DM comparison (the "QR2 is best" result) |
| 5 | `notebooks/Quantum_Reservoir_Metrics.ipynb` | Side-by-side metric table, classical vs quantum |
| 6 | `experiments/run_research_suite.py` | E1–E8: fair baselines, all 100 coupling draws, Trotter depth, shots, noise, memory |
| 7 | `experiments/overfitting_diagnostics.py` | D1–D6: train/test gap, regularisation, learning curve, selection bias, classical swap, 2018–26 holdout |
| 8 | `experiments/confidence_bands.py` | D7–D8: repeated trials with error bands, year-by-year error |
| 8b | `experiments/locked_protocol.py` | All settings chosen on 1987–1997 validation only, then scored once on 1997–2017 and 2018–2026 |
| 9 | `experiments/quantum_value_and_cost.py` | Does the quantum circuit add anything? Smallest usable circuit, fewest shots, cost |
| 10 | `hardware/qpu_shot_budget.py`, `hardware/ionq_feasibility.py` | How many shots/gates a real run needs, and what it costs |
| 11 | `hardware/qpu_experiment.py` | The actual IonQ run (local simulator → IonQ cloud simulator → real Forte-1) |

The other two files in `qrc/` are building blocks: `quantum_reservoir_trotter.py` turns the
evolution into real gates, and `quantum_reservoir_noisy.py` adds gate noise.

---

## 6. The real quantum-hardware run

`hardware/qpu_experiment.py` → `results/qpu/qpu.forte-1_m15_s1_pack3_shots400/`

- **Machine:** IonQ Forte-1 (trapped-ion quantum computer), via the IonQ cloud.
- **Months forecast:** Jul 2008 – Jun 2009 (the Lehman / financial-crisis period), 12 months.
- **Circuit:** the 10-qubit reservoir with memory K = 1 and only the 15 strongest couplings,
  one Trotter step (45 `RXX` gates for 3 packed months). Three months run side by side on 30
  qubits, so 12 months = **4 circuits × 400 shots**, costing **$80.32**.
- **Training stays on the simulator.** Only the 12 test months are measured on hardware. The
  readout is trained on simulated features, adjusted for the measured noise shrink.
- **Result:** hardware ⟨Z⟩ matched the simulator with correlation **0.996** (shrink s = 0.958),
  and the leftover differences were about the size expected from 400 shots alone. So **the
  hardware path works.** With only 12 months, it cannot tell us which model forecasts better.
- Circuit picture: `results/qpu/qpu.forte-1_m15_s1_pack3_shots400/circuit_diagram_2008-07.png`.

---

## 7. The short version of the results

1. **Replication works.** Our Python version reproduces the paper's quantum forecasts exactly,
   and in the paper's own test the QR2 model comes out on top (MSE 0.104 vs HAR 0.116).
2. **But a plain linear regression on the same 7 inputs does as well or better** (MSE 0.102).
   The difference from QR2 is not statistically significant (DM p = 0.66).
3. **The paper's choices were lucky picks.** The coupling matrix QR2 uses ranks 2nd of 20
   random draws, K = 3 is the single best memory depth, and the feature set was chosen on the
   test window. On 40 random feature sets QRC beat linear regression on the same inputs **0%** of
   the time.
4. **On truly new data (2018–2026) QRC does worse than simple models**: QR2 MSE 0.28 vs
   linear 0.18 and HAR 0.19.
5. **The entanglement does do something.** Switching couplings off (J = 0) makes it clearly
   worse. But the quantum features are ~99% explained by simple cos/sin functions of the
   inputs, so the circuit is a fixed non-linear feature map that a laptop can compute.
6. **With honestly chosen settings it is competitive but not better.** When every setting is picked on
   1987–1997 validation data only (`experiments/locked_protocol.py`), QRC ties linear regression, HAR and AR(1)
   on both 1997–2017 and 2018–2026. The paper's own settings rank 297th of 300 on that validation.
7. **Hardware is feasible and cheap** once the circuit is trimmed and shots are kept at a few
   hundred. The Forte-1 run matched the simulator.

Details: `results/diagnostics/summary.md`, `results/diagnostics/summary_d7_d8.md`,
`results/research/quantum_value_and_cost.md`, `results/diagnostics/QRC_Overfitting_Audit.docx`.

---

## 8. Glossary

| term | meaning |
|---|---|
| qubit | quantum bit; here each one holds one input feature as a rotation angle |
| ⟨Z⟩ | the average measurement result of a qubit, between −1 and +1; these are the reservoir's features |
| shot | one run + measurement of a circuit; ⟨Z⟩ is estimated by averaging many shots |
| Ising Hamiltonian | the fixed "mixing rule" that couples qubits through random strengths J<sub>ij</sub> |
| Trotter step | approximating the smooth evolution e<sup>−iHt</sup> with a short sequence of gates |
| K (memory depth) | how many past months are fed in before reading out |
| virtual node | reading out at extra moments in time to get more features (QR2 has 2) |
| ridge regression | linear regression with a penalty λ that stops it over-reacting to noise |
| noise shrink s | real hardware returns ⟨Z⟩ multiplied by roughly s < 1 because of gate errors |
| HAR | standard volatility benchmark: linear in last month, last quarter and last year's RV |
| holdout | data kept completely untouched until the final test |
