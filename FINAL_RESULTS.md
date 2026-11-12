# Final Results Compilation: Quantum Reservoir Computing for Realized Volatility Forecasting

FIRE198 · Aidan, Colin, Vikas, Yashas · compiled 2026-11-12 from the team's Weekly Meeting Notes and the
GitHub repository (`main`, `Vikas`, `Colin` branches).

> This is the organized version of everything in the weekly notes. Where an early weekly entry was later
> shown to be wrong or superseded, the **Corrections** section says so explicitly. When in doubt, the
> later result wins.

---

## 1. Introduction

We set out to reproduce and then go beyond the paper *Quantum Reservoir Computing for Realized Volatility
Forecasting* (Li & Habibnia, arXiv:2505.13933, Physical Review Research 8, 023028, 2026). The paper claims that
a 10-qubit quantum reservoir (QR2) is the statistically best of 11 models for forecasting next month's US
stock-market volatility.

Our three questions:
1. **Can we reproduce it?** The authors' code was incomplete and needed private Julia packages and a GPU.
2. **Is the claim robust?** We checked it against fair baselines, for overfitting and for selection bias, and on new 2018–2026 data.
3. **Can it run on a real quantum computer, and what does that take?**

## 2. Background (one paragraph each)

- **Realized volatility (RV):** how much the S&P 500 moved within a month. We forecast its logarithm one
  month ahead. Data: monthly, 1950–2017 (816 months) plus macro inputs (dividend and earnings yield,
  T-bill rate, default spread, industrial production, inflation, Fama–French factors), all scaled to [−1, 0].
  Volatility spikes in crises: Oct 1987 (~99% annualized), Mar 2020 (~95%), Oct 2008 (~83%), Aug 2011, Apr 2025.
- **Reservoir computing:** a fixed, untrained system turns inputs into features. Only a linear (ridge)
  regression on those features is trained.
- **The quantum reservoir:** 7 input qubits plus 3 memory qubits. Each month's 7 inputs are loaded as
  rotations RY(π·x). The qubits evolve under a fixed random Ising Hamiltonian, the inputs are reset and
  the memory is kept, and this repeats for the last 3 months. Then ⟨Z⟩ of every qubit is measured: 10 features
  (QR1) or 20 features (QR2, read at two times = "virtual nodes").
- **Evaluation:** rolling 571-month training window, 245 one-month-ahead forecasts (Aug 1997 – Dec 2017),
  scored by MSE and QLIKE, compared with Diebold–Mariano tests and the Model Confidence Set.
- **Shots:** on hardware each ⟨Z⟩ is an average of repeated ±1 measurements. The error is about 1/√shots
  (±0.05 at 400 shots, ±0.001 at 1,000,000).

---

## 3. Results

### 3.1 Reproduction (Apr 2026): **done, exact**

| step | who | outcome |
|---|---|---|
| Picked the paper with professor's input | all | QRC volatility paper (Apr 4–6) |
| Fixed the classical pipeline: 19 bugs (missing `preprocess.py`, reservoirpy API changes, a scale mismatch that made QR look worst at MSE 6.88, wrong rolling windows, QLIKE MCS fed MSE losses, DM test without HAC, …) | Vikas | pipeline runs end to end |
| Rewrote the Julia quantum simulator in Python/Qiskit; fixed the coupling-matrix axis-order bug | Aidan, Vikas | matches the authors' forecasts to **1e-6** |
| Paper metrics reproduced | – | QR1 MSE 0.1051 / QLIKE 1.443; QR2 MSE 0.1038 / QLIKE 1.400 |
| Paper's Model Confidence Set reproduced | – | QR2 p = 1.0 under MSE and QLIKE, as in the paper |

### 3.2 Toward hardware: Trotter circuit (Apr 2026)

The exact evolution e<sup>−iHt</sup> was replaced by real gates (RXX + RZ). QR2 MSE: 0.1088 (1 step, 45 RXX),
0.1068 (2 steps), 0.1057 (4 steps, 180 RXX) vs 0.1038 exact. The reservoir tolerates a crude
approximation because it only needs to be *a* good feature map, not an exact simulation (Aidan, Vikas).

### 3.3 Is the paper's claim robust? (Sep–Nov 2026): **no quantum advantage**

Same 245 test months, same inputs, same ridge readout:

| model | MSE | vs QR2 (Diebold–Mariano p) |
|---|---|---|
| **Linear regression on QR2's own inputs** | **0.1016** | 0.66 (tie) |
| QR2 (paper) | 0.1038 | – |
| QR1 (paper) | 0.1051 | 0.78 |
| Echo-state network | 0.1077 | 0.53 |
| HAR benchmark | 0.1156 | 0.06 |
| QR2 with couplings off (no entanglement) | 0.1229 | 0.001 (worse) |
| AR(1) | 0.1302 | 0.002 |
| Random walk | 0.1428 | 0.003 |

- **Selection bias explains the paper's ranking.** QR2's coupling matrix is 2nd best of 20 random draws
  (a typical draw gives 0.111), memory depth K = 3 is the single best of K = 1–5, and on 40 random input
  sets QRC lost to linear regression on the same inputs every time (Aidan, D4).
- **Not overfitted in the usual sense:** the test/train error ratio is the same as linear (1.26 vs 1.29); the
  gap comes from market regime changes (2000–02, 2008–09) (Aidan, D1–D3).
- **True holdout, 2018–2026** (never used for any choice): linear 0.177, HAR 0.187, QR1 0.217, **QR2 0.281**.
  About a third of QR2's error is May 2020, when COVID pushed industrial production 3× outside its training
  range (Aidan, D6). Vikas's independent rebuilt-data study (2018–2026, 12 models, 5 seeds) agrees: no uniquely
  superior quantum model, and plain regression ≈ QR1 (0.138 vs 0.139).
- **What the circuit actually computes:** with RY encoding every quantum feature is a polynomial in cos(πx)
  and sin(πx). Simple cos/sin features explain 99.5% of each quantum feature, and a classical cos/sin model
  scores 0.1050, slightly better than the quantum circuit's 0.1077 (Aidan, Nov 5).
- **Computation verified correct:** 416/416 forecasts recomputed to 6e-10, 67/67 outputs identical on a
  clean rerun, 25 automated tests pass, shuffled-target controls lose in 798/800 runs (Colin, Vikas).

### 3.3b Locked protocol: settings chosen without seeing any test data (Nov 12)

`experiments/locked_protocol.py` → `results/locked/`. The inputs were fixed in advance: the 7 that exist through
2026. Coupling matrix, memory K, virtual nodes V and ridge λ were all chosen on 1987–1997 validation forecasts
inside the training period. The model was then scored once per test window. (Quick mode: 5 coupling matrices,
300 configurations. The 12-matrix run was stopped when the computer ran low on memory.)

| model | 1997–2017 MSE | 2018–2026 MSE |
|---|---|---|
| Locked QRC (best on validation: K = 5, V = 1, ridge λ = 100) | 0.118 | 0.183 |
| Locked linear, same inputs (lags 1–3, λ = 100) | 0.121 | 0.185 |
| HAR | 0.116 | 0.187 |
| AR(1) | 0.130 | 0.176 |
| Paper QR2 recipe (not locked) | 0.120 | 0.281 |

- **The paper's QR2 settings rank 297th of 300 on pre-1997 validation.** That's strong evidence they were tuned on the test window.
- **Chosen honestly, QRC is competitive in both windows.** It statistically ties linear, HAR and AR(1); only the random walk is clearly worse.
- **The 2018–2026 failure of the paper recipe (0.281) is mostly its unregularized readout.** With the
  validation-chosen ridge it drops to 0.183, in the middle of the classical pack.
- Clipping inputs to their 1950–1997 range (the May 2020 IP outlier) helps slightly on 2018–2026 (0.182).

### 3.3c Hardware-friendly couplings (E6)

Replacing the paper's couplings between all 45 qubit pairs with a ring (10) or a line (9) of couplings costs little.
Median MSE over 8 random draws is about 0.110–0.116, vs about 0.108 with all pairs coupled
(`results/research/e6_native_graphs.csv`). This matters for hardware that can only couple neighbouring qubits.

### 3.4 Does the quantum model work at all? **Yes, as a forecaster, but no better than classical**

It learns real signal. Out-of-sample R² is about 0.65 (it explains about 65% of month-to-month variation beyond
the long-run average). It beats HAR, AR(1), the random walk and its own no-entanglement version on
1997–2017, and its typical monthly error (~20%) is the same as every competitive model's. It does not
beat a linear model given the same inputs, and it generalizes worse to 2018–2026.

### 3.5 Hardware (Oct–Nov 2026): **ran on IonQ Forte-1, matches simulation**

| run | where | shots | signal kept s (predicted) | corr. with simulation | cost |
|---|---|---|---|---|---|
| Noise study, paper-faithful 24-qubit circuits | local sim with IonQ noise (cloud refused >19 qubits and resets) | 1k–1M | – | – | $0 (Vikas) |
| Paper-like circuit | IonQ cloud, Aria-1 noise model | 1,000 | 0.794 (0.79) | – | $0 |
| Final design | IonQ cloud, Forte-1 noise model | 400 | 0.943 (0.962) | 0.996 | $0 |
| **Final design** | **real IonQ Forte-1 QPU** | **400** | **0.958 (0.962)** | **0.996** | **$80.32** |

Final design (Aidan, Nov 5): 1 month of memory (10 qubits, no mid-circuit reset), 15 strongest couplings,
1 Trotter step, 3 months packed per 30-qubit circuit, noise-aware readout. 12 crisis months (Jul 2008 – Jun
2009) = 4 circuits. That's down from about $2,000 for the paper-faithful circuit. Twelve months is too few to rank
models. The run shows the hardware path works and behaves as predicted.

---

## 4. Shots: where "a million shots" came from, and why it isn't needed

- **Source:** Vikas's Week 9 IonQ noise study (`Vikas` branch, `results/6-ionq-noise/`), relayed to the
  professor in the notes. It concluded that "matching the paper takes about 100,000 to 1,000,000 shots per circuit".
  At 1,000 shots QR1 fell to HAR level (0.116).
- **Why it needed so many:** that study kept the paper's readout, a regression with essentially no
  regularization (λ = 10⁻⁸). Some reservoir features barely move from month to month: their standard deviation
  is only about 0.005–0.009 (Vikas's diagnostics, `results/3-diagnostics`). At 1,000 shots each feature carries
  about ±0.03 of random error, several times bigger than those features' real signal. An unregularized
  regression still leans hard on them (its condition number is 1,000–6,000), so the noise gets amplified into
  the forecast. To push shot noise below ~0.005 you need roughly (0.03 / 0.005)² × 1,000 ≈ 36,000 or more
  shots, hence 100,000–1,000,000.
- **Same author, different answer:** Vikas's own earlier noise study on 2018–2026 (`results/4-noise-extended`)
  found that 1,000 shots changed MSE by less than the random spread. There the forecast error is much larger
  (~0.17), so a small shot-noise penalty disappears in it. On the 1997–2017 window (~0.10) the same penalty
  becomes visible.
- **The fix (Nov 5):** standardize the features and use a sensible ridge penalty (λ = 1–10), which stops
  the regression from leaning on the near-constant, noise-dominated features. Also train it on simulated features
  that already contain the hardware's noise shrink and matching shot noise. Then 300–1,000 shots per circuit is
  enough. MSE was 0.115 at 200 shots and 0.112 at 1,000, vs 0.108 with infinite shots. This was confirmed on real
  hardware at 400 shots.
- **The general principle (the teacher's point):** shot error shrinks as 1/√N, so 100× more shots buys only
  10× more precision. Beyond a few hundred shots that precision is far smaller than the forecast's own
  error (~20% per month) and than hardware bias, which shots cannot remove. More shots never hurt, but
  past ~1,000 they stop being worth the money. A million is unnecessary once the readout is designed for
  noisy inputs.

---

## 5. Inconsistencies across the team's work, and how each is resolved

(The same table is in the Aidan tab of the weekly notes, 11/12 entry.)

| # | inconsistency | resolution |
|---|---|---|
| 1 | "1M shots needed" (Vikas Wk 9) vs "400 shots worked on real hardware" (Aidan 11/5) | Both correct for their setup. 1M is needed only with the paper's unregularized readout; with a regularized, noise-aware readout a few hundred shots suffice (section 4) |
| 2 | Vikas's `results/4-noise-extended` says 1,000 shots is fine; `results/6-ionq-noise` says 100k+ is needed | Different test windows. On 2018–2026 the error (~0.17) hides the shot-noise penalty; on 1997–2017 (~0.10) it shows |
| 3 | QR2 on 2018–2026: 0.174 (Vikas benchmark) vs 0.281 (Aidan D6) | Different setups. Vikas recovered MKT/STR to 2026 and averaged 5 seeds; D6 used the 7 inputs in `1950-2026.csv` (IP instead of MKT/STR) and the paper's single coupling draw, and was hit by the May 2020 IP outlier. Both find no quantum advantage |
| 4 | "QR2 is definitively the best model" (Vikas 4/12, Yashas Wk 5/6) | QR2 0.1038 vs ARMAX 0.1039 is a tie. MCS p = 1 means "last model left", and the paper never tested linear regression on QR2's own inputs (0.1016) |
| 5 | "Results are perfectly replicated" (Aidan 4/11, repeated in Vikas Wk 5) next to MSE 0.1155 vs 0.1051 | Pre-fix numbers. The final Qiskit code matches the authors to 1e-6 |
| 6 | Trotter: "1st order fine, within 1.8%" (Aidan 4/16) vs "2nd order required" (Vikas `results/5-hardware-circuits`) | Different pass/fail tests: overall MSE vs every forecast within 0.05 of the exact model. With hardware noise, 1 step is best (Vikas `results/4`) |
| 7 | "The quantum model needs entanglement" (Aidan 10/15) | With J = 0 the features are only cos(πx); the couplings add sin-like terms a classical cos/sin model reproduces (0.1050 vs 0.1077). Not a quantum advantage |
| 8 | Colin's report: QR1 0.185, QR2 0.173 vs our ~0.10 | Different window (Vikas's 2018–2026 study vs 1997–2017). Compare models only within one window |
| 9 | Paper-faithful circuits need 24 qubits, mid-circuit resets and 1.6M shots; IonQ's cloud refused them | Resolved by the trimmed 10-qubit, 400-shot, 3-months-per-circuit design ($80.32) |

Minor notes on the weekly log itself:
- Colin Wk 4 says RV runs "from −1 to 0 … that week" and that "0's were wild days". RV is monthly, and −1…0 is a min–max scaling: RV has exactly one 0, Oct 1987 (Black Monday), the most volatile month in the sample.
- In Vikas's section, Week 5 repeats Week 4, and Weeks 6, 7 and 8 contain identical text.

---

## 6. Conclusion

1. **Reproduction: succeeded exactly**, after fixing 19 bugs and rewriting the quantum simulator in Qiskit.
2. **The paper's headline claim does not hold up:** QR2 ties a linear model on its own inputs, its
   ranking relies on lucky settings chosen on the test window, and it is the worst fitted model on 2018–2026.
3. **The quantum model still works as a forecaster** (R² ≈ 0.65, beats HAR / AR / random walk on 1997–2017)
   and **its features are effectively a classical cos/sin feature map**.
4. **Real hardware is feasible and cheap:** IonQ Forte-1 reproduced the simulator (correlation 0.996) for
   $80 with 400 shots.

**Recommended next steps** (consolidated from all four logs):
- Lock all settings on a validation split inside the training window, then score once on 1997–2017 and once
  on 2018–2026.
- Report results over many coupling matrices (median and spread), not one draw.
- Always include linear-on-same-inputs and cos/sin-feature baselines.
- Clip or rescale inputs inside each training window (fixes the May 2020 failure).
- If continuing on hardware, use the noise-aware readout and a few hundred shots, and extend beyond 12 months.

## 7. Where everything lives

| what | where |
|---|---|
| Plain-English explanation | `HOW_IT_WORKS.md` (repo root) |
| Reproduction | `pipeline/`, `notebooks/`, `results/predictions`, `results/stats` |
| Robustness diagnostics D1–D8 | `experiments/`, `results/diagnostics/summary.md`, `summary_d7_d8.md`, `QRC_Overfitting_Audit.docx` |
| Quantum value, circuit/shot reduction, cost | `experiments/quantum_value_and_cost.py`, `results/research/quantum_value_and_cost.md` |
| Hardware run | `hardware/qpu_experiment.py`, `results/qpu/qpu.forte-1_m15_s1_pack3_shots400/` |
| Million-shot IonQ noise study | `Vikas` branch, `results/6-ionq-noise/report.md` |
| Rebuilt 2018–2026 dataset study | `Vikas` branch, `results/extended-study-2026-09-24/` |
| Annotated diagnostic report | `Colin` branch, `report_annotated.html` |
