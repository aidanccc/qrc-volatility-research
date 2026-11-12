# Quantum Reservoir Computing for Realized Volatility Forecasting

Replication and stress-test of **[arXiv:2505.13933] Quantum Reservoir Computing for Realized Volatility Forecasting**
(*Physical Review Research 8, 023028, 2026*). Original repo:
[LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting](https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting).

> **New here? Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) first.** It explains the task, what the quantum
> reservoir actually does, what shots are (and why a million is pointless), and what we found, all in
> plain English. For everyone's results in one place (all branches and the weekly notes, including
> where they disagree and why), see [FINAL_RESULTS.md](FINAL_RESULTS.md).

---

## What this project does

1. **Replicates the paper.** A 10-qubit quantum reservoir (QR1/QR2) forecasts monthly US stock-market
   volatility. Our Python/Qiskit version reproduces the authors' Julia/GPU forecasts to 1e-6, and in the
   paper's own test window QR2 ranks first.
2. **Stress-tests the claim.** We compare against fair baselines on identical inputs, check for selection
   bias and overfitting, and test on brand-new 2018–2026 data. Result: QRC ties a plain linear regression
   on the same inputs (not significantly different) and does worse than simple models on the 2018–2026
   holdout.
3. **Runs it on a real quantum computer.** We trimmed the circuit and ran 12 forecast months (2008–09
   crisis) on **IonQ Forte-1**: 4 circuits × 30 qubits × 400 shots for $80.32. The hardware matched the
   simulator (correlation 0.996).

---

## Repository layout

```
.
├── HOW_IT_WORKS.md        ← plain-English guide to the whole project
├── data/
│   ├── Data.CSV            monthly inputs 1950–2017 (816 rows, normalised)
│   ├── 1950-2026.csv       extended series for the 2018–2026 true holdout
│   ├── coeff_10.jld2       100 random Ising coupling matrices (from the authors)
│   └── predict_result.csv  authors' original QR1/QR2 forecasts (reference)
│
├── qrc/                    the quantum reservoir (library imported by everything else)
│   ├── quantum_reservoir_qiskit.py   exact simulator, line-by-line translation of the Julia code
│   ├── fast_qrc.py                   same physics, ~100× faster; adds shots / readout error / noise
│   ├── quantum_reservoir_trotter.py  the evolution as a real gate circuit (RXX + RZ)
│   └── quantum_reservoir_noisy.py    the circuit with depolarizing gate noise
│
├── pipeline/               reproduce the paper
│   ├── preprocess.py                 step 1: build data/Data_raw.csv and data/dff.csv
│   └── run_qrc_simulation.py         step 2: QR1/QR2 forecasts, checked against the authors' output
│
├── notebooks/              baselines and the paper's comparison
│   ├── LSTM.ipynb                    LSTM / LSTMX
│   ├── classical_reservoir.ipynb     echo-state networks CRL / CRLX
│   ├── Reservoir_Learning.ipynb      HAR, HARX, AR, ARMAX + MCS and Diebold–Mariano tests
│   └── Quantum_Reservoir_Metrics.ipynb  classical vs quantum metric table
│
├── experiments/            our research on top of the replication
│   ├── run_research_suite.py         E1–E8: fair baselines, coupling draws, Trotter, shots, noise, memory
│   ├── overfitting_diagnostics.py    D1–D6: overfitting, selection bias, classical swap, 2018–26 holdout
│   ├── confidence_bands.py           D7–D8: repeated trials with bands, year-by-year error
│   ├── quantum_value_and_cost.py     does the circuit add value? minimal circuit, shots, cost
│   └── locked_protocol.py            all settings chosen on pre-1997 validation, scored once
│
├── hardware/               real quantum-computer work
│   ├── qpu_shot_budget.py            how many shots a hardware run needs
│   ├── ionq_feasibility.py           gate counts, noisy simulation, cost on IonQ
│   └── qpu_experiment.py             the IonQ run (local → IonQ simulator → Forte-1 QPU)
│
├── julia/                  authors' original Julia code (reference only)
└── results/                every table, plot and report (index: results/README.md)
```

---

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Running

**Run everything from the repository root.** Scripts are run as modules (`python -m folder.script`) so
they can import `qrc/`. The notebooks have a setup cell that moves to the repo root automatically.

```bash
# Reproduce the paper
python -m pipeline.preprocess
python -m pipeline.run_qrc_simulation          # writes results/predictions/qrc_predict_result.csv
# then run notebooks/LSTM.ipynb, notebooks/classical_reservoir.ipynb, notebooks/Reservoir_Learning.ipynb

# Research experiments (CPU, minutes each)
python -m experiments.run_research_suite       # --quick for a fast pass
python -m experiments.overfitting_diagnostics  # --quick
python -m experiments.confidence_bands
python -m experiments.quantum_value_and_cost
python -m experiments.locked_protocol        # --quick for 5 coupling matrices

# Hardware planning
python -m hardware.qpu_shot_budget
python -m hardware.qpu_experiment --backend local          # free, offline ideal simulator
# --backend ionq_simulator (free cloud simulator) / qpu.forte-1 (paid; needs IONQ_API_KEY,
# --dry-run-qpu gives a free quote, a real submission also needs --confirm-qpu)
```

---

## Key results

**Paper test window, Aug 1997 – Dec 2017 (245 months).** Same inputs, same rolling 571-month ridge readout.

| model | MSE | vs QR2 (DM p) |
|---|---|---|
| Linear regression, QR2's inputs, lags 1–3 | **0.1016** | 0.66 (no significant difference) |
| QR2 (paper) | 0.1038 | – |
| QR1 (paper) | 0.1051 | 0.78 |
| Echo-state network | 0.1077 | 0.53 |
| HAR | 0.1156 | 0.06 |
| QR2 with couplings off (J = 0) | 0.1229 | 0.001 (significantly worse) |
| Random walk | 0.1428 | 0.003 |

**True holdout, Jan 2018 – Aug 2026 (never used for any choice):** linear 0.177, HAR 0.187, QR1 0.217,
QR2 0.281. QRC is the worst of the fitted models.

**Why the paper looked better:** its coupling draw ranks 2/20, its memory depth K = 3 is the single best,
and its feature set was picked on the test window. On 40 random feature sets QRC beat linear regression on
the same inputs 0% of the time.

**Locked protocol (settings chosen on 1987–1997 validation only):** QRC 0.118 / 0.183 vs linear 0.121 / 0.185,
HAR 0.116 / 0.187 (1997–2017 / 2018–2026), all statistical ties. The paper's settings rank 297/300 on validation.

**Hardware (IonQ Forte-1, 12 months, 400 shots):** hardware ⟨Z⟩ vs simulator correlation 0.996, noise
shrink s = 0.958. 12 months is enough to show the hardware path works, but too few to rank models.

**Shots:** with a noise-aware readout, the gain from more shots is essentially gone by 300–1,000 shots per
circuit (see [HOW_IT_WORKS.md §4](HOW_IT_WORKS.md#4-shots-and-why-a-million-shots-is-pointless)).

Full write-ups: `results/diagnostics/summary.md`, `results/diagnostics/summary_d7_d8.md`,
`results/research/quantum_value_and_cost.md`, `results/diagnostics/QRC_Overfitting_Audit.docx`.

### Paper replication tables (original pipeline)

Qiskit QR1/QR2 vs the authors' Julia output: max absolute difference 1e-6 over 245 forecasts (float32
storage of the couplings). Metrics: QR1 MSE 0.1051 / QLIKE 1.443, QR2 MSE 0.1038 / QLIKE 1.400.
In the paper's 11-model Model Confidence Set, QR2 has p = 1.0 under both MSE and QLIKE, as the paper
reports. Note that the paper's comparison does not include a linear model on QR2's own inputs, which is the
baseline that ties it.

---

## Julia code (reference only)

`julia/Time_series.jl` and `julia/Time_serial_Finance_regression.ipynb` are the authors' original
implementation. They depend on two unpublished packages (`QuantumCircuits_demo`, `VQC_demo_cuda`) and an
NVIDIA GPU, so they cannot run here. `qrc/quantum_reservoir_qiskit.py` translates them function by function.
