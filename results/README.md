# Results index

Every file here is written by a script. Re-run the script to regenerate it.

| folder | written by | what's inside | start with |
|---|---|---|---|
| `predictions/` | `pipeline/run_qrc_simulation.py`, `qrc/quantum_reservoir_trotter.py` | QR1/QR2 forecast series (245 test months) | `qrc_predict_result.csv` |
| `plots/` | `notebooks/Reservoir_Learning.ipynb` | forecast-vs-actual plot per model (paper replication) | `QR2_model.png` |
| `stats/` | `notebooks/Reservoir_Learning.ipynb` | Diebold–Mariano statistics and p-values, 11 × 11 models | `p_value.csv` |
| `research/` | `experiments/run_research_suite.py` (E1–E8), `experiments/quantum_value_and_cost.py` (Q1–Q6) | fair baselines, coupling robustness, Trotter depth, shots, noise, memory, circuit/shot reduction, cost | `quantum_value_and_cost.md` |
| `diagnostics/` | `experiments/overfitting_diagnostics.py` (D1–D6), `experiments/confidence_bands.py` (D7–D8), `hardware/qpu_shot_budget.py` | overfitting checks, selection bias, 2018–26 holdout, repeated-trial bands, year-by-year error, shot budget | `summary.md`, `QRC_Overfitting_Audit.docx` |
| `locked/` | `experiments/locked_protocol.py` | settings chosen on 1987–1997 validation only, scored once on both test windows (quick mode, 5 coupling matrices) | `summary.md` |
| `qpu/` | `hardware/qpu_experiment.py` | one folder per hardware-path run (see below) | `qpu.forte-1_m15_s1_pack3_shots400/report.md` |

## `qpu/` run folders

Folder names are `<backend>_m<couplings kept>_s<Trotter steps>_pack<months per circuit>_shots<shots>`.

| folder | backend | what it is |
|---|---|---|
| `local_shots1000_steps2` | local Aer, ideal | first dry run of the full paper-like circuit |
| `ionq_simulator-aria-1_shots1000_steps2` | IonQ cloud simulator, Aria-1 noise model | measured the noise shrink s ≈ 0.80 |
| `local_m15_s1_pack3_shots400` | local Aer, ideal | rehearsal of the final trimmed, 3-months-per-circuit design |
| `ionq_simulator-forte-1_m15_s1_pack1_shots400` | IonQ cloud simulator, Forte-1 noise model | noisy rehearsal of the final design |
| `qpu.forte-1_m15_s1_pack3_shots400` | **real IonQ Forte-1 QPU** | the hardware run: 4 circuits, 400 shots, $80.32; includes the circuit diagram |

Each run folder has `report.md` (summary), `forecasts.csv` (12 monthly forecasts), `hardware_z.csv`
(measured ⟨Z⟩ features) and `job.json` (raw measurement counts and job ids).

## Which experiment answers which question

| question | file |
|---|---|
| Is QRC better than simple models on the same inputs? | `diagnostics/d5_classical_swap.csv`, `research/e1_baselines.csv` |
| Was the paper's coupling matrix a lucky draw? | `research/e2_coupling_robustness.png`, `diagnostics/d4_selection_bias.png` |
| Is it overfitted? | `diagnostics/d1_train_test_gap.png`, `d2_regularization_path.png`, `d3_learning_curve.png` |
| Does it hold up on new data (2018–2026)? | `diagnostics/d6_holdout.csv`, `d6_holdout_forecasts.png` |
| How big is run-to-run variation? | `diagnostics/d7_repeated_trials.png` |
| Which years does it win or lose? | `diagnostics/d8_yearly_error.png` |
| Does entanglement matter? | `research/quantum_value_and_cost.md` (J = 0 rows) |
| How many shots are needed? | `research/q4_shot_reduction.csv`, `diagnostics/qpu_shot_budget.png`, `research/e4_shots.png` |
| How small can the circuit be? | `research/q3_circuit_reduction.csv`, `research/e3_trotter.png` |
| What does hardware cost? | `research/q6_cost.csv` |
| Does real hardware match the simulator? | `qpu/qpu.forte-1_m15_s1_pack3_shots400/qpu_vs_simulation.png` |
