"""
run_qrc_simulation.py — Quantum Reservoir Simulation Runner
============================================================

Direct Python/Qiskit translation of Time_serial_Finance_regression.ipynb
(cells 4–13).  Requires quantum_reservoir_qiskit.py in the same directory.

Outputs
-------
qrc_predict_result.csv
    245-row predictions for QR1 and QR2 in the same denormalized log-RV
    format as predict_result.csv (the original Julia output).

Verification
------------
After running, compare qrc_predict_result.csv against predict_result.csv.
Differences should be near machine-epsilon if the same coupling matrices
from coeff_10.jld2 are loaded successfully.  If h5py cannot parse the
JLD2 file, fresh random matrices are generated and results will differ.
"""

import os
import numpy as np
import pandas as pd

from quantum_reservoir_qiskit import (
    # coupling matrices
    load_coupling_matrices,
    generate_coupling_matrix,
    # Hamiltonian + unitaries
    build_ising_hamiltonian,
    compute_unitaries,
    # simulation
    quantum_reservoir,
    # readout
    rolling_ridge_regression,
    # metrics
    MSE, RMSE, MAE, MAPE, compute_qlike, hitrate,
    # constants needed for denormalization
    MIN_RV, DIF,
)

# ================================================================
# Paper parameters — exact match to Julia notebook cells 9–10
# ================================================================
NQUBIT = 10      # total qubits
TAU    = 1.0     # evolution time (energy unit)
K      = 3       # memory depth
TOTAL  = 816     # total monthly observations (Jan 1950 – Dec 2017)
L_OOS  = 245     # out-of-sample length     (Aug 1997 – Dec 2017)
WS     = 0       # rolling window start     (0 = full sample)
LAMBDA = 1e-8    # ridge regularisation     (paper: 0.00000001)

# Feature sets — exact match to Julia notebook cell 11
# features1 = ["RV","MKT","DP","IP","RV_q","STR","DEF"]
# features2 = ["RV","MKT","STR","RV_q","EP","INF","DEF"]
FEATURES_QR1 = ["RV", "MKT", "DP",  "IP",   "RV_q", "STR", "DEF"]
FEATURES_QR2 = ["RV", "MKT", "STR", "RV_q", "EP",   "INF", "DEF"]

# ================================================================
# Step 1: Load Data.CSV
# Matches:
#   data, header = readdlm("Data.CSV", ',', header=true)
#   Datas = identity.(DataFrame(data, vec(header)))
# ================================================================
os.makedirs("results/predictions", exist_ok=True)

print("=" * 60)
print("Step 1 — Loading Data.CSV")
print("=" * 60)

data = pd.read_csv("data/Data.CSV", header=0, index_col=0)
print(f"  Loaded {len(data)} rows, {len(data.columns)} columns.")

# Full RV series (all 816 months) — used by ridge regression
target_full = np.array(data["RV"], dtype=np.float64)

# Out-of-sample slice for metric computation
# Matches: y = Datas."RV"[Total-L+1:Total]  (Julia 1-indexed)
y = target_full[TOTAL - L_OOS : TOTAL]

# ================================================================
# Step 2: Load coupling matrices
# Matches:
#   file = JLD2.load("coeff_10.jld2")
#   ms   = file["ms"]
# ================================================================
print("\n" + "=" * 60)
print("Step 2 — Loading coupling matrices (coeff_10.jld2)")
print("=" * 60)

ms = load_coupling_matrices("data/coeff_10.jld2")

if ms is None:
    # Fallback: generate new random matrices.
    # NOTE: results will NOT match predict_result.csv in this case.
    print("  Generating fresh random coupling matrices (paper replication")
    print("  requires the original coeff_10.jld2 to be readable).")
    ms = np.stack([
        generate_coupling_matrix(NQUBIT, J_max=1.0, seed=i)
        for i in range(100)
    ])
else:
    print(f"  Loaded ms with shape {ms.shape}.")

# Julia notebook uses ms[1,:,:] for QR1 and ms[2,:,:] (1-indexed)
# → Python 0-indexed: ms[0] and ms[1]
J_qr1 = ms[0]
J_qr2 = ms[1]

# ================================================================
# Step 3: Build Hamiltonians and compute unitary operators
# Matches:
#   Qr1 = Qreservoir(nqubit, ms[1,:,:])
#   Qr2 = Qreservoir(nqubit, ms[2,:,:])
#
# The matrix exponential replaces the GPU exponentiation in Julia:
#   U  = CuArray(exp(-im * τ  * Matrix(matrix(QR))))
#   δU = CuArray(exp(-im * δτ * Matrix(matrix(QR))))
# ================================================================
print("\n" + "=" * 60)
print("Step 3 — Building Hamiltonians and evolution operators")
print("=" * 60)

print("  QR1: building H from ms[0] ...")
H_qr1       = build_ising_hamiltonian(NQUBIT, J_qr1)
U_qr1, dU_qr1 = compute_unitaries(H_qr1, TAU, virtual_node=1)
print(f"  QR1: H shape {H_qr1.shape}, U shape {U_qr1.shape}")

print("  QR2: building H from ms[1] ...")
H_qr2       = build_ising_hamiltonian(NQUBIT, J_qr2)
U_qr2, dU_qr2 = compute_unitaries(H_qr2, TAU, virtual_node=2)
print(f"  QR2: H shape {H_qr2.shape}, U shape {U_qr2.shape}")

# ================================================================
# Step 4: Run quantum reservoir simulations
# Matches:
#   signal1 = Quantum_Reservoir(Datas, features1, Qr1, B, K, 1, τ, nqubit)
#   signal2 = Quantum_Reservoir(Datas, features2, Qr2, B, K, 2, τ, nqubit)
#
# signal1 shape: (10,  816)   — QR1: 10 observables × 1 virtual node
# signal2 shape: (20,  816)   — QR2: 10 observables × 2 virtual nodes
# ================================================================
print("\n" + "=" * 60)
print("Step 4 — Running quantum reservoir simulations")
print("=" * 60)

print("  QR1 (virtual_node=1, 7 features) ...")
signal1 = quantum_reservoir(
    data, FEATURES_QR1, U_qr1, dU_qr1, K, virtual_node=1, nqubit=NQUBIT
)
print(f"  QR1 signal shape: {signal1.shape}")

print("\n  QR2 (virtual_node=2, 7 features) ...")
signal2 = quantum_reservoir(
    data, FEATURES_QR2, U_qr2, dU_qr2, K, virtual_node=2, nqubit=NQUBIT
)
print(f"  QR2 signal shape: {signal2.shape}")

# ================================================================
# Step 5: Rolling ridge regression readout
# Matches cells 12 and 13 in Time_serial_Finance_regression.ipynb:
#
#   for j in 1:L
#     y_train  = Datas."RV"[ws+j : ws+wi+j-1]
#     x_train  = signal[:, ws+j : ws+wi+j-1]
#     W_paras[j,:] = y_train' * x_train' *
#                    inv(x_train * x_train' + 1e-8 * I)
#     Pre[j]   = dot(W_paras[j,:], signal[:, ws+wi+j])
# ================================================================
print("\n" + "=" * 60)
print("Step 5 — Rolling ridge regression (245 windows × 571 training rows)")
print("=" * 60)

print("  QR1 readout ...")
Pre1 = rolling_ridge_regression(
    signal1, target_full, TOTAL, L_OOS, ws=WS, lambda_reg=LAMBDA
)

print("  QR2 readout ...")
Pre2 = rolling_ridge_regression(
    signal2, target_full, TOTAL, L_OOS, ws=WS, lambda_reg=LAMBDA
)

# ================================================================
# Step 6: Print metrics and compare against Julia reference values
# Matches the println() calls in cells 12–13 of the Julia notebook
# ================================================================
print("\n" + "=" * 60)
print("Step 6 — Metrics  (Julia reference values in parentheses)")
print("=" * 60)

print("\n  --- QR1 ---")
print(f"  Hit rate : {hitrate(Pre1, y):.4f}   (Julia: 0.4408)")
print(f"  MSE      : {MSE(Pre1,     y):.4f}   (Julia: 0.1051)")
print(f"  RMSE     : {RMSE(Pre1,    y):.4f}   (Julia: 0.3242)")
print(f"  MAE      : {MAE(Pre1,     y):.4f}   (Julia: 0.2488)")
print(f"  MAPE     : {MAPE(Pre1,    y):.4f}   (Julia: 8.2824)")
print(f"  QLIKE    : {compute_qlike(Pre1, y):.4f}   (Julia: 1.4428)")

print("\n  --- QR2 ---")
print(f"  Hit rate : {hitrate(Pre2, y):.4f}   (Julia: 0.4571)")
print(f"  MSE      : {MSE(Pre2,     y):.4f}   (Julia: 0.1038)")
print(f"  RMSE     : {RMSE(Pre2,    y):.4f}   (Julia: 0.3221)")
print(f"  MAE      : {MAE(Pre2,     y):.4f}   (Julia: 0.2426)")
print(f"  MAPE     : {MAPE(Pre2,    y):.4f}   (Julia: 8.1731)")
print(f"  QLIKE    : {compute_qlike(Pre2, y):.4f}   (Julia: 1.4004)")

# ================================================================
# Step 7: Save output and compare to predict_result.csv
# ================================================================
print("\n" + "=" * 60)
print("Step 7 — Saving predictions and verifying against predict_result.csv")
print("=" * 60)

# Denormalize to the same log-RV scale used in predict_result.csv:
#   denorm = (norm + 1) * DIF + MIN_RV
# Matches the Julia denormalization in Time_series.jl lines 120-126
Pre1_denorm = (Pre1 + 1) * DIF + MIN_RV
Pre2_denorm = (Pre2 + 1) * DIF + MIN_RV

# Save with the same column order as predict_result.csv (QR2 first, then QR1)
out_df = pd.DataFrame({"QR2": Pre2_denorm, "QR1": Pre1_denorm})
out_df.to_csv("results/predictions/qrc_predict_result.csv", index=False)
print("  Saved: results/predictions/qrc_predict_result.csv")

# Load original Julia output and compare
try:
    ref = pd.read_csv("data/predict_result.csv")
    diff_qr1 = np.abs(out_df["QR1"].values - ref["QR1"].values)
    diff_qr2 = np.abs(out_df["QR2"].values - ref["QR2"].values)

    print("\n  Absolute difference vs predict_result.csv (original Julia output):")
    print(f"  QR1 — max: {diff_qr1.max():.6f}   mean: {diff_qr1.mean():.6f}")
    print(f"  QR2 — max: {diff_qr2.max():.6f}   mean: {diff_qr2.mean():.6f}")
    print()
    if diff_qr1.max() < 1e-4 and diff_qr2.max() < 1e-4:
        print("  ✓ Replication successful — differences are within numerical tolerance.")
    else:
        print("  ! Differences are larger than 1e-4.")
        print("    Likely cause: coeff_10.jld2 was not readable so fresh coupling")
        print("    matrices were used.  Check the h5py output in Step 2.")
except FileNotFoundError:
    print("  predict_result.csv not found — skipping comparison.")
