"""
quantum_reservoir_noisy.py — Noisy Quantum Reservoir Simulation
================================================================

Extends quantum_reservoir_trotter.py by adding depolarizing noise
after each evolution step, simulating the effect of real hardware
gate errors and decoherence on forecasting performance.

NOISE MODEL
-----------
After each evolution step (U and dU), independent single-qubit
depolarizing noise is applied to every qubit:

    ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

This is implemented via Kraus operators using Qiskit's DensityMatrix.evolve().
No shot sampling is needed — expectation values are computed exactly from
the noisy density matrix, making this a clean noise channel simulation.

NOISE LEVEL REFERENCE POINTS
-----------------------------
    p=0.0    — noiseless (should match quantum_reservoir_trotter results)
    p=0.001  — optimistic near-term hardware
    p=0.005  — near IBM Eagle single-qubit gate error budget
    p=0.01   — near IBM Eagle two-qubit gate error budget
    p=0.05   — heavy noise / pessimistic hardware

RESEARCH QUESTION
-----------------
At what noise level does QR2 lose its advantage over HARX/ARMAX
in the MCS test? If current hardware sits below that threshold,
the circuit is viable on real devices today.
"""

import os
import numpy as np
import pandas as pd

from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace

from qrc.quantum_reservoir_qiskit import (
    MAX_RV, MIN_RV, COE, DIF,
    load_coupling_matrices,
    generate_coupling_matrix,
    build_z_observables,
    encode_input,
    rolling_ridge_regression,
    MSE, RMSE, MAE, MAPE, compute_qlike, hitrate,
)

from qrc.quantum_reservoir_trotter import build_trotter_evolution


# ============================================================
# Depolarizing noise channel
# ============================================================

def _depolarizing_kraus(p: float) -> Kraus:
    """
    Build single-qubit depolarizing Kraus channel with error rate p.

    Kraus operators:
        K0 = sqrt(1-p) * I     (no error)
        K1 = sqrt(p/3)  * X    (bit flip)
        K2 = sqrt(p/3)  * Y    (bit+phase flip)
        K3 = sqrt(p/3)  * Z    (phase flip)

    p=0 → identity channel, p=1 → maximally mixed state.
    """
    return Kraus([
        np.sqrt(1 - p) * np.eye(2, dtype=complex),
        np.sqrt(p / 3) * np.array([[0,  1 ], [1,  0 ]], dtype=complex),
        np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0 ]], dtype=complex),
        np.sqrt(p / 3) * np.array([[1,  0 ], [0, -1 ]], dtype=complex),
    ])


def apply_depolarizing_noise(
    rho    : DensityMatrix,
    nqubit : int,
    p      : float,
) -> DensityMatrix:
    """
    Apply independent single-qubit depolarizing noise to every qubit.

    Called after each evolution step to model the accumulated effect
    of gate errors and decoherence across one Trotter step.

    Parameters
    ----------
    rho    : DensityMatrix — current quantum state
    nqubit : int           — number of qubits in rho
    p      : float         — per-qubit depolarizing error rate (0 to 1)

    Returns
    -------
    DensityMatrix — state after noise is applied
    """
    if p == 0.0:
        return rho
    channel = _depolarizing_kraus(p)
    for i in range(nqubit):
        rho = rho.evolve(channel, qargs=[i])
    return rho


# ============================================================
# Noisy quantum reservoir simulation
# ============================================================

def quantum_reservoir_noisy(
    data         : pd.DataFrame,
    features     : list,
    J            : np.ndarray,
    tau          : float,
    K_delay      : int,
    virtual_node : int,
    nqubit       : int,
    n_trotter    : int   = 4,
    p_noise      : float = 0.0,
) -> np.ndarray:
    """
    Trotterized quantum reservoir with depolarizing noise after each step.

    Drop-in replacement for quantum_reservoir_trotter() with one addition:
    after every call to rho.evolve(U_op) or rho.evolve(dU_op), depolarizing
    noise with rate p_noise is applied to all nqubit qubits.

    p_noise=0.0 exactly reproduces quantum_reservoir_trotter() results.

    Parameters
    ----------
    data         : pd.DataFrame — full dataset (816 rows)
    features     : list of str  — input feature column names
    J            : np.ndarray   — (nqubit x nqubit) Ising coupling matrix
    tau          : float        — total evolution time (1.0)
    K_delay      : int          — memory depth (3)
    virtual_node : int          — virtual nodes (1=QR1, 2=QR2)
    nqubit       : int          — total qubits (10)
    n_trotter    : int          — Trotter steps (default 4)
    p_noise      : float        — per-qubit depolarizing rate per step

    Returns
    -------
    np.ndarray — shape (nqubit * virtual_node, n_samples)
    """
    n_input   = len(features)
    n_hidden  = nqubit - n_input
    N         = nqubit
    n_samples = len(data)

    # Build circuits and precompute Operators once
    U_circ, dU_circ = build_trotter_evolution(nqubit, J, tau, n_trotter, virtual_node)
    U_op  = Operator(U_circ)
    dU_op = Operator(dU_circ)

    observables = build_z_observables(nqubit)
    input_qargs = list(range(n_input))

    output = np.zeros((N * virtual_node, n_samples))

    for l in range(K_delay, n_samples):
        if l % 200 == 0:
            print(f"    time step {l} / {n_samples}")

        rho_hidden = DensityMatrix.from_label('0' * n_hidden)

        for k in range(K_delay, 0, -1):
            input_vals = np.array([data.iloc[l - k][feat] for feat in features])
            rho_input  = encode_input(input_vals, n_input)
            rho_joint  = rho_hidden.tensor(rho_input)

            if k != 1:
                # Intermediate step: evolve, apply noise, partial trace
                rho_joint  = rho_joint.evolve(U_op)
                rho_joint  = apply_depolarizing_noise(rho_joint, nqubit, p_noise)
                rho_hidden = partial_trace(rho_joint, input_qargs)
            else:
                # Final step: evolve dU per virtual node, apply noise, measure
                it = 0
                for _ in range(virtual_node):
                    rho_joint = rho_joint.evolve(dU_op)
                    rho_joint = apply_depolarizing_noise(rho_joint, nqubit, p_noise)
                    for obs in observables:
                        output[it, l] = rho_joint.expectation_value(obs).real
                        it += 1

    return output


# ============================================================
# Noise sweep runner
# ============================================================

def run_noise_sweep(
    data_path    : str   = "data/Data.CSV",
    jld2_path    : str   = "data/coeff_10.jld2",
    n_trotter    : int   = 4,
    noise_levels : list  = None,
    save_csv     : bool  = True,
) -> dict:
    """
    Sweep over noise levels for QR2 and record forecasting metrics.

    Uses QR2 (virtual_node=2) — the better-performing and more
    Trotter-robust model from the sweep results.

    Parameters
    ----------
    n_trotter    : int  — Trotter steps to use (4 recommended)
    noise_levels : list — p_noise values to sweep over
    save_csv     : bool — save results to results/predictions/noise_sweep_qr2.csv

    Returns
    -------
    dict — {p_noise: metrics_dict} for each noise level
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05]

    NQUBIT = 10
    TAU    = 1.0
    K      = 3
    TOTAL  = 816
    L_OOS  = 245
    WS     = 0
    LAMBDA = 1e-8

    FEATURES_QR2 = ["RV", "MKT", "STR", "RV_q", "EP", "INF", "DEF"]

    os.makedirs("results/predictions", exist_ok=True)

    print("Loading data ...")
    data        = pd.read_csv(data_path, header=0, index_col=0)
    target_full = np.array(data["RV"], dtype=np.float64)
    y           = target_full[TOTAL - L_OOS : TOTAL]

    print("Loading coupling matrices ...")
    ms = load_coupling_matrices(jld2_path)
    if ms is None:
        ms = np.stack([generate_coupling_matrix(NQUBIT, seed=i) for i in range(100)])
    J_qr2 = ms[1]

    results = {}

    for p in noise_levels:
        print(f"\n{'='*60}")
        print(f"QR2  p_noise={p}  n_trotter={n_trotter}")
        print(f"{'='*60}")

        signal = quantum_reservoir_noisy(
            data, FEATURES_QR2, J_qr2, TAU, K, 2, NQUBIT,
            n_trotter=n_trotter, p_noise=p,
        )
        Pre = rolling_ridge_regression(signal, target_full, TOTAL, L_OOS, WS, LAMBDA)

        metrics = {
            'p_noise'  : p,
            'hit_rate' : hitrate(Pre, y),
            'mse'      : MSE(Pre, y),
            'rmse'     : RMSE(Pre, y),
            'mae'      : MAE(Pre, y),
            'mape'     : MAPE(Pre, y),
            'qlike'    : compute_qlike(Pre, y),
        }
        results[p] = metrics

        print(f"  Hit rate : {metrics['hit_rate']:.4f}   (exact ref: 0.4571)")
        print(f"  MSE      : {metrics['mse']:.4f}   (exact ref: 0.1038)")
        print(f"  RMSE     : {metrics['rmse']:.4f}   (exact ref: 0.3221)")
        print(f"  MAE      : {metrics['mae']:.4f}   (exact ref: 0.2426)")
        print(f"  MAPE     : {metrics['mape']:.4f}   (exact ref: 8.1731)")
        print(f"  QLIKE    : {metrics['qlike']:.4f}   (exact ref: 1.4004)")

    # Summary table
    print(f"\n{'='*60}")
    print("NOISE SWEEP SUMMARY — QR2")
    print(f"{'='*60}")
    print(f"{'p_noise':<10} {'MSE':<10} {'QLIKE':<10} {'Hit Rate':<10}")
    print(f"{'(exact)':<10} {'0.1038':<10} {'1.4004':<10} {'0.4571':<10}")
    print("-" * 40)
    for p, m in results.items():
        print(f"{p:<10} {m['mse']:<10.4f} {m['qlike']:<10.4f} {m['hit_rate']:<10.4f}")

    if save_csv:
        rows = [{'p_noise': p, **m} for p, m in results.items()]
        pd.DataFrame(rows).to_csv(
            "results/predictions/noise_sweep_qr2.csv", index=False
        )
        print("\nSaved: results/predictions/noise_sweep_qr2.csv")

    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_noise_sweep(n_trotter=4)
