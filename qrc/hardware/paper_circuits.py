"""Paper-faithful gate circuits for running QR1/QR2 on IonQ hardware.

The model is the one in Time_series.jl and Time_serial_Finance_regression.ipynb:
the authors' coeff_10.jld2 couplings (ms[1] for QR1, ms[2] for QR2 in Julia),
their feature lists, tau = 1, K = 3 monthly inputs, RY(pi * x) encoding, Z
readout and the rolling ridge readout with lambda = 1e-8.

Two things differ from the authors' code, and only these two:
  - Evolution: the exact matrix exponential e^{-i t H} becomes a Trotter circuit
    of RXX and RZ gates (first or second order).
  - Memory hand-off: the partial trace over the input qubits becomes a reset of
    those qubits followed by re-encoding. Each virtual-node readout of QR2 is a
    separate circuit, because measuring ends the run.

Nothing in this module contacts IonQ or any other service.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qrc.reservoir import (
    load_coupling_matrices, build_ising_hamiltonian, compute_unitaries,
    quantum_reservoir, rolling_ridge_regression, MIN_RV, DIF,
)

ROOT = Path(__file__).resolve().parents[2]
NQUBIT, N_INPUT, TAU, K = 10, 7, 1.0, 3
TOTAL, L_OOS, WS, LAMBDA = 816, 245, 0, 1e-8

# Notebook cell 11: Qreservoir(nqubit, ms[1,:,:]) and ms[2,:,:] (1-indexed).
PAPER_MODELS = {
    'QR1': {'features': ['RV', 'MKT', 'DP', 'IP', 'RV_q', 'STR', 'DEF'], 'coupling_index': 0, 'virtual_nodes': 1},
    'QR2': {'features': ['RV', 'MKT', 'STR', 'RV_q', 'EP', 'INF', 'DEF'], 'coupling_index': 1, 'virtual_nodes': 2},
}

# Gates IonQ accepts in its QIS gate set, plus reset and measurement.
IONQ_QIS_BASIS = ['rxx', 'rz', 'ry', 'reset', 'measure']


def paper_inputs():
    """Authors' Data.CSV, RV target and coupling matrices, unmodified."""
    data = pd.read_csv(ROOT / 'data/Data.CSV', header=0, index_col=0)
    ms = load_coupling_matrices(str(ROOT / 'data/coeff_10.jld2'))
    if ms is None:
        raise RuntimeError('coeff_10.jld2 could not be read; generated couplings are not paper-faithful')
    couplings = {name: ms[spec['coupling_index']] for name, spec in PAPER_MODELS.items()}
    return data, data['RV'].to_numpy(np.float64), couplings


def _xx_layer(qc, J, dt):
    # Same pair order and coupling cut-off as quantum_reservoir_trotter.build_trotter_step.
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                qc.rxx(2.0 * J[i, j] * dt, i, j)


def _z_layer(qc, dt):
    # e^{-i dt Z} = RZ(2 dt); the field strength is 1 on every qubit.
    for q in range(qc.num_qubits):
        qc.rz(2.0 * dt, q)


def trotter_block(J, duration, steps, order):
    """Gate circuit approximating e^{-i duration H}, H = sum J_ij X_i X_j + sum Z_i.

    order 1: [Z . XX]^steps, XX gates first (matches quantum_reservoir_trotter.py).
    order 2: [Z/2 . XX . Z/2]^steps (Strang splitting). Neighbouring half Z layers
             merge, so it uses exactly the same RXX gates as order 1 and only
             steps + 1 RZ layers.
    """
    if order not in (1, 2) or steps < 1:
        raise ValueError('order must be 1 or 2 and steps >= 1')
    dt = duration / steps
    qc = QuantumCircuit(J.shape[0])
    if order == 1:
        for _ in range(steps):
            _xx_layer(qc, J, dt)
            _z_layer(qc, dt)
        return qc
    _z_layer(qc, dt / 2)
    for step in range(steps):
        _xx_layer(qc, J, dt)
        _z_layer(qc, dt if step < steps - 1 else dt / 2)
    return qc


def month_values(data, features, l):
    """Inputs for forecast column l: rows l-3, l-2, l-1 (oldest first), as in Quantum_Reservoir."""
    return data[features].iloc[l - K:l].to_numpy(np.float64)


def history_circuit(values, J, steps, order, virtual_nodes, endpoint):
    """Full three-month reservoir run measured after `endpoint` virtual-node evolutions.

    Input qubits 0-6 carry features; qubits 7-9 are the memory. After each of the
    first two months the input qubits are reset and re-encoded, which leaves the
    memory in the same state as the authors' partial trace.
    """
    u = trotter_block(J, TAU, steps, order)
    du = trotter_block(J, TAU / virtual_nodes, steps, order)
    qc = QuantumCircuit(NQUBIT, NQUBIT)
    for lag, row in enumerate(values):
        if lag:
            qc.reset(range(N_INPUT))
        for q, value in enumerate(row):
            qc.ry(float(np.pi * value), q)
        if lag < K - 1:
            qc.compose(u, inplace=True)
        else:
            for _ in range(endpoint):
                qc.compose(du, inplace=True)
    qc.measure(range(NQUBIT), range(NQUBIT))
    return qc


def month_circuits(model, l, steps, order, data=None, couplings=None):
    """Every circuit needed for one forecast column: one per virtual node."""
    if data is None or couplings is None:
        data, _, couplings = paper_inputs()
    spec = PAPER_MODELS[model]
    values = month_values(data, spec['features'], l)
    return [history_circuit(values, couplings[model], steps, order, spec['virtual_nodes'], e)
            for e in range(1, spec['virtual_nodes'] + 1)]


def ideal_features(data, model, J, steps=None, order=None):
    """Noiseless reservoir features; steps=None gives the authors' exact evolution.

    Uses the production partial-trace simulator with the Trotter blocks as its
    unitaries. tests/test_paper_circuits.py checks this equals simulating the
    reset-based history circuits.
    """
    spec = PAPER_MODELS[model]
    v = spec['virtual_nodes']
    if steps is None:
        u, du = compute_unitaries(build_ising_hamiltonian(NQUBIT, J), TAU, v)
    else:
        u = Operator(trotter_block(J, TAU, steps, order)).data
        du = Operator(trotter_block(J, TAU / v, steps, order)).data
    return quantum_reservoir(data, spec['features'], u, du, K, v, NQUBIT)


def paper_forecasts(signal, target):
    """Rolling ridge readout of the notebook, returned on the log-RV scale like predict_result.csv."""
    pred = rolling_ridge_regression(signal, target, TOTAL, L_OOS, WS, LAMBDA)
    return pred, (pred + 1) * DIF + MIN_RV
