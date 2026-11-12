"""
fast_qrc.py — Vectorized quantum reservoir simulator (same physics, ~100x faster)
==================================================================================

Numerically equivalent to quantum_reservoir() in quantum_reservoir_qiskit.py,
but exploits two facts about the protocol:

  1. The input register is always a *pure* RY-product state and the hidden
     register is only 3 qubits (rank <= 8).  So the 1024x1024 joint density
     matrix can be stored as <= 8 state vectors  (rho = sum_r |psi_r><psi_r|)
     and evolved as U @ psi instead of U @ rho @ U^dagger.
  2. Every time step l is independent (hidden state is reset each step), so
     all ~813 time steps are evolved together in one batched matmul.

It also adds the hardware-realism knobs the exact simulator lacks:

  - shots        : finite-shot estimation of <Z_i> (None = exact expectation)
  - readout_err  : symmetric bit-flip readout error per qubit
  - memory_noise : decoherence channel on the hidden (memory) qubits after every
                   intermediate evolution + partial trace
                   ("dephasing" | "amp_damping" | "depolarizing", strength p)

Hardware equivalence: "partial trace out the input qubits, re-encode the next
input" is exactly what a circuit does with mid-circuit RESET on the input
qubits followed by fresh RY gates (see build_hardware_circuit()).  So sampling
the final probability distribution here == running that circuit on a
noiseless device.

Qubit convention matches quantum_reservoir_qiskit.py (Qiskit little-endian):
  qubits 0..n_in-1 = input, n_in..nqubit-1 = hidden; joint index = h*2^n_in + i
"""

import numpy as np
from qiskit import QuantumCircuit

from qrc.quantum_reservoir_qiskit import rolling_ridge_regression, MSE, RMSE, MAE, MAPE, compute_qlike, hitrate


# ------------------------------------------------------------------
# Input encoding: RY(pi * x_i)|0> on each input qubit, batched over time
# ------------------------------------------------------------------
def input_states(X: np.ndarray) -> np.ndarray:
    """X: (T, n_in) feature values -> (T, 2^n_in) real state vectors."""
    theta = X * np.pi / 2.0            # RY(pi*x)|0> = cos(pi x/2)|0> + sin(pi x/2)|1>
    c, s = np.cos(theta), np.sin(theta)
    T, n_in = X.shape
    psi = np.ones((T, 1))
    for q in range(n_in - 1, -1, -1):  # highest qubit is the leftmost kron factor
        psi = np.einsum("ta,tb->tab", psi, np.stack([c[:, q], s[:, q]], 1)).reshape(T, -1)
    return psi


def z_signs(nqubit: int) -> np.ndarray:
    """(2^n, n) matrix: zsign[idx, i] = +1 if bit i of idx is 0 else -1."""
    idx = np.arange(2 ** nqubit)[:, None]
    return 1.0 - 2.0 * ((idx >> np.arange(nqubit)[None, :]) & 1)


# ------------------------------------------------------------------
# Memory-qubit noise channels (Kraus form, applied to (T, d_h, d_h))
# ------------------------------------------------------------------
_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)


def single_qubit_kraus(kind: str, p: float) -> list:
    if kind == "dephasing":           # rho -> (1-p) rho + p Z rho Z
        return [np.sqrt(1 - p) * _I, np.sqrt(p) * _Z]
    if kind == "amp_damping":         # T1-type energy relaxation
        return [np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, np.sqrt(p)], [0, 0]])]
    if kind == "depolarizing":        # rho -> (1-p) rho + p I/2
        return [np.sqrt(1 - 3 * p / 4) * _I] + [np.sqrt(p / 4) * P for P in (_X, _Y, _Z)]
    raise ValueError(kind)


def apply_memory_noise(rho: np.ndarray, n_hidden: int, kind: str, p: float) -> np.ndarray:
    if kind is None or p == 0:
        return rho
    ks = single_qubit_kraus(kind, p)
    for q in range(n_hidden):
        full = [np.kron(np.kron(np.eye(2 ** (n_hidden - 1 - q)), K), np.eye(2 ** q)) for K in ks]
        rho = sum(K @ rho @ K.conj().T for K in full)
    return rho


# ------------------------------------------------------------------
# Core reservoir
# ------------------------------------------------------------------
def run_reservoir(data, features, U, dU, K_delay=3, virtual_node=1, nqubit=10,
                  shots=None, readout_err=0.0, memory_noise=(None, 0.0), rng=None):
    """
    Drop-in (much faster) replacement for quantum_reservoir().
    Returns signal of shape (nqubit * virtual_node, n_samples); first K_delay
    columns are zero, exactly like the original.
    """
    rng = np.random.default_rng(rng)
    X = data[features].to_numpy(dtype=float)
    n_samples, n_in = X.shape
    n_h = nqubit - n_in
    d_h, d_in = 2 ** n_h, 2 ** n_in
    ls = np.arange(K_delay, n_samples)
    T = len(ls)

    # hidden register, stored as factor A with rho_h = A A^dagger; starts |0..0>
    A = np.zeros((T, d_h, 1), dtype=complex)
    A[:, 0, 0] = 1.0

    for k in range(K_delay, 0, -1):
        psi_in = input_states(X[ls - k])                                   # (T, d_in)
        Psi = np.einsum("thr,ti->thir", A, psi_in).reshape(T, d_h * d_in, -1)
        r = Psi.shape[2]
        if k != 1:
            Psi = (U @ Psi.transpose(1, 0, 2).reshape(d_h * d_in, -1)).reshape(-1, T, r).transpose(1, 0, 2)
            Psi = Psi.reshape(T, d_h, d_in, r)
            rho_h = np.einsum("thir,tgir->thg", Psi, Psi.conj())             # partial trace over inputs
            rho_h = apply_memory_noise(rho_h, n_h, *memory_noise)
            w, V = np.linalg.eigh(rho_h)
            A = V * np.sqrt(np.clip(w, 0, None))[:, None, :]
        else:
            zs = z_signs(nqubit)
            out = np.zeros((nqubit * virtual_node, n_samples))
            for v in range(virtual_node):
                Psi = (dU @ Psi.transpose(1, 0, 2).reshape(d_h * d_in, -1)).reshape(-1, T, r).transpose(1, 0, 2)
                probs = np.sum(np.abs(Psi) ** 2, axis=2)                     # (T, 2^n) computational-basis probs
                p1 = probs @ ((1 - zs) / 2)                                  # P(bit_i = 1), exact marginals
                p1 = p1 * (1 - readout_err) + (1 - p1) * readout_err
                if shots is not None:
                    # binomial per qubit: exact marginals, ignores cross-qubit shot correlations
                    p1 = rng.binomial(shots, np.clip(p1, 0, 1)) / shots
                out[v * nqubit:(v + 1) * nqubit, ls] = (1 - 2 * p1).T
            return out


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------
TOTAL, L_OOS, LAMBDA = 816, 245, 1e-8
FEATURES_QR1 = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]
FEATURES_QR2 = ["RV", "MKT", "STR", "RV_q", "EP", "INF", "DEF"]


def forecast(signal, target):
    return rolling_ridge_regression(signal, target, TOTAL, L_OOS, 0, LAMBDA)


def metrics(pred, actual):
    return {"MSE": MSE(pred, actual), "RMSE": RMSE(pred, actual), "MAE": MAE(pred, actual),
            "MAPE": MAPE(pred, actual), "QLIKE": compute_qlike(pred, actual), "HitRate": hitrate(pred, actual)}


# ------------------------------------------------------------------
# Hardware circuit for ONE time step (what you would submit to a QPU)
# ------------------------------------------------------------------
def build_hardware_circuit(x_window, evo_U: QuantumCircuit, evo_dU: QuantumCircuit,
                           n_in: int, nqubit: int, n_virtual_steps: int) -> QuantumCircuit:
    """
    x_window: (K_delay, n_in) inputs, oldest first (rows l-K .. l-1).
    Implements the reservoir protocol with mid-circuit RESET instead of partial trace.
    Measures all qubits after `n_virtual_steps` applications of dU
    (virtual node v needs its own circuit with n_virtual_steps = v).
    """
    qc = QuantumCircuit(nqubit, nqubit)
    inputs = list(range(n_in))
    for k, x in enumerate(x_window):
        if k > 0:
            qc.reset(inputs)                      # == partial trace + fresh |0> on inputs
        for i, val in enumerate(x):
            qc.ry(float(val) * np.pi, i)
        if k < len(x_window) - 1:
            qc.compose(evo_U, inplace=True)
        else:
            for _ in range(n_virtual_steps):
                qc.compose(evo_dU, inplace=True)
    qc.measure(range(nqubit), range(nqubit))
    return qc
