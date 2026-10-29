"""Circuits for the IonQ noise study of the paper-faithful QR1/QR2 reservoir.

Two equivalent forms of the same reservoir history (RY encode -> evolve, per month):

- reset_history_circuit: 10 qubits, the 7 input qubits are reset between months.
  Used for the local Aer simulation of every forecast month.
- unrolled_history_circuit: no resets. Each month gets 7 fresh input qubits and
  the 3 memory qubits are shared, so n months need 7n + 3 qubits. IonQ's cloud
  simulator lists no reset gate (and fails reset circuits), so this is the cloud form.

Old input qubits are never touched again, so both equal the authors' partial
trace. Gate content (RY, RXX, RZ) is identical; only qubit labels differ. The
Trotter blocks are qrc.hardware.paper_circuits.trotter_block (order 2 here).

The paper protocol uses 3 months (24 unrolled qubits). The cloud check also uses
1- and 2-month histories (10 and 17 qubits), because the cloud simulator did not
run any job with 20 or more qubits.
"""
import numpy as np
from qiskit import QuantumCircuit

from qrc.hardware.paper_circuits import PAPER_MODELS, N_INPUT, NQUBIT, TAU, K, trotter_block

ORDER = 2


def history_values(data, features, l, months=K):
    """Inputs for feature column l: the `months` rows before l, oldest first."""
    return data[features].iloc[l - months:l].to_numpy(np.float64)


def block_qubits(b, months):
    """Physical qubits standing in for reservoir qubits 0..9 during month b of an unrolled circuit."""
    memory = list(range(N_INPUT * months, N_INPUT * months + NQUBIT - N_INPUT))
    return list(range(N_INPUT * b, N_INPUT * (b + 1))) + memory


def _blocks(J, steps, virtual_nodes):
    return trotter_block(J, TAU, steps, ORDER), trotter_block(J, TAU / virtual_nodes, steps, ORDER)


def unrolled_history_circuit(values, J, steps, virtual_nodes, endpoint):
    """Reset-free circuit measured after `endpoint` virtual-node evolutions of the last month."""
    n = len(values)
    u, du = _blocks(J, steps, virtual_nodes)
    qc = QuantumCircuit(N_INPUT * n + NQUBIT - N_INPUT, NQUBIT)
    for b, row in enumerate(values):
        qubits = block_qubits(b, n)
        for q, value in enumerate(row):
            qc.ry(float(np.pi * value), qubits[q])
        for _ in range(1 if b < n - 1 else endpoint):
            qc.compose(u if b < n - 1 else du, qubits=qubits, inplace=True)
    qc.measure(block_qubits(n - 1, n), range(NQUBIT))     # clbit q = reservoir qubit q
    return qc


def reset_history_circuit(values, J, steps, virtual_nodes):
    """10-qubit reset circuit that saves the outcome distribution after each virtual node.

    One density-matrix run then gives what separate measured circuits per
    endpoint would give (the same as qrc.hardware.paper_circuits.history_circuit).
    """
    u, du = _blocks(J, steps, virtual_nodes)
    qc = QuantumCircuit(NQUBIT)
    for b, row in enumerate(values):
        if b:
            qc.reset(range(N_INPUT))
        for q, value in enumerate(row):
            qc.ry(float(np.pi * value), q)
        if b < len(values) - 1:
            qc.compose(u, inplace=True)
        else:
            for node in range(virtual_nodes):
                qc.compose(du, inplace=True)
                qc.save_probabilities(label=f'node{node}')
    return qc


def cloud_circuits(model, l, steps, months, data, couplings):
    """Unrolled circuits (one per virtual node) for feature column l, named for the cache."""
    spec = PAPER_MODELS[model]
    values = history_values(data, spec['features'], l, months)
    circuits = []
    for e in range(1, spec['virtual_nodes'] + 1):
        qc = unrolled_history_circuit(values, couplings[model], steps, spec['virtual_nodes'], e)
        qc.name = f'{model}_s{steps}_k{months}_l{l}_e{e}'
        circuits.append(qc)
    return circuits


def z_from_counts(counts, nbits=NQUBIT):
    """<Z_q> from counts or probabilities keyed by bitstrings (clbit 0 rightmost)."""
    z, total = np.zeros(nbits), 0.0
    for key, n in counts.items():
        bits = (int(key.replace(' ', ''), 2) >> np.arange(nbits)) & 1
        z += n * (1 - 2 * bits)
        total += n
    return z / total
