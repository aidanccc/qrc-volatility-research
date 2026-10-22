"""Local emulation of IonQ Aria-1 noise for the 10-qubit reservoir circuits.

The uploaded IonQ demo used the cloud simulator with noise_model="aria-1".
This module builds an equivalent-in-spirit local Aer noise model from IonQ's
public Aria-1 characterization records (snapshot in aria1_characterizations.json):

- rxx (Molmer-Sorensen-type 2-qubit gate): 2-qubit depolarizing error
- ry (1-qubit rotation): 1-qubit depolarizing error
- rz: noiseless (virtual phase update on IonQ hardware)
- SPAM: symmetric readout bit-flip applied to the measured distribution
- all-to-all connectivity, so circuits run untranspiled (no SWAPs)

Fidelities are treated as average gate fidelities F, converted to the
depolarizing parameter lambda = (1 - F) * d / (d - 1).
"""
from pathlib import Path
import json

import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error

SNAPSHOT = Path(__file__).with_name('aria1_characterizations.json')


def aria1_rates(path=SNAPSHOT):
    """Median 1q/2q/SPAM fidelities over complete, non-outlier daily records."""
    records = json.loads(Path(path).read_text())['records']
    ok = [r for r in records
          if r['fidelity']['2q']['median'] is not None
          and r['fidelity']['spam']['median'] is not None
          and r['fidelity']['1q']['median'] >= 0.99]   # drops 0.79 calibration outliers
    f1 = float(np.median([r['fidelity']['1q']['median'] for r in ok]))
    f2 = float(np.median([r['fidelity']['2q']['median'] for r in ok]))
    spam = float(np.median([r['fidelity']['spam']['median'] for r in ok]))
    return {'fidelity_1q': f1, 'fidelity_2q': f2, 'fidelity_spam': spam,
            'records_used': len(ok), 'records_total': len(records),
            'first_date': min(r['date'] for r in ok)[:10], 'last_date': max(r['date'] for r in ok)[:10]}


def error_parameters(scale=1.0, rates=None):
    """Depolarizing and readout probabilities at `scale` x Aria-1 error."""
    rates = rates or aria1_rates()
    p1 = 2 * (1 - rates['fidelity_1q']) * scale
    p2 = 4 / 3 * (1 - rates['fidelity_2q']) * scale
    readout = (1 - rates['fidelity_spam']) * scale
    assert 0 <= p1 <= 1 and 0 <= p2 <= 16 / 15 and 0 <= readout < 0.5
    return {'scale': scale, 'depolarizing_1q': p1, 'depolarizing_2q': p2, 'readout_flip': readout}


def build_noise_model(scale=1.0, rates=None):
    """Aer NoiseModel; returns None at scale 0 (ideal)."""
    if scale == 0:
        return None
    p = error_parameters(scale, rates)
    model = NoiseModel(basis_gates=['ry', 'rz', 'rxx', 'reset'])
    model.add_all_qubit_quantum_error(depolarizing_error(p['depolarizing_1q'], 1), ['ry'])
    model.add_all_qubit_quantum_error(depolarizing_error(p['depolarizing_2q'], 2), ['rxx'])
    return model


def apply_readout(probabilities, flip, nqubit=10):
    """Apply an independent symmetric bit-flip to each qubit of a 2**n distribution."""
    if flip == 0:
        return np.asarray(probabilities, dtype=float)
    confusion = np.array([[1 - flip, flip], [flip, 1 - flip]])
    p = np.asarray(probabilities, dtype=float).reshape([2] * nqubit)
    for axis in range(nqubit):
        p = np.moveaxis(np.tensordot(confusion, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


def z_expectations(probabilities, nqubit=10):
    """<Z_q> for q = 0..n-1 from a little-endian (Qiskit) basis distribution."""
    bits = (np.arange(2 ** nqubit)[:, None] >> np.arange(nqubit)) & 1
    return (1 - 2 * bits).T @ np.asarray(probabilities)


def sample_z(probabilities, shots, rng, nqubit=10):
    """Finite-shot <Z_q> estimates from one multinomial draw of `shots` measurements."""
    p = np.clip(probabilities, 0, None)
    counts = rng.multinomial(shots, p / p.sum())
    return z_expectations(counts / shots, nqubit)
