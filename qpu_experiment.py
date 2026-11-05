"""
qpu_experiment.py — Minimal, budgeted real-hardware run of the quantum reservoir
================================================================================

Design (justified by qpu_shot_budget.py):
  * Model QR1 with memory depth K=1: 10 qubits, no mid-circuit reset, one
    circuit per month (QR1 has one virtual node). Exact-simulation test MSE
    0.1077 (K=3: 0.1051), still better than HAR (0.1156).
  * Evolution e^{-i tau H} as an order-2 Trotter circuit of RXX/RZ (IonQ native
    QIS gates) — same construction as Vikas's qrc/hardware/paper_circuits.py.
  * Only CAL_MONTHS + TEST_MONTHS circuits go to the backend:
      - calibration months (inside the training window) measure the hardware
        shrinkage s = slope of hardware <Z> vs exact <Z>;
      - test months are forecast with a noise-aware ridge readout trained on
        simulated features shrunk by the measured s plus matching shot noise.
  * Test window: 2008-07..2009-06 (Lehman / GFC), inside the paper's test period.

Backends:
  --backend local            ideal Aer sampling (free, offline)
  --backend ionq_simulator   IonQ cloud simulator, --noise ideal|aria-1|forte-1
  --backend qpu.forte-1      REAL HARDWARE: refuses to submit unless --confirm-qpu
  --dry-run                  build + validate circuits, print the cost, submit nothing

The API key is read from the IONQ_API_KEY environment variable only; it is never
written to disk. Job ids and counts are cached in results/qpu/<run>/ so a run
can be resumed / re-analysed without resubmitting.

  python qpu_experiment.py --dry-run
  IONQ_API_KEY=... python qpu_experiment.py --backend ionq_simulator --noise aria-1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_reservoir_qiskit import MSE, COE
from overfitting_diagnostics import DATA, WI, qrc_features, lagged, rolling_fit, MS
from fast_qrc import FEATURES_QR1

NQ, N_IN, TAU, K = 10, 7, 1.0, 1
J = MS[0]                                   # QR1's coupling (paper ms[1] in Julia indexing)
TEST = pd.date_range("2008-07-31", "2009-06-30", freq="ME")
N_CAL = 8


# ----------------------------------------------------------------------------
# circuits
# ----------------------------------------------------------------------------
def trotter_block(duration, steps):
    """Order-2 (Strang) Trotter circuit for e^{-i t H}, H = sum J_ij X_i X_j + sum Z_i."""
    dt = duration / steps
    qc = QuantumCircuit(NQ)

    def z(d):
        for q in range(NQ):
            qc.rz(2.0 * d, q)

    z(dt / 2)
    for s in range(steps):
        for i in range(NQ):
            for j in range(i + 1, NQ):
                if abs(J[i, j]) > 1e-12:
                    qc.rxx(2.0 * J[i, j] * dt, i, j)
        z(dt if s < steps - 1 else dt / 2)
    return qc


def month_circuit(row, steps, measure=True):
    """K=1 reservoir step for one month: RY(pi x) encode the 7 inputs, evolve, measure all 10."""
    qc = QuantumCircuit(NQ, NQ if measure else 0)
    for q, x in enumerate(row):
        qc.ry(float(np.pi * x), q)
    qc.compose(trotter_block(TAU, steps), inplace=True)
    if measure:
        qc.measure(range(NQ), range(NQ))
    return qc


def z_from_counts(counts):
    z, tot = np.zeros(NQ), 0
    for key, n in counts.items():
        bits = (int(key.replace(" ", ""), 2) >> np.arange(NQ)) & 1
        z += n * (1 - 2 * bits)
        tot += n
    return z / tot


def z_exact_circuit(row, steps):
    sv = Statevector(month_circuit(row, steps, measure=False))
    p = sv.probabilities()
    bits = (np.arange(2 ** NQ)[:, None] >> np.arange(NQ)) & 1
    return p @ (1 - 2 * bits)


# ----------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------
def get_backend(name, noise):
    if name == "local":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    from qiskit_ionq import IonQProvider
    key = os.environ.get("IONQ_API_KEY")
    if not key:
        raise SystemExit("Set IONQ_API_KEY in the environment (it is never stored in this repo).")
    backend = IonQProvider(key).get_backend(name)
    if name == "ionq_simulator":
        backend.set_options(noise_model=noise)
    return backend


def run_counts(backend, circuits, shots, cache):
    cache.mkdir(parents=True, exist_ok=True)
    rec_path = cache / "job.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    if "counts" in rec:
        return rec["counts"]
    if "job_id" not in rec:
        job = backend.run(circuits, shots=shots)
        rec = {"job_id": job.job_id(), "backend": backend.name, "shots": shots, "submitted": time.time(),
               "names": [qc.name for qc in circuits]}
        rec_path.write_text(json.dumps(rec, indent=1))
        print(f"submitted job {rec['job_id']} ({len(circuits)} circuits x {shots} shots) to {backend.name}")
        result = job.result()
    else:
        print(f"resuming job {rec['job_id']}")
        result = backend.retrieve_job(rec["job_id"]).result()
    rec["counts"] = [dict(result.get_counts(i)) for i in range(len(circuits))]
    rec["completed"] = time.time()
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec["counts"]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", help="local | ionq_simulator | qpu.forte-1 | qpu.aria-1")
    ap.add_argument("--noise", default="aria-1", help="IonQ simulator noise model: ideal | aria-1 | forte-1")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2, help="order-2 Trotter steps (QR1 'chosen' = 2)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-qpu", action="store_true", help="required to submit to real hardware")
    args = ap.parse_args()
    if args.backend.startswith("qpu") and not (args.confirm_qpu or args.dry_run):
        raise SystemExit("Refusing to submit to real hardware without --confirm-qpu (this spends QPU credits).")

    X = DATA[FEATURES_QR1].to_numpy(float)
    y = DATA["RV"].to_numpy(float)
    idx = {d: i for i, d in enumerate(DATA.index)}
    test_rows = [idx[d] for d in TEST]
    start = test_rows[0]
    cal_rows = [int(r) for r in np.linspace(start - WI + 5, start - 1, N_CAL)]   # spread over the first training window
    rows = cal_rows + test_rows

    # exact simulated features (K=1) for every month, and validation of the gate circuit
    F = qrc_features(DATA, FEATURES_QR1, 1, 0, k_delay=K)
    z_circ = np.array([z_exact_circuit(X[r - 1], args.steps) for r in rows])
    trot_err = np.abs(z_circ - F[rows]).max()
    circuits = []
    for r in rows:
        qc = month_circuit(X[r - 1], args.steps)
        qc.name = f"QR1_K1_s{args.steps}_row{r}_{DATA.index[r]:%Y-%m}"
        circuits.append(qc)
    ops = circuits[0].count_ops()
    total_shots = len(circuits) * args.shots
    print(f"{len(circuits)} circuits ({N_CAL} calibration + {len(test_rows)} test months), {NQ} qubits, "
          f"depth {circuits[0].depth()}, gates {dict(ops)}")
    print(f"total shots {total_shots:,}; two-qubit gate-shots {ops.get('rxx', 0) * total_shots:,}")
    print(f"max |<Z> Trotter circuit - exact evolution| = {trot_err:.4f}")
    if args.dry_run:
        return

    tag = f"{args.backend}{'-' + args.noise if args.backend == 'ionq_simulator' else ''}_shots{args.shots}_steps{args.steps}"
    out = Path("results/qpu") / tag
    counts = run_counts(get_backend(args.backend, args.noise), circuits, args.shots, out)
    Zhw = np.array([z_from_counts(c) for c in counts])

    # calibration: hardware shrinkage s (slope through origin vs exact)
    zc, ze = Zhw[:N_CAL].ravel(), F[cal_rows].ravel()
    s_hat = float(ze @ zc / (ze @ ze))
    print(f"measured shrinkage s = {s_hat:.3f}")

    # noise-aware ridge readout: train on simulated features shrunk by s_hat + matching shot noise
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(50):
        Ftr = 1 - 2 * rng.binomial(args.shots, np.clip((1 - s_hat * F) / 2, 0, 1)) / args.shots
        Ftr[:K] = 0
        p = []
        for j, r in enumerate(test_rows):
            A, b, x = Ftr[r - WI:r], y[r - WI:r], Zhw[N_CAL + j]
            mu, sd = A.mean(0), A.std(0)
            sd[sd < 1e-12] = 1
            A, x = (A - mu) / sd, (x - mu) / sd
            w = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (b - b.mean()))
            p.append(x @ w + b.mean())
        preds.append(p)
    pred_hw = np.mean(preds, 0)

    yt = y[test_rows]
    n = len(test_rows)
    exact = rolling_fit(F, y, start, n, lam=1.0, intercept=True, standardize=True)[0]
    har = rolling_fit(DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float), y, start, n, intercept=True)[0]
    lin = rolling_fit(lagged(DATA, FEATURES_QR1, [1, 2, 3]), y, start, n, intercept=True)[0]
    res = pd.DataFrame({"month": TEST.strftime("%Y-%m"), "realized": yt, f"QR1 K=1 on {tag}": pred_hw,
                        "QR1 K=1 exact simulation": exact, "HAR": har, "Linear, same inputs, lags 1-3": lin})
    res.to_csv(out / "forecasts.csv", index=False)
    feat_err = np.abs(Zhw[N_CAL:] - s_hat * F[test_rows]).mean()
    lines = [f"# Hardware-path run: {tag}\n",
             f"- circuits: {len(circuits)} ({N_CAL} calibration + {n} test), {NQ} qubits, depth {circuits[0].depth()}, "
             f"{ops.get('rxx', 0)} RXX gates each, {args.shots} shots",
             f"- Trotter circuit vs exact evolution: max |dZ| = {trot_err:.4f}",
             f"- measured feature shrinkage s = {s_hat:.3f}",
             f"- test months: mean |hardware Z - s*exact Z| = {feat_err:.4f} "
             f"(pure shot noise would be ~{np.sqrt(2 / np.pi / args.shots):.4f})",
             f"\nMSE over {TEST[0]:%Y-%m}..{TEST[-1]:%Y-%m} ({n} months, Lehman/GFC):\n",
             "| model | MSE |", "|---|---|"]
    for c in res.columns[2:]:
        lines.append(f"| {c} | {MSE(res[c].to_numpy(), yt):.4f} |")
    lines.append("\n12 months is a small sample: this run tests that the hardware path works and matches the "
                 "simulator's prediction, not which model is better.")
    txt = "\n".join(lines)
    print(txt)
    (out / "report.md").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
