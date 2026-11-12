"""
qpu_experiment.py — Budgeted real-hardware run of the quantum reservoir
=======================================================================

Design (from qpu_shot_budget.py and quantum_value_and_cost.py):
  * QR1 inputs, memory K=1: one month = one 10-qubit block, no mid-circuit reset.
  * Reservoir circuit: RY(pi x) encoding, then ONE order-2 Trotter step of the
    QR1 Ising Hamiltonian keeping only the `--couplings` strongest J_ij
    (default 15 RXX gates). The readout is trained on exact simulated features
    of this same circuit, so the circuit need not reproduce e^{-iHt}.
  * `--pack` months share one circuit on disjoint qubits (3 x 10 = 30 of
    Forte-1's 36), so the per-circuit minimum price is paid once per 3 months.
  * Feature shrinkage s from hardware noise is measured on the test circuits
    themselves (hardware <Z> vs simulated <Z>: no target data involved). The
    ridge readout is noise-aware: trained on simulated features shrunk by s
    plus matching shot noise; lambda = 10, standardized, intercept.
  * Test window 2008-07..2009-06 (Lehman / GFC), inside the paper's test period.

Backends:
  --backend local              ideal Aer sampling (free, offline)
  --backend ionq_simulator     IonQ cloud simulator, --noise ideal|aria-1|forte-1 (free)
  --backend qpu.forte-1        REAL HARDWARE. --dry-run-qpu gets IonQ's exact, unbilled
                               quote; a real submission additionally needs --confirm-qpu.
Debiasing (error mitigation) is switched off on hardware: it raises the minimum
price and the noise-aware readout already accounts for noise.

The API key is read from IONQ_API_KEY only and never written to disk. Job ids
and counts are cached in results/qpu/<run>/ so runs resume without resubmitting.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qrc.quantum_reservoir_qiskit import MSE
from experiments.overfitting_diagnostics import DATA, WI, lagged, rolling_fit
from qrc.fast_qrc import FEATURES_QR1, z_signs
from experiments.quantum_value_and_cost import circuit as reservoir_circuit, prune, probs_for

NQ, N_IN = 10, 7
TEST = pd.date_range("2008-07-31", "2009-06-30", freq="ME")
LAM = 10.0
ZS = z_signs(NQ)


def block_circuit(Jm, steps):
    return reservoir_circuit(Jm, steps)


def packed_circuit(rows_x, Jm, steps):
    """len(rows_x) independent months on disjoint 10-qubit blocks, all measured."""
    nb = len(rows_x)
    qc = QuantumCircuit(NQ * nb, NQ * nb)
    blk = block_circuit(Jm, steps)
    for b, x in enumerate(rows_x):
        qs = list(range(NQ * b, NQ * (b + 1)))
        for q, v in enumerate(x):
            qc.ry(float(np.pi * v), qs[q])
        qc.compose(blk, qubits=qs, inplace=True)
    qc.measure(range(NQ * nb), range(NQ * nb))
    return qc


def z_blocks(counts, nb):
    """<Z> for each 10-qubit block from counts over NQ*nb clbits (clbit 0 rightmost)."""
    z, tot = np.zeros((nb, NQ)), 0
    for key, n in counts.items():
        v = int(key.replace(" ", ""), 2)
        bits = (v >> np.arange(NQ * nb)) & 1
        z += n * (1 - 2 * bits).reshape(nb, NQ)
        tot += n
    return z / tot


def get_backend(name, noise):
    if name == "local":
        from qiskit_aer import AerSimulator
        return AerSimulator(method="matrix_product_state")      # packed blocks are unentangled: MPS is cheap
    from qiskit_ionq import IonQProvider
    key = os.environ.get("IONQ_API_KEY")
    if not key:
        raise SystemExit("Set IONQ_API_KEY in the environment (it is never stored in this repo).")
    backend = IonQProvider(key).get_backend(name)
    if name == "ionq_simulator":
        backend.set_options(noise_model=noise)
    return backend


def run_kwargs(backend_name):
    if backend_name.startswith("qpu"):
        from qiskit_ionq import ErrorMitigation
        return {"error_mitigation": ErrorMitigation.NO_DEBIASING}
    return {}


def dry_run_quote(backend, circuits, shots, backend_name):
    import requests
    key = os.environ["IONQ_API_KEY"]
    total, ids = 0.0, []
    for qc in circuits:
        job = backend.run(qc, shots=shots, dry_run=True, **run_kwargs(backend_name))
        jid = job.job_id()
        for _ in range(30):
            r = requests.get(f"https://api.ionq.co/v0.4/jobs/{jid}/cost",
                             headers={"Authorization": f"apiKey {key}"}, timeout=30).json()
            if r.get("estimated_cost") or r.get("cost"):
                break
            time.sleep(2)
        val = (r.get("estimated_cost") or r.get("cost") or {}).get("value")
        print(f"  dry run {jid}: {r}")
        total += float(val or 0)
        ids.append(jid)
    return total, ids


def run_counts(backend, circuits, shots, cache, backend_name):
    cache.mkdir(parents=True, exist_ok=True)
    rec_path = cache / "job.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    if "counts" in rec:
        return rec["counts"]
    if "job_ids" not in rec:
        jobs = [backend.run(qc, shots=shots, **run_kwargs(backend_name)) for qc in circuits]
        rec = {"job_ids": [j.job_id() for j in jobs], "backend": backend_name, "shots": shots,
               "submitted": time.time(), "names": [qc.name for qc in circuits]}
        rec_path.write_text(json.dumps(rec, indent=1))
        print(f"submitted {len(jobs)} jobs ({shots} shots each) to {backend_name}: {rec['job_ids']}")
    else:
        print(f"resuming jobs {rec['job_ids']}")
        jobs = [backend.retrieve_job(j) for j in rec["job_ids"]]
    rec["counts"] = [dict(j.result().get_counts()) for j in jobs]
    rec["completed"] = time.time()
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec["counts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", help="local | ionq_simulator | qpu.forte-1")
    ap.add_argument("--noise", default="aria-1")
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--couplings", type=int, default=15)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--pack", type=int, default=3, help="months per circuit (10 qubits each)")
    ap.add_argument("--dry-run", action="store_true", help="build circuits and print gate counts only")
    ap.add_argument("--dry-run-qpu", action="store_true", help="get IonQ's unbilled price quote")
    ap.add_argument("--confirm-qpu", action="store_true", help="required to submit to real hardware")
    args = ap.parse_args()
    is_qpu = args.backend.startswith("qpu")
    if is_qpu and not (args.confirm_qpu or args.dry_run or args.dry_run_qpu):
        raise SystemExit("Refusing to submit to real hardware without --confirm-qpu (this spends QPU credits).")
    if args.backend == "ionq_simulator" and args.pack * NQ > 19:
        print("note: IonQ's simulator did not run >19-qubit jobs for this project; using --pack 1")
        args.pack = 1

    Jm = prune(args.couplings)
    X = DATA[FEATURES_QR1].to_numpy(float)
    y = DATA["RV"].to_numpy(float)
    idx = {d: i for i, d in enumerate(DATA.index)}
    test_rows = [idx[d] for d in TEST]
    start, n = test_rows[0], len(test_rows)

    # exact simulated features of THIS circuit for every month (training + reference)
    P = probs_for(Operator(block_circuit(Jm, args.steps)).data)
    F = P @ ZS
    groups = [test_rows[i:i + args.pack] for i in range(0, n, args.pack)]
    circuits = []
    for g in groups:
        qc = packed_circuit([X[r - 1] for r in g], Jm, args.steps)
        qc.name = f"QR1K1_m{args.couplings}_s{args.steps}_" + "_".join(f"{DATA.index[r]:%Y%m}" for r in g)
        circuits.append(qc)
    ops = circuits[0].count_ops()
    n1 = ops.get("ry", 0) + ops.get("rz", 0)
    print(f"{len(circuits)} circuits x {circuits[0].num_qubits} qubits ({args.pack} months each), "
          f"{ops.get('rxx', 0)} RXX + {n1} 1q gates per circuit, {args.shots} shots, depth {circuits[0].depth()}")
    if args.dry_run:
        return

    backend = get_backend(args.backend, args.noise)
    if args.dry_run_qpu:
        total, _ = dry_run_quote(backend, circuits, args.shots, args.backend)
        print(f"\nIonQ quote for the full run: ${total:.2f} (dry run: nothing executed or charged)")
        return

    tag = (f"{args.backend}{'-' + args.noise if args.backend == 'ionq_simulator' else ''}"
           f"_m{args.couplings}_s{args.steps}_pack{args.pack}_shots{args.shots}")
    out = Path("results/qpu") / tag
    counts = run_counts(backend, circuits, args.shots, out, args.backend)
    Zhw = np.vstack([z_blocks(c, len(g)) for c, g in zip(counts, groups)])      # (n, 10)

    Fe = F[test_rows]
    s_hat = float((Fe.ravel() @ Zhw.ravel()) / (Fe.ravel() @ Fe.ravel()))
    corr = float(np.corrcoef(Fe.ravel(), Zhw.ravel())[0, 1])

    rng = np.random.default_rng(0)
    preds = []
    for _ in range(50):
        Ftr = 1 - 2 * rng.binomial(args.shots, np.clip((1 - s_hat * F) / 2, 0, 1)) / args.shots
        p = []
        for j, r in enumerate(test_rows):
            A, b, x = Ftr[r - WI:r], y[r - WI:r], Zhw[j]
            mu, sd = A.mean(0), A.std(0)
            sd[sd < 1e-12] = 1
            A, x = (A - mu) / sd, (x - mu) / sd
            w = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ (b - b.mean()))
            p.append(x @ w + b.mean())
        preds.append(p)
    pred_hw = np.mean(preds, 0)

    yt = y[test_rows]
    sim = rolling_fit(F, y, start, n, lam=LAM, intercept=True, standardize=True)[0]
    har = rolling_fit(DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float), y, start, n, intercept=True)[0]
    lin = rolling_fit(lagged(DATA, FEATURES_QR1, [1, 2, 3]), y, start, n, intercept=True)[0]
    res = pd.DataFrame({"month": TEST.strftime("%Y-%m"), "realized": yt, f"QR1 on {tag}": pred_hw,
                        "same circuit, exact simulation": sim, "HAR": har, "Linear, same inputs, lags 1-3": lin})
    res.to_csv(out / "forecasts.csv", index=False)
    pd.DataFrame(Zhw, index=TEST.strftime("%Y-%m")).to_csv(out / "hardware_z.csv")
    lines = [f"# Hardware-path run: {tag}\n",
             f"- {len(circuits)} circuits x {circuits[0].num_qubits} qubits, {ops.get('rxx', 0)} RXX each, "
             f"{args.shots} shots, debiasing off",
             f"- measured feature shrinkage s = {s_hat:.3f}; corr(hardware <Z>, simulated <Z>) = {corr:.3f}",
             f"- mean |hardware Z - s * simulated Z| = {np.abs(Zhw - s_hat * Fe).mean():.4f} "
             f"(shot noise alone ~{np.sqrt(2 / np.pi / args.shots):.4f})",
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
