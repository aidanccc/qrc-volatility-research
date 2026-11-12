"""
quantum_value_and_cost.py — Is the quantum part doing anything, how accurate is
the model in plain terms, and how cheap can a real-QPU run be made?
==============================================================================

All on QR1 with memory K=1 (the 10-qubit, reset-free hardware candidate) unless
stated; paper test window 1997-08..2017-12 (245 months), 571-month rolling ridge.

Q1  Accuracy in plain terms: out-of-sample R^2 vs the rolling mean, typical
    multiplicative error exp(RMSE), share of months within +-25%.
Q2  What does the quantum circuit add?
      - J=0 (no couplings => each <Z> is just cos(pi x_i): no entanglement)
      - classical trig feature maps: cos/sin of inputs (1-body) and all their
        pairwise products (2-body)
      - how much of each quantum feature those classical bases explain (R^2)
Q3  Circuit reduction: keep only the m strongest couplings, 1 Trotter step.
    The readout is trained on simulated features of the SAME circuit, so the
    circuit need not reproduce e^{-iHt}; it only has to be a good reservoir.
Q4  Shot reduction: noise-aware ridge readout, local readout-noise model
    (Z shrinks by s, ZZ by s^2; s scaled to the gate count from the IonQ
    aria-1 cloud measurement s=0.794 at 90 RXX), shots 50..1000, Z-only vs
    Z + ZZ correlators (free: same shots).
Q5  Validate the noise model on the IonQ cloud counts already collected.
Q6  Cost table for candidate designs.

Writes results/research/quantum_value_and_cost.md (+ csv/png).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qrc.quantum_reservoir_qiskit import MSE, COE, MIN_RV, DIF
from experiments.overfitting_diagnostics import (DATA, WI, L_OOS, MS, C1, C2, C3, C4, INK, MUTED, md_table, lagged,
                                     rolling_fit, unitaries)
from qrc.fast_qrc import FEATURES_QR1, FEATURES_QR2, input_states, z_signs

OUT = Path("results/research")
NQ, N_IN = 10, 7
J = MS[0]
Y = DATA["RV"].to_numpy(float)
START = len(DATA) - L_OOS
YT = Y[START:]
X = DATA[FEATURES_QR1].to_numpy(float)
ZS = z_signs(NQ)                                      # (1024, 10)
PAIRS = list(combinations(range(NQ), 2))
ZZS = np.stack([ZS[:, i] * ZS[:, j] for i, j in PAIRS], 1)
S_REF, RXX_REF = 0.794, 90                            # IonQ cloud aria-1, QR1 K=1, 2 Trotter steps
LINES = []

# pricing (USD). Azure Quantum / Google listing for IonQ Forte, Nov 2026; IonQ research-credit estimator
PRICE = {"Forte commercial (no error mitigation)": dict(g2=0.001121, g1=0.00022, minimum=25.79),
         "Forte commercial (error mitigation on, default)": dict(g2=0.001121, g1=0.00022, minimum=168.20),
         "IonQ research-credit estimator": dict(g2=0.0003, g1=0.0, minimum=1.0)}


def log(s=""):
    print(s)
    LINES.append(s)


# ----------------------------------------------------------------------------
def circuit(Jm, steps):
    """Order-2 Trotter circuit of H = sum Jm_ij X_i X_j + sum Z_i over tau=1 (no measurement)."""
    dt = 1.0 / steps
    qc = QuantumCircuit(NQ)
    for q in range(NQ):
        qc.rz(dt, q)
    for s in range(steps):
        for i, j in PAIRS:
            if abs(Jm[i, j]) > 1e-12:
                qc.rxx(2 * Jm[i, j] * dt, i, j)
        for q in range(NQ):
            qc.rz(2 * dt if s < steps - 1 else dt, q)
    return qc


def prune(m):
    """Keep the m largest-|J| couplings."""
    vals = sorted(PAIRS, key=lambda p: -abs(J[p]))[:m]
    Jm = np.zeros_like(J)
    for i, j in vals:
        Jm[i, j] = Jm[j, i] = J[i, j]
    return Jm


def probs_for(Ufull):
    """K=1 outcome distribution for every month: hidden |000>, inputs RY(pi x_{t-1})."""
    psi = input_states(X[:-1])                                     # rows 0..T-2 encode month t-1
    P = np.abs(psi @ Ufull[:, :2 ** N_IN].T) ** 2                  # (T-1, 1024)
    out = np.zeros((len(X), 2 ** NQ))
    out[1:] = P
    out[0, 0] = 1
    return out


def bitflip(P, f):
    """Independent readout bit-flip f on every qubit (Z -> (1-2f) Z, ZZ -> (1-2f)^2 ZZ)."""
    if f == 0:
        return P
    T = P.reshape((-1,) + (2,) * NQ)
    M = np.array([[1 - f, f], [f, 1 - f]])
    for ax in range(1, NQ + 1):
        T = np.moveaxis(np.tensordot(T, M, axes=([ax], [0])), -1, ax)
    return T.reshape(P.shape)


def feats(P, zz=False):
    F = P @ ZS
    return np.column_stack([F, P @ ZZS]) if zz else F


def ridge_preds(Ftr, Fte, lam=1.0):
    p = np.zeros(L_OOS)
    for j in range(L_OOS):
        t = START + j
        A, b, x = Ftr[t - WI:t], Y[t - WI:t], Fte[t]
        mu, sd = A.mean(0), A.std(0)
        sd[sd < 1e-12] = 1
        A, x = (A - mu) / sd, (x - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (b - b.mean()))
        p[j] = x @ w + b.mean()
    return p


def plain(pred):
    """Plain-language accuracy of forecasts of normalized log-RV."""
    to_log = lambda v: (v + 1) * DIF + MIN_RV
    e = to_log(pred) - to_log(YT)
    mean_fc = np.array([Y[START + j - WI:START + j].mean() for j in range(L_OOS)])
    return {"MSE": MSE(pred, YT), "OOS R2 vs rolling mean": 1 - MSE(pred, YT) / MSE(mean_fc, YT),
            "typical error factor exp(RMSE)": float(np.exp(np.sqrt(np.mean(e ** 2)))),
            "median abs % error in RV": float(np.median(np.abs(np.exp(e) - 1)) * 100),
            "months within +-25%": float(np.mean(np.abs(np.exp(e) - 1) < 0.25) * 100)}


# ----------------------------------------------------------------------------
def q1_q2():
    log("# Is the quantum part doing anything, and how accurate is it?\n")
    U_exact = unitaries(0, 1)[1]
    P = probs_for(U_exact)
    Fq = feats(P)
    from experiments.overfitting_diagnostics import qrc_features
    ref = qrc_features(DATA, FEATURES_QR1, 1, 0, k_delay=1)
    assert np.abs(Fq[1:] - ref[1:]).max() < 1e-8, np.abs(Fq[1:] - ref[1:]).max()
    x1 = lagged(DATA, FEATURES_QR1, [1])
    trig1 = np.column_stack([np.cos(np.pi * x1), np.sin(np.pi * x1)])
    trig2 = np.column_stack([trig1] + [trig1[:, a] * trig1[:, b] for a, b in combinations(range(trig1.shape[1]), 2)])
    F0 = probs_for(unitaries(0, 1, zero_coupling=True)[1]) @ ZS
    models = {
        "QR1 K=1 quantum (10 Z features)": Fq,
        "QR1 K=1 quantum + ZZ correlators (55)": feats(P, zz=True),
        "J=0, no entanglement (= cos(pi x), 10)": F0,
        "Classical: cos/sin of inputs (14)": trig1,
        "Classical: + pairwise products (105)": trig2,
        "Linear, same inputs, lag 1 (7)": x1,
        "Linear, same inputs, lags 1-3 (21)": lagged(DATA, FEATURES_QR1, [1, 2, 3]),
        "HAR (3)": DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float),
    }
    rows = {}
    for name, F in models.items():
        for lam in [1.0]:
            rows[name] = plain(ridge_preds(F, F, lam))
    tab = pd.DataFrame(rows).T
    tab.index.name = "model (same ridge readout, lambda=1, standardized)"
    tab.to_csv(OUT / "q1_q2_quantum_value.csv")
    log("Same 245 test months, same rolling ridge readout for every model. 'K=1' = one month of memory "
        "(the hardware candidate).\n")
    log(md_table(tab.sort_values("MSE")))

    # how classical are the quantum features?
    r2 = {}
    for bname, B in [("cos/sin of inputs (1-body)", trig1), ("+ pairwise products (2-body)", trig2)]:
        B1 = np.column_stack([np.ones(len(B)), B])[1:]
        vals = []
        for i in range(Fq.shape[1]):
            f = Fq[1:, i]
            if f.std() < 1e-10:
                continue
            w = np.linalg.lstsq(B1, f, rcond=None)[0]
            vals.append(1 - np.var(f - B1 @ w) / np.var(f))
        r2[bname] = {"median R2": np.median(vals), "min R2": np.min(vals)}
    r2 = pd.DataFrame(r2).T
    r2.index.name = "classical basis explaining each quantum <Z>"
    log("\n" + md_table(r2, "{:.4f}"))
    log("\nNote: with RY encoding, every quantum feature is exactly a polynomial in cos(pi x_i), sin(pi x_i) "
        "(up to 7-body products), and a 10-qubit circuit is simulated exactly on a laptop. So the "
        "question is only whether the circuit picks a *useful* fixed feature map, not whether it is "
        "classically hard.")
    return P


def q3_reduction():
    log("\n# Circuit reduction: fewer couplings, 1 Trotter step\n")
    rows = []
    for steps in [1, 2]:
        for m in [0, 5, 10, 15, 20, 30, 45]:
            qc = circuit(prune(m), steps)
            P = probs_for(Operator(qc).data)
            n2 = qc.count_ops().get("rxx", 0)
            s = S_REF ** (n2 / RXX_REF)
            F = feats(P)
            rows.append({"couplings kept": m, "Trotter steps": steps, "RXX gates": n2,
                         "est. noise shrink s": s, "exact MSE": MSE(ridge_preds(F, F), YT)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q3_circuit_reduction.csv", index=False)
    log("Readout trained on exact simulated features of the same circuit. s = assumed shrink "
        f"{S_REF}^(RXX/{RXX_REF}).\n")
    log(md_table(df.set_index("couplings kept")))
    return df


def q4_shots(designs):
    log("\n# Shot reduction (noise-aware ridge, local readout-noise model)\n")
    rng = np.random.default_rng(1)
    rows = []
    for dname, (m, steps) in designs.items():
        qc = circuit(prune(m), steps)
        n2 = qc.count_ops().get("rxx", 0)
        s = S_REF ** (n2 / RXX_REF)
        Pn = bitflip(probs_for(Operator(qc).data), (1 - s) / 2)
        for zz in [False, True]:
            for lam in [1.0, 10.0]:
                for shots in [50, 100, 200, 300, 500, 1000]:
                    mses = []
                    for _ in range(5):
                        tr = np.array([rng.multinomial(shots, p) for p in Pn]) / shots
                        te = np.array([rng.multinomial(shots, p) for p in Pn[START:]]) / shots
                        Ftr, Fte = feats(tr, zz), feats(np.vstack([tr[:START], te]), zz)
                        mses.append(MSE(ridge_preds(Ftr, Fte, lam), YT))
                    rows.append({"design": dname, "RXX": n2, "s": s, "features": "Z + ZZ" if zz else "Z",
                                 "lambda": lam, "shots": shots, "MSE": np.mean(mses), "MSE sd": np.std(mses)})
        print(f"  {dname} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q4_shot_reduction.csv", index=False)
    best = df.loc[df.groupby(["design", "shots"])["MSE"].idxmin()]
    piv = best.pivot(index="design", columns="shots", values="MSE")
    log("Best of {Z, Z+ZZ} x {lambda 1, 10} per cell (mean of 5 shot draws). HAR = 0.1156, "
        "linear (lags 1-3) on QR1 inputs in Q2 table.\n")
    log(md_table(piv))
    log("\nWhich readout won (features, lambda):\n")
    log(md_table(best.assign(choice=best["features"] + ", " + best["lambda"].astype(str))
                 .pivot(index="design", columns="shots", values="choice"), "{}"))
    return df


def q5_validate():
    log("\n# Noise-model check against the IonQ cloud (aria-1) counts\n")
    path = Path("results/qpu/ionq_simulator-aria-1_shots1000_steps2/job.json")
    if not path.exists():
        log("(no cloud counts found)")
        return
    rec = json.loads(path.read_text())
    rows_ = [int(n.split("_row")[1].split("_")[0]) for n in rec["names"]]
    P = probs_for(Operator(circuit(J, 2)).data)
    Zx, ZZx = P[rows_] @ ZS, P[rows_] @ ZZS
    Zh, ZZh = [], []
    for c in rec["counts"]:
        tot = sum(c.values())
        p = np.zeros(2 ** NQ)
        for k, v in c.items():
            p[int(k.replace(" ", ""), 2)] += v / tot
        Zh.append(p @ ZS)
        ZZh.append(p @ ZZS)
    Zh, ZZh = np.array(Zh), np.array(ZZh)
    sz = (Zx.ravel() @ Zh.ravel()) / (Zx.ravel() @ Zx.ravel())
    szz = (ZZx.ravel() @ ZZh.ravel()) / (ZZx.ravel() @ ZZx.ravel())
    corr = np.corrcoef(Zx.ravel(), Zh.ravel())[0, 1]
    log(f"- 20 circuits, 1000 shots: Z shrink s = {sz:.3f}, ZZ shrink = {szz:.3f} (model assumes s^2 = {sz ** 2:.3f}); "
        f"corr(hardware Z, exact Z) = {corr:.3f}")


def q6_cost(designs, shots_list, n_test=12):
    log("\n# Cost of a hardware run (USD)\n")
    log(f"{n_test} test months; calibration circuits dropped (s is estimated from the test circuits "
        "themselves against their simulated features: no target data is used). Gate counts from the "
        "circuits above (+7 RY +RZ layers +10 measurements).\n")
    rows = []
    for dname, (m, steps) in designs.items():
        qc = circuit(prune(m), steps)
        ops = qc.count_ops()
        n2, n1 = ops.get("rxx", 0), ops.get("rz", 0) + N_IN
        for shots in shots_list:
            r = {"design": dname, "RXX": n2, "shots": shots}
            for pname, pr in PRICE.items():
                per = max(pr["minimum"], shots * (n2 * pr["g2"] + n1 * pr["g1"]))
                r[pname] = per * n_test
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q6_cost.csv", index=False)
    log(md_table(df.set_index("design"), "{:,.0f}"))
    log("\nPrices: Azure Quantum / Google listing for IonQ Forte (per 2-qubit gate-shot $0.001121, "
        "1-qubit $0.00022, minimum per circuit $25.79 without / $168.20 with error mitigation) and the "
        "IonQ research-credit resource estimator ($0.0003 per 2-qubit gate-shot, $1 minimum). "
        "Confirm with IonQ's dry-run estimate before submitting.")


def main():
    q1_q2()
    red = q3_reduction()
    designs = {"paper-like: 45 couplings, 2 steps": (45, 2), "45 couplings, 1 step": (45, 1),
               "15 couplings, 1 step": (15, 1), "10 couplings, 1 step": (10, 1), "5 couplings, 1 step": (5, 1)}
    shots = q4_shots(designs)
    q5_validate()
    q6_cost(designs, [100, 300, 1000])
    (OUT / "quantum_value_and_cost.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for steps, c in [(1, C1), (2, C2)]:
        d = red[red["Trotter steps"] == steps]
        ax.plot(d["RXX gates"], d["exact MSE"], marker="o", color=c, label=f"{steps} Trotter step(s)")
    ax.axhline(0.1156, color=INK, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], 0.1156, "HAR ", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("RXX gates per circuit (cost driver)")
    ax.set_ylabel("test MSE (noiseless, exact features)")
    ax.set_title("Circuit reduction: pruning couplings", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    best = shots.loc[shots.groupby(["design", "shots"])["MSE"].idxmin()]
    for (dname, g), c in zip(best.groupby("design", sort=False), [C1, C2, C3, C4, MUTED]):
        ax.plot(g["shots"], g["MSE"], marker="o", color=c, label=dname)
    ax.axhline(0.1156, color=INK, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("shots per circuit")
    ax.set_ylabel("test MSE (noise + shots)")
    ax.set_title("Shot reduction with noise-aware readout", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle("QR1, 10 qubits, K=1: cutting hardware cost", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "quantum_value_and_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
