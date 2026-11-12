"""
ionq_feasibility.py — Can the QR2 reservoir run on IonQ hardware, and how small can the circuit get?
=====================================================================================================

Built on run_research_suite.py / fast_qrc.py (same data, couplings, readout).
Writes CSVs, a plot and a summary to results/hardware/.

  H1  variants   Candidate circuit reductions (Trotter order/steps, virtual nodes,
                 sparse or line couplings): IonQ gate counts and noiseless forecast MSE.
  H2  gate noise Full 813-step Aer density-matrix run of every variant under
                 IonQ-like depolarizing noise (2q error 0.1% / 0.4% / 0.6%) + readout error.
  H3  shots      Finite shots on top of the noisy features. The ridge penalty is
                 chosen on a validation split INSIDE the first training window
                 (months 1..571), never on the 1997-2017 test window.
  H4  cost       Circuits, shots and on-demand price for a full run and for a pilot.

IonQ mapping: RXX(theta) is one native MS gate (arbitrary angle), RZ is virtual
(free), RY is single-qubit GPi/GPi2.  So the 2-qubit gate count = RXX count.

Usage
-----
  python -m hardware.ionq_feasibility            # ~15-30 min (Aer runs dominate)
  python -m hardware.ionq_feasibility --quick    # 120 time steps per noisy run, MSE columns skipped
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qrc.quantum_reservoir_qiskit import rolling_ridge_regression, MSE
from qrc.fast_qrc import run_reservoir, z_signs
from experiments.run_research_suite import (
    DATA, TARGET, Y, MS, CONFIGS, NQ, TAU, K, WI, TOTAL, L_OOS,
    trotter_circuits, graph_coupling, md_table, C1, C2, C3, INK, MUTED,
)

OUT = "results/hardware"
FEATS, _, MI = CONFIGS["QR2"]
N_IN = len(FEATS)
X = DATA[FEATS].to_numpy(float)
STEPS = np.arange(K, TOTAL)                      # 813 time steps that need a circuit
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SHOTS = [300, 1000, 3000, 10000, None]
SEEDS = range(5)
READOUT = 0.005                                   # IonQ SPAM ~0.5%

# IonQ-like 2-qubit error rates (1q error = 2q / 20).  Published fidelities move;
# these bracket Aria (~99.4%), Forte (~99.6%) and the newest systems (~99.9%).
NOISE = {"ideal": 0.0, "2q 0.1% (newest)": 0.001, "2q 0.4% (Forte-like)": 0.004, "2q 0.6% (Aria-like)": 0.006}

# AWS Braket on-demand list prices (USD) at time of writing — check before budgeting.
PRICE = {"IonQ Aria": (0.30, 0.03), "IonQ Forte": (0.30, 0.08)}   # (per task, per shot)

SUMMARY = []


def log(s=""):
    print(s)
    SUMMARY.append(s)


# ----------------------------------------------------------------------------
# circuit variants
# ----------------------------------------------------------------------------
def top_couplings(J, m):
    """Keep only the m strongest couplings (no renormalisation: same field/coupling scale)."""
    a = np.abs(np.triu(J, 1))
    keep = a >= np.sort(a[a > 0])[::-1][m - 1]
    out = np.where(keep, J, 0.0)
    return out + out.T


def variants():
    J = MS[MI]
    return {
        # name: (J, V, n_trotter, order)
        "paper: all-to-all, Strang n=2, V=2": (J, 2, 2, 2),
        "all-to-all, Lie n=1, V=2": (J, 2, 1, 1),
        "all-to-all, Lie n=1, V=1": (J, 1, 1, 1),
        "top-15 couplings, Lie n=1, V=1": (top_couplings(J, 15), 1, 1, 1),
        "top-9 couplings, Lie n=1, V=1": (top_couplings(J, 9), 1, 1, 1),
        "line (9 couplings), Lie n=1, V=1": (graph_coupling("line", 0), 1, 1, 1),
    }


def step_circuit(x_window, Uc, dUc, v, save=True):
    """One time step: K encode/evolve rounds with RESET on the inputs, then v x dU."""
    qc = QuantumCircuit(NQ)
    for k, x in enumerate(x_window):
        if k:
            qc.reset(range(N_IN))
        for i, val in enumerate(x):
            qc.ry(float(val) * np.pi, i)
        if k < len(x_window) - 1:
            qc.compose(Uc, inplace=True)
        else:
            for _ in range(v):
                qc.compose(dUc, inplace=True)
    if save:
        qc.save_probabilities()
    return qc


def n2q(qc):
    return qc.count_ops().get("rxx", 0)


# ----------------------------------------------------------------------------
# readout
# ----------------------------------------------------------------------------
def pick_lambda(sig):
    """Rolling ridge on months 1..571 only: train 451, validate on the last 120."""
    best = None
    for lam in LAMBDAS:
        p = rolling_ridge_regression(sig[:, :WI], TARGET[:WI], WI, 120, 0, lam)
        e = MSE(p, TARGET[WI - 120:WI])
        if best is None or e < best[1]:
            best = (lam, e)
    return best[0]


def test_mse(sig, lam):
    return MSE(rolling_ridge_regression(sig, TARGET, TOTAL, L_OOS, 0, lam), Y)


def sample(p1, shots, rng):
    p1 = p1 * (1 - READOUT) + (1 - p1) * READOUT
    if shots is not None:
        p1 = rng.binomial(shots, np.clip(p1, 0, 1)) / shots
    return 1 - 2 * p1


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    steps = STEPS[:120] if args.quick else STEPS
    zs = z_signs(NQ)

    log("# IonQ feasibility of the QR2 reservoir\n")
    log(f"Test window {DATA.index[WI]:%Y-%m}..{DATA.index[-1]:%Y-%m}; readout error {READOUT:.1%}; "
        f"ridge lambda chosen on months 1..{WI} only.\n")

    # ---- H1: gate counts + noiseless MSE (Trotter error included) ----
    rows, circs = [], {}
    for name, (J, V, n, order) in variants().items():
        Uc, dUc = trotter_circuits(J, V, n, order)
        circs[name] = (Uc, dUc, V)
        sig = run_reservoir(DATA, FEATS, Operator(Uc).data, Operator(dUc).data, K, V, NQ)
        g = [n2q(step_circuit(X[100 - K:100], Uc, dUc, v + 1, save=False)) for v in range(V)]
        rows.append({"variant": name, "MS gates / circuit": max(g), "circuits / step": V,
                     "MS gates / step": sum(g), "noiseless MSE": test_mse(sig, 1e-8)})
    h1 = pd.DataFrame(rows).set_index("variant")
    log("## H1 — Circuit variants (noiseless, exact expectations, paper lambda 1e-8)\n")
    log(md_table(h1, "{:.4g}"))

    # ---- H2: Aer density-matrix runs under IonQ-like noise ----
    p1_cache = {}
    for name, (Uc, dUc, V) in circs.items():
        qcs = [step_circuit(X[l - K:l], Uc, dUc, v + 1) for l in steps for v in range(V)]
        for nlabel, e2 in NOISE.items():
            t = time.time()
            nm = NoiseModel()
            if e2 > 0:
                nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["rxx"])
                nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 20, 1), ["ry", "rz"])
            sim = AerSimulator(method="density_matrix", noise_model=nm, max_parallel_experiments=0)
            res = sim.run(transpile(qcs, sim, optimization_level=0)).result()
            probs = np.array([res.data(i)["probabilities"] for i in range(len(qcs))])
            p1 = probs @ ((1 - zs) / 2)                         # (n_circ, NQ) P(bit = 1)
            full = np.full((NQ * V, TOTAL), 0.5)
            full[:, steps] = p1.reshape(len(steps), V * NQ).T  # same row order as run_reservoir
            full[:, :K] = 0.5                                  # first K columns -> <Z> = 0
            p1_cache[(name, nlabel)] = full
            print(f"  aer {name} | {nlabel}: {time.time() - t:.0f}s")
    np.savez_compressed(f"{OUT}/h2_noisy_marginals.npz",
                        **{f"{a}||{b}": v for (a, b), v in p1_cache.items()})

    # feature survival: slope/corr of noisy <Z> vs ideal over the computed steps
    surv = []
    for (name, nlabel), p1 in p1_cache.items():
        ideal = 1 - 2 * p1_cache[(name, "ideal")][:, steps]
        noisy = 1 - 2 * p1[:, steps]
        surv.append({"variant": name, "noise": nlabel, "signal retained": (noisy * ideal).sum() / (ideal ** 2).sum(),
                     "corr": np.corrcoef(noisy.ravel(), ideal.ravel())[0, 1]})
    surv = pd.DataFrame(surv)
    surv.to_csv(f"{OUT}/h2_feature_survival.csv", index=False)
    log("\n## H2 — How much of the reservoir signal survives gate noise\n")
    log(md_table(surv.pivot(index="variant", columns="noise", values="signal retained")[list(NOISE)], "{:.3f}"))

    if args.quick:
        _write_summary()
        return

    # ---- H3: shots + readout on top of gate noise, lambda picked on training data ----
    rows = []
    for (name, nlabel), p1 in p1_cache.items():
        for shots in SHOTS:
            for s in (SEEDS if shots else [0]):
                sig = sample(p1, shots, np.random.default_rng(s))
                sig[:, :K] = 0
                lam = pick_lambda(sig)
                rows.append({"variant": name, "noise": nlabel, "shots": shots or np.inf, "seed": s,
                             "lambda": lam, "MSE": test_mse(sig, lam), "MSE (paper lambda)": test_mse(sig, 1e-8)})
    h3 = pd.DataFrame(rows)
    h3.to_csv(f"{OUT}/h3_shots_noise.csv", index=False)
    med = h3.groupby(["variant", "noise", "shots"]).MSE.median().reset_index()
    log("\n## H3 — Median test MSE (5 shot seeds), lambda chosen inside the training window\n")
    for nlabel in NOISE:
        t = med[med.noise == nlabel].pivot(index="variant", columns="shots", values="MSE")
        t.columns = [("exact" if np.isinf(c) else f"{int(c)} shots") for c in t.columns]
        t.index.name = nlabel
        log(md_table(t.loc[list(circs)]) + "\n")
    lin = pd.read_csv("results/diagnostics/d5_classical_swap.csv").set_index("Model").MSE
    log(f"References: linear same inputs lags 1-3 = {lin.iloc[0]:.4f}, QR2 paper exact = {lin['QR2']:.4f}, "
        f"HAR = {lin.get('HAR', np.nan):.4f}.")

    # ---- H4: cost ----
    rows = []
    for name, (_, _, V) in circs.items():
        for shots in (1000, 3000, 10000):
            for label, n_steps in (("full run (813 steps)", len(STEPS)), ("pilot (40 steps)", 40)):
                n_c = n_steps * V
                row = {"variant": name, "run": label, "shots": shots, "circuits": n_c, "total shots": n_c * shots}
                for dev, (pt, ps) in PRICE.items():
                    row[f"{dev} $"] = n_c * (pt + ps * shots)
                rows.append(row)
    h4 = pd.DataFrame(rows)
    h4.to_csv(f"{OUT}/h4_cost.csv", index=False)
    log("\n## H4 — On-demand cost (AWS Braket list price; credits/reservations differ)\n")
    show = h4[h4.shots == 1000].set_index("variant")
    log(md_table(show[show.run.str.startswith("full")].drop(columns=["run", "shots"]), "{:,.0f}"))
    log("")
    log(md_table(show[show.run.str.startswith("pilot")].drop(columns=["run", "shots"]), "{:,.0f}"))

    _plot(h1, med, lin)
    _write_summary()


def _plot(h1, med, lin):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    names = list(h1.index)
    ax.barh(range(len(names)), h1["MS gates / circuit"], color=C1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("2-qubit (MS) gates per circuit")
    ax.set_title("Circuit size", loc="left", fontsize=11)
    ax = axes[1]
    for name, c in zip(names, [INK, C1, C2, C3, MUTED, "#9b59b6"]):
        d = med[(med.variant == name) & (med.noise == "2q 0.4% (Forte-like)") & np.isfinite(med.shots)]
        ax.plot(d.shots, d.MSE, "o-", color=c, label=name, lw=1.5, ms=4)
    ax.axhline(lin.iloc[0], color=C2, ls="--", lw=1)
    ax.text(300, lin.iloc[0], " linear, same inputs", va="bottom", fontsize=8, color=C2)
    ax.set_xscale("log")
    ax.set_xlabel("shots per circuit")
    ax.set_ylabel("test MSE 1997-2017")
    ax.set_title("Forte-like noise (0.4% 2q) + 0.5% readout", loc="left", fontsize=11)
    ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/h_feasibility.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def _write_summary():
    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")


if __name__ == "__main__":
    main()
