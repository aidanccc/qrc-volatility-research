"""
run_research_suite.py — Hardware-readiness + data-validity experiments for QRC
==============================================================================

Built on fast_qrc.py (same physics as the exact simulator, ~2 s per run).
Writes CSVs, plots and a summary to results/research/.

Experiments
-----------
  E1  baselines     Is QRC actually better than simple models on the SAME
                    out-of-sample window?  (random walk, AR(1), HAR, linear
                    regression on QRC's own inputs, classical random reservoir)
                    + Diebold-Mariano tests + cumulative-loss plot over time.
  E2  couplings     Were ms[0]/ms[1] lucky draws?  Run all 100 coupling
                    matrices from coeff_10.jld2 and look at the distribution.
  E3  trotter       Forecast error vs Trotter steps, 1st order (Lie) vs
                    2nd order (Strang) — which gives the shallowest usable circuit?
  E4  shots         Finite measurement shots + readout error: how many shots
                    does a QPU need, and how much QPU time is that?
  E5  noise         Decoherence on the memory qubits (dephasing / T1 damping /
                    depolarizing) — does noise hurt, or help (noise-as-resource)?
  E6  native        Hardware-native reservoirs: line / ring couplings (no SWAPs
                    on IBM heavy-hex) vs the paper's all-to-all couplings.
  E8  memory       Ablation of memory depth K (1 = no memory) — is the
                    quantum memory doing any work?
  E7  circuit       Build the real per-time-step circuit (mid-circuit reset),
                    check it on Qiskit Aer against the fast simulator, transpile
                    to IBM heavy-hex and all-to-all (trapped ion) gate sets,
                    and measure how gate noise degrades the reservoir features.

Usage
-----
  python run_research_suite.py                 # everything (~15 min on CPU)
  python run_research_suite.py --only e1,e4    # a subset
  python run_research_suite.py --quick         # fewer seeds/matrices (~4 min)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from quantum_reservoir_qiskit import (
    load_coupling_matrices, generate_coupling_matrix, build_ising_hamiltonian, compute_unitaries, MSE,
)
from quantum_reservoir_trotter import build_trotter_evolution
from fast_qrc import (
    run_reservoir, forecast, metrics, build_hardware_circuit,
    TOTAL, L_OOS, LAMBDA, FEATURES_QR1, FEATURES_QR2,
)

OUT = "results/research"
NQ, TAU, K = 10, 1.0, 3
WI = TOTAL - L_OOS

# categorical slots (dataviz reference palette, light mode) + neutral ink
C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 10,
    "figure.dpi": 130, "savefig.bbox": "tight", "legend.frameon": False,
})

# ----------------------------------------------------------------------------
# shared data
# ----------------------------------------------------------------------------
DATA = pd.read_csv("data/Data.CSV", index_col=0, parse_dates=True)
TARGET = DATA["RV"].to_numpy(float)
Y = TARGET[WI:]
OOS_DATES = DATA.index[WI:]
MS = load_coupling_matrices("data/coeff_10.jld2")
CONFIGS = {"QR1": (FEATURES_QR1, 1, 0), "QR2": (FEATURES_QR2, 2, 1)}   # features, virtual nodes, ms index

SUMMARY = []


def log(s=""):
    print(s)
    SUMMARY.append(s)


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(rows)


def exact_unitaries(J, V):
    return compute_unitaries(build_ising_hamiltonian(NQ, J), TAU, V)


def qrc_run(J, features, V, U=None, dU=None, k_delay=K, **kw):
    if U is None:
        U, dU = exact_unitaries(J, V)
    sig = run_reservoir(DATA, features, U, dU, k_delay, V, NQ, **kw)
    return sig, forecast(sig, TARGET)


def sample_signal(sig, shots, readout_err, rng):
    """Finite-shot / readout-error version of an exact <Z> signal (post-processing only)."""
    valid = np.any(sig != 0, axis=0)
    p1 = (1 - sig) / 2
    p1 = p1 * (1 - readout_err) + (1 - p1) * readout_err
    if shots is not None:
        p1 = rng.binomial(shots, np.clip(p1, 0, 1)) / shots
    out = 1 - 2 * p1
    out[:, ~valid] = 0
    return out


def diebold_mariano(e1, e2, lag=6):
    """DM test, squared loss, Newey-West variance. Negative stat => model 1 better."""
    d = e1 ** 2 - e2 ** 2
    n, dc = len(d), d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * (1 - norm.cdf(abs(stat)))


# ----------------------------------------------------------------------------
# E1 — baselines on identical rolling window
# ----------------------------------------------------------------------------
def rolling_linear(Xmat, intercept=True):
    """Same rolling protocol as the QRC readout: train on rows j..j+WI-1, predict row WI+j."""
    if intercept:
        Xmat = np.column_stack([np.ones(len(Xmat)), Xmat])
    preds = np.zeros(L_OOS)
    for j in range(L_OOS):
        A, b = Xmat[j:j + WI], TARGET[j:j + WI]
        w = np.linalg.solve(A.T @ A + LAMBDA * np.eye(A.shape[1]), A.T @ b)
        preds[j] = Xmat[WI + j] @ w
    return preds


def lagged(features, lags):
    X = DATA[features].to_numpy(float)
    out = np.zeros((len(X), len(features) * len(lags)))
    for n, k in enumerate(lags):
        out[k:, n * len(features):(n + 1) * len(features)] = X[:-k]
    return out


def classical_reservoir(features, n_nodes, seed):
    """Fixed random nonlinear features of the same 3-lag window (echo-state-style, no recurrence)."""
    rng = np.random.default_rng(seed)
    Xl = lagged(features, [1, 2, 3])
    W = rng.normal(0, 1 / np.sqrt(Xl.shape[1]), (Xl.shape[1], n_nodes)) * 3
    b = rng.uniform(-1, 1, n_nodes)
    F = np.tanh(Xl @ W + b)
    F[:K] = 0
    return forecast(F.T, TARGET)


def e1_baselines(quick):
    log("\n## E1 — Is QRC better than simple baselines on the same test window?\n")
    preds = {}
    for name, (feats, V, mi) in CONFIGS.items():
        preds[name] = qrc_run(MS[mi], feats, V)[1]
    preds["Random walk"] = DATA["RV1"].to_numpy(float)[WI:]
    preds["AR(1)"] = rolling_linear(DATA[["RV1"]].to_numpy(float))
    preds["HAR"] = rolling_linear(DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float))
    preds["Linear (QR2 inputs, lag 1)"] = rolling_linear(lagged(FEATURES_QR2, [1]))
    preds["Linear (QR2 inputs, lags 1-3)"] = rolling_linear(lagged(FEATURES_QR2, [1, 2, 3]))
    seeds = range(10 if quick else 30)
    cr = np.array([classical_reservoir(FEATURES_QR2, 20, s) for s in seeds])
    cr_mse = np.array([MSE(p, Y) for p in cr])
    preds["Classical random reservoir (median seed)"] = cr[np.argsort(cr_mse)[len(cr_mse) // 2]]

    rows = {}
    e_qr2 = preds["QR2"] - Y
    for name, p in preds.items():
        m = metrics(p, Y)
        if name != "QR2":
            m["DM stat vs QR2"], m["DM p-value"] = diebold_mariano(e_qr2, p - Y)
        rows[name] = m
    df = pd.DataFrame(rows).T
    df.index.name = "Model"
    df = df.sort_values("MSE")
    df.to_csv(f"{OUT}/e1_baselines.csv")
    log(md_table(df[["MSE", "QLIKE", "MAE", "HitRate", "DM stat vs QR2", "DM p-value"]].fillna(np.nan)))
    log(f"\nClassical random reservoir, 20 nodes, {len(cr_mse)} seeds: MSE median {np.median(cr_mse):.4f}, "
        f"best {cr_mse.min():.4f}, worst {cr_mse.max():.4f}")
    log("DM stat < 0 with p < 0.05 => QR2 significantly better than that model (squared-error loss).")

    # cumulative squared-error advantage of QR2 over HAR and over linear-on-same-inputs
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, c in [("HAR", C1), ("Linear (QR2 inputs, lags 1-3)", C2), ("Random walk", C3)]:
        adv = np.cumsum((preds[name] - Y) ** 2 - e_qr2 ** 2)
        ax.plot(OOS_DATES, adv, color=c, label=f"vs {name}")
        ax.annotate(f"vs {name}", (OOS_DATES[-1], adv[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30"), color="#e6e5e0", alpha=0.7, lw=0)
    ax.text(pd.Timestamp("2008-10-01"), ax.get_ylim()[1] * 0.92, "2008 crisis", fontsize=8, color=MUTED)
    ax.set_title("Cumulative squared-error advantage of QR2 (above 0 = QR2 ahead)", loc="left", fontsize=11)
    ax.set_ylabel("cumulative SE difference")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(f"{OUT}/e1_cumulative_advantage.png")
    plt.close(fig)
    return preds


# ----------------------------------------------------------------------------
# E2 — coupling-matrix robustness
# ----------------------------------------------------------------------------
def e2_couplings(quick):
    log("\n## E2 — Were the paper's coupling matrices lucky draws?\n")
    n = 20 if quick else len(MS)
    rows = []
    t0 = time.time()
    for i in range(n):
        for name, (feats, V, _) in CONFIGS.items():
            p = qrc_run(MS[i], feats, V)[1]
            rows.append({"matrix": i, "config": name, "MSE": MSE(p, Y), "QLIKE": metrics(p, Y)["QLIKE"]})
        if i % 10 == 9:
            print(f"    {i + 1}/{n} matrices ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e2_coupling_robustness.csv", index=False)
    for name, (_, _, mi) in CONFIGS.items():
        s = df[df.config == name].set_index("matrix")["MSE"]
        rank = (s < s[mi]).sum() + 1
        log(f"- {name}: paper matrix ms[{mi}] MSE {s[mi]:.4f} ranks {rank}/{len(s)}; "
            f"distribution median {s.median():.4f}, 5-95% [{s.quantile(.05):.4f}, {s.quantile(.95):.4f}]")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(df.MSE.min(), df.MSE.max(), 30)
    for (name, (_, _, mi)), c in zip(CONFIGS.items(), [C1, C2]):
        s = df[df.config == name].set_index("matrix")["MSE"]
        ax.hist(s, bins=bins, color=c, alpha=0.55, label=f"{name} ({len(s)} random couplings)", edgecolor="white")
        ax.axvline(s[mi], color=c, lw=2, ls="--")
        ax.annotate(f"paper {name}", (s[mi], ax.get_ylim()[1] * 0.9), xytext=(3, 0), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("out-of-sample MSE")
    ax.set_ylabel("count")
    ax.set_title("Forecast MSE across all coupling matrices", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/e2_coupling_robustness.png")
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E3 — Trotter order / depth
# ----------------------------------------------------------------------------
def strang_evolution(J, t, n):
    """2nd-order (Strang) Trotter: [RZ(dt/2) . RXX(dt) . RZ(dt/2)]^n — same RXX count as 1st order."""
    qc, dt = QuantumCircuit(NQ), t / n
    for _ in range(n):
        for i in range(NQ):
            qc.rz(dt, i)                      # RZ(theta) = exp(-i theta/2 Z)  => exp(-i dt/2 Z)
        for i in range(NQ):
            for j in range(i + 1, NQ):
                if abs(J[i, j]) > 1e-12:
                    qc.rxx(2 * J[i, j] * dt, i, j)
        for i in range(NQ):
            qc.rz(dt, i)
    return qc


def trotter_circuits(J, V, n, order):
    if order == 1:
        return build_trotter_evolution(NQ, J, TAU, n, V)
    return strang_evolution(J, TAU, n), strang_evolution(J, TAU / V, n)


def e3_trotter(quick):
    log("\n## E3 — How shallow can the circuit be? (Trotter steps, 1st vs 2nd order)\n")
    steps = [1, 2, 3, 4, 6, 8]
    rows = []
    for name, (feats, V, mi) in CONFIGS.items():
        J = MS[mi]
        U_ex = expm(-1j * TAU * build_ising_hamiltonian(NQ, J))
        mse_exact = MSE(qrc_run(J, feats, V)[1], Y)
        for order in (1, 2):
            for n in steps:
                Uc, dUc = trotter_circuits(J, V, n, order)
                U, dU = Operator(Uc).data, Operator(dUc).data
                p = qrc_run(J, feats, V, U=U, dU=dU)[1]
                rows.append({"config": name, "order": order, "n_trotter": n,
                             "rxx_per_U": Uc.count_ops().get("rxx", 0), "depth_U": Uc.depth(),
                             "operator_error": np.linalg.norm(U_ex - U, 2),
                             "MSE": MSE(p, Y), "MSE_vs_exact_%": 100 * (MSE(p, Y) / mse_exact - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e3_trotter.csv", index=False)
    show = df.pivot_table(index=["config", "n_trotter"], columns="order", values=["operator_error", "MSE_vs_exact_%"])
    show.columns = [f"{a} (order {b})" for a, b in show.columns]
    show.index = [f"{c} n={n}" for c, n in show.index]
    show.index.name = "config"
    log(md_table(show, "{:.3f}"))
    for name in CONFIGS:
        for order in (1, 2):
            d = df[(df.config == name) & (df.order == order) & (df["MSE_vs_exact_%"].abs() < 5)]
            if len(d):
                log(f"- {name}, order {order}: minimum n_trotter within 5% of exact MSE = {d.n_trotter.min()}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, name in zip(axes, CONFIGS):
        for order, c in [(1, C1), (2, C2)]:
            d = df[(df.config == name) & (df.order == order)]
            ax.plot(d.n_trotter, d["MSE_vs_exact_%"], "o-", color=c, ms=5, label=f"order {order}")
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhspan(-5, 5, color="#e6e5e0", alpha=0.6, lw=0)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("Trotter steps per evolution")
    axes[0].set_ylabel("MSE change vs exact (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Forecast degradation from Trotterization (grey band = ±5%)", x=0.02, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/e3_trotter.png")
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E4 — shots and readout error
# ----------------------------------------------------------------------------
def e4_shots(quick):
    log("\n## E4 — How many measurement shots does a QPU need?\n")
    shot_list = [100, 300, 1000, 3000, 10000, 100000, None]
    readouts = [0.0, 0.01, 0.03]
    seeds = range(5 if quick else 20)
    rows = []
    for name, (feats, V, mi) in CONFIGS.items():
        sig = qrc_run(MS[mi], feats, V)[0]
        for ro in readouts:
            for shots in shot_list:
                for s in seeds:
                    rng = np.random.default_rng(1000 * s + 7)
                    p = forecast(sample_signal(sig, shots, ro, rng), TARGET)
                    rows.append({"config": name, "readout_err": ro, "shots": shots or np.inf,
                                 "seed": s, "MSE": MSE(p, Y), "QLIKE": metrics(p, Y)["QLIKE"]})
                    if shots is None and ro == 0:
                        break
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e4_shots.csv", index=False)
    agg = df.groupby(["config", "readout_err", "shots"]).MSE.agg(["mean", "std"]).reset_index()
    show = agg[agg.readout_err == 0].pivot(index="shots", columns="config", values="mean")
    show.index = [("exact" if np.isinf(s) else f"{int(s)}") for s in show.index]
    show.index.name = "shots (readout err 0)"
    log(md_table(show))
    har = MSE(rolling_linear(DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float)), Y)
    log(f"\nHAR MSE for reference: {har:.4f}")
    for name, (_, V, _) in CONFIGS.items():
        d = agg[(agg.config == name) & (agg.readout_err == 0)]
        ok = d[d["mean"] < har]
        need = ok.shots.min() if len(ok) else None
        if need is not None and np.isfinite(need):
            circuits = (TOTAL - K) * V
            total_shots = circuits * need
            log(f"- {name}: beats HAR from ~{int(need)} shots/circuit -> {circuits} circuits x {int(need)} shots "
                f"= {total_shots / 1e6:.2f}M shots per full backtest (~{total_shots * 250e-6 / 60:.0f} min QPU time at "
                f"~250 µs/shot incl. mid-circuit resets; rough estimate)")
        else:
            log(f"- {name}: does not beat HAR at any finite shot count tested")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (name, ro), c, ls in [(("QR2", 0.0), C1, "-"), (("QR2", 0.03), C1, ":"), (("QR1", 0.0), C2, "-"),
                              (("QR1", 0.03), C2, ":")]:
        d = agg[(agg.config == name) & (agg.readout_err == ro) & np.isfinite(agg.shots)]
        ax.errorbar(d.shots, d["mean"], yerr=d["std"], color=c, ls=ls, marker="o", ms=4, capsize=2,
                    label=f"{name}, readout err {ro:.0%}")
        ex = agg[(agg.config == name) & np.isinf(agg.shots)]["mean"].iloc[0]
        ax.axhline(ex, color=c, lw=1, alpha=0.4)
    ax.axhline(har, color=INK, lw=1.2, ls="--")
    ax.annotate("HAR", (100, har), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("shots per circuit")
    ax.set_ylabel("out-of-sample MSE")
    ax.set_title("Forecast error vs measurement shots (thin lines = exact expectation)", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/e4_shots.png")
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E5 — memory-qubit decoherence (noise as resource?)
# ----------------------------------------------------------------------------
def e5_noise(quick):
    log("\n## E5 — Does decoherence on the memory qubits hurt or help?\n")
    ps = [0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5] if not quick else [0.0, 0.03, 0.1, 0.3, 0.5]
    rows = []
    for name, (feats, V, mi) in CONFIGS.items():
        U, dU = exact_unitaries(MS[mi], V)
        for kind in ["dephasing", "amp_damping", "depolarizing"]:
            for p in ps:
                pr = qrc_run(MS[mi], feats, V, U=U, dU=dU, memory_noise=(kind, p))[1]
                rows.append({"config": name, "channel": kind, "p": p, "MSE": MSE(pr, Y),
                             "QLIKE": metrics(pr, Y)["QLIKE"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e5_noise.csv", index=False)
    show = df.pivot_table(index="p", columns=["config", "channel"], values="MSE")
    show.columns = [f"{c} {k}" for c, k in show.columns]
    log(md_table(show))
    for name in CONFIGS:
        base = df[(df.config == name) & (df.p == 0)].MSE.iloc[0]
        best = df[df.config == name].loc[lambda d: d.MSE.idxmin()]
        log(f"- {name}: noiseless MSE {base:.4f}; best {best.MSE:.4f} with {best.channel} p={best.p} "
            f"({100 * (best.MSE / base - 1):+.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, name in zip(axes, CONFIGS):
        for kind, c in [("dephasing", C1), ("amp_damping", C2), ("depolarizing", C3)]:
            d = df[(df.config == name) & (df.channel == kind)]
            ax.plot(d.p, d.MSE, "o-", color=c, ms=4, label=kind.replace("_", " "))
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("noise strength p per memory step")
    axes[0].set_ylabel("out-of-sample MSE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Memory-qubit decoherence vs forecast error", x=0.02, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/e5_noise.png")
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E6 — hardware-native coupling graphs
# ----------------------------------------------------------------------------
def graph_coupling(kind, seed):
    if kind == "all-to-all":
        return generate_coupling_matrix(NQ, 1.0, seed=10_000 + seed)
    rng = np.random.default_rng(20_000 + seed)
    m = np.zeros((NQ, NQ))
    edges = [(i, i + 1) for i in range(NQ - 1)] + ([(NQ - 1, 0)] if kind == "ring" else [])
    for i, j in edges:
        m[i, j] = m[j, i] = rng.random()
    return m / np.max(np.linalg.eigvalsh(m))        # same normalisation as coeff_matrix()


def e6_native(quick):
    log("\n## E6 — Hardware-native reservoirs (line / ring couplings)\n")
    seeds = range(8 if quick else 25)
    rows = []
    for kind in ["all-to-all", "ring", "line"]:
        for s in seeds:
            J = graph_coupling(kind, s)
            for name, (feats, V, _) in CONFIGS.items():
                p = qrc_run(J, feats, V)[1]
                rows.append({"graph": kind, "seed": s, "config": name, "MSE": MSE(p, Y),
                             "QLIKE": metrics(p, Y)["QLIKE"], "rxx_per_trotter_step": int((np.triu(J, 1) != 0).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e6_native_graphs.csv", index=False)
    show = df.groupby(["graph", "config"]).agg(MSE_median=("MSE", "median"), MSE_best=("MSE", "min"),
                                                QLIKE_median=("QLIKE", "median"),
                                                RXX_per_step=("rxx_per_trotter_step", "first")).reset_index()
    show.index = show.graph + " " + show.config
    show.index.name = "reservoir"
    log(md_table(show.drop(columns=["graph", "config"])))

    fig, ax = plt.subplots(figsize=(7, 3.3))
    groups = [(g, c) for g in ["all-to-all", "ring", "line"] for c in CONFIGS]
    data = [df[(df.graph == g) & (df.config == c)].MSE for g, c in groups]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=INK))
    for patch, (g, c) in zip(bp["boxes"], groups):
        patch.set_facecolor(C1 if c == "QR1" else C2)
        patch.set_alpha(0.6)
        patch.set_edgecolor("white")
    ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n{c}" for g, c in groups], fontsize=8)
    ax.set_ylabel("out-of-sample MSE")
    ax.set_title("Random reservoirs by coupling graph (line/ring need no SWAPs)", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/e6_native_graphs.png")
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------
# E7 — real circuits: Aer validation, transpilation, gate-noise feature fidelity
# ----------------------------------------------------------------------------
def z_from_counts(counts, nq):
    tot = sum(counts.values())
    z = np.zeros(nq)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]              # qubit 0 = rightmost char
        z += c * np.array([1 - 2 * int(b[i]) for i in range(nq)])
    return z / tot


def e7_circuit(quick):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    log("\n## E7 — The actual hardware circuit\n")
    name, (feats, V, mi) = "QR2", CONFIGS["QR2"]
    n_trot = 2
    J = MS[mi]
    Uc, dUc = trotter_circuits(J, V, n_trot, 2)
    X = DATA[feats].to_numpy(float)
    ls = list(range(300, 300 + (4 if quick else 10)))
    shots = 20000

    # (a) Aer shot-based run of the reset circuit vs fast simulator with the same Trotter unitaries
    sig = run_reservoir(DATA, feats, Operator(Uc).data, Operator(dUc).data, K, V, NQ)
    sim = AerSimulator(seed_simulator=11)
    circs = [build_hardware_circuit(X[l - K:l], Uc, dUc, len(feats), NQ, v + 1) for l in ls for v in range(V)]
    tc = transpile(circs, sim, optimization_level=0)
    res = sim.run(tc, shots=shots).result()
    diffs = []
    for n, (l, v) in enumerate([(l, v) for l in ls for v in range(V)]):
        z_aer = z_from_counts(res.get_counts(n), NQ)
        diffs.append(np.abs(z_aer - sig[v * NQ:(v + 1) * NQ, l]))
    diffs = np.array(diffs)
    log(f"(a) Circuit correctness: {len(circs)} circuits (mid-circuit reset protocol) on Aer, {shots} shots. "
        f"max |<Z>_Aer - <Z>_fastsim| = {diffs.max():.4f}, mean {diffs.mean():.4f} "
        f"(pure shot noise would give ~{np.sqrt(1 / shots):.4f} per value) -> "
        f"{'MATCHES' if diffs.max() < 5 / np.sqrt(shots) else 'MISMATCH'}")

    # (b) transpile one full time-step circuit to hardware gate sets
    qc = build_hardware_circuit(X[ls[0] - K:ls[0]], Uc, dUc, len(feats), NQ, V)
    rows = []
    targets = {
        "all-to-all, trapped-ion style (rxx native)": dict(basis_gates=["rxx", "rz", "ry", "rx", "reset", "measure"]),
        "IBM heavy-hex (ecr, sx, rz)": dict(basis_gates=["ecr", "rz", "sx", "x", "reset", "measure"],
                                            coupling_map=CouplingMap.from_heavy_hex(3)),
    }
    line_J = graph_coupling("line", 0)
    Ul, dUl = trotter_circuits(line_J, V, n_trot, 2)
    qc_line = build_hardware_circuit(X[ls[0] - K:ls[0]], Ul, dUl, len(feats), NQ, V)
    for label, circ in [("paper all-to-all J", qc), ("line J (E6)", qc_line)]:
        for tname, kw in targets.items():
            t = transpile(circ, optimization_level=1 if quick else 2, seed_transpiler=3, **kw)
            ops = t.count_ops()
            n2q = sum(v for k, v in ops.items() if k in ("rxx", "ecr", "cx", "cz"))
            rows.append({"reservoir": label, "target": tname, "2q_gates": n2q, "depth": t.depth(),
                         **{f"est_fidelity@{e:.1%}": (1 - e) ** n2q for e in (0.001, 0.003, 0.01)}})
    tdf = pd.DataFrame(rows)
    tdf.to_csv(f"{OUT}/e7_transpile.csv", index=False)
    tdf.index = tdf.reservoir + " -> " + tdf.target
    tdf.index.name = f"one QR2 time step, Strang n_trotter={n_trot}, V=2"
    log("\n(b) Gate cost per circuit (estimated fidelity = (1 - 2q error)^#2q gates):\n")
    log(md_table(tdf.drop(columns=["reservoir", "target"]), "{:.3g}"))

    # (c) gate noise: how much of the reservoir signal survives?
    log("\n(c) Feature survival under gate-level depolarizing noise (all-to-all basis, readout err 1%):\n")
    base_c = transpile(circs, basis_gates=["rxx", "rz", "ry", "rx", "reset", "measure"], optimization_level=1)
    ideal = np.concatenate([sig[v * NQ:(v + 1) * NQ, l] for l in ls for v in range(V)])
    frows = []
    for e2 in [0.0, 0.001, 0.003, 0.01]:
        nm = NoiseModel()
        if e2 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["rxx"])
            nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 10, 1), ["rz", "ry", "rx"])
        nm.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.01, 0.99]]))
        r = AerSimulator(noise_model=nm, seed_simulator=5).run(base_c, shots=4000).result()
        noisy = np.concatenate([z_from_counts(r.get_counts(n), NQ) for n in range(len(base_c))])
        slope = (noisy @ ideal) / (ideal @ ideal)
        frows.append({"2q error": e2, "corr(noisy, ideal)": np.corrcoef(noisy, ideal)[0, 1],
                      "signal retained (slope)": slope, "RMS error": np.sqrt(np.mean((noisy - ideal) ** 2))})
    fdf = pd.DataFrame(frows).set_index("2q error")
    fdf.to_csv(f"{OUT}/e7_gate_noise_features.csv")
    log(md_table(fdf))
    return tdf, fdf


# ----------------------------------------------------------------------------
# E8 — does the quantum memory matter?
# ----------------------------------------------------------------------------
def e8_memory(quick):
    log("\n## E8 — Does the reservoir's memory (K past inputs) matter?\n")
    rows = {}
    for kd in [1, 2, 3, 4, 5]:
        r = {}
        for name, (feats, V, mi) in CONFIGS.items():
            p = qrc_run(MS[mi], feats, V, k_delay=kd)[1]
            r[f"{name} MSE"], r[f"{name} QLIKE"] = MSE(p, Y), metrics(p, Y)["QLIKE"]
        rows[f"K={kd}"] = r
    df = pd.DataFrame(rows).T
    df.index.name = "memory depth"
    df.to_csv(f"{OUT}/e8_memory.csv")
    log(md_table(df))
    log("K=1 = no memory: the reservoir only sees last month's inputs (paper uses K=3).")
    return df


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e1,e2,e3,e4,e5,e6,e7,e8")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = args.only.lower().split(",")
    log("# QRC research suite results")
    log(f"Test window {OOS_DATES[0]:%Y-%m} .. {OOS_DATES[-1]:%Y-%m} ({L_OOS} months), rolling training window "
        f"{WI} months, {'quick' if args.quick else 'full'} mode.")
    exps = {"e1": e1_baselines, "e2": e2_couplings, "e3": e3_trotter, "e4": e4_shots,
            "e5": e5_noise, "e6": e6_native, "e7": e7_circuit, "e8": e8_memory}
    for key, fn in exps.items():
        if key in todo:
            t = time.time()
            print(f"\n>>> running {key} ...")
            fn(args.quick)
            print(f"    ({time.time() - t:.0f}s)")
    suffix = "" if args.only == ap.get_default("only") else "_" + "_".join(todo)
    with open(f"{OUT}/summary{suffix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")
    print(f"\nWrote {OUT}/summary{suffix}.md")


if __name__ == "__main__":
    main()
