"""
overfitting_diagnostics.py — Is QRC overfitted, and does the quantum circuit add anything?
==========================================================================================

Built on fast_qrc.py (same physics as the exact simulator). Writes CSVs, plots
and summary.md to results/diagnostics/.

The QRC readout is closed-form ridge regression (no iterative training), so the
"loss curves" here are the closed-form analogues:

  D1  train/test gap      in-sample vs out-of-sample MSE for every rolling fit,
                          compared with the OLS optimism expected from p/n alone
  D2  regularization path train & test MSE vs ridge lambda (capacity knob)
  D3  learning curve      train & test MSE vs training-window length
  D4  selection bias      how much of QR2's edge comes from picking the coupling
                          matrix / feature set / K that happened to do well on the
                          test window (paired QRC-vs-linear on random feature sets)
  D5  classical swap      replace the quantum circuit with minimal classical
                          models, identical inputs + readout + rolling protocol;
                          plus "how classical are the quantum features?" (R^2)
  D6  true holdout        2018-01..2026-08 from 1950-2026.csv, never used for any
                          model choice. MKT/STR are missing after 2017, so the
                          pre-specified holdout feature set is every QR1 u QR2
                          feature that exists there (7 = one per input qubit).

Usage
-----
  python -m experiments.overfitting_diagnostics            # everything (~10 min CPU)
  python -m experiments.overfitting_diagnostics --quick    # fewer seeds / subsets
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from qrc.quantum_reservoir_qiskit import load_coupling_matrices, build_ising_hamiltonian, compute_unitaries, MSE, COE
from qrc.fast_qrc import run_reservoir, metrics, FEATURES_QR1, FEATURES_QR2

OUT = "results/diagnostics"
NQ, TAU, K, LAMBDA = 10, 1.0, 3, 1e-8
WI, L_OOS = 571, 245                       # paper protocol: 571-month rolling window, 245 test months
CONFIGS = {"QR1": (FEATURES_QR1, 1, 0), "QR2": (FEATURES_QR2, 2, 1)}   # features, virtual nodes, ms index
FEATURES_HOLD = ["RV", "DP", "IP", "RV_q", "DEF", "EP", "INF"]          # QR1 u QR2 minus MKT, STR
CANDIDATES = ["MKT", "DP", "IP", "RV_q", "STR", "DEF", "EP", "INF", "SMB", "HML", "TB"]

# dataviz reference palette (light), validated: blue, orange, aqua, yellow + neutral ink
C1, C2, C3, C4, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#8a8984"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 10,
    "figure.dpi": 130, "savefig.bbox": "tight", "legend.frameon": False,
})

DATA = pd.read_csv("data/Data.CSV", index_col=0, parse_dates=True)
FULL = pd.read_csv("data/1950-2026.csv", index_col=0, parse_dates=True)
MS = load_coupling_matrices("data/coeff_10.jld2")
SUMMARY = []


def log(s=""):
    print(s)
    SUMMARY.append(s)


def md_table(df, floatfmt="{:.4f}"):
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------
# feature builders — every model returns a (T, p) matrix, row t uses data < t only
# ----------------------------------------------------------------------------
_UCACHE = {}


def unitaries(mi, V, zero_coupling=False):
    key = (mi, V, zero_coupling)
    if key not in _UCACHE:
        J = np.zeros((NQ, NQ)) if zero_coupling else MS[mi]
        _UCACHE[key] = compute_unitaries(build_ising_hamiltonian(NQ, J), TAU, V)
    return _UCACHE[key]


def qrc_features(df, features, V, mi, k_delay=K, zero_coupling=False):
    U, dU = unitaries(mi, V, zero_coupling)
    return run_reservoir(df, features, U, dU, k_delay, V, NQ).T


def lagged(df, features, lags):
    X = df[features].to_numpy(float)
    out = np.zeros((len(X), len(features) * len(lags)))
    for n, k in enumerate(lags):
        out[k:, n * len(features):(n + 1) * len(features)] = X[:-k]
    return out


def cos_random_features(df, features, n, seed):
    """Classical stand-in for the RY encoding: cos of random projections of the 3-lag window."""
    rng = np.random.default_rng(seed)
    Xl = lagged(df, features, [1, 2, 3])
    W = rng.normal(0, 1 / np.sqrt(Xl.shape[1]), (Xl.shape[1], n))
    F = np.cos(np.pi * Xl @ W + rng.uniform(0, 2 * np.pi, n))
    F[:K] = 0
    return F


def esn_features(df, features, n, seed, rho=0.9, leak=0.5, in_scale=0.5):
    """Minimal leaky echo-state network (recurrent tanh). State at row t has seen inputs up to t-1."""
    rng = np.random.default_rng(seed)
    X = df[features].to_numpy(float)
    Win = rng.uniform(-in_scale, in_scale, (n, X.shape[1]))
    W = rng.normal(0, 1, (n, n))
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h, F = np.zeros(n), np.zeros((len(X), n))
    for t in range(1, len(X)):
        h = (1 - leak) * h + leak * np.tanh(Win @ X[t - 1] + W @ h)
        F[t] = h
    return F


# ----------------------------------------------------------------------------
# rolling readout (generalises fast_qrc.forecast: any window, intercept, in-sample loss)
# ----------------------------------------------------------------------------
def rolling_fit(F, y, start, n_test, wi=WI, lam=LAMBDA, intercept=False, standardize=False):
    """Fit on rows [t-wi, t), predict row t, for t = start .. start+n_test-1.
    intercept=False + standardize=False + lam=1e-8 reproduces the paper's QRC readout exactly.
    Returns (predictions, in-sample MSE of each fit, on the same log-RV scale as MSE())."""
    preds, ins = np.zeros(n_test), np.zeros(n_test)
    for j in range(n_test):
        t = start + j
        A, b, x = F[t - wi:t], y[t - wi:t], F[t]
        if standardize:
            mu, sd = A.mean(0), A.std(0)
            sd[sd < 1e-12] = 1
            A, x = (A - mu) / sd, (x - mu) / sd
        if intercept:                                           # unpenalized intercept via centering
            am, bm = A.mean(0), b.mean()
            w = np.linalg.solve((A - am).T @ (A - am) + lam * np.eye(A.shape[1]), (A - am).T @ (b - bm))
            c = bm - am @ w
        else:
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
            c = 0.0
        preds[j] = x @ w + c
        ins[j] = np.mean((A @ w + c - b) ** 2) * COE        # same denormalized scale as MSE()
    return preds, ins


def diebold_mariano(e_ref, e_other, lag=6):
    """Squared-loss DM. Stat < 0 => reference model (first arg) is better."""
    d = e_ref ** 2 - e_other ** 2
    n, dc = len(d), d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def block_bootstrap_ci(d, block=12, n_boot=4000, seed=0):
    """Moving-block bootstrap 95% CI of mean(d)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    starts = rng.integers(0, n - block + 1, (n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    m = d[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


# shared model zoo on a given dataframe / window ---------------------------------
def model_zoo(df, features, configs, start, n_test, seeds):
    """Every model on identical inputs and rolling protocol. Returns {name: (preds, insample, n_params)}."""
    y = df["RV"].to_numpy(float)
    res = {}
    for name, (feats, V, mi) in configs.items():
        F = qrc_features(df, feats, V, mi)
        res[name] = (*rolling_fit(F, y, start, n_test), F.shape[1])
        res[f"{name} + intercept"] = (*rolling_fit(F, y, start, n_test, intercept=True), F.shape[1] + 1)
        F0 = qrc_features(df, feats, V, mi, zero_coupling=True)
        res[f"{name} with J=0 (no entanglement)"] = (*rolling_fit(F0, y, start, n_test, intercept=True), F0.shape[1])
    ref_V = list(configs.values())[-1][1]
    n_nodes = NQ * ref_V
    res["Random walk"] = (df["RV1"].to_numpy(float)[start:start + n_test], np.full(n_test, np.nan), 0)
    for name, F in [("AR(1)", df[["RV1"]].to_numpy(float)),
                    ("HAR", df[["RV1", "RV_q", "RV_a"]].to_numpy(float)),
                    ("Linear, same inputs, lag 1", lagged(df, features, [1])),
                    ("Linear, same inputs, lags 1-3", lagged(df, features, [1, 2, 3]))]:
        res[name] = (*rolling_fit(F, y, start, n_test, intercept=True), F.shape[1] + 1)
    for label, fn in [("Cosine random features", cos_random_features), ("Echo-state network", esn_features)]:
        runs = [rolling_fit(fn(df, features, n_nodes, s), y, start, n_test, intercept=True) for s in seeds]
        mses = np.array([MSE(p, y[start:start + n_test]) for p, _ in runs])
        med = np.argsort(mses)[len(mses) // 2]
        res[f"{label} ({n_nodes} nodes, median seed)"] = (*runs[med], n_nodes + 1)
        res[f"_{label} all seeds"] = mses
    return res


def score_table(res, y_test, ref):
    rows = {}
    e_ref = res[ref][0] - y_test
    for name, val in res.items():
        if name.startswith("_"):
            continue
        p, ins, npar = val
        m = metrics(p, y_test)
        r = {"MSE": m["MSE"], "QLIKE": m["QLIKE"], "MAE": m["MAE"], "params": npar,
             "train MSE": np.nanmean(ins), "test/train": m["MSE"] / np.nanmean(ins)}
        if name != ref:
            d = (p - y_test) ** 2 - e_ref ** 2
            r["DM stat vs " + ref], r["DM p"] = diebold_mariano(e_ref, p - y_test)
            lo, hi = block_bootstrap_ci(d)
            r["dMSE vs " + ref] = d.mean()
            r["95% CI lo"], r["95% CI hi"] = lo, hi
        rows[name] = r
    df = pd.DataFrame(rows).T
    df.index.name = "Model"
    return df.sort_values("MSE")


# ----------------------------------------------------------------------------
# D1 + D5 — train/test gap and classical replacement on the paper's test window
# ----------------------------------------------------------------------------
def d1_d5_main_window(quick):
    log("\n## D1 + D5 — Train/test gap, and replacing the quantum circuit (paper test window 1997-08..2017-12)\n")
    seeds = range(10 if quick else 30)
    y = DATA["RV"].to_numpy(float)
    start = len(DATA) - L_OOS
    res = model_zoo(DATA, FEATURES_QR2, CONFIGS, start, L_OOS, seeds)
    tab = score_table(res, y[start:], "QR2")
    tab.to_csv(f"{OUT}/d5_classical_swap.csv")
    log("Every model: same 245 test months, same 571-month rolling window, ridge readout. "
        "Non-QRC models use QR2's 7 inputs. dMSE > 0 => worse than QR2; CI = 12-month block bootstrap.\n")
    log(md_table(tab[["MSE", "QLIKE", "params", "train MSE", "test/train", "dMSE vs QR2", "95% CI lo",
                      "95% CI hi", "DM p"]]))
    for label in ["Cosine random features", "Echo-state network"]:
        m = res[f"_{label} all seeds"]
        log(f"- {label}: {len(m)} seeds, MSE median {np.median(m):.4f}, best {m.min():.4f}, "
            f"share of seeds beating QR2 {np.mean(m < tab.loc['QR2', 'MSE']):.0%}")
    p = res["QR2"][2]
    log(f"- Expected test/train ratio from parameter count alone (OLS, n={WI}, p={p}): "
        f"(n+p)/(n-p) = {(WI + p) / (WI - p):.3f}. Ratios far above this come from regime change "
        f"(the test window contains 2000-02 and 2008-09), not from the readout memorising noise.")

    # how classical are the quantum features? R^2 of each QR2 feature from classical transforms of the inputs
    F_q = qrc_features(DATA, FEATURES_QR2, 2, 1)[K:]
    Xl = lagged(DATA, FEATURES_QR2, [1, 2, 3])[K:]
    bases = {"linear in 3 lags": Xl,
             "+ cos/sin(pi x) of 3 lags": np.column_stack([Xl, np.cos(np.pi * Xl), np.sin(np.pi * Xl)])}
    rows = {}
    for bname, B in bases.items():
        B1 = np.column_stack([np.ones(len(B)), B])
        r2 = []
        for i in range(F_q.shape[1]):
            f = F_q[:, i]
            if f.std() < 1e-10:
                continue
            w = np.linalg.lstsq(B1, f, rcond=None)[0]
            r2.append(1 - np.var(f - B1 @ w) / np.var(f))
        rows[bname] = {"median R2": np.median(r2), "min R2": np.min(r2), "features": len(r2)}
    r2df = pd.DataFrame(rows).T
    r2df.index.name = "classical basis"
    r2df.to_csv(f"{OUT}/d5_quantum_feature_r2.csv")
    log("\nHow much of each QR2 reservoir feature is a simple classical function of the inputs (in-sample R^2):\n")
    log(md_table(r2df, "{:.3f}"))

    # D1 plot: train vs test MSE dumbbell
    show = tab.drop(index="Random walk")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(show) + 1.2))
    ys = np.arange(len(show))[::-1]
    for yy, (name, r) in zip(ys, show.iterrows()):
        ax.plot([r["train MSE"], r["MSE"]], [yy, yy], color="#d6d5cf", lw=2, zorder=1)
        ax.scatter(r["train MSE"], yy, s=50, color=C1, zorder=3, edgecolor="white", lw=1.5)
        ax.scatter(r["MSE"], yy, s=50, color=C2, zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"x{r['test/train']:.2f}", (r["MSE"], yy), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, show.index, fontsize=8)
    ax.scatter([], [], color=C1, label="train (in-sample) MSE")
    ax.scatter([], [], color=C2, label="test (out-of-sample) MSE")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    ax.set_xlabel("MSE")
    ax.set_title("D1  Train vs test error (label = test/train ratio)", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/d1_train_test_gap.png")
    plt.close(fig)

    # D1 over time: in-sample MSE of each rolling fit vs rolling 12m OOS MSE
    dates = DATA.index[start:]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for name, c in [("QR2", C1), ("Linear, same inputs, lags 1-3", C2)]:
        p, ins, _ = res[name]
        oos12 = pd.Series((p - y[start:]) ** 2 * COE, index=dates).rolling(12).mean()
        ax.plot(dates, ins, color=c, lw=1.5, ls="--")
        ax.plot(dates, oos12, color=c, lw=1.5, label=f"{name}")
    ax.plot([], [], color=MUTED, ls="--", label="dashed = in-sample MSE of that month's fit")
    ax.plot([], [], color=MUTED, label="solid = trailing 12-month test MSE")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("MSE")
    ax.set_title("D1  Train loss is flat; test loss spikes in crises (non-stationarity)", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/d1_loss_over_time.png")
    plt.close(fig)

    # D5 plot: test MSE by model, seeded models as strips
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(tab) + 1.4))
    ys = np.arange(len(tab))[::-1]
    for yy, (name, r) in zip(ys, tab.iterrows()):
        quantum = name.startswith("QR")
        ax.scatter(r["MSE"], yy, s=60, color=C1 if quantum else C2, zorder=3, edgecolor="white", lw=1.5)
        for label in ["Cosine random features", "Echo-state network"]:
            if name.startswith(label):
                m = res[f"_{label} all seeds"]
                ax.scatter(m, np.full(len(m), yy) + np.random.default_rng(0).uniform(-.18, .18, len(m)),
                           s=10, color=C2, alpha=0.35, zorder=2)
    ax.axvline(tab.loc["QR2", "MSE"], color=C1, lw=1, ls="--")
    ax.set_yticks(ys, tab.index, fontsize=8)
    ax.scatter([], [], color=C1, label="quantum reservoir")
    ax.scatter([], [], color=C2, label="classical (small dots = other random seeds)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_xlabel("out-of-sample MSE, 1997-08..2017-12 (dashed = QR2)")
    ax.set_title("D5  Quantum circuit replaced by minimal classical models", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/d5_classical_swap.png")
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# D2 — regularization path, D3 — learning curve
# ----------------------------------------------------------------------------
def d2_d3_curves(quick):
    log("\n## D2 — Regularization path (ridge lambda = capacity knob)\n")
    y = DATA["RV"].to_numpy(float)
    start = len(DATA) - L_OOS
    feats = {"QR2 (20 quantum features)": qrc_features(DATA, FEATURES_QR2, 2, 1),
             "Linear, QR2 inputs, lags 1-3 (21)": lagged(DATA, FEATURES_QR2, [1, 2, 3]),
             "Cosine random features (20)": cos_random_features(DATA, FEATURES_QR2, 20, 0)}
    lams = np.logspace(-8, 3, 12 if quick else 23)
    rows = []
    for name, F in feats.items():
        for lam in lams:
            p, ins = rolling_fit(F, y, start, L_OOS, lam=lam, intercept=True, standardize=True)
            rows.append({"model": name, "lambda": lam, "train MSE": ins.mean(), "test MSE": MSE(p, y[start:])})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/d2_regularization_path.csv", index=False)
    log("(Features standardized inside each training window, unpenalized intercept, so lambda is comparable.)\n")
    for name in feats:
        d = df[df.model == name]
        b = d.loc[d["test MSE"].idxmin()]
        log(f"- {name}: test MSE at lambda=1e-8 {d['test MSE'].iloc[0]:.4f}; best {b['test MSE']:.4f} "
            f"at lambda={b['lambda']:.2g} ({100 * (b['test MSE'] / d['test MSE'].iloc[0] - 1):+.1f}%)")
    log("If heavy regularization helped a lot, the readout would be overfitting. "
        "(Picking lambda on the test window is itself selection; this is a diagnostic only.)")

    fig, axes = plt.subplots(1, len(feats), figsize=(11, 3.3), sharey=True)
    for ax, name in zip(axes, feats):
        d = df[df.model == name]
        ax.plot(d["lambda"], d["train MSE"], color=C1, marker="o", ms=4, label="train")
        ax.plot(d["lambda"], d["test MSE"], color=C2, marker="o", ms=4, label="test")
        ax.set_xscale("log")
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("ridge lambda")
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=8)
    fig.suptitle("D2  Loss vs regularization strength", x=0.02, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/d2_regularization_path.png")
    plt.close(fig)

    log("\n## D3 — Learning curve (training-window length)\n")
    wis = [36, 60, 96, 144, 240, 360, 480, 571]
    rows = []
    for name, F in feats.items():
        for wi in wis:
            p, ins = rolling_fit(F, y, start, L_OOS, wi=wi, intercept=True)
            rows.append({"model": name, "window": wi, "train MSE": ins.mean(), "test MSE": MSE(p, y[start:])})
    df3 = pd.DataFrame(rows)
    df3.to_csv(f"{OUT}/d3_learning_curve.csv", index=False)
    show = df3.pivot(index="window", columns="model", values="test MSE")
    show.index.name = "train window (months)"
    log("Test MSE (same 245 test months) vs training-window length:\n")
    log(md_table(show))

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for (name, c) in zip(feats, [C1, C2, C3]):
        d = df3[df3.model == name]
        ax.plot(d.window, d["test MSE"], color=c, marker="o", ms=4, label=f"{name}: test")
        ax.plot(d.window, d["train MSE"], color=c, ls="--", lw=1.5)
    ax.plot([], [], color=MUTED, ls="--", label="dashed = train")
    ax.set_xscale("log")
    ax.set_ylim(0, min(0.4, df3["test MSE"].max() * 1.05))
    ax.set_xlabel("training window (months, log scale)")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.set_title("D3  Learning curves: gap closes as data grows", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/d3_learning_curve.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# D4 — selection bias
# ----------------------------------------------------------------------------
def d4_selection(quick, lin_mse, har_mse):
    log("\n## D4 — Selection bias: couplings, memory depth, feature set\n")
    y = DATA["RV"].to_numpy(float)
    start = len(DATA) - L_OOS
    e2 = pd.read_csv("results/research/e2_coupling_robustness.csv")
    for name, (_, _, mi) in CONFIGS.items():
        s = e2[e2.config == name].set_index("matrix")["MSE"]
        log(f"- {name} over {len(s)} coupling matrices: paper's ms[{mi}] ranks {(s < s[mi]).sum() + 1}/{len(s)} "
            f"(MSE {s[mi]:.4f}); median {s.median():.4f}; share beating same-input linear "
            f"({lin_mse:.4f}) {np.mean(s < lin_mse):.0%}; share beating HAR ({har_mse:.4f}) {np.mean(s < har_mse):.0%}")
    e8 = pd.read_csv("results/research/e8_memory.csv", index_col=0)
    log(f"- QR2 memory depth K=1..5 MSE: {', '.join(f'{v:.4f}' for v in e8['QR2 MSE'])} "
        f"(paper K=3 is the single best point)")

    # paired QRC vs linear across random 7-feature input sets (RV always included)
    rng = np.random.default_rng(42)
    n_sets = 15 if quick else 40
    rows = []
    t0 = time.time()
    seen = set()
    while len(rows) < n_sets:
        feats = ["RV"] + sorted(rng.choice(CANDIDATES, 6, replace=False).tolist())
        if tuple(feats) in seen:
            continue
        seen.add(tuple(feats))
        F = qrc_features(DATA, feats, 2, 1)
        pq = rolling_fit(F, y, start, L_OOS)[0]
        pqi = rolling_fit(F, y, start, L_OOS, intercept=True)[0]
        pl = rolling_fit(lagged(DATA, feats, [1, 2, 3]), y, start, L_OOS, intercept=True)[0]
        rows.append({"features": " ".join(feats), "QRC MSE": MSE(pq, y[start:]),
                     "QRC + intercept MSE": MSE(pqi, y[start:]), "Linear MSE": MSE(pl, y[start:])})
        if len(rows) % 10 == 0:
            print(f"    {len(rows)}/{n_sets} feature sets ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["QRC - Linear"] = df["QRC MSE"] - df["Linear MSE"]
    df.to_csv(f"{OUT}/d4_feature_sets.csv", index=False)
    log(f"- {len(df)} random 7-input sets (RV + 6 of {len(CANDIDATES)}), QR2 recipe vs linear on the SAME inputs: "
        f"QRC better in {np.mean(df['QRC - Linear'] < 0):.0%}; median QRC-linear MSE {df['QRC - Linear'].median():+.4f}; "
        f"QRC MSE median {df['QRC MSE'].median():.4f} vs linear {df['Linear MSE'].median():.4f}")
    log(f"- Same with an intercept added to the QRC readout (the paper's has none): QRC better in "
        f"{np.mean(df['QRC + intercept MSE'] < df['Linear MSE']):.0%}; median QRC+int MSE "
        f"{df['QRC + intercept MSE'].median():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    s = e2[e2.config == "QR2"].set_index("matrix")["MSE"]
    ax.hist(s, bins=25, color=C1, alpha=0.75, edgecolor="white")
    for v, lab, c in [(s[1], "paper ms[1]", C1), (lin_mse, "linear, same inputs", C2), (har_mse, "HAR", INK)]:
        ax.axvline(v, color=c, ls="--", lw=1.5)
        ax.annotate(lab, (v, ax.get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points", fontsize=8,
                    color=INK, rotation=90, va="top")
    ax.set_xlabel("QR2 test MSE")
    ax.set_ylabel("coupling matrices")
    ax.set_title(f"QR2 across {len(s)} random couplings", loc="left", fontsize=10)
    ax = axes[1]
    lim = [min(df["QRC MSE"].min(), df["Linear MSE"].min()) * 0.97, max(df["QRC MSE"].max(), df["Linear MSE"].max()) * 1.03]
    ax.plot(lim, lim, color=MUTED, lw=1)
    ax.scatter(df["Linear MSE"], df["QRC MSE"], s=36, color=C1, edgecolor="white", lw=1.2)
    ax.text(lim[0], lim[1], "  above line: linear wins", va="top", fontsize=8, color=MUTED)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("linear (same inputs) test MSE")
    ax.set_ylabel("QRC test MSE")
    ax.set_title("Random input sets: QRC vs linear, paired", loc="left", fontsize=10)
    fig.suptitle("D4  Selection bias", x=0.02, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/d4_selection_bias.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# D6 — true holdout 2018-2026
# ----------------------------------------------------------------------------
def d6_holdout(quick):
    log("\n## D6 — True holdout 2018-01..2026-08 (data never used for any modelling choice)\n")
    y = FULL["RV"].to_numpy(float)
    start = 816
    n_test = len(FULL) - start
    configs = {"QR1 recipe (holdout inputs)": (FEATURES_HOLD, 1, 0), "QR2 recipe (holdout inputs)": (FEATURES_HOLD, 2, 1)}
    seeds = range(10 if quick else 30)
    res = model_zoo(FULL, FEATURES_HOLD, configs, start, n_test, seeds)
    tab = score_table(res, y[start:], "QR2 recipe (holdout inputs)")
    tab.to_csv(f"{OUT}/d6_holdout.csv")
    log(f"Inputs (pre-specified, MKT/STR unavailable after 2017): {FEATURES_HOLD}. Coupling ms[0]/ms[1], V, K, lambda "
        f"exactly as in the paper. {n_test} test months.\n")
    ref = "QR2 recipe (holdout inputs)"
    log(md_table(tab[["MSE", "QLIKE", "train MSE", "test/train", f"dMSE vs {ref}", "95% CI lo", "95% CI hi", "DM p"]]))

    # same inputs on the paper window, for the before/after comparison
    old = model_zoo(DATA, FEATURES_HOLD, configs, len(DATA) - L_OOS, L_OOS, range(5))
    yo = DATA["RV"].to_numpy(float)[-L_OOS:]
    rows = {}
    for name in ["QR1 recipe (holdout inputs)", "QR2 recipe (holdout inputs)", "Linear, same inputs, lags 1-3", "HAR",
                 "AR(1)", "Random walk"]:
        rows[name] = {"MSE 1997-2017": MSE(old[name][0], yo), "MSE 2018-2026": MSE(res[name][0], y[start:])}
    cmp = pd.DataFrame(rows).T
    cmp.index.name = "Model (holdout inputs)"
    cmp.to_csv(f"{OUT}/d6_before_after.csv")
    log("\nSame inputs, paper test window vs true holdout:\n")
    log(md_table(cmp))

    # coupling robustness on holdout
    n_mat = 15 if quick else 40
    hm = []
    for i in range(n_mat):
        F = qrc_features(FULL, FEATURES_HOLD, 2, i)
        hm.append(MSE(rolling_fit(F, y, start, n_test)[0], y[start:]))
    hm = np.array(hm)
    lin = tab.loc["Linear, same inputs, lags 1-3", "MSE"]
    log(f"\n- QR2 recipe over {n_mat} coupling matrices on the holdout: median {np.median(hm):.4f}, "
        f"best {hm.min():.4f}; share beating same-input linear ({lin:.4f}) {np.mean(hm < lin):.0%}")
    pd.DataFrame({"matrix": range(n_mat), "MSE": hm}).to_csv(f"{OUT}/d6_holdout_couplings.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    dates = FULL.index[start:]
    ax.plot(dates, y[start:], color=INK, lw=1.2, label="realized (RV)")
    for name, c in [(ref, C1), ("Linear, same inputs, lags 1-3", C2), ("HAR", C3)]:
        ax.plot(dates, res[name][0], color=c, lw=1.5, label=name)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_ylabel("log RV (normalized)")
    ax.set_title("D6  Holdout forecasts 2018-2026", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/d6_holdout_forecasts.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log("# Overfitting diagnostics and classical-replacement test")
    t = time.time()
    res = d1_d5_main_window(args.quick)
    y = DATA["RV"].to_numpy(float)[-L_OOS:]
    d2_d3_curves(args.quick)
    d4_selection(args.quick, MSE(res["Linear, same inputs, lags 1-3"][0], y), MSE(res["HAR"][0], y))
    d6_holdout(args.quick)
    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")
    print(f"\nWrote {OUT}/summary.md ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
