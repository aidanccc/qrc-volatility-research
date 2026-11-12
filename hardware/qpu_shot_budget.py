"""
qpu_shot_budget.py — How many shots does a real-QPU run actually need?
======================================================================

Hardware deployment scheme: only the forecast (test) months are measured on the
QPU, so the QPU cost is (test months) x (virtual nodes) circuits, not 813. The
training-window features come from the simulator.

What the QPU returns for a month is modelled as
    z_hw = s * z_exact + finite-shot sampling noise,
where s < 1 is the feature shrinkage from gate noise. Vikas's IonQ study found
s ~ 0.73 (local Aria-1 model) to ~ 0.79 (IonQ cloud aria-1), QR1, 1-month history.

Three readout schemes, all on the paper's 245 test months:
  naive        readout trained on exact simulated features (paper ridge, lambda=1e-8)
  noise-aware  training features get the SAME simulated shrink + shot noise as
               the hardware (paper ridge, lambda=1e-8)
  noise-aware + ridge
               as above, standardized features, intercept, lambda=1 (fixed a
               priori, not tuned on the test window)
Plus a robustness case: train with s=0.73, hardware gives s=0.79 (noise-model
mismatch seen between the local model and IonQ's cloud).

Sweeps memory depth K (K=1 -> 10 qubits, no reset needed; K=3 -> 24-qubit
unrolled circuit) and shots. Writes results/diagnostics/qpu_shot_budget.* .
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qrc.quantum_reservoir_qiskit import MSE
from experiments.overfitting_diagnostics import (md_table, DATA, OUT, L_OOS, WI, LAMBDA, C1, C2, C3, C4, INK, MUTED,
                                     qrc_features, lagged, rolling_fit)
from qrc.fast_qrc import FEATURES_QR1, FEATURES_QR2

SHOTS = [100, 300, 1000, 3000, 10000, None]
S_HW = 0.79
N_SEEDS = 5
CONFIGS = {"QR1": (FEATURES_QR1, 1, 0), "QR2": (FEATURES_QR2, 2, 1)}
SCHEMES = ["naive", "noise-aware", "noise-aware + ridge", "noise-aware + ridge, s mismatch"]


def measure(F, s, shots, rng, K):
    """Simulated QPU readout of every row: shrink by s, then binomial shot noise."""
    Z = s * F
    if shots is not None:
        Z = 1 - 2 * rng.binomial(shots, np.clip((1 - Z) / 2, 0, 1)) / shots
    Z[:K] = 0
    return Z


def predict(F_train, F_test, y, start, n, ridge):
    """Rolling fit on F_train rows [t-WI, t), predict with F_test row t."""
    preds = np.zeros(n)
    for j in range(n):
        t = start + j
        A, b, x = F_train[t - WI:t], y[t - WI:t], F_test[t]
        if ridge:
            mu, sd = A.mean(0), A.std(0)
            sd[sd < 1e-12] = 1
            A, x = (A - mu) / sd, (x - mu) / sd
            bm = b.mean()
            w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ (b - bm))
            preds[j] = x @ w + bm
        else:
            w = np.linalg.solve(A.T @ A + LAMBDA * np.eye(A.shape[1]), A.T @ b)
            preds[j] = x @ w
    return preds


def main():
    y = DATA["RV"].to_numpy(float)
    start, n = len(DATA) - L_OOS, L_OOS
    yt = y[start:]
    lin = MSE(rolling_fit(lagged(DATA, FEATURES_QR2, [1, 2, 3]), y, start, n, intercept=True)[0], yt)
    har = MSE(rolling_fit(DATA[["RV1", "RV_q", "RV_a"]].to_numpy(float), y, start, n, intercept=True)[0], yt)
    rng = np.random.default_rng(0)
    rows = []
    for model, (feats, V, mi) in CONFIGS.items():
        for K in [1, 2, 3]:
            F = qrc_features(DATA, feats, V, mi, k_delay=K)
            for shots in SHOTS:
                for scheme in SCHEMES:
                    mses = []
                    for _ in range(1 if shots is None and scheme == "naive" else N_SEEDS):
                        hw = measure(F, S_HW, shots, rng, K)                  # what the QPU would give
                        if scheme == "naive":
                            tr = F
                        elif scheme.endswith("mismatch"):
                            tr = measure(F, 0.73, shots, rng, K)
                        else:
                            tr = measure(F, S_HW, shots, rng, K)              # independent simulated draw
                        mses.append(MSE(predict(tr, hw, y, start, n, "ridge" in scheme), yt))
                    rows.append({"model": model, "K": K, "qubits (unrolled)": 7 * K + 3, "circuits per month": V,
                                 "scheme": scheme, "shots": np.inf if shots is None else shots,
                                 "MSE mean": np.mean(mses), "MSE sd": np.std(mses)})
            print(f"{model} K={K} done")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/qpu_shot_budget.csv", index=False)

    lines = ["# QPU shot budget: simulator-trained readout, test months measured on hardware\n",
             f"245 test months (1997-08..2017-12). Hardware model: features shrunk by s={S_HW} (IonQ aria-1 cloud "
             f"level) plus binomial shot noise. Benchmarks: HAR {har:.4f}, linear on QR2 inputs {lin:.4f}. "
             f"Mean of {N_SEEDS} shot draws.\n"]
    for model in CONFIGS:
        for scheme in SCHEMES:
            piv = df[(df.model == model) & (df.scheme == scheme)].pivot(index="K", columns="shots", values="MSE mean")
            piv.columns = ["exact" if c == np.inf else f"{int(c):,}" for c in piv.columns]
            lines.append(f"\n**{model}, {scheme}** — test MSE by memory depth K (rows) and shots per circuit (columns)\n")
            lines.append(md_table(piv))
    lines.append("\nFewest shots per circuit that still beat HAR on average:\n")
    for model in CONFIGS:
        for scheme in SCHEMES[1:]:
            for K in [1, 2, 3]:
                sub = df[(df.model == model) & (df.K == K) & (df.scheme == scheme) & np.isfinite(df.shots)]
                ok = sub[sub["MSE mean"] < har].sort_values("shots")
                lines.append(f"- {model}, K={K} ({7 * K + 3} qubits), {scheme}: "
                             + (f"{int(ok.shots.iloc[0]):,}" if len(ok) else "not within 10,000"))
    txt = "\n".join(lines)
    print(txt)
    with open(f"{OUT}/summary_qpu_shot_budget.md", "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # figure: noise-aware + ridge, per model, MSE vs shots for each K
    xs = [s for s in SHOTS if s] + [30000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, model in zip(axes, CONFIGS):
        for K, c in zip([1, 2, 3], [C1, C2, C3]):
            for scheme, ls in [("noise-aware + ridge", "-"), ("noise-aware", ":")]:
                sub = df[(df.model == model) & (df.K == K) & (df.scheme == scheme)].sort_values("shots")
                ax.plot(xs, sub["MSE mean"], color=c, ls=ls, marker="o", ms=3,
                        label=f"K={K} ({7 * K + 3} qubits), {scheme}")
        ax.axhline(har, color=INK, lw=1, ls="--")
        ax.axhline(lin, color=C4, lw=1, ls="--")
        ax.text(105, har, "HAR", va="bottom", fontsize=8)
        ax.text(105, lin, "linear, same inputs", va="top", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{s:,}" for s in xs[:-1]] + ["exact"])
        ax.set_xlabel("shots per circuit")
        ax.set_title(f"{model}", loc="left", fontsize=10)
    axes[0].set_ylim(0.095, 0.16)
    axes[0].set_ylabel("test MSE, 245 months")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"QPU shot budget: noise-aware readout, hardware shrink s={S_HW}", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/qpu_shot_budget.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
