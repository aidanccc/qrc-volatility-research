"""
locked_protocol.py — Choose every setting before seeing any test data, then score once.
=======================================================================================
The paper's coupling matrix, memory depth K and inputs look like the best of many
options tried on the 1997-2017 test window (D4 in overfitting_diagnostics.py). This
experiment removes that selection bias:

  1. Inputs are fixed in advance: FEATURES_HOLD (the 7 QR1/QR2 inputs that exist
     through 2026), so one locked model can be scored on both test windows.
  2. Every choice — coupling matrix, memory depth K, virtual nodes V, readout type
     and ridge lambda — is made on a VALIDATION window inside the paper's training
     period: one-month-ahead forecasts for 1987-08..1997-07 (120 months), each fit
     on the preceding 451 months. No month from 1997-08 onward is touched.
  3. The linear baseline gets the same treatment (lags 1..3, same readout options).
  4. The locked models are then scored exactly once on
        A) 1997-08..2017-12 (the paper's test window, 245 months), and
        B) 2018-01..2026-08 (true holdout, 104 months),
     with the paper's 571-month rolling window.
  5. As a guard against a lucky single pick, the average forecast of the top-5
     validation configurations is also reported.
  6. Input clipping (each input limited to its 1950-01..1997-07 range, known before
     any test month) is reported as a pre-specified variant of the locked model.

Writes results/locked/{summary.md, validation_grid.csv, test_scores.csv, test_mse.png}.

Usage:  python -m experiments.locked_protocol            (~5-10 min CPU)
        python -m experiments.locked_protocol --quick    (5 coupling matrices)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qrc.quantum_reservoir_qiskit import MSE
from qrc.fast_qrc import metrics
from experiments.overfitting_diagnostics import (FULL, FEATURES_HOLD, WI, C1, C2, C3, C4, INK, MUTED,
                                                 qrc_features, lagged, rolling_fit, diebold_mariano,
                                                 block_bootstrap_ci, md_table)

OUT = "results/locked"
Y = FULL["RV"].to_numpy(float)
VAL_START, VAL_N, VAL_WI = 451, 120, 451     # validation forecasts 1987-08..1997-07, 451-month window
PAPER_START, PAPER_N = 571, 245              # 1997-08..2017-12
HOLD_START, HOLD_N = 816, len(FULL) - 816    # 2018-01..2026-08
TRAIN_END = PAPER_START                      # rows [0, 571) = everything known before the paper window

# readout options: the paper's (raw features, no intercept, lambda=1e-8) and standardized ridge + intercept
READOUTS = [("paper", 1e-8)] + [("ridge", lam) for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0)]

SUMMARY = []


def log(s=""):
    print(s)
    SUMMARY.append(s)


def fit(F, start, n, wi, readout):
    kind, lam = readout
    if kind == "paper":
        return rolling_fit(F, Y, start, n, wi=wi, lam=lam)[0]
    return rolling_fit(F, Y, start, n, wi=wi, lam=lam, intercept=True, standardize=True)[0]


def clipped(df):
    lo = df.iloc[:TRAIN_END][FEATURES_HOLD].min()
    hi = df.iloc[:TRAIN_END][FEATURES_HOLD].max()
    out = df.copy()
    out[FEATURES_HOLD] = df[FEATURES_HOLD].clip(lo, hi, axis=1)
    return out


def score_windows(preds_by_model, ref):
    """preds_by_model: {name: {"A": preds, "B": preds}} -> tidy table with DM vs ref."""
    rows = []
    for win, start, n in [("A", PAPER_START, PAPER_N), ("B", HOLD_START, HOLD_N)]:
        y = Y[start:start + n]
        e_ref = preds_by_model[ref][win] - y
        for name, d in preds_by_model.items():
            p = d[win]
            m = metrics(p, y)
            r = {"window": "1997-2017" if win == "A" else "2018-2026", "model": name,
                 "MSE": m["MSE"], "QLIKE": m["QLIKE"]}
            if name != ref:
                dd = (p - y) ** 2 - e_ref ** 2
                r["dMSE vs " + ref] = MSE(p, y) - MSE(preds_by_model[ref][win], y)
                r["DM p"] = diebold_mariano(e_ref, p - y)[1]
                lo, hi = block_bootstrap_ci(dd)
                r["sign"] = "worse" if lo > 0 else ("better" if hi < 0 else "tie")
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n_mat = 5 if args.quick else 12
    Ks, Vs = [1, 2, 3, 4, 5], [1, 2]
    yv = Y[VAL_START:VAL_START + VAL_N]

    log("# Locked protocol: every setting chosen before any test month is seen\n")
    log(f"Inputs fixed in advance: {FEATURES_HOLD}. Validation = one-month-ahead forecasts "
        f"{FULL.index[VAL_START]:%Y-%m}..{FULL.index[VAL_START + VAL_N - 1]:%Y-%m} ({VAL_N} months, "
        f"{VAL_WI}-month window), entirely inside the paper's training period. Grid: {n_mat} coupling matrices x "
        f"K {Ks} x V {Vs} x {len(READOUTS)} readouts = {n_mat * len(Ks) * len(Vs) * len(READOUTS)} quantum configurations.\n")

    # ---- 1. validation grid (quantum) ----
    grid, feats_cache = [], {}
    for mi in range(n_mat):
        for K in Ks:
            for V in Vs:
                F = qrc_features(FULL, FEATURES_HOLD, V, mi, k_delay=K)
                feats_cache[(mi, K, V)] = F
                for ro in READOUTS:
                    grid.append({"matrix": mi, "K": K, "V": V, "readout": ro[0], "lambda": ro[1],
                                 "val MSE": MSE(fit(F, VAL_START, VAL_N, VAL_WI, ro), yv)})
        print(f"  matrix {mi} done ({time.time() - t0:.0f}s)")
    g = pd.DataFrame(grid).sort_values("val MSE").reset_index(drop=True)
    g.to_csv(f"{OUT}/validation_grid.csv", index=False)

    best = g.iloc[0]
    paper_like = g[(g.matrix == 1) & (g.K == 3) & (g.V == 2) & (g.readout == "paper")]
    log("## 1. What validation chose\n")
    log(md_table(g.head(8).set_index("matrix"), floatfmt="{:.4g}"))
    if len(paper_like):
        rank = paper_like.index[0] + 1
        log(f"\nThe paper's QR2 recipe (matrix 1, K=3, V=2, paper readout) ranks {rank} of {len(g)} on validation "
            f"(val MSE {paper_like['val MSE'].iloc[0]:.4f} vs best {best['val MSE']:.4f}).")
    for col in ["K", "V", "readout"]:
        top = g.head(20)[col].value_counts()
        log(f"- Top-20 configurations by {col}: " + ", ".join(f"{k}: {v}" for k, v in top.items()))

    # ---- 2. validation for the linear baseline ----
    lin_grid = []
    for L in (1, 2, 3):
        F = lagged(FULL, FEATURES_HOLD, list(range(1, L + 1)))
        for ro in READOUTS[1:]:                       # linear always gets an intercept
            lin_grid.append({"lags": L, "lambda": ro[1],
                             "val MSE": MSE(fit(F, VAL_START, VAL_N, VAL_WI, ro), yv)})
    lg = pd.DataFrame(lin_grid).sort_values("val MSE").reset_index(drop=True)
    lb = lg.iloc[0]
    log(f"\nLinear baseline chosen on validation: lags 1..{int(lb['lags'])}, ridge lambda {lb['lambda']:g} "
        f"(val MSE {lb['val MSE']:.4f}).")

    # ---- 3. score once on both test windows ----
    def q_preds(row, df=FULL):
        mi, K, V = int(row["matrix"]), int(row["K"]), int(row["V"])
        F = feats_cache[(mi, K, V)] if df is FULL else qrc_features(df, FEATURES_HOLD, V, mi, k_delay=K)
        ro = (row["readout"], row["lambda"])
        return {"A": fit(F, PAPER_START, PAPER_N, WI, ro), "B": fit(F, HOLD_START, HOLD_N, WI, ro)}

    models = {}
    models["Locked QRC (best on validation)"] = q_preds(best)
    top5 = [q_preds(g.iloc[i]) for i in range(5)]
    models["Locked QRC (top-5 average)"] = {w: np.mean([t[w] for t in top5], axis=0) for w in "AB"}
    models["Locked QRC, inputs clipped to 1950-1997 range"] = q_preds(best, clipped(FULL))
    if len(paper_like):
        models["Paper QR2 recipe (not locked, for reference)"] = q_preds(paper_like.iloc[0])
    Fl = lagged(FULL, FEATURES_HOLD, list(range(1, int(lb["lags"]) + 1)))
    lro = ("ridge", lb["lambda"])
    models["Locked linear (same inputs)"] = {"A": fit(Fl, PAPER_START, PAPER_N, WI, lro),
                                             "B": fit(Fl, HOLD_START, HOLD_N, WI, lro)}
    Flc = lagged(clipped(FULL), FEATURES_HOLD, list(range(1, int(lb["lags"]) + 1)))
    models["Locked linear, inputs clipped"] = {"A": fit(Flc, PAPER_START, PAPER_N, WI, lro),
                                               "B": fit(Flc, HOLD_START, HOLD_N, WI, lro)}
    for name, F in [("HAR", FULL[["RV1", "RV_q", "RV_a"]].to_numpy(float)), ("AR(1)", FULL[["RV1"]].to_numpy(float))]:
        models[name] = {"A": rolling_fit(F, Y, PAPER_START, PAPER_N, intercept=True)[0],
                        "B": rolling_fit(F, Y, HOLD_START, HOLD_N, intercept=True)[0]}
    rv1 = FULL["RV1"].to_numpy(float)
    models["Random walk"] = {"A": rv1[PAPER_START:PAPER_START + PAPER_N], "B": rv1[HOLD_START:HOLD_START + HOLD_N]}

    ref = "Locked linear (same inputs)"
    sc = score_windows(models, ref)
    sc.to_csv(f"{OUT}/test_scores.csv", index=False)
    log("\n## 2. Scored once on each test window\n")
    log(f"Reference for dMSE / Diebold-Mariano: {ref}. 'sign' uses a 12-month block-bootstrap 95% CI "
        f"(tie = interval includes 0).\n")
    for win in ["1997-2017", "2018-2026"]:
        t = sc[sc.window == win].drop(columns="window").set_index("model").sort_values("MSE")
        log(f"\n**{win}**\n")
        log(md_table(t, floatfmt="{:.4f}"))

    # ---- 4. figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    order = list(models)
    colors = {n: (C1 if "QRC" in n or "QR2" in n else C2 if "linear" in n else MUTED) for n in order}
    for ax, win in zip(axes, ["1997-2017", "2018-2026"]):
        t = sc[sc.window == win].set_index("model").loc[order]
        ax.barh(range(len(order)), t["MSE"], color=[colors[n] for n in order])
        ax.set_yticks(range(len(order)), order if ax is axes[0] else [""] * len(order), fontsize=8)
        ax.invert_yaxis()
        ax.axvline(t.loc[ref, "MSE"], color=INK, lw=1, ls="--")
        ax.set_title(win, loc="left", fontsize=11)
        ax.set_xlabel("test MSE (lower is better)")
        lo = t["MSE"].min()
        ax.set_xlim(lo * 0.9, t["MSE"].max() * 1.03)
    fig.suptitle("Locked protocol: settings chosen on 1987-1997 validation, scored once", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/test_mse.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")
    print(f"\nWrote {OUT}/summary.md ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
