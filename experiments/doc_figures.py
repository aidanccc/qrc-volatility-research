"""
doc_figures.py — Plain diagnostic graphs for the write-up (replacing tables with pictures)
=========================================================================================
Uses only light computation (two reservoir runs) plus saved results:

  F1 forecasts_vs_actual   locked quantum model and linear vs what actually happened, 1997-2026
  F2 rolling_loss          12-month rolling MSE over time (the "loss over time" view)
  F3 cumulative_loss_gap   running total of (linear error^2 - quantum error^2): who is ahead, when
  F4 error_distribution    predicted vs actual scatter + histogram of % errors (what MSE means)
  F5 shots_vs_error        forecast error vs shots: paper readout vs our regularized readout
  F6 locked_protocol       where the paper's settings rank among 300, and the locked test scores
  F7 trotter_steps         forecast error vs Trotter steps (circuit depth)

Writes results/figures/*.png.   Usage:  python -m experiments.doc_figures
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from qrc.quantum_reservoir_qiskit import MSE, MIN_RV, DIF
from experiments.overfitting_diagnostics import FULL, FEATURES_HOLD, WI, qrc_features, lagged, rolling_fit

OUT = "results/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 160, "savefig.bbox": "tight",
})

Y = FULL["RV"].to_numpy(float)
START, N = 571, len(FULL) - 571            # 1997-08 .. 2026-08, one continuous run
DATES = FULL.index[START:]
CRISES = [("2000-03", "2002-10", "dot-com"), ("2008-09", "2009-06", "financial crisis"),
          ("2020-02", "2020-06", "COVID"), ("2022-01", "2022-10", "rate shock")]
to_log = lambda v: (v + 1) * DIF + MIN_RV   # normalized -> log RV, as in the repo's metrics


def shade(ax, label=True):
    for a, b, name in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.7, lw=0, zorder=0)
        if label:
            ax.text(pd.Timestamp(a), 1.0, name, transform=ax.get_xaxis_transform(), fontsize=8,
                    color=MUTED, va="bottom")
    ax.axvline(pd.Timestamp("2018-01-01"), color=INK, lw=1, ls=":")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", name)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- forecasts (locked settings from results/locked; paper QR2 recipe for contrast) ----
    g = pd.read_csv("results/locked/validation_grid.csv")
    b = g.iloc[0]
    Fq = qrc_features(FULL, FEATURES_HOLD, int(b.V), int(b.matrix), k_delay=int(b.K))
    q = rolling_fit(Fq, Y, START, N, lam=b["lambda"], intercept=True, standardize=True)[0]
    Fl = lagged(FULL, FEATURES_HOLD, [1, 2, 3])
    lin = rolling_fit(Fl, Y, START, N, lam=100.0, intercept=True, standardize=True)[0]
    har = rolling_fit(FULL[["RV1", "RV_q", "RV_a"]].to_numpy(float), Y, START, N, intercept=True)[0]
    Fp = qrc_features(FULL, FEATURES_HOLD, 2, 1, k_delay=3)
    paper = rolling_fit(Fp, Y, START, N)[0]
    y = Y[START:]
    eq, el, eh, ep = to_log(q) - to_log(y), to_log(lin) - to_log(y), to_log(har) - to_log(y), to_log(paper) - to_log(y)

    # F1 forecasts vs actual
    fig, ax = plt.subplots(figsize=(11, 4.2))
    shade(ax)
    ax.plot(DATES, to_log(y), color=INK, lw=1.4, label="actual")
    ax.plot(DATES, to_log(q), color=BLUE, lw=1.6, label="quantum reservoir (locked settings)")
    ax.plot(DATES, to_log(lin), color=ORANGE, lw=1.4, ls="--", label="linear regression, same inputs")
    ax.set_ylabel("log realized volatility")
    ax.set_title("Forecasts vs what actually happened, 1997-2026")
    ax.text(pd.Timestamp("2018-03-01"), 0.03, "new data (never used to choose settings) →", transform=ax.get_xaxis_transform(),
            fontsize=8, color=TEXT2)
    ax.legend(loc="upper right", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    save(fig, "F1_forecasts_vs_actual")

    # F2 rolling 12-month loss
    roll = lambda e: pd.Series(e ** 2, index=DATES).rolling(12).mean()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    shade(ax)
    for e, c, ls, name in [(eq, BLUE, "-", "quantum (locked)"), (el, ORANGE, "--", "linear"), (eh, AQUA, ":", "HAR")]:
        r = roll(e)
        ax.plot(DATES, r, color=c, ls=ls, label=name)
    ax.set_ylabel("12-month rolling MSE")
    ax.set_title("Loss over time: every model fails in the same crises")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    save(fig, "F2_rolling_loss")

    # F3 cumulative loss gap
    fig, ax = plt.subplots(figsize=(11, 3.8))
    shade(ax)
    gap_l = np.cumsum(el ** 2 - eq ** 2)
    gap_p = np.cumsum(el ** 2 - ep ** 2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(DATES, gap_l, color=BLUE, label="quantum, locked settings")
    ax.plot(DATES, gap_p, color=BLUE, ls="--", alpha=0.6, label="quantum, paper's settings")
    ax.annotate("quantum ahead", xy=(0.005, 0.93), xycoords="axes fraction", fontsize=9, color=TEXT2)
    ax.annotate("linear ahead", xy=(0.005, 0.03), xycoords="axes fraction", fontsize=9, color=TEXT2)
    ax.set_ylabel("cumulative (linear − quantum)\nsquared error")
    ax.set_title("Running score: drifts near zero (a tie) until the paper's version collapses in 2020")
    ax.legend(loc="lower left", bbox_to_anchor=(0.12, 0))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    save(fig, "F3_cumulative_loss_gap")

    # F4 error distribution
    pct = (np.exp(eq) - 1) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lim = [to_log(y).min() - 0.1, to_log(y).max() + 0.1]
    a1.plot(lim, lim, color=MUTED, lw=1, ls="--")
    a1.scatter(to_log(y), to_log(q), s=16, color=BLUE, alpha=0.6, edgecolor=SURFACE, lw=0.5)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("actual log RV"); a1.set_ylabel("forecast log RV")
    a1.set_title("Forecast vs actual, 1997-2026")
    base = np.array([Y[t - WI:t].mean() for t in range(START, START + N)])     # rolling 571-month average forecast
    r2 = 1 - np.mean(eq ** 2) / np.mean((to_log(base) - to_log(y)) ** 2)
    a1.text(0.03, 0.92, f"out-of-sample R² ≈ {r2:.2f}\n(skill vs always forecasting the 571-month average)",
            transform=a1.transAxes, fontsize=9, color=TEXT2, va="top")
    bins = np.arange(-80, 160, 10)
    a2.hist(np.clip(pct, -80, 150), bins=bins, color=BLUE, edgecolor=SURFACE, lw=2)
    a2.axvline(0, color=INK, lw=1)
    med = np.median(np.abs(pct))
    a2.set_xlabel("forecast error in volatility (%)  (+ = forecast too high)")
    a2.set_ylabel("months")
    a2.set_title("How big the monthly misses are")
    a2.text(0.98, 0.92, f"median miss {med:.0f}%\n{np.mean(np.abs(pct) < 25):.0%} of months within ±25%\n"
            f"MSE {MSE(q, y):.3f}", transform=a2.transAxes, ha="right", va="top", fontsize=9, color=TEXT2)
    save(fig, "F4_error_distribution")

    # F5 shots vs error
    v = pd.read_csv(f"{OUT}/data/vikas_shot_sweep.csv")
    v = v[(v.model == "QR1") & (v.noise == "aria-1")]
    s = pd.read_csv("results/research/q4_shot_reduction.csv")
    s = s[(s.design == "paper-like: 45 couplings, 2 steps") & (s.features == "Z")].groupby("shots").MSE.min()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axhline(0.1156, color=AQUA, ls=":", lw=1.5)
    ax.text(1.6e6, 0.1156, " HAR benchmark", color=TEXT2, fontsize=8, va="center")
    ax.axhline(0.1051, color=MUTED, ls="--", lw=1)
    ax.text(1.6e6, 0.1051, " infinite shots", color=TEXT2, fontsize=8, va="center")
    ax.plot(v.shots, v.mse, color=ORANGE, marker="o", ms=6, label="paper's readout (no regularization) — Vikas's IonQ study")
    ax.plot(s.index, s.values, color=BLUE, marker="s", ms=6, label="our regularized, noise-aware readout")
    ax.axvline(400, color=INK, lw=1, ls=":")
    ax.text(420, 0.134, "400 shots: what our real\nIonQ Forte-1 run used", fontsize=9, color=TEXT2)
    ax.set_xscale("log")
    ax.set_xlim(40, 1.5e6)
    ax.set_xlabel("shots per circuit (log scale)")
    ax.set_ylabel("forecast MSE, 1997-2017 (QR1)")
    ax.set_title("Shots: the paper's readout needs ~1,000,000; a regularized one is done by ~500")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.62))
    save(fig, "F5_shots_vs_error")

    # F6 locked protocol
    sc = pd.read_csv("results/locked/test_scores.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    a1 = axes[0]
    a1.hist(g["val MSE"], bins=30, color=MUTED, edgecolor=SURFACE, lw=1.5)
    pv = g[(g.matrix == 1) & (g.K == 3) & (g.V == 2) & (g.readout == "paper")]["val MSE"].iloc[0]
    rank = int((g["val MSE"] < pv).sum()) + 1
    a1.axvline(g["val MSE"].min(), color=BLUE, lw=2)
    a1.axvline(pv, color=ORANGE, lw=2, ls="--")
    a1.text(g["val MSE"].min(), a1.get_ylim()[1] * 0.92, " chosen", color=TEXT2, fontsize=9)
    a1.text(pv, a1.get_ylim()[1] * 0.80, f"paper's settings\nrank {rank} of {len(g)} ", color=TEXT2, fontsize=9, ha="right")
    a1.set_xlabel("practice-run MSE, 1987-1997")
    a1.set_ylabel("number of model versions")
    a1.set_title(f"1. Rank {len(g)} versions on pre-1997 data")
    keep = [("Locked QRC (best on validation)", "quantum, locked", BLUE),
            ("Locked linear (same inputs)", "linear, same inputs", ORANGE), ("HAR", "HAR", AQUA),
            ("AR(1)", "AR(1)", MUTED), ("Paper QR2 recipe (not locked, for reference)", "quantum, paper's settings", INK)]
    for ax, w, title in [(axes[1], "1997-2017", "2. Test once: 1997-2017"), (axes[2], "2018-2026", "3. Test once: 2018-2026 (new data)")]:
        vals = [sc[(sc.window == w) & (sc.model == m)].MSE.iloc[0] for m, _, _ in keep]
        ax.barh(range(len(keep)), vals, color=[c for _, _, c in keep], height=0.6, edgecolor=SURFACE, lw=2)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, color=TEXT2)
        ax.set_yticks(range(len(keep)), [l for _, l, _ in keep] if ax is axes[1] else [""] * len(keep))
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("test MSE (lower is better)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "F6_locked_protocol")

    # F7 Trotter steps (numbers from the 4/16 sweep)
    steps = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for vals, exact, c, ls, name in [([0.1153, 0.1091, 0.1064], 0.1051, ORANGE, "--", "QR1"),
                                     ([0.1088, 0.1068, 0.1057], 0.1038, BLUE, "-", "QR2")]:
        ax.plot(steps, vals, color=c, ls=ls, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(exact, color=c, lw=1, alpha=0.5, ls=":")
        ax.text(4.1, vals[-1], f" {name}", color=TEXT2, va="center")
    ax.set_xticks(steps, ["1 step\n45 gates", "2 steps\n90 gates", "4 steps\n180 gates"])
    ax.set_xlim(0.7, 4.6)
    ax.set_ylabel("forecast MSE, 1997-2017")
    ax.set_title("Circuit depth: more Trotter steps → closer to exact (dotted)")
    save(fig, "F7_trotter_steps")


if __name__ == "__main__":
    main()
