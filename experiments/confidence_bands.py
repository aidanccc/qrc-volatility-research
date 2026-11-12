"""
confidence_bands.py — D7 repeated trials + D8 year-by-year error analysis
=========================================================================

D7  Repeated trials with confidence bands.
    The QRC has one random ingredient: the Ising coupling matrix J. We redraw it
    N_RUNS times (the paper used one draw) and refit with the exact paper recipe,
    then record train and test MSE for every run. The echo-state network is
    repeated with N_RUNS seeds as the classical analogue. Deterministic models
    (linear, HAR) get a band from a 12-month block bootstrap of the test months,
    which measures the uncertainty that comes from the data itself.
    Two windows: paper test window 1997-08..2017-12 (paper's QR2 inputs) and
    true holdout 2018-01..2026-08 (7 inputs available through 2026).

D8  Year-by-year error, 1997-08..2026-08, one continuous run.
    Same 7 inputs throughout so the years are comparable; QR2 band = 5-95%
    across the N_RUNS coupling draws.

Writes CSVs, PNGs and summary_d7_d8.md to results/diagnostics/.
Usage:  python -m experiments.confidence_bands   (~2-4 min CPU)
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qrc.quantum_reservoir_qiskit import MSE, COE
from experiments.overfitting_diagnostics import (DATA, FULL, OUT, L_OOS, FEATURES_HOLD, C1, C2, C3, C4, INK, MUTED,
                                     qrc_features, esn_features, lagged, rolling_fit, block_bootstrap_ci,
                                     md_table)
from qrc.fast_qrc import FEATURES_QR2

N_RUNS = 20
PAPER_MI = 1                       # QR2 coupling used in the paper
EVENTS = [("2000-03", "Dot-com\ncrash"), ("2008-09", "Lehman /\nGFC"), ("2011-08", "Euro debt /\nUS downgrade"),
          ("2018-02", "Volmageddon"), ("2020-03", "COVID-19"), ("2022-03", "Fed hikes /\ninflation")]
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def band(x):
    return np.percentile(x, 5), np.percentile(x, 95)


# ----------------------------------------------------------------------------
# D7 — repeated trials
# ----------------------------------------------------------------------------
def repeated_trials(df, features, start, n_test, label):
    y = df["RV"].to_numpy(float)
    yt = y[start:start + n_test]
    rows = []
    for i in range(N_RUNS):
        p, ins = rolling_fit(qrc_features(df, features, 2, i), y, start, n_test)          # exact paper readout
        rows.append({"window": label, "model": "QR2 (coupling draw)", "run": i, "train MSE": ins.mean(),
                     "test MSE": MSE(p, yt), "paper draw": i == PAPER_MI})
        p, ins = rolling_fit(esn_features(df, features, 20, i), y, start, n_test, intercept=True)
        rows.append({"window": label, "model": "Echo-state network (seed)", "run": i, "train MSE": ins.mean(),
                     "test MSE": MSE(p, yt), "paper draw": False})
    det = {"Linear, same inputs, lags 1-3": lagged(df, features, [1, 2, 3]),
           "HAR": df[["RV1", "RV_q", "RV_a"]].to_numpy(float)}
    boot = {}
    for name, F in det.items():
        p, ins = rolling_fit(F, y, start, n_test, intercept=True)
        e2 = (p - yt) ** 2 * COE
        boot[name] = block_bootstrap_ci(e2)
        rows.append({"window": label, "model": name, "run": 0, "train MSE": ins.mean(), "test MSE": e2.mean(),
                     "paper draw": False})
    # data-volatility band for the paper's own QR2 draw
    p, _ = rolling_fit(qrc_features(df, features, 2, PAPER_MI), y, start, n_test)
    boot["QR2 (paper draw)"] = block_bootstrap_ci((p - yt) ** 2 * COE)
    return pd.DataFrame(rows), boot


def d7():
    log("# D7 — Repeated trials: train vs test error with confidence bands\n")
    log(f"QR2 refit with {N_RUNS} independent coupling-matrix draws (the paper reports one); echo-state network "
        f"refit with {N_RUNS} seeds. Model band = 5-95% across runs. Data band = 95% CI from a 12-month block "
        f"bootstrap of the test months.\n")
    windows = [(DATA, FEATURES_QR2, len(DATA) - L_OOS, L_OOS, "1997-08..2017-12 (paper window)"),
               (FULL, FEATURES_HOLD, 816, len(FULL) - 816, "2018-01..2026-08 (true holdout)")]
    allrows, boots = [], {}
    for df, feats, start, n, label in windows:
        t = time.time()
        r, b = repeated_trials(df, feats, start, n, label)
        allrows.append(r)
        boots[label] = b
        print(f"  {label}: {time.time() - t:.0f}s")
    res = pd.concat(allrows, ignore_index=True)
    res["test/train"] = res["test MSE"] / res["train MSE"]
    res.to_csv(f"{OUT}/d7_repeated_trials.csv", index=False)

    summ = []
    for (w, m), g in res.groupby(["window", "model"], sort=False):
        lo, hi = band(g["test MSE"]) if len(g) > 1 else boots[w][m]
        summ.append({"window": w, "model": m, "runs": len(g), "train MSE": g["train MSE"].mean(),
                     "test MSE mean": g["test MSE"].mean(), "test MSE sd": g["test MSE"].std() if len(g) > 1 else np.nan,
                     "band lo": lo, "band hi": hi, "band type": "5-95% across runs" if len(g) > 1 else "block bootstrap 95%",
                     "test/train": g["test/train"].mean()})
        if m.startswith("QR2"):
            lo, hi = boots[w]["QR2 (paper draw)"]
            pd_mse = g.loc[g["paper draw"], "test MSE"].iloc[0]
            summ.append({"window": w, "model": "QR2 (paper draw only)", "runs": 1,
                         "train MSE": g.loc[g["paper draw"], "train MSE"].iloc[0], "test MSE mean": pd_mse,
                         "test MSE sd": np.nan, "band lo": lo, "band hi": hi, "band type": "block bootstrap 95%",
                         "test/train": g.loc[g["paper draw"], "test/train"].iloc[0]})
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}/d7_repeated_trials_summary.csv", index=False)
    for w in summ.window.unique():
        log(f"\n### {w}\n")
        log(md_table(summ[summ.window == w].drop(columns="window").set_index("model")))
    for w in res.window.unique():
        q = res[(res.window == w) & (res.model == "QR2 (coupling draw)")]
        lin = res[(res.window == w) & (res.model == "Linear, same inputs, lags 1-3")]["test MSE"].iloc[0]
        har = res[(res.window == w) & (res.model == "HAR")]["test MSE"].iloc[0]
        log(f"\n- {w}: QR2 draws beating linear {np.mean(q['test MSE'] < lin):.0%}, beating HAR "
            f"{np.mean(q['test MSE'] < har):.0%}; paper draw ranks "
            f"{int((q['test MSE'] < q.loc[q['paper draw'], 'test MSE'].iloc[0]).sum()) + 1}/{len(q)}")

    # figure: per window, train (hollow) and test (filled) per model with bands
    order = ["QR2 (coupling draw)", "Echo-state network (seed)", "Linear, same inputs, lags 1-3", "HAR"]
    colors = {order[0]: C1, order[1]: C2, order[2]: C3, order[3]: C4}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, w in zip(axes, res.window.unique()):
        for j, m in enumerate(order):
            g = res[(res.window == w) & (res.model == m)]
            yy = len(order) - 1 - j
            jit = rng.uniform(-0.12, 0.12, len(g)) if len(g) > 1 else np.zeros(1)
            s = summ[(summ.window == w) & (summ.model == m)].iloc[0]
            ax.plot([s["band lo"], s["band hi"]], [yy, yy], color=colors[m], lw=7, alpha=0.22, solid_capstyle="butt")
            ax.scatter(g["train MSE"], yy + 0.28 + jit * 0.5, s=18, facecolor="white", edgecolor=colors[m], lw=1.2)
            ax.scatter(g["test MSE"], yy + jit, s=22, color=colors[m], edgecolor="white", lw=0.6, zorder=3)
            if m.startswith("QR2"):
                pdm = g[g["paper draw"]]
                ax.scatter(pdm["test MSE"], [yy], marker="D", s=46, color=INK, zorder=4)
        ax.set_title(w, loc="left", fontsize=10)
        ax.set_xlabel("MSE (log RV)")
    axes[0].set_yticks(range(len(order))[::-1], ["QR2\n20 coupling draws", "Echo-state net\n20 seeds",
                                                 "Linear\n(same inputs)", "HAR"], fontsize=8)
    axes[1].scatter([], [], facecolor="white", edgecolor=MUTED, label="train MSE (hollow, raised)")
    axes[1].scatter([], [], color=MUTED, label="test MSE (filled)")
    axes[1].scatter([], [], marker="D", color=INK, label="the paper's coupling draw")
    axes[1].plot([], [], color=MUTED, lw=7, alpha=0.3, label="band: 5-95% of runs / bootstrap 95% CI")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("D7  Repeated trials: train vs test error, with confidence bands", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/d7_repeated_trials.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# D8 — year-by-year error, continuous 1997-2026
# ----------------------------------------------------------------------------
def d8():
    log("\n# D8 — Year-by-year forecast error, 1997-08..2026-08 (continuous run, 7 inputs available throughout)\n")
    df, feats, start = FULL, FEATURES_HOLD, 571
    y = df["RV"].to_numpy(float)
    n = len(df) - start
    yt = y[start:]
    dates = df.index[start:]
    preds = {}
    qr2_runs = np.array([rolling_fit(qrc_features(df, feats, 2, i), y, start, n)[0] for i in range(N_RUNS)])
    preds["QR2 (paper draw)"] = qr2_runs[PAPER_MI]
    preds["QR1 (paper draw)"] = rolling_fit(qrc_features(df, feats, 1, 0), y, start, n)[0]
    preds["Linear, same inputs, lags 1-3"] = rolling_fit(lagged(df, feats, [1, 2, 3]), y, start, n, intercept=True)[0]
    preds["HAR"] = rolling_fit(df[["RV1", "RV_q", "RV_a"]].to_numpy(float), y, start, n, intercept=True)[0]
    preds["AR(1)"] = rolling_fit(df[["RV1"]].to_numpy(float), y, start, n, intercept=True)[0]
    yr = dates.year
    sq = pd.DataFrame({k: (v - yt) ** 2 * COE for k, v in preds.items()}, index=dates)
    yearly = sq.groupby(yr).mean()
    runs_sq = (qr2_runs - yt) ** 2 * COE
    runs_year = pd.DataFrame(runs_sq.T, index=dates).groupby(yr).mean()
    yearly["QR2 band 5%"] = runs_year.quantile(0.05, axis=1)
    yearly["QR2 band 95%"] = runs_year.quantile(0.95, axis=1)
    yearly["months"] = sq.groupby(yr).size()
    yearly["realized RV mean"] = pd.Series(yt, index=dates).groupby(yr).mean()
    models = list(preds)
    yearly["best model"] = yearly[models].idxmin(axis=1)
    yearly.index.name = "year"
    yearly.to_csv(f"{OUT}/d8_yearly_error.csv")
    log("Mean squared error per calendar year (lower is better). 1997 = Aug-Dec only, 2026 = Jan-Aug only.\n")
    log(md_table(yearly[models + ["QR2 band 5%", "QR2 band 95%", "best model"]]))
    wins = yearly["best model"].value_counts()
    log("\nYears won (lowest MSE): " + ", ".join(f"{k} {v}" for k, v in wins.items()))
    d = yearly["QR2 (paper draw)"] - yearly["Linear, same inputs, lags 1-3"]
    log(f"QR2 beats same-input linear in {int((d < 0).sum())}/{len(d)} years. Largest QR2 losses: "
        + ", ".join(f"{y_} ({v:+.3f})" for y_, v in d.sort_values(ascending=False).head(3).items()))
    share = sq.loc["2020-01":"2020-12"].sum() / sq.sum()
    log("Share of each model's total 1997-2026 squared error that comes from 2020 alone: "
        + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))

    # figure: top = realized RV with events, bottom = yearly MSE with QR2 band
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=False, gridspec_kw={"height_ratios": [1, 1.25]})
    a0.plot(dates, yt, color=INK, lw=1)
    a0.axvspan(pd.Timestamp("2018-01-01"), dates[-1], color=C4, alpha=0.10)
    a0.text(pd.Timestamp("2018-03-01"), a0.get_ylim()[1], "true holdout", va="top", fontsize=8, color=MUTED)
    for k, (d_, lab) in enumerate(EVENTS):
        t = pd.Timestamp(d_)
        a0.axvline(t, color=MUTED, lw=0.8, ls=":")
        a0.annotate(lab, (t, yt.min()), xytext=(2, 2 + 24 * (k % 2)), textcoords="offset points", fontsize=7,
                    color=MUTED)
    a0.set_ylabel("log RV (normalized)")
    a0.set_title("S&P 500 realized volatility (target) and major market events", loc="left", fontsize=10)
    x = yearly.index.to_numpy()
    a1.fill_between(x, yearly["QR2 band 5%"], yearly["QR2 band 95%"], color=C1, alpha=0.18, lw=0,
                    label=f"QR2, 5-95% across {N_RUNS} coupling draws")
    for m, c, ls in [("QR2 (paper draw)", C1, "-"), ("Linear, same inputs, lags 1-3", C2, "-"), ("HAR", C3, "--"),
                     ("AR(1)", MUTED, ":")]:
        a1.plot(x, yearly[m], color=c, ls=ls, marker="o", ms=3, label=m)
    a1.axvspan(2017.5, x[-1] + 0.5, color=C4, alpha=0.10)
    a1.set_yscale("log")
    a1.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    a1.set_ylabel("MSE per year (log scale)")
    a1.set_xlabel("calendar year")
    a1.legend(fontsize=7.5, ncol=3, loc="upper left")
    a1.set_title("D8  Forecast error by year", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d8_yearly_error.png")
    plt.close(fig)


def main():
    t = time.time()
    d7()
    d8()
    with open(f"{OUT}/summary_d7_d8.md", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\nWrote {OUT}/summary_d7_d8.md ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
