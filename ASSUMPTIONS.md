# Dataset assumptions

- This is a faithful reconstruction and retrospective extension, not exact author-code replication or an as-of historical trading backtest.
- The original dataset and original notebooks are immutable. Prepared v2 corrects derived identities and fills verified factor gaps; the initial diagnostic snapshot remains available.
- Macro release lags/vintages and the extended dataset’s raw construction are unavailable. Predictors are lagged by a month, but this does not prove historical publication availability.
- Legacy RV constants convert the extended dataset’s RV to log RV; the small post-2017 disagreement with independent prices remains explicit. Do not pool protocols with different targets.
- Preserve paper feature sets; freeze DP/TB differencing from the historical preprocessing convention instead of selecting stationarity transforms on test data. For HARX/ARMAX use the ten macro predictors, explicitly excluding notebook mutation artifacts (duplicated lag columns).
- Use original normalized quantum inputs without silently applying the modern price-feature scaler. Modern scaling remains frozen at 2017. No target winsorization. Statistical outliers are retained.
- Factor scale validation is specified in docs-dataset/AUDIT.md. FIZ-to-CIZ construction changes are reported; unavailable August factors remain missing.


- Five-seed extension ensembles use seeded generated coupling matrices, paired across QR1/QR2 as in the modern infrastructure. The isolated legacy verification uses the supplied author coupling matrices. The extension is therefore not a claim of identical author reservoir realizations.

## 1 October 2026 — diagnostic study assumptions

The user authorized switching this folder to `aidanccc/qrc-volatility-research` on shared `Vikas` and setting aside the personal checkout. The prior personal-only restriction is superseded for this project. Backup details are in `analysis/diagnostics/MIGRATION.md`.

This study permits analysis scripts, diagrams, and documentation only. It leaves model code, original/prepared data, and published results unchanged. The canonical analysis input is the existing prepared extended v2 snapshot; the modern target is not pooled with it. Seed 0 is used for regenerated reservoir features and state/circuit diagnostics, while forecast figures reuse seeds 0–4 and both published windows. Twelve histories are selected at evenly spaced valid forecast positions, using input availability rather than target values. First-three-row feature placeholders are omitted from descriptive plots, not altered in production.

Full logical resource counts assume the mathematically equivalent seven-input reset channel and separate full-history repetitions for virtual-node measurement endpoints. A local deterministic density-matrix test verifies that reset/re-encoding matches partial trace/replacement. This does not establish IonQ backend support, native gate count, timing, or reset-noise fidelity. Marginal shot standard errors describe ideal measurement sampling only and are not forecast confidence intervals or IonQ-noise results.

Regime thresholds are pre-2018 log-RV tertiles; error grouping by realized target regime is retrospective. Feature range flags use each feature’s historical extrema, not the inaccurate blanket [-1,0] comment in the encoder. No clipping, imputation, normalization change, or readout optimization is applied. Hidden-state mixedness is not interpreted as proof of useful memory or entanglement.

The observed Trotter gate ordering is retained: chronological XX then Z corresponds to the matrix product Z × XX. The reversed formula in its docstring is documented as a mismatch, not silently repaired. Historical prediction agreement is checked from existing artifacts; this run additionally validates one original-coupling three-input instrumentation example per QR model, but does not rerun the full original benchmark.
