# Colin extension assumptions

- This is a faithful reconstruction and retrospective extension, not exact author-code replication or an as-of historical trading backtest.
- Raw Colin input and original notebooks are immutable. Prepared v2 corrects derived identities and fills verified factor gaps; the initial diagnostic snapshot remains available.
- Macro release lags/vintages and Colin's raw construction are unavailable. Predictors are lagged by a month, but this does not prove historical publication availability.
- Legacy RV constants convert Colin's RV to log RV; the small post-2017 disagreement with independent prices remains explicit. Do not pool protocols with different targets.
- Preserve paper feature sets; freeze DP/TB differencing from the historical preprocessing convention instead of selecting stationarity transforms on test data. For HARX/ARMAX use the ten macro predictors, explicitly excluding notebook mutation artifacts (duplicated lag columns).
- Use original normalized quantum inputs without silently applying the modern price-feature scaler. Modern scaling remains frozen at 2017. No target winsorization. Statistical outliers are retained.
- Factor scale validation is specified in docs-colin/AUDIT.md. FIZ-to-CIZ construction changes are reported; unavailable August factors remain missing.
