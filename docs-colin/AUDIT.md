# Colin dataset audit — September 24, 2026

Source: aidanccc/qrc-volatility-research commit ca78df0cae90a658d5b2696e6641bcfa4c3a0356 (Colin Chen, September 24, 10:09 EDT). The sole change adds `1950-2026.csv`; no construction code or metadata accompanied it.

The 920 monthly rows span January 1950–August 2026. All 816 overlapping rows match `data/Data.CSV` within 1.2e-16. All four factor columns are absent for 104 extension months (416 cells). No duplicate dates were found.

## Reconstruction

The canonical prepared snapshot is `data/snapshots/colin-2026-09-24-v2`. Colin's original file is unchanged. Official December 2024 FIZ archives validate affine normalization from pre-1984 observations against 1984–2017. Acceptance requires 90% exact agreement overall, 85% in each half, and the inverse values on the published 0.01 percentage-point rounding grid. Source revisions are residuals, not justification to regress different definitions onto one another. Each mapping, residual and source hash is retained.

The initial diagnostic snapshot `colin-2026-09-24` used an unnecessarily strict maximum revision threshold and rejected SMB/STR. It is preserved; it is not an experiment input. Investigation of FIZ historical agreement distinguished normalization recovery from source revisions, motivating v2 before any forecast scoring. STR's largest historical raw revision is 0.16 percentage points. These mappings are reconstructed, not Colin-confirmed metadata.

Use FIZ factors through December 2024 and current CIZ factors afterward. This preserves historical construction as far as available but introduces the documented provider change in January 2025; the analysis is retrospective. Sources currently end July 2026. 412 factor values were recovered; four August cells remain missing. August target forecasts can use July inputs, but September exogenous forecasts cannot use missing August inputs.

There are three RV1/RV2 boundary inconsistencies. RV_q and RV_a are separately affine-normalized averages of preceding 3/12 normalized RV values, not interchangeable with modern unshifted trailing features. Historical relations agree within 1e-10; reconstructing them corrects 208 extension cells. All 211 corrections are listed. Initial legacy zero padding is preserved and excluded by training-history requirements.

138 level/change flags exceed six pre-2018 median absolute deviation scales. These are diagnostic flags, not errors; no flagged row is discarded. Negative scaled DP values alone do not demonstrate negative raw dividend yields.

The legacy inverse RV normalization differs from modern price-derived log RV by up to 4.776e-5 after 2017. The exact normalization used for Colin's extension is unprovided. Separate target-specific results are necessary; no pooled target comparison or claim of bitwise equivalence.

## Provenance and history

`shared-history.json` inventories every commit reachable from cloned shared branches; `personal-history.json` inventories all locally reachable personal commits, including parent links and changed paths. Ref snapshots are retained. Shared development ended in April's Trotter maintenance until Colin's September dataset addition. Personal commits 6e7ddf0 and eca995c added the modern pipeline/results; 56058f7 established personal-only routing and e88ceec merged that work. The current user explicitly authorized this dedicated shared-repository branch, superseding that routing for this task only.

The modern package, CLI, dependency pins and baseline tests are ported from personal commit 6e7ddf0 (and subsequent report maintenance visible in personal history). Personal uncommitted follow-up files are not silently included. Original notebooks and simulator stay intact.

Official sources: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html and its December 2024 FIZ archive; paper https://arxiv.org/html/2505.13933v2. Downloaded provider bytes remain local; hashes and derived research data are published.
