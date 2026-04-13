"""
preprocess.py — Data Preprocessing for Realized Volatility Forecasting
========================================================================

This script is the FIRST step in the reproduction pipeline for:
    "Quantum Reservoir Computing for Realized Volatility Forecasting"
    (arXiv:2505.13933, Physical Review Research 8, 023028, 2026)

PURPOSE:
    Reads the raw dataset (Data.CSV) and produces two derived files that
    all downstream notebooks depend on:

    1. Data_raw.csv  — A direct copy of Data.CSV. Several notebooks
       (LSTM.ipynb, classical_reservoir.ipynb, Reservoir_Learning.ipynb)
       expect this filename, so we create it for compatibility.

    2. dff.csv       — The "differenced" dataset. For each column in
       Data.CSV, we run an Augmented Dickey-Fuller (ADF) stationarity
       test. If the column is non-stationary (p-value > 0.05), we
       replace it with its first difference and rename it with a
       "diff_" prefix. Stationary columns are kept as-is. All NaN
       values (created by differencing) are filled with 0.

WHY THIS IS NEEDED:
    The original repo did not include a script to generate these files,
    but both LSTM.ipynb and classical_reservoir.ipynb fail at their
    very first cell without them. This script unblocks the entire
    Python-side pipeline.

STATIONARITY LOGIC:
    The ADF test (Augmented Dickey-Fuller) tests the null hypothesis
    that a unit root is present in the time series (i.e., the series
    is non-stationary). If p-value > 0.05, we cannot reject the null,
    so we difference the column to make it stationary.

    In the S&P 500 dataset, two columns are typically differenced:
        - DP (Dividend-Price ratio) → becomes diff_DP
        - TB (T-bill rate)         → becomes diff_TB
    All other columns (RV, MKT, EP, SMB, HML, DEF, IP, INF, STR,
    RV_q, RV_a, RV1, RV2) are already stationary.

USAGE:
    Run from the repository root directory:
        python preprocess.py

    This must be run BEFORE any of the Jupyter notebooks.

INPUTS:
    - Data.CSV  (816 rows x 16 columns, monthly S&P 500 data, 1950-2017)

OUTPUTS:
    - Data_raw.csv  (identical copy of Data.CSV)
    - dff.csv       (differenced dataset with NaN filled to 0)
"""

import pandas as pd
import statsmodels.api as sm


def check_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on a time series.

    The ADF test checks whether a time series has a unit root (i.e.,
    is non-stationary). A low p-value (< 0.05) indicates the series
    is stationary and does NOT need differencing.

    Parameters
    ----------
    series : pd.Series
        The time series to test. NaN values are dropped before testing.

    Returns
    -------
    float
        The p-value from the ADF test. If > 0.05, the series should
        be differenced to achieve stationarity.
    """
    result = sm.tsa.adfuller(series.dropna())
    return result[1]  # p_value


def main():
    # ----------------------------------------------------------------
    # Step 1: Load the base dataset
    # ----------------------------------------------------------------
    # Data.CSV contains 816 monthly observations (Jan 1950 - Dec 2017)
    # with 16 columns: Date (index), DP, EP, MKT, SMB, HML, TB, DEF,
    # IP, INF, RV, STR, RV_q, RV_a, RV1, RV2
    #
    # The RV column is log(realized volatility), normalized to [-1, 0].
    # All other columns are macro-financial predictors.
    df = pd.read_csv("Data.CSV", header=0, index_col=0)
    print(f"Loaded Data.CSV: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------------------------------------------
    # Step 2: Write Data_raw.csv (same content, different name)
    # ----------------------------------------------------------------
    # Multiple notebooks expect "Data_raw.csv" as their input filename.
    # This is just a renamed copy of Data.CSV.
    df.to_csv("Data_raw.csv")
    print("Written: Data_raw.csv")

    # ----------------------------------------------------------------
    # Step 3: Build the differenced DataFrame (dff)
    # ----------------------------------------------------------------
    # For each column, run the ADF stationarity test:
    #   - If p-value > 0.05 (non-stationary): replace with first
    #     difference and prefix the column name with "diff_"
    #   - If p-value <= 0.05 (stationary): keep the column as-is
    #
    # This mirrors the logic in Reservoir_Learning.ipynb cell 12,
    # which performs the same ADF test and differencing inline.
    dff = pd.DataFrame(index=df.index)
    for col in df.columns:
        p_value = check_stationarity(df[col])
        if p_value > 0.05:
            # Non-stationary: apply first difference to make stationary
            # diff() computes x[t] - x[t-1], creating NaN at t=0
            dff[f"diff_{col}"] = df[col].diff()
            print(f"  {col}: p={p_value:.4f} > 0.05 -> differenced as diff_{col}")
        else:
            # Already stationary: keep as-is
            dff[col] = df[col]
            print(f"  {col}: p={p_value:.4f} <= 0.05 -> kept as-is")

    # Fill NaN values with 0. The first row will have NaN for any
    # differenced column (since diff() has no predecessor for row 0).
    dff = dff.fillna(0)

    # ----------------------------------------------------------------
    # Step 4: Write dff.csv
    # ----------------------------------------------------------------
    dff.to_csv("dff.csv")
    print(f"Written: dff.csv ({dff.shape[0]} rows, {dff.shape[1]} columns)")
    print(f"Columns: {list(dff.columns)}")


if __name__ == "__main__":
    main()
