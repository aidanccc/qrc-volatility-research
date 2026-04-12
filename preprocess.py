"""
preprocess.py

Generates Data_raw.csv and dff.csv from Data.CSV.

Data_raw.csv: a direct copy of Data.CSV (some notebooks expect this name).

dff.csv: Data_raw.csv with non-stationary columns replaced by their first
         differences (prefixed diff_), stationary columns kept as-is, and all
         NaN values filled with 0.  This mirrors the ADF-differencing logic in
         Reservoir_Learning.ipynb cells 4 and 12.

Run from the repo root:
    python preprocess.py
"""

import pandas as pd
import statsmodels.api as sm


def check_stationarity(series):
    """Return ADF p-value for a series (NaNs dropped before testing)."""
    result = sm.tsa.adfuller(series.dropna())
    return result[1]  # p_value


def main():
    # Load base dataset
    df = pd.read_csv("Data.CSV", header=0, index_col=0)
    print(f"Loaded Data.CSV: {df.shape[0]} rows, {df.shape[1]} columns")

    # Write Data_raw.csv (same content, different name expected by some notebooks)
    df.to_csv("Data_raw.csv")
    print("Written: Data_raw.csv")

    # Build dff: ADF-difference non-stationary columns
    dff = pd.DataFrame(index=df.index)
    for col in df.columns:
        p_value = check_stationarity(df[col])
        if p_value > 0.05:
            dff[f"diff_{col}"] = df[col].diff()
            print(f"  {col}: p={p_value:.4f} > 0.05 → differenced as diff_{col}")
        else:
            dff[col] = df[col]
            print(f"  {col}: p={p_value:.4f} ≤ 0.05 → kept as-is")

    dff = dff.fillna(0)
    dff.to_csv("dff.csv")
    print(f"Written: dff.csv ({dff.shape[0]} rows, {dff.shape[1]} columns)")
    print(f"Columns: {list(dff.columns)}")


if __name__ == "__main__":
    main()
