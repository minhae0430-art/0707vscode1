import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def carreau_model(gamma, eta0, lambda_, n):
    return eta0 / (1 + (lambda_ * gamma) ** n)  # simplified Carreau form


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    xls = pd.ExcelFile(path)
    print("Sheets:", xls.sheet_names)

    df = pd.read_excel(path, sheet_name=xls.sheet_names[0])
    print(df.head(8).to_string())

    # Use the first two numeric data columns from the sheet as shear-rate and viscosity.
    # This is robust to the header layout used in the uploaded workbook.
    numeric_cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors='coerce')
        if s.notna().sum() >= 5:
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise ValueError("Could not find enough numeric data columns in the Excel sheet")

    gamma_col = numeric_cols[1]
    eta_col = numeric_cols[2] if len(numeric_cols) >= 3 else numeric_cols[1]

    gamma = pd.to_numeric(df[gamma_col], errors='coerce').dropna().to_numpy(dtype=float)
    eta = pd.to_numeric(df[eta_col], errors='coerce').dropna().to_numpy(dtype=float)

    if len(gamma) < 4 or len(eta) < 4:
        raise ValueError("Not enough valid data points were found in the Excel sheet")

    # Keep only the overlapping points
    mask = np.isfinite(gamma) & np.isfinite(eta)
    gamma = gamma[mask]
    eta = eta[mask]

    if len(gamma) < 4:
        raise ValueError("Not enough valid data points were found after filtering")

    return gamma, eta


def fit_carreau(path):
    gamma, eta = load_data(path)

    # Initial guess
    p0 = [np.max(eta), 0.1, 0.8]

    # Fit with bounds for stability
    bounds = ([1e-12, 1e-12, 1e-3], [1e6, 1e6, 2.0])
    popt, _ = curve_fit(carreau_model, gamma, eta, p0=p0, bounds=bounds, maxfev=100000)

    eta0_fit, lambda_fit, n_fit = popt
    print("Fitted parameters")
    print(f"eta0 = {eta0_fit:.6f}")
    print(f"lambda = {lambda_fit:.6f}")
    print(f"n = {n_fit:.6f}")

    # Plot
    x_fit = np.geomspace(np.min(gamma), np.max(gamma), 400)
    y_fit = carreau_model(x_fit, *popt)
    plt.figure(figsize=(6, 4))
    plt.loglog(gamma, eta, 'o', label='data')
    plt.loglog(x_fit, y_fit, '-', label='Carreau fit')
    plt.xlabel('shear rate (1/s)')
    plt.ylabel('viscosity (Pa·s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('carreau_fit.png', dpi=200)
    print('Saved plot to carreau_fit.png')

    return popt


if __name__ == '__main__':
    xlsx_path = 'PEO(200M)1wt_carraue_Fitting.xlsx'
    fit_carreau(xlsx_path)
