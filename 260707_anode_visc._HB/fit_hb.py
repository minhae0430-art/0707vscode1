import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def herschel_bulkley(gamma, tau_y, k, n):
    return tau_y + k * np.power(gamma, n)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Excel file not found: {path}')

    df = pd.read_excel(path, sheet_name=0, header=None)

    stress_col = None
    rate_col = None

    for col_idx in range(df.shape[1]):
        values = [str(v).strip().lower() for v in df.iloc[:3, col_idx].tolist() if pd.notna(v)]
        joined = ' '.join(values)
        if 'stress' in joined and 'shear' in joined:
            stress_col = col_idx
        if 'rate' in joined and 'shear' in joined:
            rate_col = col_idx

    if stress_col is None or rate_col is None:
        raise ValueError('Could not identify shear stress and shear rate columns from the Excel sheet')

    data = df.iloc[2:].copy()
    stress = pd.to_numeric(data.iloc[:, stress_col], errors='coerce')
    rate = pd.to_numeric(data.iloc[:, rate_col], errors='coerce')

    mask = np.isfinite(stress) & np.isfinite(rate) & (rate > 0) & (stress > 0)
    stress = stress[mask].to_numpy(dtype=float)
    rate = rate[mask].to_numpy(dtype=float)

    if len(stress) < 5:
        raise ValueError('Not enough valid data points for fitting')

    order = np.argsort(rate)
    return rate[order], stress[order]


def fit_hb(path):
    gamma, tau = load_data(path)

    tau_y0 = max(np.mean(tau), 1e-3)
    k0 = 1.0
    n0 = 0.5
    p0 = [tau_y0, k0, n0]
    bounds = [(1e-3, 1e3), (1e-12, 1e6), (0.1, 2.0)]

    def relative_error_sum(params):
        tau_y, k, n = params
        pred = herschel_bulkley(gamma, tau_y, k, n)
        return np.sum(np.abs((pred - tau) / tau))

    result = minimize(relative_error_sum, p0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200000})
    if not result.success:
        raise RuntimeError(f'Relative error optimization failed: {result.message}')

    tau_y_fit, k_fit, n_fit = result.x

    print('Fitted Herschel-Bulkley parameters (relative error sum):')
    print(f'tau_y = {tau_y_fit:.6f} Pa')
    print(f'K = {k_fit:.6f}')
    print(f'n = {n_fit:.6f}')
    print(f'relative error sum = {result.fun:.6f}')

    x_fit = np.geomspace(np.min(gamma), np.max(gamma), 400)
    y_fit = herschel_bulkley(x_fit, tau_y_fit, k_fit, n_fit)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(gamma, tau, 'o', color='tab:blue', label='data')
    ax.loglog(x_fit, y_fit, '-', color='tab:red', label='model')
    ax.set_title('260707_Anode_HB')
    ax.set_xlabel('shear rate (1/s)')
    ax.set_ylabel('shear stress (Pa)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    fig.tight_layout()

    output_dir = os.path.dirname(path)
    output_path = os.path.join(output_dir, '260707_Anode_HB_fit.png')
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f'Saved figure to {output_path}')

    return np.array([tau_y_fit, k_fit, n_fit])


if __name__ == '__main__':
    excel_path = r'c:\Users\밍혜\Desktop\VScode\260707_anode_visc._HB\260707_anode_visc.HB.xlsx'
    fit_hb(excel_path)
