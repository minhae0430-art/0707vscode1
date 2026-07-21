import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def herschel_bulkley(gamma, tau_y, k, n):
    return tau_y + k * np.power(gamma, n)


def load_and_average(path, rtol=2e-3):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Excel file not found: {path}')

    df = pd.read_excel(path, sheet_name=0, header=None)

    stress_cols = []
    rate_cols = []
    for col_idx in range(df.shape[1]):
        values = [str(v).strip().lower() for v in df.iloc[:3, col_idx].tolist() if pd.notna(v)]
        joined = ' '.join(values)
        if 'shear' in joined and 'stress' in joined:
            stress_cols.append(col_idx)
        if 'shear' in joined and 'rate' in joined:
            rate_cols.append(col_idx)

    if len(stress_cols) == 0 or len(rate_cols) == 0:
        raise ValueError('Could not identify shear stress and shear rate columns from the Excel sheet')

    if len(stress_cols) != len(rate_cols):
        raise ValueError(f'Found {len(stress_cols)} stress columns but {len(rate_cols)} rate columns')

    data = df.iloc[2:].copy()

    pairs = []
    for stress_col, rate_col in zip(stress_cols, rate_cols):
        stress = pd.to_numeric(data.iloc[:, stress_col], errors='coerce')
        rate = pd.to_numeric(data.iloc[:, rate_col], errors='coerce')
        mask = np.isfinite(stress) & np.isfinite(rate) & (rate > 0) & (stress > 0)
        stress_vals = stress[mask].to_numpy(dtype=float)
        rate_vals = rate[mask].to_numpy(dtype=float)

        if len(stress_vals) < 3:
            continue

        for rr, ss in zip(rate_vals, stress_vals):
            pairs.append((float(rr), float(ss)))

    if len(pairs) < 5:
        raise ValueError('Not enough valid data points after averaging repeat measurements')

    # cluster nearby rates using relative tolerance rtol
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    clusters = []
    cur_cluster = [pairs_sorted[0]]
    for rr, ss in pairs_sorted[1:]:
        prev_rr = cur_cluster[-1][0]
        if rr <= prev_rr * (1.0 + rtol):
            cur_cluster.append((rr, ss))
        else:
            clusters.append(cur_cluster)
            cur_cluster = [(rr, ss)]
    clusters.append(cur_cluster)

    gamma = np.array([np.mean([p[0] for p in c]) for c in clusters], dtype=float)
    tau = np.array([np.mean([p[1] for p in c]) for c in clusters], dtype=float)

    order = np.argsort(gamma)
    return gamma[order], tau[order]


def fit_hb(path):
    gamma, tau = load_and_average(path)

    def relative_squared_error_sum(params):
        tau_y, k, n = params
        pred = herschel_bulkley(gamma, tau_y, k, n)
        return np.sum(np.square((pred - tau) / tau))

    initial_guesses = [
        [max(np.mean(tau) * 0.5, 1e-3), 1.0, 0.7],
        [max(np.min(tau) * 0.5, 1e-3), 2.0, 0.8],
        [0.2, 2.0, 0.7],
    ]
    bounds = [(1e-6, 1e6), (1e-6, 1e6), (0.1, 2.0)]

    best_result = None
    for p0 in initial_guesses:
        result = minimize(
            relative_squared_error_sum,
            p0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200000},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if not best_result.success:
        raise RuntimeError(f'Optimization failed: {best_result.message}')

    tau_y_fit, k_fit, n_fit = best_result.x
    error_sum = best_result.fun

    print('Fitted Herschel-Bulkley parameters (sum of squared relative errors):')
    print(f'tau_y = {tau_y_fit:.6f} Pa')
    print(f'K = {k_fit:.6f}')
    print(f'n = {n_fit:.6f}')
    print(f'error sum = {error_sum:.6f}')

    x_fit = np.geomspace(np.min(gamma), np.max(gamma), 400)
    y_fit = herschel_bulkley(x_fit, tau_y_fit, k_fit, n_fit)

    fig, ax = plt.subplots(figsize=(8, 5))
    # plot averaged data only
    ax.loglog(gamma, tau, 'o', color='tab:blue', label='data', markersize=7, markeredgewidth=0.6, markeredgecolor='white', zorder=4)
    ax.loglog(x_fit, y_fit, '-', color='tab:red', label='model', zorder=3)
    ax.set_title('260720_graphite_HB_Fitting')
    ax.set_xlabel('shear rate (1/s)')
    ax.set_ylabel('shear stress (Pa)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)

    fig.tight_layout()

    output_dir = os.path.dirname(path)
    output_path = os.path.join(output_dir, '260720_graphite_HB_Fitting.png')
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved figure to {output_path}')

    return np.array([tau_y_fit, k_fit, n_fit], dtype=float)


if __name__ == '__main__':
    excel_path = r'c:\Users\밍혜\Desktop\VScode\260721_anode_visc._HB\260720_anode_HB_Fitting.xlsx'
    fit_hb(excel_path)
