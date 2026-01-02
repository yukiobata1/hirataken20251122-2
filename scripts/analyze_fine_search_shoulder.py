#!/usr/bin/env python3
"""
Analyze fine_search_shoulder results: calculate S(Q) from RDF and compare with experimental data.
Based on overlay_exp_bimodal_v3.py methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import re

# Number density (atoms/Å³)
RHO = 0.0522


def load_exp_sq(filepath):
    """Load experimental S(Q) data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    q_list, sq_list = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 2:
            try:
                q_list.append(float(parts[0].strip()))
                sq_list.append(float(parts[1].strip()))
            except ValueError:
                continue

    return np.array(q_list), np.array(sq_list)


def load_rdf_robust(filepath):
    """Load RDF file robustly."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) <= 2:
            continue
        data_lines.append(line)

    return np.loadtxt(StringIO("\n".join(data_lines)))


def calc_sq_from_gr(r, g, rho=RHO):
    """Calculate S(Q) from g(r) using Fourier transform."""
    Q = np.linspace(0.5, 15.0, 500)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1

    for i, q in enumerate(Q):
        if q < 1e-6:
            continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q

    return Q, S


def parse_params(filename):
    """Parse parameters from filename like out_sig12_115_ga1_50.rdf

    Returns:
        sig12: sigma12 ratio (e.g., 1.15)
        ga1_frac: Ga1 fraction (e.g., 0.50 for ga1_50)
    """
    match = re.search(r'sig12_(\d+)_ga1_(\d+)', filename)
    if match:
        sig12 = int(match.group(1)) / 100.0
        ga1_frac = int(match.group(2)) / 100.0  # ga1_50 -> 0.50
        return sig12, ga1_frac
    return None, None


def main():
    output_dir = Path('outputs/fine_search_shoulder')
    results_dir = output_dir / 'analysis'
    results_dir.mkdir(exist_ok=True)

    # Load experimental data
    exp_q, exp_sq = load_exp_sq('data/sq_real_data.csv')
    print(f"Experimental data: {len(exp_q)} points, Q range: {exp_q.min():.2f} - {exp_q.max():.2f}")

    # Find all RDF files
    rdf_files = sorted(output_dir.glob('out_*.rdf'))

    if not rdf_files:
        print("No RDF files found!")
        return

    print(f"Found {len(rdf_files)} RDF files")

    results = []

    for rdf_file in rdf_files:
        sig12, ga1_frac = parse_params(rdf_file.name)
        if sig12 is None:
            continue

        try:
            data = load_rdf_robust(rdf_file)
            r = data[:, 1]

            if data.shape[1] >= 7:
                # LAMMPS RDF columns for "compute rdf 200 1 1 2 2 1 2":
                # col 2: g(1-1) = Ga1-Ga1
                # col 4: g(2-2) = Ga2-Ga2
                # col 6: g(1-2) = Ga1-Ga2
                g11 = data[:, 2]  # Ga1-Ga1
                g22 = data[:, 4]  # Ga2-Ga2
                g12 = data[:, 6]  # Ga1-Ga2

                # Weighted total g(r) by Ga1/Ga2 fractions (from filename)
                x1 = ga1_frac       # Ga1 fraction
                x2 = 1 - ga1_frac   # Ga2 fraction
                g_total = x1**2 * g11 + x2**2 * g22 + 2 * x1 * x2 * g12
            else:
                g_total = data[:, 2]

            # Calculate S(Q)
            Q, S = calc_sq_from_gr(r, g_total)

            # Interpolate simulation to experimental Q points
            S_interp = np.interp(exp_q, Q, S)

            # R-factor (full range)
            r_factor = np.sum(np.abs(exp_sq - S_interp)) / np.sum(np.abs(exp_sq))

            # RMSE in peak region (1.5-5.0 Å⁻¹)
            mask = (exp_q >= 1.5) & (exp_q <= 5.0)
            rmse = np.sqrt(np.mean((exp_sq[mask] - S_interp[mask])**2))

            # RMSE in shoulder region (2.8-3.5 Å⁻¹)
            mask_shoulder = (exp_q >= 2.8) & (exp_q <= 3.5)
            if np.sum(mask_shoulder) > 0:
                rmse_shoulder = np.sqrt(np.mean((exp_sq[mask_shoulder] - S_interp[mask_shoulder])**2))
            else:
                rmse_shoulder = rmse

            results.append({
                'file': rdf_file.name,
                'sig12': sig12,
                'ga1_frac': ga1_frac,
                'r_factor': r_factor,
                'rmse': rmse,
                'rmse_shoulder': rmse_shoulder,
                'Q': Q,
                'S': S,
                'r': r,
                'g11': g11,
                'g22': g22,
                'g12': g12,
                'g_total': g_total
            })

            print(f"  Loaded: {rdf_file.name}, R={r_factor:.4f}")

        except Exception as e:
            print(f"  Error loading {rdf_file}: {e}")

    # Sort by R-factor
    results_sorted = sorted(results, key=lambda x: x['r_factor'])

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS (sorted by R-factor)")
    print("=" * 80)
    print(f"{'sig12':>8} {'Ga1%':>8} {'R-factor':>10} {'RMSE':>10} {'RMSE_shoulder':>15}")
    print("-" * 80)

    for r in results_sorted:
        print(f"{r['sig12']:>8.2f} {r['ga1_frac']*100:>8.0f} {r['r_factor']:>10.4f} {r['rmse']:>10.4f} {r['rmse_shoulder']:>15.4f}")

    best = results_sorted[0]
    print("-" * 80)
    print(f"BEST: sig12={best['sig12']:.2f}, Ga1={best['ga1_frac']*100:.0f}% (R={best['r_factor']:.4f})")

    # Save CSV
    with open(results_dir / 'metrics_summary.csv', 'w') as f:
        f.write('sig12,ga1_frac,r_factor,rmse,rmse_shoulder\n')
        for r in results_sorted:
            f.write(f"{r['sig12']:.2f},{r['ga1_frac']:.2f},{r['r_factor']:.6f},{r['rmse']:.6f},{r['rmse_shoulder']:.6f}\n")

    # ========== Individual plots ==========
    for res in results:
        sig12 = res['sig12']
        ga1_frac = res['ga1_frac']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # S(Q) comparison
        ax = axes[0]
        ax.plot(res['Q'], res['S'], 'b-', lw=2, label='Simulation')
        ax.scatter(exp_q, exp_sq, color='black', s=10, alpha=0.7, label='Experiment', zorder=5)
        ax.set_xlabel('Q (Å⁻¹)')
        ax.set_ylabel('S(Q)')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.5)
        ax.axvspan(2.5, 3.5, color='orange', alpha=0.15, label='Shoulder region')
        ax.legend()
        ax.set_title(f'σ₁₂={sig12:.2f}x, Ga1={ga1_frac*100:.0f}%')
        ax.grid(True, alpha=0.3)

        # Partial g(r)
        ax = axes[1]
        ax.plot(res['r'], res['g11'], 'b-', label='Ga1-Ga1', linewidth=1.5)
        ax.plot(res['r'], res['g22'], 'r-', label='Ga2-Ga2', linewidth=1.5)
        ax.plot(res['r'], res['g12'], 'g-', label='Ga1-Ga2', linewidth=1.5)
        ax.plot(res['r'], res['g_total'], 'k--', label='Total', linewidth=2)
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Partial RDFs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Metrics text
        metrics_text = f"R-factor: {res['r_factor']:.4f}\nRMSE: {res['rmse']:.4f}\nRMSE(shoulder): {res['rmse_shoulder']:.4f}"
        ax.text(0.95, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_name = f'sq_comparison_sig12_{round(sig12*100)}_ga1_{round(ga1_frac*100)}.png'
        plt.savefig(results_dir / plot_name, dpi=150)
        plt.close()

    # ========== Gallery plot ==========
    sig12_vals = sorted(set(r['sig12'] for r in results))
    ga1_vals = sorted(set(r['ga1_frac'] for r in results))

    n_rows = len(sig12_vals)
    n_cols = len(ga1_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), squeeze=False)

    for res in results:
        i = sig12_vals.index(res['sig12'])
        j = ga1_vals.index(res['ga1_frac'])
        ax = axes[i, j]

        ax.plot(res['Q'], res['S'], 'b-', lw=1.5)
        ax.scatter(exp_q, exp_sq, color='black', s=5, alpha=0.5, zorder=5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3.5)
        ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

        ax.set_title(f'σ₁₂={res["sig12"]:.2f}x, Ga1={res["ga1_frac"]*100:.0f}%\nR={res["r_factor"]:.3f}', fontsize=8)
        ax.tick_params(labelsize=6)

        if i == n_rows - 1:
            ax.set_xlabel('Q (Å⁻¹)', fontsize=8)
        if j == 0:
            ax.set_ylabel('S(Q)', fontsize=8)

    plt.suptitle('Fine Search Shoulder: S(Q) vs Experiment', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'gallery_all_sq.png', dpi=200)
    plt.close()
    print(f"\nGallery saved: {results_dir / 'gallery_all_sq.png'}")

    # ========== Heatmap ==========
    if len(sig12_vals) > 1 and len(ga1_vals) > 1:
        rfactor_matrix = np.full((len(sig12_vals), len(ga1_vals)), np.nan)
        for res in results:
            i = sig12_vals.index(res['sig12'])
            j = ga1_vals.index(res['ga1_frac'])
            rfactor_matrix[i, j] = res['r_factor']

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rfactor_matrix, aspect='auto', origin='lower', cmap='viridis_r')
        ax.set_xticks(range(len(ga1_vals)))
        ax.set_xticklabels([f'{v*100:.0f}%' for v in ga1_vals])
        ax.set_yticks(range(len(sig12_vals)))
        ax.set_yticklabels([f'{v:.2f}x' for v in sig12_vals])
        ax.set_xlabel('Ga1 fraction')
        ax.set_ylabel('σ₁₂ ratio')
        ax.set_title('R-factor Heatmap: S(Q) vs Experimental')
        plt.colorbar(im, label='R-factor')

        best_i = sig12_vals.index(best['sig12'])
        best_j = ga1_vals.index(best['ga1_frac'])
        ax.plot(best_j, best_i, 'r*', markersize=15, markeredgecolor='white')

        plt.tight_layout()
        plt.savefig(results_dir / 'rfactor_heatmap.png', dpi=150)
        plt.close()
        print(f"Heatmap saved: {results_dir / 'rfactor_heatmap.png'}")

    # ========== Best fit overlay ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best['Q'], best['S'], 'b-', lw=2, label=f'Best: σ₁₂={best["sig12"]:.2f}x, Ga1={best["ga1_frac"]*100:.0f}%')
    ax.scatter(exp_q, exp_sq, color='black', s=20, alpha=0.7, label='Experiment', zorder=5)
    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('S(Q)', fontsize=12)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axvspan(2.5, 3.5, color='orange', alpha=0.15, label='Shoulder region')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Best Fit: R-factor = {best["r_factor"]:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'best_fit_overlay.png', dpi=150)
    plt.close()
    print(f"Best fit overlay saved: {results_dir / 'best_fit_overlay.png'}")

    print(f"\nAll results saved to: {results_dir}")


if __name__ == '__main__':
    main()
