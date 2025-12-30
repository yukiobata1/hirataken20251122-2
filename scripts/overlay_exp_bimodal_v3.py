#!/usr/bin/env python3
"""
Overlay experimental S(Q) data with bimodal Ga V3 simulation results.
V3 varies sigma12 (Ga1-Ga2 interaction) and Ga1 fraction.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Parameters
SIGMA_BASE = 2.70
SIGMA_RATIO_GA1 = 1.1  # Ga1-Ga1: 2.97 A
SIGMA_RATIO_GA2 = 0.9  # Ga2-Ga2: 2.43 A

# V3 grid parameters
SIGMA12_RATIOS = [1.10, 1.15, 1.20]
GA1_FRACTIONS = [0.35, 0.50, 0.65]


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


def calc_sq_from_gr(r, g, rho=0.0522):
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


def main():
    # Load experimental data
    exp_q, exp_sq = load_exp_sq('data/sq_real_data.csv')
    print(f"Experimental data: {len(exp_q)} points, Q range: {exp_q.min():.2f} - {exp_q.max():.2f}")

    # Load all simulation results
    results = {}
    for sig12_ratio in SIGMA12_RATIOS:
        for ga1_frac in GA1_FRACTIONS:
            sig12_label = round(sig12_ratio * 100)
            ga1_label = round(ga1_frac * 100)
            rdf_file = f"outputs/bimodal_search_v3/out_sig12_{sig12_label}_ga1_{ga1_label}.rdf"

            try:
                data = load_rdf_robust(rdf_file)
                r = data[:, 1]

                if data.shape[1] >= 7:
                    g11 = data[:, 2]
                    g22 = data[:, 4]
                    g12 = data[:, 6]

                    x1 = ga1_frac
                    x2 = 1 - ga1_frac
                    g_total = x1**2 * g11 + x2**2 * g22 + 2 * x1 * x2 * g12
                else:
                    g_total = data[:, 2]

                Q, S = calc_sq_from_gr(r, g_total)
                results[(sig12_ratio, ga1_frac)] = {'r': r, 'g_total': g_total, 'Q': Q, 'S': S}
                print(f"  Loaded: {rdf_file}")
            except Exception as e:
                print(f"  Error loading {rdf_file}: {e}")

    # ========== Plot 1: 3x3 grid comparison ==========
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, sig12_ratio in enumerate(SIGMA12_RATIOS):
        for j, ga1_frac in enumerate(GA1_FRACTIONS):
            ax = axes[i, j]
            key = (sig12_ratio, ga1_frac)

            if key in results:
                res = results[key]
                Q, S = res['Q'], res['S']

                ax.plot(Q, S, 'b-', lw=2, label='Simulation')
                ax.scatter(exp_q, exp_sq, color='black', s=10, alpha=0.7, label='Experiment', zorder=5)

                # Peak annotation
                peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
                peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
                peak_s = S[(Q > 1.5) & (Q < 4.0)][peak_idx]
                ax.plot(peak_q, peak_s, 'ro', markersize=6)

            ax.set_xlim(0, 12)
            ax.set_ylim(0, 3.5)
            ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f"Ga1:{ga1_frac:.0%}", fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"sig12={sig12_ratio:.2f}x\nS(Q)", fontsize=11)
            if i == 2:
                ax.set_xlabel("Q (A^-1)", fontsize=11)
            if i == 0 and j == 2:
                ax.legend(fontsize=8, loc='upper right')

    plt.suptitle("Bimodal Ga V3: S(Q) vs Experiment\n"
                 f"Ga1-Ga1={SIGMA_BASE*SIGMA_RATIO_GA1:.2f}A, Ga2-Ga2={SIGMA_BASE*SIGMA_RATIO_GA2:.2f}A",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output1 = "outputs/bimodal_search_v3/bimodal_v3_grid_with_exp.png"
    plt.savefig(output1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output1}")

    # ========== Plot 2: Overlay by sigma12 (3 panels) ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'red', 'green']

    for i, sig12_ratio in enumerate(SIGMA12_RATIOS):
        ax = axes2[i]
        sig12_val = SIGMA_BASE * sig12_ratio

        for j, ga1_frac in enumerate(GA1_FRACTIONS):
            key = (sig12_ratio, ga1_frac)
            if key in results:
                res = results[key]
                label = f"Ga1:{ga1_frac:.0%}"
                ax.plot(res['Q'], res['S'], color=colors[j], lw=2, label=label)

        ax.scatter(exp_q, exp_sq, color='black', s=15, alpha=0.7, label='Experiment', zorder=5)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel("Q (A^-1)", fontsize=12)
        ax.set_ylabel("S(Q)", fontsize=12)
        ax.set_title(f"sigma12 = {sig12_val:.2f} A ({sig12_ratio:.2f}x)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    plt.suptitle("Bimodal Ga V3: Effect of Ga1 Fraction at Different sigma12", fontsize=13)
    plt.tight_layout()

    output2 = "outputs/bimodal_search_v3/bimodal_v3_by_sigma12_with_exp.png"
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output2}")

    # ========== Plot 3: Overlay by Ga1 fraction (3 panels) ==========
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['purple', 'orange', 'cyan']

    for j, ga1_frac in enumerate(GA1_FRACTIONS):
        ax = axes3[j]

        for i, sig12_ratio in enumerate(SIGMA12_RATIOS):
            key = (sig12_ratio, ga1_frac)
            if key in results:
                res = results[key]
                sig12_val = SIGMA_BASE * sig12_ratio
                label = f"sig12={sig12_val:.2f}A"
                ax.plot(res['Q'], res['S'], color=colors[i], lw=2, label=label)

        ax.scatter(exp_q, exp_sq, color='black', s=15, alpha=0.7, label='Experiment', zorder=5)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel("Q (A^-1)", fontsize=12)
        ax.set_ylabel("S(Q)", fontsize=12)
        ax.set_title(f"Ga1 = {ga1_frac:.0%}, Ga2 = {1-ga1_frac:.0%}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    plt.suptitle("Bimodal Ga V3: Effect of sigma12 at Different Ga1 Fractions", fontsize=13)
    plt.tight_layout()

    output3 = "outputs/bimodal_search_v3/bimodal_v3_by_ga1frac_with_exp.png"
    plt.savefig(output3, dpi=150, bbox_inches='tight')
    print(f"Saved: {output3}")

    # ========== Summary: Best fit analysis ==========
    print("\n" + "=" * 70)
    print("FITTING SUMMARY")
    print("=" * 70)

    # Calculate R-factor for each simulation
    r_factors = {}
    for key, res in results.items():
        sig12_ratio, ga1_frac = key
        Q_sim, S_sim = res['Q'], res['S']

        # Interpolate simulation to experimental Q points
        S_sim_interp = np.interp(exp_q, Q_sim, S_sim)

        # R-factor
        r_factor = np.sum(np.abs(exp_sq - S_sim_interp)) / np.sum(np.abs(exp_sq))
        r_factors[key] = r_factor

        sig12_val = SIGMA_BASE * sig12_ratio
        print(f"  sig12={sig12_val:.2f}A, Ga1={ga1_frac:.0%}: R = {r_factor:.4f}")

    # Best fit
    best_key = min(r_factors, key=r_factors.get)
    best_sig12, best_ga1 = best_key
    print(f"\nBest fit: sigma12={SIGMA_BASE*best_sig12:.2f}A ({best_sig12:.2f}x), "
          f"Ga1={best_ga1:.0%} (R={r_factors[best_key]:.4f})")


if __name__ == "__main__":
    main()
