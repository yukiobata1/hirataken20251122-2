#!/usr/bin/env python3
"""
Overlay experimental S(Q) data with bimodal Ga simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Parameters
SIGMA_BASE = 2.70
SIGMA_RATIO_GA1 = 1.1
SIGMA_RATIO_GA2 = 0.9
GA1_FRACTIONS = [0.50, 0.75, 0.25]


def load_exp_sq(filepath):
    """Load experimental S(Q) data from user_provided_sq_data.txt"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    q_list, sq_list = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Format: Q, S(Q)
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
    exp_q, exp_sq = load_exp_sq('data/user_provided_sq_data.txt')
    print(f"Experimental data: {len(exp_q)} points, Q range: {exp_q.min():.2f} - {exp_q.max():.2f}")

    # Load simulation results
    results = {}
    for ga1_frac in GA1_FRACTIONS:
        label = f"ga1_{int(ga1_frac*100):02d}"
        rdf_file = f"outputs/bimodal_search/out_{label}.rdf"

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
            results[ga1_frac] = {'r': r, 'g_total': g_total, 'Q': Q, 'S': S}
            print(f"  Loaded: {rdf_file}")
        except Exception as e:
            print(f"  Error loading {rdf_file}: {e}")

    # Create comparison plot (similar to bimodal_ga_comparison.png but with exp data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['blue', 'red', 'green']

    for idx, ga1_frac in enumerate(GA1_FRACTIONS):
        if ga1_frac not in results:
            continue

        res = results[ga1_frac]
        Q, S = res['Q'], res['S']
        r, g = res['r'], res['g_total']

        color = colors[idx % len(colors)]
        label = f"Ga1:{ga1_frac:.0%}, Ga2:{1-ga1_frac:.0%}"

        # Top row: S(Q)
        axes[0, idx].plot(Q, S, color=color, lw=2, label='Simulation')
        axes[0, idx].scatter(exp_q, exp_sq, color='black', s=15, alpha=0.7, label='Experiment', zorder=5)
        axes[0, idx].set_xlim(0, 12)
        axes[0, idx].set_ylim(0, 3.5)
        axes[0, idx].set_xlabel("Q (A^-1)", fontsize=12)
        axes[0, idx].set_ylabel("S(Q)", fontsize=12)
        axes[0, idx].set_title(label, fontsize=14, fontweight='bold')
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].legend(fontsize=9, loc='upper right')
        axes[0, idx].axvspan(2.5, 3.5, color='orange', alpha=0.15)

        # Peak annotation
        peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
        peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
        peak_s = S[(Q > 1.5) & (Q < 4.0)][peak_idx]
        axes[0, idx].plot(peak_q, peak_s, 'ko', markersize=8)
        axes[0, idx].text(peak_q + 0.3, peak_s, f"Q={peak_q:.2f}", fontsize=10)

        # Bottom row: g(r) (no experimental g(r) data)
        axes[1, idx].plot(r, g, color=color, lw=2, label='Simulation')
        axes[1, idx].set_xlim(0, 10)
        axes[1, idx].set_ylim(0, 3.5)
        axes[1, idx].set_xlabel("r (A)", fontsize=12)
        axes[1, idx].set_ylabel("g(r)", fontsize=12)
        axes[1, idx].set_title(f"g(r) - {label}", fontsize=12)
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].legend(fontsize=9)

    plt.suptitle(f"Bimodal Ga Model vs Experiment\n"
                 f"sigma(Ga1)={SIGMA_RATIO_GA1}x, sigma(Ga2)={SIGMA_RATIO_GA2}x",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output1 = "outputs/bimodal_search/bimodal_ga_comparison_with_exp.png"
    plt.savefig(output1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output1}")

    # Create overlay plot (similar to bimodal_ga_overlay.png but with exp data)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, ga1_frac in enumerate(GA1_FRACTIONS):
        if ga1_frac not in results:
            continue
        res = results[ga1_frac]
        label = f"Ga1:{ga1_frac:.0%}, Ga2:{1-ga1_frac:.0%}"
        color = colors[idx % len(colors)]

        ax1.plot(res['Q'], res['S'], color=color, lw=2, label=label)
        ax2.plot(res['r'], res['g_total'], color=color, lw=2, label=label)

    # Add experimental S(Q)
    ax1.scatter(exp_q, exp_sq, color='black', s=20, alpha=0.8, label='Experiment', zorder=5)

    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 3.5)
    ax1.set_xlabel("Q (A^-1)", fontsize=12)
    ax1.set_ylabel("S(Q)", fontsize=12)
    ax1.set_title("S(Q) Comparison", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlabel("r (A)", fontsize=12)
    ax2.set_ylabel("g(r)", fontsize=12)
    ax2.set_title("g(r) Comparison", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Bimodal Ga vs Experiment\n"
                 f"sigma(Ga1)={SIGMA_BASE*SIGMA_RATIO_GA1:.2f}A, "
                 f"sigma(Ga2)={SIGMA_BASE*SIGMA_RATIO_GA2:.2f}A",
                 fontsize=12)
    plt.tight_layout()

    output2 = "outputs/bimodal_search/bimodal_ga_overlay_with_exp.png"
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output2}")

    plt.show()


if __name__ == "__main__":
    main()
