#!/usr/bin/env python3
"""
Analyze single-component LJ simulation results and create comparison figures.
Calculates S(Q) from g(r) and compares with experimental data.

Usage:
    python scripts/analyze_and_plot.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

# ================= CONFIGURATION =================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "423K"
DATA_DIR = Path("/home/yuki/lammps_settings_obata/data/experimental")
THESIS_FIG_DIR = Path("/home/yuki/lammps_settings_obata/thesis/figures")

# Number density for Ga at 150°C (atoms/Å³)
RHO = 0.0522
# ================================================


def load_exp_sq():
    """Load experimental S(Q) data"""
    filepath = DATA_DIR / "sq_real_data.csv"
    data = np.loadtxt(filepath, delimiter=',')

    # Remove duplicates by averaging
    q_unique = []
    sq_unique = []
    for q in np.unique(data[:, 0]):
        mask = data[:, 0] == q
        q_unique.append(q)
        sq_unique.append(np.mean(data[mask, 1]))

    return np.array(q_unique), np.array(sq_unique)


def load_exp_gr():
    """Load experimental g(r) data"""
    filepath = DATA_DIR / "g_exp_150C_cleaned.dat"
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0], data[:, 1]


def load_sim_rdf(filepath):
    """Load simulation RDF from LAMMPS output"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                int(parts[0])  # First column is row index
                data_lines.append(line)
            except ValueError:
                continue

    if not data_lines:
        return None, None

    data = np.loadtxt(StringIO("\n".join(data_lines)))
    return data[:, 1], data[:, 2]  # r, g(r)


def calc_sq_from_gr(r, g, rho=RHO):
    """Calculate S(Q) from g(r) using Fourier transform with Lorch window"""
    Q = np.linspace(0.1, 15.0, 500)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    r_max = r[-1]

    # Lorch window function to reduce truncation artifacts
    def lorch(r_val):
        x = np.pi * r_val / r_max
        return np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)

    for i, q in enumerate(Q):
        if q < 1e-6:
            continue
        window = lorch(r)
        integrand = (g - 1.0) * r * np.sin(q * r) * window
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q

    return Q, S


def create_figures(r_sim, g_sim, q_sim, s_sim):
    """Create comparison figures with experimental data"""

    # Load experimental data
    q_exp, s_exp = load_exp_sq()
    r_exp, g_exp = load_exp_gr()

    print(f"  Experimental S(Q): {len(q_exp)} points")
    print(f"  Experimental g(r): {len(r_exp)} points")
    print(f"  Simulation g(r): {len(r_sim)} points")

    # Find peak values for comparison
    sim_gr_peak_mask = (r_sim > 2.5) & (r_sim < 3.5)
    exp_gr_peak_mask = (r_exp > 2.5) & (r_exp < 3.5)
    sim_gr_peak = np.max(g_sim[sim_gr_peak_mask]) if np.any(sim_gr_peak_mask) else 0
    exp_gr_peak = np.max(g_exp[exp_gr_peak_mask]) if np.any(exp_gr_peak_mask) else 0

    sim_sq_peak_mask = (q_sim > 2.0) & (q_sim < 3.0)
    exp_sq_peak_mask = (q_exp > 2.0) & (q_exp < 3.0)
    sim_sq_peak = np.max(s_sim[sim_sq_peak_mask]) if np.any(sim_sq_peak_mask) else 0
    exp_sq_peak = np.max(s_exp[exp_sq_peak_mask]) if np.any(exp_sq_peak_mask) else 0

    print(f"\n  Peak comparison:")
    print(f"    g(r): Sim={sim_gr_peak:.2f}, Exp={exp_gr_peak:.2f}, Ratio={sim_gr_peak/exp_gr_peak:.2f}")
    print(f"    S(Q): Sim={sim_sq_peak:.2f}, Exp={exp_sq_peak:.2f}, Ratio={sim_sq_peak/exp_sq_peak:.2f}")

    # ========== Figure 1: g(r) comparison ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(r_sim, g_sim, 'b-', lw=2, label='LJ Simulation (423 K)', zorder=3)
    ax1.scatter(r_exp, g_exp, color='black', s=15, alpha=0.7, label='Experiment (150°C)', zorder=5)

    ax1.set_xlabel('r (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, max(sim_gr_peak, exp_gr_peak) * 1.2)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)
    ax1.set_title('Radial Distribution Function: Simulation vs Experiment (423 K)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to outputs and thesis
    fig1.savefig(OUTPUT_DIR / 'rdf_comparison_423K.png', dpi=150)
    fig1.savefig(THESIS_FIG_DIR / 'rdf_comparison_with_exp.png', dpi=150)
    plt.close(fig1)
    print(f"\n  Saved: rdf_comparison_423K.png")

    # ========== Figure 2: S(Q) comparison ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(q_sim, s_sim, 'b-', lw=2, label='LJ Simulation (423 K)', zorder=3)
    ax2.scatter(q_exp, s_exp, color='black', s=15, alpha=0.7, label='Experiment (150°C)', zorder=5)

    # Highlight shoulder region
    ax2.axvspan(2.8, 3.5, color='orange', alpha=0.15, label='Shoulder region')

    ax2.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('S(Q)', fontsize=12)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, max(sim_sq_peak, exp_sq_peak) * 1.3)
    ax2.legend(fontsize=11)
    ax2.set_title('Structure Factor: Single-component LJ vs Experiment (423 K)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add annotation for shoulder
    ax2.annotate('Shoulder\n(not reproduced)', xy=(3.1, 1.5), xytext=(4.5, 2.2),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()

    fig2.savefig(OUTPUT_DIR / 'sq_comparison_423K.png', dpi=150)
    fig2.savefig(THESIS_FIG_DIR / 'sq_comparison_with_exp.png', dpi=150)
    plt.close(fig2)
    print(f"  Saved: sq_comparison_423K.png")

    # ========== Calculate metrics ==========
    # Interpolate simulation to experimental Q points
    s_sim_interp = np.interp(q_exp, q_sim, s_sim)

    # R-factor
    r_factor = np.sum(np.abs(s_exp - s_sim_interp)) / np.sum(np.abs(s_exp))

    # RMSE
    rmse = np.sqrt(np.mean((s_exp - s_sim_interp)**2))

    # Shoulder region RMSE
    shoulder_mask = (q_exp >= 2.8) & (q_exp <= 3.5)
    rmse_shoulder = np.sqrt(np.mean((s_exp[shoulder_mask] - s_sim_interp[shoulder_mask])**2))

    print(f"\n  Metrics:")
    print(f"    R-factor: {r_factor:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Shoulder RMSE: {rmse_shoulder:.4f}")

    # Save metrics
    with open(OUTPUT_DIR / 'metrics.txt', 'w') as f:
        f.write(f"Single-component LJ at 423 K\n")
        f.write(f"R-factor: {r_factor:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"Shoulder RMSE: {rmse_shoulder:.6f}\n")
        f.write(f"g(r) peak: Sim={sim_gr_peak:.3f}, Exp={exp_gr_peak:.3f}\n")
        f.write(f"S(Q) peak: Sim={sim_sq_peak:.3f}, Exp={exp_sq_peak:.3f}\n")

    return r_factor, rmse


def main():
    print("=" * 70)
    print("Analysis: Single-component LJ Simulation at 423 K")
    print("=" * 70)

    # Find RDF file
    rdf_file = OUTPUT_DIR / "ga_single_423K.rdf"

    if not rdf_file.exists():
        print(f"ERROR: RDF file not found: {rdf_file}")
        print("Run the simulation first: python scripts/run_single_component_423K.py")
        return 1

    print(f"  Loading: {rdf_file}")

    # Load simulation data
    r_sim, g_sim = load_sim_rdf(rdf_file)

    if r_sim is None:
        print("ERROR: Failed to load RDF data")
        return 1

    print(f"  Loaded {len(r_sim)} data points")

    # Calculate S(Q)
    print("\n  Calculating S(Q) from g(r)...")
    q_sim, s_sim = calc_sq_from_gr(r_sim, g_sim)

    # Save S(Q) data
    sq_data = np.column_stack([q_sim, s_sim])
    np.savetxt(OUTPUT_DIR / 'sq_423K.dat', sq_data, header='Q(A^-1)  S(Q)', fmt='%.6f')
    print(f"  Saved: sq_423K.dat")

    # Create figures
    print("\n  Creating comparison figures...")
    create_figures(r_sim, g_sim, q_sim, s_sim)

    print("\n" + "=" * 70)
    print("Analysis completed!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Thesis figures updated: {THESIS_FIG_DIR}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
