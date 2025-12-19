#!/usr/bin/env python3
"""
Grid Search for Bimodal Ga Structure (Ga1/Ga2 model)

Based on the approach suggested by Prof. Hirata:
- Two types of Ga atoms with different sigma values
- Ga1-Ga1: sigma * 1.1 (larger)
- Ga2-Ga2: sigma * 0.9 (smaller)
- Ga1-Ga2: sigma * 1.0 (original)
- Epsilon: fixed
- Composition ratios: 0.5:0.5, 0.75:0.25, 0.25:0.75

This aims to reproduce the high-Q shoulder on the first S(Q) peak
observed in liquid Ga (Fig.3 of Amon et al. J. Phys. Chem. C 2023).
"""

import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from io import StringIO

# ================= CONFIGURATION =================
# Base Parameters (Ga from EPSR paper)
SIGMA_BASE = 2.70  # Angstrom
EPSILON_BASE = 0.430  # kcal/mol (converted from ~1.8 kJ/mol)
TEMP = 423.15  # 150 C

# Bimodal sigma ratios (fixed as suggested)
SIGMA_RATIO_GA1 = 1.1  # Ga1-Ga1 (larger)
SIGMA_RATIO_GA2 = 0.9  # Ga2-Ga2 (smaller)
# Ga1-Ga2 uses Lorentz-Berthelot mixing: (sigma1 + sigma2) / 2

# Composition ratios to test (Ga1 fraction)
GA1_FRACTIONS = [0.50, 0.75, 0.25]

# LAMMPS Command (KOKKOS GPU)
LMP_CMD = "lmp -k on g 1 -sf kk"
# =============================================


def write_lammps_input(filename, output_prefix, ga1_frac):
    """
    Write LAMMPS input file for bimodal Ga simulation.

    ga1_frac: fraction of Ga1 atoms (rest is Ga2)
    """
    # Sigma values
    sigma1 = SIGMA_BASE * SIGMA_RATIO_GA1  # Ga1-Ga1
    sigma2 = SIGMA_BASE * SIGMA_RATIO_GA2  # Ga2-Ga2
    sigma12 = (sigma1 + sigma2) / 2.0      # Ga1-Ga2 (Lorentz-Berthelot)

    # Epsilon values (all fixed)
    eps1 = EPSILON_BASE
    eps2 = EPSILON_BASE
    eps12 = np.sqrt(eps1 * eps2)  # Lorentz-Berthelot

    # Random seed based on composition
    seed = int(ga1_frac * 10000) + 12345

    content = f"""# Bimodal Ga Grid Search: Ga1 fraction = {ga1_frac:.2f}
# Ga1-Ga1 sigma = {sigma1:.4f} A (x1.1)
# Ga2-Ga2 sigma = {sigma2:.4f} A (x0.9)
# Ga1-Ga2 sigma = {sigma12:.4f} A (mixed)
# Epsilon = {EPSILON_BASE:.4f} kcal/mol (fixed)

# KOKKOS Initialization
package         kokkos neigh full newton off

units           real
atom_style      atomic
boundary        p p p

read_data       inputs/data.ga_base_2types

# Set type fractions: Ga1={ga1_frac:.2f}, Ga2={1-ga1_frac:.2f}
# First set all to type 1, then convert fraction to type 2
set             group all type 1
set             group all type/fraction 2 {1-ga1_frac:.2f} {seed}

# Force Field (LJ)
pair_style      lj/cut/kk 12.0
pair_coeff      1 1 {eps1:.4f} {sigma1:.4f}
pair_coeff      2 2 {eps2:.4f} {sigma2:.4f}
pair_coeff      1 2 {eps12:.4f} {sigma12:.4f}

# Settings
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
velocity        all create {TEMP} {seed} dist gaussian

thermo          1000
thermo_style    custom step temp press density

# Minimization
minimize        1.0e-4 1.0e-6 10000 100000
reset_timestep  0

# Equilibration (50000 steps)
timestep        2.0
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             50000
unfix           nvt
reset_timestep  0

# Production (100000 steps)
fix             nvt all nvt temp {TEMP} {TEMP} 100.0

# RDF calculation: 1-1, 2-2, 1-2 pairs
compute         myrdf all rdf 200 1 1 2 2 1 2 cutoff 12.0
fix             rdfout all ave/time 100 1000 100000 c_myrdf[*] file {output_prefix}.rdf mode vector

run             100000
"""
    with open(filename, 'w') as f:
        f.write(content)


def prepare_base_data():
    """Prepare 2-type base data file from single Ga structure."""
    if not os.path.exists('inputs/data.ga_1000'):
        raise FileNotFoundError("inputs/data.ga_1000 not found!")

    with open('inputs/data.ga_1000', 'r') as f:
        content = f.read()

    # Modify for 2 atom types
    content = content.replace("1 atom types", "2 atom types")
    if "Masses" in content:
        content = content.replace("1 69.723", "1 69.723\n2 69.723")

    with open('inputs/data.ga_base_2types', 'w') as f:
        f.write(content)
    print("Created inputs/data.ga_base_2types")


def calc_sq_from_gr(r, g, rho=0.0522):
    """
    Calculate S(Q) from g(r) using Fourier transform.
    rho = 0.0522 atoms/A^3 for Ga at 150C
    """
    Q = np.linspace(0.5, 20.0, 400)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1

    for i, q in enumerate(Q):
        if q < 1e-6:
            continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q

    return Q, S


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


def main():
    os.makedirs("outputs/bimodal_search", exist_ok=True)

    # Prepare base data
    try:
        prepare_base_data()
    except Exception as e:
        print(f"Warning: {e}")

    results = {}

    print("=" * 70)
    print("Bimodal Ga Grid Search")
    print("=" * 70)
    print(f"Temperature: {TEMP} K (150 C)")
    print(f"Base sigma: {SIGMA_BASE} A")
    print(f"Ga1-Ga1 sigma ratio: {SIGMA_RATIO_GA1} (= {SIGMA_BASE * SIGMA_RATIO_GA1:.3f} A)")
    print(f"Ga2-Ga2 sigma ratio: {SIGMA_RATIO_GA2} (= {SIGMA_BASE * SIGMA_RATIO_GA2:.3f} A)")
    print(f"Ga1-Ga2 sigma: Lorentz-Berthelot mixing")
    print(f"Epsilon: {EPSILON_BASE} kcal/mol (fixed)")
    print(f"Ga1 fractions to test: {GA1_FRACTIONS}")
    print("=" * 70)

    for ga1_frac in GA1_FRACTIONS:
        label = f"ga1_{int(ga1_frac*100):02d}"
        print(f"\nProcessing: Ga1={ga1_frac:.0%}, Ga2={1-ga1_frac:.0%} ... ", end="", flush=True)

        input_file = f"outputs/bimodal_search/in.{label}"
        output_prefix = f"outputs/bimodal_search/out_{label}"

        write_lammps_input(input_file, output_prefix, ga1_frac)

        cmd = f"{LMP_CMD} -in {input_file} -log outputs/bimodal_search/log.{label}"
        try:
            subprocess.run(cmd, shell=True, check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         timeout=600)
            print("Done.")
        except subprocess.CalledProcessError as e:
            print(f"Failed! {e}")
            continue
        except subprocess.TimeoutExpired:
            print("Timeout!")
            continue

        # Analyze RDF
        try:
            data = load_rdf_robust(f"{output_prefix}.rdf")
            r = data[:, 1]

            # Columns: index, r, g11, c11, g22, c22, g12, c12
            if data.shape[1] >= 7:
                g11 = data[:, 2]  # Ga1-Ga1
                g22 = data[:, 4]  # Ga2-Ga2
                g12 = data[:, 6]  # Ga1-Ga2

                # Weighted total g(r)
                # For composition x1:x2, total g(r) = x1^2*g11 + x2^2*g22 + 2*x1*x2*g12
                x1 = ga1_frac
                x2 = 1 - ga1_frac
                g_total = x1**2 * g11 + x2**2 * g22 + 2 * x1 * x2 * g12
            else:
                g_total = data[:, 2]

            Q, S = calc_sq_from_gr(r, g_total)

            # Store results including partial g(r)
            results[ga1_frac] = {
                'r': r, 'g_total': g_total, 'Q': Q, 'S': S,
                'g11': g11 if data.shape[1] >= 7 else None,
                'g22': g22 if data.shape[1] >= 7 else None,
                'g12': g12 if data.shape[1] >= 7 else None,
            }

            # Find peak position
            peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
            peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
            print(f"  -> S(Q) peak at Q = {peak_q:.2f} A^-1")

        except Exception as e:
            print(f"Analysis Error: {e}")

    # Plotting
    print("\n" + "=" * 70)
    print("Generating Summary Plots...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['blue', 'red', 'green']
    labels_list = []

    for idx, ga1_frac in enumerate(GA1_FRACTIONS):
        if ga1_frac not in results:
            continue

        res = results[ga1_frac]
        Q, S = res['Q'], res['S']
        r, g = res['r'], res['g_total']

        color = colors[idx % len(colors)]
        label = f"Ga1:{ga1_frac:.0%}, Ga2:{1-ga1_frac:.0%}"
        labels_list.append(label)

        # Top row: S(Q)
        axes[0, idx].plot(Q, S, color=color, lw=2)
        axes[0, idx].set_xlim(0, 12)
        axes[0, idx].set_ylim(0, 3.5)
        axes[0, idx].set_xlabel("Q (A^-1)", fontsize=12)
        axes[0, idx].set_ylabel("S(Q)", fontsize=12)
        axes[0, idx].set_title(label, fontsize=14, fontweight='bold')
        axes[0, idx].grid(True, alpha=0.3)

        # Highlight shoulder region (Q ~ 2.5-3.5)
        axes[0, idx].axvspan(2.5, 3.5, color='orange', alpha=0.15, label='Shoulder region')

        # Mark peak
        peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
        peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
        peak_s = S[(Q > 1.5) & (Q < 4.0)][peak_idx]
        axes[0, idx].plot(peak_q, peak_s, 'ko', markersize=8)
        axes[0, idx].text(peak_q + 0.3, peak_s, f"Q={peak_q:.2f}", fontsize=10)

        # Bottom row: g(r)
        axes[1, idx].plot(r, g, color=color, lw=2, label='Total')
        if res['g11'] is not None:
            axes[1, idx].plot(r, res['g11'], '--', color='darkblue', lw=1, alpha=0.7, label='Ga1-Ga1')
            axes[1, idx].plot(r, res['g22'], '--', color='darkred', lw=1, alpha=0.7, label='Ga2-Ga2')
            axes[1, idx].plot(r, res['g12'], '--', color='darkgreen', lw=1, alpha=0.7, label='Ga1-Ga2')
        axes[1, idx].set_xlim(0, 10)
        axes[1, idx].set_ylim(0, 3.5)
        axes[1, idx].set_xlabel("r (A)", fontsize=12)
        axes[1, idx].set_ylabel("g(r)", fontsize=12)
        axes[1, idx].set_title(f"g(r) - {label}", fontsize=12)
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].legend(fontsize=8, loc='upper right')

    plt.suptitle(f"Bimodal Ga Model: sigma(Ga1)={SIGMA_RATIO_GA1}x, sigma(Ga2)={SIGMA_RATIO_GA2}x\n"
                 f"T={TEMP:.1f}K, eps={EPSILON_BASE} kcal/mol (fixed)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = "outputs/bimodal_search/bimodal_ga_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Also create overlay comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, ga1_frac in enumerate(GA1_FRACTIONS):
        if ga1_frac not in results:
            continue
        res = results[ga1_frac]
        label = f"Ga1:{ga1_frac:.0%}, Ga2:{1-ga1_frac:.0%}"
        color = colors[idx % len(colors)]

        ax1.plot(res['Q'], res['S'], color=color, lw=2, label=label)
        ax2.plot(res['r'], res['g_total'], color=color, lw=2, label=label)

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

    plt.suptitle(f"Bimodal Ga: Effect of Ga1/Ga2 Ratio\n"
                 f"sigma(Ga1)={SIGMA_BASE*SIGMA_RATIO_GA1:.2f}A, "
                 f"sigma(Ga2)={SIGMA_BASE*SIGMA_RATIO_GA2:.2f}A",
                 fontsize=12)
    plt.tight_layout()

    output_file2 = "outputs/bimodal_search/bimodal_ga_overlay.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")

    # Summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETED")
    print("=" * 70)
    print(f"Successfully completed: {len(results)}/{len(GA1_FRACTIONS)} simulations")
    print("\nResults summary:")
    for ga1_frac in GA1_FRACTIONS:
        if ga1_frac in results:
            Q, S = results[ga1_frac]['Q'], results[ga1_frac]['S']
            peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
            peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
            peak_s = S[(Q > 1.5) & (Q < 4.0)][peak_idx]
            print(f"  Ga1={ga1_frac:.0%}: S(Q) peak at Q={peak_q:.2f} A^-1, S_max={peak_s:.2f}")


if __name__ == "__main__":
    main()
