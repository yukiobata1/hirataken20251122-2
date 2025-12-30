#!/usr/bin/env python3
"""
Grid Search for Bimodal Ga Structure (Ga1/Ga2 model) - Version 2

Modified based on Prof. Hirata's suggestion:
- Ga1-Ga1: sigma * 1.1 (larger)
- Ga2-Ga2: sigma * 0.9 (smaller)
- Ga1-Ga2: sigma * 0.9 or 1.1 (NOT 1.0 to avoid filling the gap)

The idea: If Ga1-Ga2 is at the midpoint (1.0), it fills the gap between
the two peaks and makes splitting harder to observe.
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

# Bimodal sigma ratios
SIGMA_RATIO_GA1 = 1.1  # Ga1-Ga1 (larger)
SIGMA_RATIO_GA2 = 0.9  # Ga2-Ga2 (smaller)

# NEW: Ga1-Ga2 sigma ratios to test (not using Lorentz-Berthelot)
SIGMA_RATIO_GA12_LIST = [0.9, 1.1]  # Test both: closer to Ga2 or closer to Ga1

# Composition ratios to test (Ga1 fraction)
GA1_FRACTIONS = [0.50, 0.75, 0.25]

# LAMMPS Command (KOKKOS GPU)
LMP_CMD = "lmp -k on g 1 -sf kk"
# =============================================


def write_lammps_input(filename, output_prefix, ga1_frac, sigma_ratio_12):
    """
    Write LAMMPS input file for bimodal Ga simulation.

    ga1_frac: fraction of Ga1 atoms (rest is Ga2)
    sigma_ratio_12: sigma ratio for Ga1-Ga2 interaction (0.9 or 1.1)
    """
    # Sigma values
    sigma1 = SIGMA_BASE * SIGMA_RATIO_GA1   # Ga1-Ga1
    sigma2 = SIGMA_BASE * SIGMA_RATIO_GA2   # Ga2-Ga2
    sigma12 = SIGMA_BASE * sigma_ratio_12   # Ga1-Ga2 (explicit, NOT L-B)

    # Epsilon values (all fixed)
    eps1 = EPSILON_BASE
    eps2 = EPSILON_BASE
    eps12 = np.sqrt(eps1 * eps2)  # Lorentz-Berthelot for epsilon

    # Random seed based on composition and sigma12
    seed = int(ga1_frac * 10000 + sigma_ratio_12 * 1000) + 12345

    content = f"""# Bimodal Ga Grid Search V2: Ga1 fraction = {ga1_frac:.2f}
# Ga1-Ga1 sigma = {sigma1:.4f} A (x{SIGMA_RATIO_GA1})
# Ga2-Ga2 sigma = {sigma2:.4f} A (x{SIGMA_RATIO_GA2})
# Ga1-Ga2 sigma = {sigma12:.4f} A (x{sigma_ratio_12}) << KEY CHANGE
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
    output_dir = "outputs/bimodal_search_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare base data
    try:
        prepare_base_data()
    except Exception as e:
        print(f"Warning: {e}")

    results = {}

    print("=" * 70)
    print("Bimodal Ga Grid Search V2 - Testing Ga1-Ga2 sigma ratios")
    print("=" * 70)
    print(f"Temperature: {TEMP} K (150 C)")
    print(f"Base sigma: {SIGMA_BASE} A")
    print(f"Ga1-Ga1 sigma ratio: {SIGMA_RATIO_GA1} (= {SIGMA_BASE * SIGMA_RATIO_GA1:.3f} A)")
    print(f"Ga2-Ga2 sigma ratio: {SIGMA_RATIO_GA2} (= {SIGMA_BASE * SIGMA_RATIO_GA2:.3f} A)")
    print(f"Ga1-Ga2 sigma ratios to test: {SIGMA_RATIO_GA12_LIST}")
    print(f"Ga1 fractions to test: {GA1_FRACTIONS}")
    print(f"Epsilon: {EPSILON_BASE} kcal/mol (fixed)")
    print("=" * 70)

    # Run simulations for each combination
    for sigma_ratio_12 in SIGMA_RATIO_GA12_LIST:
        print(f"\n{'='*70}")
        print(f"Ga1-Ga2 sigma ratio = {sigma_ratio_12} (= {SIGMA_BASE * sigma_ratio_12:.3f} A)")
        print(f"{'='*70}")

        for ga1_frac in GA1_FRACTIONS:
            label = f"sig12_{int(sigma_ratio_12*100):03d}_ga1_{int(ga1_frac*100):02d}"
            print(f"\nProcessing: sigma12={sigma_ratio_12}, Ga1={ga1_frac:.0%}, Ga2={1-ga1_frac:.0%} ... ", end="", flush=True)

            input_file = f"{output_dir}/in.{label}"
            output_prefix = f"{output_dir}/out_{label}"

            write_lammps_input(input_file, output_prefix, ga1_frac, sigma_ratio_12)

            cmd = f"{LMP_CMD} -in {input_file} -log {output_dir}/log.{label}"
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
                    x1 = ga1_frac
                    x2 = 1 - ga1_frac
                    g_total = x1**2 * g11 + x2**2 * g22 + 2 * x1 * x2 * g12
                else:
                    g_total = data[:, 2]
                    g11 = g22 = g12 = None

                Q, S = calc_sq_from_gr(r, g_total)

                # Store results
                key = (sigma_ratio_12, ga1_frac)
                results[key] = {
                    'r': r, 'g_total': g_total, 'Q': Q, 'S': S,
                    'g11': g11, 'g22': g22, 'g12': g12,
                    'sigma_ratio_12': sigma_ratio_12,
                    'ga1_frac': ga1_frac,
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

    # Plot 1: Comparison by sigma_ratio_12
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors_frac = {'0.50': 'blue', '0.75': 'red', '0.25': 'green'}
    linestyles_sig = {0.9: '-', 1.1: '--'}

    # Top-left: S(Q) for sigma12=0.9
    ax = axes[0, 0]
    for ga1_frac in GA1_FRACTIONS:
        key = (0.9, ga1_frac)
        if key in results:
            res = results[key]
            label = f"Ga1:{ga1_frac:.0%}"
            color = colors_frac[f"{ga1_frac:.2f}"]
            ax.plot(res['Q'], res['S'], color=color, lw=2, label=label)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("Q (A^-1)", fontsize=12)
    ax.set_ylabel("S(Q)", fontsize=12)
    ax.set_title(f"S(Q): Ga1-Ga2 sigma = 0.9x ({SIGMA_BASE*0.9:.2f} A)\n(closer to Ga2)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    # Top-right: S(Q) for sigma12=1.1
    ax = axes[0, 1]
    for ga1_frac in GA1_FRACTIONS:
        key = (1.1, ga1_frac)
        if key in results:
            res = results[key]
            label = f"Ga1:{ga1_frac:.0%}"
            color = colors_frac[f"{ga1_frac:.2f}"]
            ax.plot(res['Q'], res['S'], color=color, lw=2, label=label)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("Q (A^-1)", fontsize=12)
    ax.set_ylabel("S(Q)", fontsize=12)
    ax.set_title(f"S(Q): Ga1-Ga2 sigma = 1.1x ({SIGMA_BASE*1.1:.2f} A)\n(closer to Ga1)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    # Bottom-left: g(r) for sigma12=0.9
    ax = axes[1, 0]
    for ga1_frac in GA1_FRACTIONS:
        key = (0.9, ga1_frac)
        if key in results:
            res = results[key]
            label = f"Ga1:{ga1_frac:.0%}"
            color = colors_frac[f"{ga1_frac:.2f}"]
            ax.plot(res['r'], res['g_total'], color=color, lw=2, label=label)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("r (A)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_title(f"g(r): Ga1-Ga2 sigma = 0.9x", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom-right: g(r) for sigma12=1.1
    ax = axes[1, 1]
    for ga1_frac in GA1_FRACTIONS:
        key = (1.1, ga1_frac)
        if key in results:
            res = results[key]
            label = f"Ga1:{ga1_frac:.0%}"
            color = colors_frac[f"{ga1_frac:.2f}"]
            ax.plot(res['r'], res['g_total'], color=color, lw=2, label=label)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("r (A)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_title(f"g(r): Ga1-Ga2 sigma = 1.1x", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Bimodal Ga Model V2: Effect of Ga1-Ga2 sigma\n"
                 f"Ga1-Ga1={SIGMA_RATIO_GA1}x, Ga2-Ga2={SIGMA_RATIO_GA2}x, T={TEMP:.1f}K",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f"{output_dir}/bimodal_ga_v2_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Plot 2: Direct comparison sigma12=0.9 vs 1.1 for each composition
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ga1_frac in enumerate(GA1_FRACTIONS):
        ax = axes2[idx]
        for sigma_ratio_12 in SIGMA_RATIO_GA12_LIST:
            key = (sigma_ratio_12, ga1_frac)
            if key in results:
                res = results[key]
                ls = linestyles_sig[sigma_ratio_12]
                label = f"σ12={sigma_ratio_12}x"
                ax.plot(res['Q'], res['S'], ls, lw=2, label=label)

        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel("Q (A^-1)", fontsize=12)
        ax.set_ylabel("S(Q)", fontsize=12)
        ax.set_title(f"Ga1:{ga1_frac:.0%}, Ga2:{1-ga1_frac:.0%}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvspan(2.5, 3.5, color='orange', alpha=0.15)

    plt.suptitle(f"Effect of Ga1-Ga2 sigma: 0.9x vs 1.1x comparison\n"
                 f"(σ11={SIGMA_RATIO_GA1}x, σ22={SIGMA_RATIO_GA2}x fixed)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file2 = f"{output_dir}/bimodal_ga_v2_sig12_comparison.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")

    # Summary
    print("\n" + "=" * 70)
    print("GRID SEARCH V2 COMPLETED")
    print("=" * 70)
    print(f"Successfully completed: {len(results)}/{len(SIGMA_RATIO_GA12_LIST) * len(GA1_FRACTIONS)} simulations")
    print("\nResults summary:")
    for sigma_ratio_12 in SIGMA_RATIO_GA12_LIST:
        print(f"\n  Ga1-Ga2 sigma = {sigma_ratio_12}x ({SIGMA_BASE * sigma_ratio_12:.2f} A):")
        for ga1_frac in GA1_FRACTIONS:
            key = (sigma_ratio_12, ga1_frac)
            if key in results:
                Q, S = results[key]['Q'], results[key]['S']
                peak_idx = np.argmax(S[(Q > 1.5) & (Q < 4.0)])
                peak_q = Q[(Q > 1.5) & (Q < 4.0)][peak_idx]
                peak_s = S[(Q > 1.5) & (Q < 4.0)][peak_idx]
                print(f"    Ga1={ga1_frac:.0%}: S(Q) peak at Q={peak_q:.2f} A^-1, S_max={peak_s:.2f}")


if __name__ == "__main__":
    main()
