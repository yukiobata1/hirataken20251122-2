#!/usr/bin/env python3
"""
Run Ga-only LJ simulation and generate S(Q) plot.

This script:
1. Runs LAMMPS simulation for pure Gallium at 150°C
2. Computes S(Q) from the RDF output
3. Generates plots for g(r) and S(Q)

Usage:
    python scripts/run_ga_lj_sim_and_analyze.py [--gpu]
"""

import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


def run_lammps_simulation(gpu=False):
    """Run LAMMPS simulation."""
    input_file = 'inputs/in.ga_lj_150C'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print("=" * 60)
    print("Running LAMMPS simulation...")
    print("=" * 60)

    # Determine LAMMPS command
    if gpu:
        lammps_cmd = 'lmp_gpu'
    else:
        lammps_cmd = 'lmp'

    # Run LAMMPS
    try:
        result = subprocess.run(
            [lammps_cmd, '-in', input_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"LAMMPS returned exit code {result.returncode}")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            sys.exit(1)

        print("✓ LAMMPS simulation completed successfully")
        return True

    except FileNotFoundError:
        print(f"Error: LAMMPS executable '{lammps_cmd}' not found.")
        print("Make sure LAMMPS is installed and in your PATH.")
        sys.exit(1)


def load_rdf_ga(filepath):
    """
    Load RDF file for pure Ga system.
    Expected LAMMPS output columns: Row Bin R g(r) c(r)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        data_lines.append(line)

    if not data_lines:
        print(f"Error: No data found in {filepath}")
        sys.exit(1)

    data = np.loadtxt(StringIO("\n".join(data_lines)))
    return data


def calculate_sq(r, g, rho=0.0522):
    """
    Calculate S(Q) from g(r) using Fourier transform.

    S(Q) = 1 + (4π * ρ / Q) * ∫ (g(r) - 1) * r * sin(Qr) dr

    Parameters:
    - r: radial distances
    - g: pair distribution function g(r)
    - rho: number density (atoms/Å³), default 0.0522 for Ga at 150°C

    Returns:
    - Q: scattering vector
    - S: structure factor
    """
    Q = np.linspace(0.1, 20.0, 500)
    S = np.ones_like(Q)

    dr = r[1] - r[0] if len(r) > 1 else 0.1

    for i, q in enumerate(Q):
        if q < 1e-6:
            continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + (4.0 * np.pi * rho / q) * np.sum(integrand) * dr

    return Q, S


def plot_results(r, g, Q, S, output_dir='outputs'):
    """Generate g(r) and S(Q) plots."""

    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: g(r)
    ax1.plot(r, g, 'b-', linewidth=2, label='g(r)')
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='No correlation')
    ax1.set_xlabel('r (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.set_title('Pair Distribution Function - Liquid Ga at 150°C', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 3)

    # Plot 2: S(Q)
    ax2.plot(Q, S, 'r-', linewidth=2, label='S(Q)')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Ideal gas')
    ax2.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('S(Q)', fontsize=12)
    ax2.set_title('Static Structure Factor - Liquid Ga at 150°C', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-0.5, 3)

    # Highlight first peak region
    ax2.axvspan(1.8, 2.5, color='yellow', alpha=0.1)

    output_file = os.path.join(output_dir, 'ga_lj_sq.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()

    # Also save S(Q) data to file
    sq_data_file = os.path.join(output_dir, 'ga_lj_sq.txt')
    np.savetxt(sq_data_file, np.column_stack([Q, S]),
               header='Q (Ang^-1)  S(Q)', comments='# ')
    print(f"✓ S(Q) data saved to: {sq_data_file}")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Ga LJ simulation and generate S(Q) plot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--rdf-file', default='rdf.dat', help='RDF output file')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--skip-simulation', action='store_true',
                       help='Skip simulation and analyze existing RDF file')

    args = parser.parse_args()

    # Run simulation unless skipped
    if not args.skip_simulation:
        run_lammps_simulation(gpu=args.gpu)

    # Check if RDF file exists
    if not os.path.exists(args.rdf_file):
        print(f"Error: RDF file not found: {args.rdf_file}")
        print("Make sure simulation ran successfully.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Loading RDF and calculating S(Q)...")
    print("=" * 60)

    # Load RDF data
    data = load_rdf_ga(args.rdf_file)

    # Extract columns
    # LAMMPS RDF output: Row Bin R g(r) c(r)
    # Columns:            0   1  2  3    4
    r = data[:, 2]  # r values
    g = data[:, 3]  # g(r)

    print(f"Loaded RDF data: {len(r)} points from r=0 to r={r[-1]:.2f} Å")

    # Calculate S(Q)
    # Density for Ga at 150°C: 6.04 g/cm³ = 0.0522 atoms/Å³
    rho = 0.0522
    Q, S = calculate_sq(r, g, rho=rho)

    print(f"Calculated S(Q): {len(Q)} points from Q=0.1 to Q={Q[-1]:.2f} Å⁻¹")
    print(f"Number density: {rho:.4f} atoms/Å³")

    # Generate plots
    plot_results(r, g, Q, S, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
