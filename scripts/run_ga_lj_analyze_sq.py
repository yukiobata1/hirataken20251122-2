#!/usr/bin/env python3
"""
Analyze Ga LJ simulation and generate S(Q) plot.

This script reads RDF from LAMMPS output and computes S(Q).

Usage:
    python scripts/run_ga_lj_analyze_sq.py [--rdf-file rdf.dat] [--output-dir outputs]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


def load_rdf_ga(filepath):
    """
    Load RDF file for pure Ga system.
    Expected LAMMPS output columns: Row Bin R g(r) c(r)

    For time-averaged LAMMPS output with multiple timesteps,
    only reads the last (most converged) timestep.
    """
    print(f"Loading RDF from: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find all timestep headers (lines with format "timestep num_rows")
    timestep_indices = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 2 and not line.startswith('#'):
            try:
                # Check if both parts are integers
                int(parts[0])
                int(parts[1])
                timestep_indices.append(i)
            except ValueError:
                continue

    if timestep_indices:
        # Read only the last timestep
        last_timestep_idx = timestep_indices[-1]
        num_rows = int(lines[last_timestep_idx].split()[1])
        print(f"Found {len(timestep_indices)} timesteps in file")
        print(f"Reading last timestep (line {last_timestep_idx + 1}): {lines[last_timestep_idx].strip()}")

        # Extract data lines for last timestep
        data_lines = []
        for i in range(last_timestep_idx + 1, min(last_timestep_idx + 1 + num_rows, len(lines))):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                data_lines.append(line)
    else:
        # Fallback: no timestep headers found, read all data
        print("No timestep headers found, reading all data lines")
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
    - r: radial distances (Å)
    - g: pair distribution function g(r)
    - rho: number density (atoms/Å³), default 0.0522 for Ga at 150°C
           (6.04 g/cm³ / 69.723 g/mol * 6.022e23)

    Returns:
    - Q: scattering vector (Å⁻¹)
    - S: structure factor S(Q)
    """
    print(f"Calculating S(Q)...")
    print(f"  Number density: {rho:.6f} atoms/Å³")

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

    print(f"Generating plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: g(r)
    ax1.plot(r, g, 'b-', linewidth=2.5, label='g(r)')
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
    ax1.set_xlabel('r (Å)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('g(r)', fontsize=12, fontweight='bold')
    ax1.set_title('Pair Distribution Function\nLiquid Ga at 150°C (LJ)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 3)

    # Add annotations for key features
    first_peak_idx = np.argmax(g[(r > 2.0) & (r < 3.5)]) + np.where(r > 2.0)[0][0]
    if first_peak_idx < len(r):
        ax1.plot(r[first_peak_idx], g[first_peak_idx], 'ro', markersize=6)
        ax1.text(r[first_peak_idx], g[first_peak_idx] + 0.15,
                f'r ≈ {r[first_peak_idx]:.2f} Å', ha='center', fontsize=9)

    # Plot 2: S(Q)
    ax2.plot(Q, S, 'r-', linewidth=2.5, label='S(Q)')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1.5, label='Ideal gas')
    ax2.set_xlabel('Q (Å⁻¹)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('S(Q)', fontsize=12, fontweight='bold')
    ax2.set_title('Static Structure Factor\nLiquid Ga at 150°C (LJ)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-0.5, 3)

    # Highlight first peak region
    first_peak_q = 2.0 * np.pi / r[first_peak_idx] if first_peak_idx < len(r) else 2.2
    ax2.axvspan(first_peak_q - 0.3, first_peak_q + 0.3, color='yellow', alpha=0.1)
    ax2.plot(first_peak_q, 1.0 + (4.0 * np.pi * 0.0522 / first_peak_q), 'ro', markersize=6)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'ga_lj_sq.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()

    # Save S(Q) data to file
    sq_data_file = os.path.join(output_dir, 'ga_lj_sq.txt')
    header = 'Q (Ang^-1)     S(Q)'
    np.savetxt(sq_data_file, np.column_stack([Q, S]), header=header, comments='# ')
    print(f"✓ S(Q) data saved to: {sq_data_file}")

    # Also save g(r) data
    gr_data_file = os.path.join(output_dir, 'ga_lj_gr.txt')
    header = 'r (Ang)        g(r)'
    np.savetxt(gr_data_file, np.column_stack([r, g]), header=header, comments='# ')
    print(f"✓ g(r) data saved to: {gr_data_file}")


def print_statistics(r, g, Q, S):
    """Print statistics about S(Q) and g(r)."""
    print("")
    print("=" * 60)
    print("Statistics")
    print("=" * 60)

    # g(r) statistics
    first_peak_idx = np.argmax(g[(r > 2.0) & (r < 3.5)]) + np.where(r > 2.0)[0][0]
    print(f"\ng(r) statistics:")
    print(f"  First peak position: r ≈ {r[first_peak_idx]:.3f} Å")
    print(f"  First peak height: g(r) ≈ {g[first_peak_idx]:.3f}")
    print(f"  Data range: r = [{r[0]:.3f}, {r[-1]:.3f}] Å")

    # S(Q) statistics
    first_sq_peak_idx = np.argmax(S[(Q > 1.0) & (Q < 3.0)]) + np.where(Q > 1.0)[0][0]
    print(f"\nS(Q) statistics:")
    print(f"  First peak position: Q ≈ {Q[first_sq_peak_idx]:.3f} Å⁻¹")
    print(f"  First peak height: S(Q) ≈ {S[first_sq_peak_idx]:.3f}")
    print(f"  Data range: Q = [{Q[0]:.3f}, {Q[-1]:.3f}] Å⁻¹")

    # Coordination number estimate
    # CN ≈ ρ * ∫ g(r) 4πr² dr (up to first minimum)
    first_min_idx = np.where((r > r[first_peak_idx]) & (g < 1.0))[0]
    if len(first_min_idx) > 0:
        r_cutoff = r[first_min_idx[0]]
        r_int = r[r <= r_cutoff]
        g_int = g[r <= r_cutoff]
        cn = 4 * np.pi * 0.0522 * np.trapezoid(g_int * r_int**2, r_int)
        print(f"\nCoordination number estimate:")
        print(f"  First minimum at r ≈ {r_cutoff:.3f} Å")
        print(f"  Estimated CN ≈ {cn:.2f}")

    print("")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze Ga LJ simulation and generate S(Q) plot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--rdf-file', default='rdf.dat', help='RDF output file from LAMMPS')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--density', type=float, default=0.0522,
                       help='Number density (atoms/Å³)')

    args = parser.parse_args()

    print("=" * 60)
    print("Ga LJ Simulation Analysis")
    print("=" * 60)
    print()

    # Check if RDF file exists
    if not os.path.exists(args.rdf_file):
        print(f"Error: RDF file not found: {args.rdf_file}")
        print("\nMake sure to run the simulation first:")
        print("  ./run_ga_lj_sim.sh")
        sys.exit(1)

    # Load RDF data
    data = load_rdf_ga(args.rdf_file)

    # Extract columns
    # LAMMPS RDF output: Row R g(r) c(r)
    # Columns:            0   1  2    3
    if data.shape[1] < 3:
        print(f"Error: RDF file has insufficient columns: {data.shape[1]}")
        sys.exit(1)

    r = data[:, 1]  # r values (Å)
    g = data[:, 2]  # g(r)

    print(f"Loaded RDF data:")
    print(f"  Points: {len(r)}")
    print(f"  r range: {r[0]:.3f} - {r[-1]:.3f} Å")
    print()

    # Calculate S(Q)
    Q, S = calculate_sq(r, g, rho=args.density)

    print(f"Calculated S(Q):")
    print(f"  Points: {len(Q)}")
    print(f"  Q range: {Q[0]:.3f} - {Q[-1]:.3f} Å⁻¹")
    print()

    # Generate plots
    plot_results(r, g, Q, S, output_dir=args.output_dir)

    # Print statistics
    print_statistics(r, g, Q, S)

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
