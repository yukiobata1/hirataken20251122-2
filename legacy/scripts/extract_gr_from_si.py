#!/usr/bin/env python3
"""
Utility script to extract and process g(r) data from Supporting Information
or to manually input g(r) data
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def sort_and_save_gr_data(input_file, output_file='data/g_exp.dat', plot=True):
    """
    Load, sort, and save g(r) data

    Parameters
    ----------
    input_file : str
        Input data file (2 columns: r, g(r))
    output_file : str
        Output data file (default: 'data/g_exp.dat')
    plot : bool
        Whether to plot the data (default: True)
    """
    # Load data
    print(f"Loading data from {input_file}...")
    data = np.loadtxt(input_file)

    r = data[:, 0]
    g = data[:, 1]

    print(f"Loaded {len(r)} data points")
    print(f"r range: {r.min():.2f} - {r.max():.2f} Å")
    print(f"g(r) range: {g.min():.3f} - {g.max():.3f}")

    # Sort by r
    sorted_indices = np.argsort(r)
    r_sorted = r[sorted_indices]
    g_sorted = g[sorted_indices]

    # Check for duplicates
    unique_r, unique_indices = np.unique(r_sorted, return_index=True)
    if len(unique_r) < len(r_sorted):
        print(f"Warning: Found {len(r_sorted) - len(unique_r)} duplicate r values")
        print("Keeping only unique values...")
        r_sorted = r_sorted[unique_indices]
        g_sorted = g_sorted[unique_indices]

    # Save sorted data
    with open(output_file, 'w') as f:
        f.write('# Experimental G(r) data for Ga-In alloy\n')
        f.write('# Column 1: r (Å)\n')
        f.write('# Column 2: G(r)\n')
        f.write('# Data extracted from Amon et al., J. Phys. Chem. C 2023, 127, 16687-16694\n')
        for ri, gi in zip(r_sorted, g_sorted):
            f.write(f'{ri:.10f} {gi:.10f}\n')

    print(f"\nSorted data saved to {output_file}")

    # Plot if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(r_sorted, g_sorted, 'o-', markersize=3)
        plt.xlabel('r (Å)', fontsize=12)
        plt.ylabel('G(r)', fontsize=12)
        plt.title('Experimental Pair Distribution Function', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = output_file.replace('.dat', '_plot.png')
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.show()

    return r_sorted, g_sorted


def manual_input_gr_data(output_file='data/g_exp_manual.dat'):
    """
    Template for manually inputting g(r) data

    Parameters
    ----------
    output_file : str
        Output data file
    """
    print("Manual g(r) data input template")
    print("=" * 60)
    print("Replace the example data below with your actual data")
    print("=" * 60)

    # Example template data (replace with actual values)
    r_data = np.array([
        2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
        3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.5, 5.0, 5.5, 6.0
    ])

    g_data = np.array([
        0.0, 0.1, 0.3, 0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.2,
        1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ])

    # Save template
    with open(output_file, 'w') as f:
        f.write('# Template for manual g(r) data input\n')
        f.write('# REPLACE THE VALUES BELOW WITH YOUR ACTUAL DATA\n')
        f.write('# Column 1: r (Å)\n')
        f.write('# Column 2: G(r)\n')
        for ri, gi in zip(r_data, g_data):
            f.write(f'{ri:.6f} {gi:.6f}\n')

    print(f"Template saved to {output_file}")
    print("Edit this file with your actual g(r) values")


def interpolate_gr_data(input_file, r_min=2.0, r_max=12.0, N=200, output_file='data/g_exp_interp.dat'):
    """
    Interpolate g(r) data to a uniform grid

    Parameters
    ----------
    input_file : str
        Input g(r) data file
    r_min : float
        Minimum r for interpolation (Å)
    r_max : float
        Maximum r for interpolation (Å)
    N : int
        Number of interpolation points
    output_file : str
        Output file
    """
    # Load data
    data = np.loadtxt(input_file)
    r = data[:, 0]
    g = data[:, 1]

    # Create uniform grid
    r_uniform = np.linspace(r_min, r_max, N)

    # Interpolate
    g_uniform = np.interp(r_uniform, r, g, left=g[0], right=g[-1])

    # Save
    with open(output_file, 'w') as f:
        f.write('# Interpolated G(r) data\n')
        f.write(f'# Uniform grid: {r_min} - {r_max} Å, {N} points\n')
        f.write('# Column 1: r (Å)\n')
        f.write('# Column 2: G(r)\n')
        for ri, gi in zip(r_uniform, g_uniform):
            f.write(f'{ri:.10f} {gi:.10f}\n')

    print(f"Interpolated data saved to {output_file}")
    print(f"Grid: {r_min} - {r_max} Å with {N} points")

    return r_uniform, g_uniform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process g(r) data')
    parser.add_argument('--input', '-i', type=str, help='Input g(r) data file')
    parser.add_argument('--output', '-o', type=str, default='data/g_exp.dat',
                        help='Output file (default: data/g_exp.dat)')
    parser.add_argument('--manual', action='store_true',
                        help='Create manual input template')
    parser.add_argument('--interpolate', action='store_true',
                        help='Interpolate to uniform grid')
    parser.add_argument('--rmin', type=float, default=2.0,
                        help='Min r for interpolation (default: 2.0)')
    parser.add_argument('--rmax', type=float, default=12.0,
                        help='Max r for interpolation (default: 12.0)')
    parser.add_argument('--npoints', type=int, default=200,
                        help='Number of interpolation points (default: 200)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not plot data')

    args = parser.parse_args()

    if args.manual:
        manual_input_gr_data(args.output)
    elif args.interpolate:
        if not args.input:
            print("Error: --input required for interpolation")
        else:
            interpolate_gr_data(args.input, args.rmin, args.rmax, args.npoints, args.output)
    elif args.input:
        sort_and_save_gr_data(args.input, args.output, plot=not args.no_plot)
    else:
        print("No action specified. Use --input, --manual, or --interpolate")
        parser.print_help()
