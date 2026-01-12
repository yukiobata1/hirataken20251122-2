#!/usr/bin/env python3
"""
Calculate Structure Factor S(Q) from RDF g(r) data
Using Fourier transform relationship

S(Q) = 1 + 4πρ ∫ r² [g(r) - 1] sin(Qr)/(Qr) dr

Usage:
    python calculate_sq.py rdf_293K.dat 6.11
    python calculate_sq.py rdf_293K.dat 6.11 --plot
    python calculate_sq.py --all  # Process all temperature files
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.integrate import simpson
except ImportError:
    # Fallback for older scipy versions
    from scipy.integrate import simps as simpson
import argparse
import os
import glob

# Physical constants
AVOGADRO = 6.02214076e23  # mol^-1
GA_MASS = 69.723  # g/mol


def read_rdf_file(filename):
    """
    Read LAMMPS RDF output file
    
    Returns:
        r: distance array (Å)
        g_r: g(r) array
    """
    print(f"Reading {filename}...")
    
    # Skip header lines and read data
    data_lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Skip LAMMPS header (lines with non-numeric first field)
            parts = line.split()
            try:
                # Try to parse as numbers
                float(parts[0])
                # Check if this is actual RDF data (has r and g(r))
                if len(parts) >= 3:
                    data_lines.append(parts)
            except ValueError:
                continue
    
    if not data_lines:
        raise ValueError(f"No valid data found in {filename}")
    
    # Parse data
    # LAMMPS RDF format: index r g(r) coord(r)
    data = np.array(data_lines, dtype=float)
    
    # Columns: 0=bin, 1=r, 2=g(r), 3=coord (cumulative)
    if data.shape[1] >= 3:
        r = data[:, 1]
        g_r = data[:, 2]
    else:
        raise ValueError(f"Unexpected data format in {filename}")
    
    print(f"  Data range: r = {r[0]:.3f} to {r[-1]:.3f} Å")
    print(f"  Number of points: {len(r)}")
    print(f"  g(r) peak value: {np.max(g_r):.3f} at r = {r[np.argmax(g_r)]:.3f} Å")
    
    return r, g_r


def density_to_number_density(density_g_cm3, molar_mass=GA_MASS):
    """
    Convert mass density (g/cm³) to number density (atoms/Å³)
    
    ρ_n = ρ_m × N_A / M
    
    Args:
        density_g_cm3: mass density in g/cm³
        molar_mass: molar mass in g/mol
    
    Returns:
        number density in atoms/Å³
    """
    # g/cm³ → atoms/cm³ → atoms/Å³
    # 1 cm³ = 10^24 Å³
    rho_n = density_g_cm3 * AVOGADRO / molar_mass / 1e24
    return rho_n


def calculate_sq(r, g_r, rho_n, q_min=0.1, q_max=15.0, n_q=500):
    """
    Calculate S(Q) from g(r) using Fourier transform
    
    S(Q) = 1 + 4πρ ∫ r² [g(r) - 1] sin(Qr)/(Qr) dr
    
    Args:
        r: distance array (Å)
        g_r: g(r) array
        rho_n: number density (atoms/Å³)
        q_min: minimum Q value (Å⁻¹)
        q_max: maximum Q value (Å⁻¹)
        n_q: number of Q points
    
    Returns:
        Q: wave vector array (Å⁻¹)
        S_Q: structure factor array
    """
    print(f"\nCalculating S(Q)...")
    print(f"  Number density: {rho_n:.6f} atoms/Å³")
    print(f"  Q range: {q_min} to {q_max} Å⁻¹")
    
    Q = np.linspace(q_min, q_max, n_q)
    S_Q = np.zeros(n_q)
    
    # Window function to reduce truncation effects
    # Using Lorch modification
    r_max = r[-1]
    
    for i, q in enumerate(Q):
        if q < 1e-10:
            # Avoid division by zero at Q=0
            S_Q[i] = 1.0
            continue
        
        # Integrand: r² × [g(r) - 1] × sin(Qr)/(Qr)
        # With Lorch window function: sin(πr/r_max) / (πr/r_max)
        
        qr = q * r
        
        # Avoid division by zero at r=0
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_qr = np.where(qr > 1e-10, np.sin(qr) / qr, 1.0)
            
            # Lorch window function
            pi_r_rmax = np.pi * r / r_max
            lorch = np.where(pi_r_rmax > 1e-10, np.sin(pi_r_rmax) / pi_r_rmax, 1.0)
        
        # Integrand
        integrand = r**2 * (g_r - 1.0) * sinc_qr * lorch
        
        # Numerical integration using Simpson's rule
        integral = simpson(integrand, x=r)
        
        S_Q[i] = 1.0 + 4.0 * np.pi * rho_n * integral
    
    print(f"  S(Q) peak value: {np.max(S_Q):.3f} at Q = {Q[np.argmax(S_Q)]:.3f} Å⁻¹")
    
    return Q, S_Q


def plot_results(r, g_r, Q, S_Q, temperature, output_prefix):
    """Create plots for g(r) and S(Q)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot g(r)
    ax1 = axes[0]
    ax1.plot(r, g_r, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('r (Å)', fontsize=14)
    ax1.set_ylabel('g(r)', fontsize=14)
    ax1.set_title(f'Radial Distribution Function at {temperature}', fontsize=16)
    ax1.set_xlim(0, max(r))
    ax1.set_ylim(0, max(g_r) * 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Annotate first peak
    peak_idx = np.argmax(g_r)
    ax1.annotate(f'1st peak: r={r[peak_idx]:.2f} Å\ng(r)={g_r[peak_idx]:.2f}',
                xy=(r[peak_idx], g_r[peak_idx]),
                xytext=(r[peak_idx]+1, g_r[peak_idx]*0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    # Plot S(Q)
    ax2 = axes[1]
    ax2.plot(Q, S_Q, 'r-', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Q (Å⁻¹)', fontsize=14)
    ax2.set_ylabel('S(Q)', fontsize=14)
    ax2.set_title(f'Structure Factor at {temperature}', fontsize=16)
    ax2.set_xlim(0, max(Q))
    ax2.set_ylim(0, max(S_Q) * 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Annotate first peak
    # Skip initial region for peak finding
    start_idx = np.argmax(Q > 1.0)
    peak_idx = start_idx + np.argmax(S_Q[start_idx:])
    ax2.annotate(f'1st peak: Q={Q[peak_idx]:.2f} Å⁻¹\nS(Q)={S_Q[peak_idx]:.2f}',
                xy=(Q[peak_idx], S_Q[peak_idx]),
                xytext=(Q[peak_idx]+2, S_Q[peak_idx]*0.8),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_gr_sq.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_prefix}_gr_sq.png")
    plt.show()


def save_sq_data(Q, S_Q, output_file):
    """Save S(Q) data to file"""
    header = "# Q(A^-1)  S(Q)\n"
    header += f"# Calculated from g(r) using Fourier transform\n"
    
    with open(output_file, 'w') as f:
        f.write(header)
        for q, s in zip(Q, S_Q):
            f.write(f"{q:.6f}  {s:.6f}\n")
    
    print(f"S(Q) data saved: {output_file}")


def process_single_file(rdf_file, density, do_plot=True):
    """Process a single RDF file"""
    
    # Extract temperature from filename if possible
    basename = os.path.basename(rdf_file)
    if 'K' in basename:
        temp_str = basename.split('_')[1].replace('.dat', '')
    else:
        temp_str = "unknown"
    
    print(f"\n{'='*60}")
    print(f"Processing: {rdf_file}")
    print(f"Temperature: {temp_str}")
    print(f"Density: {density} g/cm³")
    print('='*60)
    
    # Read g(r)
    r, g_r = read_rdf_file(rdf_file)
    
    # Convert density
    rho_n = density_to_number_density(density)
    
    # Calculate S(Q)
    Q, S_Q = calculate_sq(r, g_r, rho_n)
    
    # Output filename
    output_prefix = rdf_file.replace('rdf_', 'sq_').replace('.dat', '')
    
    # Save S(Q) data
    save_sq_data(Q, S_Q, f"{output_prefix}.dat")
    
    # Plot if requested
    if do_plot:
        plot_results(r, g_r, Q, S_Q, temp_str, output_prefix)
    
    return Q, S_Q


def process_all_temperatures(density_dict=None):
    """Process all RDF files in current directory"""
    
    # Default densities for each temperature
    if density_dict is None:
        density_dict = {
            '293K': 6.11,
            '300K': 6.10,
            '400K': 5.98,
            '500K': 5.88,
            '600K': 5.78,
            '700K': 5.69,
            '800K': 5.60,
            '1000K': 5.45,
            '1200K': 5.32,
            '1500K': 5.15,
        }
    
    # Find all RDF files
    rdf_files = sorted(glob.glob('rdf_*K.dat'))
    
    if not rdf_files:
        print("No RDF files found (pattern: rdf_*K.dat)")
        return
    
    print(f"Found {len(rdf_files)} RDF files")
    
    results = {}
    
    for rdf_file in rdf_files:
        # Extract temperature
        basename = os.path.basename(rdf_file)
        temp_str = basename.replace('rdf_', '').replace('.dat', '')
        
        # Get density
        if temp_str in density_dict:
            density = density_dict[temp_str]
        else:
            print(f"Warning: No density for {temp_str}, using 6.0 g/cm³")
            density = 6.0
        
        Q, S_Q = process_single_file(rdf_file, density, do_plot=False)
        results[temp_str] = (Q, S_Q)
    
    # Create comparison plot
    if len(results) > 1:
        plot_comparison(results)
    
    return results


def plot_comparison(results):
    """Create comparison plot for multiple temperatures"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(results)))
    
    # Sort by temperature
    temps = sorted(results.keys(), key=lambda x: int(x.replace('K', '')))
    
    for i, temp in enumerate(temps):
        Q, S_Q = results[temp]
        
        # Read corresponding g(r) for comparison
        rdf_file = f"rdf_{temp}.dat"
        if os.path.exists(rdf_file):
            r, g_r = read_rdf_file(rdf_file)
            axes[0].plot(r, g_r, color=colors[i], linewidth=1.5, label=temp)
        
        axes[1].plot(Q, S_Q, color=colors[i], linewidth=1.5, label=temp)
    
    # g(r) plot
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('r (Å)', fontsize=14)
    axes[0].set_ylabel('g(r)', fontsize=14)
    axes[0].set_title('Radial Distribution Function', fontsize=16)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 10)
    
    # S(Q) plot
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Q (Å⁻¹)', fontsize=14)
    axes[1].set_ylabel('S(Q)', fontsize=14)
    axes[1].set_title('Structure Factor', fontsize=16)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 12)
    
    plt.tight_layout()
    plt.savefig('sq_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: sq_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Structure Factor S(Q) from RDF g(r) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_sq.py rdf_293K.dat 6.11
  python calculate_sq.py rdf_293K.dat 6.11 --plot
  python calculate_sq.py --all
  python calculate_sq.py --all --densities "293K:6.11,600K:5.78,1000K:5.45"
        """
    )
    
    parser.add_argument('rdf_file', nargs='?', help='Input RDF file')
    parser.add_argument('density', nargs='?', type=float, help='Density in g/cm³')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Process all RDF files')
    parser.add_argument('--densities', type=str, 
                       help='Density values as "T1:d1,T2:d2,..." e.g., "293K:6.11,600K:5.78"')
    parser.add_argument('--q-max', type=float, default=15.0, help='Maximum Q value (Å⁻¹)')
    parser.add_argument('--q-min', type=float, default=0.1, help='Minimum Q value (Å⁻¹)')
    
    args = parser.parse_args()
    
    if args.all:
        # Parse custom densities if provided
        density_dict = None
        if args.densities:
            density_dict = {}
            for item in args.densities.split(','):
                temp, dens = item.split(':')
                density_dict[temp.strip()] = float(dens)
        
        process_all_temperatures(density_dict)
    
    elif args.rdf_file and args.density:
        process_single_file(args.rdf_file, args.density, do_plot=args.plot)
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick example:")
        print("  python calculate_sq.py rdf_293K.dat 6.11 --plot")
        print("="*60)


if __name__ == "__main__":
    main()
