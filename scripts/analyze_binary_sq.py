import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from io import StringIO

def load_rdf_robust(filepath):
    """
    Robustly load RDF file, skipping headers.
    Expected LAMMPS output for 3 pairs (1-1, 2-2, 1-2):
    Row Bin R g11 c11 g22 c22 g12 c12
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        # Data lines usually have many columns (at least 4)
        if len(parts) < 4: continue
        data_lines.append(line)
        
    return np.loadtxt(StringIO("\n".join(data_lines)))

def calc_sq_partial(r, g, rho=0.05):
    """Calculate S(Q) from g(r)"""
    Q = np.linspace(0.5, 20.0, 300)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    
    for i, q in enumerate(Q):
        if q < 1e-6: continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q
    return Q, S

def analyze_file(rdf_file):
    if not os.path.exists(rdf_file):
        print(f"Error: File {rdf_file} not found.")
        return

    print(f"Analyzing {rdf_file}...")
    
    try:
        data = load_rdf_robust(rdf_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # LAMMPS output columns for "compute rdf 1 1 2 2 1 2"
    # Col 0: Row, Col 1: Bin
    r = data[:, 1]
    
    # Check column count to identify structure
    # typically: Row Bin R g11 c11 g22 c22 g12 c12
    # Indices:   0   1   2 3   4   5   6   7   8
    
    if data.shape[1] >= 8:
        g11 = data[:, 3]
        g22 = data[:, 5]
        g12 = data[:, 7]
    else:
        print("Error: RDF file does not contain enough columns for partial analysis.")
        print(f"Columns found: {data.shape[1]}")
        return

    # Total g(r) assuming 50:50 mixture
    # w11 = 0.25, w22 = 0.25, w12 = 0.5
    g_total = 0.25*g11 + 0.25*g22 + 0.5*g12
    
    # Calculate S(Q)
    print("Calculating Structure Factors...")
    Q, S11 = calc_sq_partial(r, g11)
    _, S22 = calc_sq_partial(r, g22)
    _, S12 = calc_sq_partial(r, g12)
    _, Stot = calc_sq_partial(r, g_total)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: g(r)
    ax1.plot(r, g11, 'b-', label='1-1 (Large)', alpha=0.6)
    ax1.plot(r, g22, 'r-', label='2-2 (Small)', alpha=0.6)
    ax1.plot(r, g12, 'g-', label='1-2 (Cross)', alpha=0.6)
    ax1.plot(r, g_total, 'k--', label='Total', lw=1.5)
    ax1.set_xlabel('r (Å)')
    ax1.set_ylabel('g(r)')
    ax1.set_title('Partial Pair Distribution Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    
    # Plot 2: S(Q)
    ax2.plot(Q, S11, 'b-', label='S11 (Large)', alpha=0.6)
    ax2.plot(Q, S22, 'r-', label='S22 (Small)', alpha=0.6)
    ax2.plot(Q, S12, 'g-', label='S12 (Cross)', alpha=0.6)
    ax2.plot(Q, Stot, 'k-', label='Total S(Q)', lw=2)
    
    # Highlight shoulder region
    ax2.axvspan(2.2, 3.2, color='orange', alpha=0.1, label='Shoulder Region')
    
    ax2.set_xlabel('Q (Å⁻¹)')
    ax2.set_ylabel('S(Q)')
    ax2.set_title('Partial Structure Factors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(-1, 4)
    
    output_png = rdf_file.replace('.rdf', '_analysis.png')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Analysis saved to: {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_binary_sq.py <path_to_rdf_file>")
        print("Example: python3 scripts/analyze_binary_sq.py grid_outputs/out_s80_e200.rdf")
    else:
        analyze_file(sys.argv[1])
