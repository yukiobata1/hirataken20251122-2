import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from io import StringIO

# ================= CONFIGURATION =================
CUSTOM_DATA_FILE = 'custom_sq_data.txt'
OLD_SIM_FILE = 'grid_outputs/out_s100_e100.rdf' # 1-component LJ baseline
RHO = 0.052 # Density of Ga for S(Q) calculation
# ============================================

def load_rdf_robust(filepath):
    """Robustly load RDF file, skipping headers and comments."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return None
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) <= 2: continue
        data_lines.append(line)
    return np.loadtxt(StringIO("\n".join(data_lines)))

def calc_sq_from_g(r, g, rho):
    """Calculate S(Q) from g(r)."""
    Q = np.linspace(0.5, 20.0, 300)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    for i, q in enumerate(Q):
        if q < 1e-6:
            integrand = (g - 1.0) * r**2
            S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr
        else:
            integrand = (g - 1.0) * r * np.sin(q * r)
            S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q
    return Q, S

def main():
    print("Creating Comparison Plot...")

    # 1. Load Custom Data (Experimental S(Q))
    print(f"Loading Custom Data from {CUSTOM_DATA_FILE}...")
    try:
        # Assuming data is Q, S format with comma separation based on provided text
        custom_data = np.loadtxt(CUSTOM_DATA_FILE, delimiter=',')
        Q_custom = custom_data[:, 0]
        S_custom = custom_data[:, 1]
    except Exception as e:
        print(f"Error loading custom data: {e}")
        # Try space delimiter just in case
        try:
             custom_data = np.loadtxt(CUSTOM_DATA_FILE)
             Q_custom = custom_data[:, 0]
             S_custom = custom_data[:, 1]
        except:
             sys.exit(1)

    # 2. Load Old Simulation Data (1-component LJ)
    print(f"Loading Simulation Data from {OLD_SIM_FILE}...")
    old_sim_data = load_rdf_robust(OLD_SIM_FILE)
    
    Q_sim, S_sim = None, None
    if old_sim_data is not None:
        try:
            r_sim = old_sim_data[:, 1]
            if old_sim_data.shape[1] >= 7:
                g_sim = 0.25*old_sim_data[:, 2] + 0.25*old_sim_data[:, 4] + 0.5*old_sim_data[:, 6]
            else:
                g_sim = old_sim_data[:, 2]
            Q_sim, S_sim = calc_sq_from_g(r_sim, g_sim, RHO)
        except Exception as e:
            print(f"Error processing simulation data: {e}")

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Custom Data (Experiment)
    # The provided data seems to have some negative Q values or noise?
    # Let's filter for Q > 0 for standard S(Q) plot
    mask = Q_custom > 0
    ax.plot(Q_custom[mask], S_custom[mask], 'k-', lw=2.5, label='Experiment (Target)')

    # Plot Simulation Data
    if Q_sim is not None:
        ax.plot(Q_sim, S_sim, 'r--', lw=2, label='Simulation (LJ, s1.0 e1.0)')

    # Highlight shoulder region
    ax.axvspan(2.5, 3.5, color='orange', alpha=0.15, label='Shoulder Region')

    ax.set_title('Structure Factor S(Q) Comparison', fontsize=16)
    ax.set_xlabel('Q (Å⁻¹)', fontsize=14)
    ax.set_ylabel('S(Q)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 3.5)

    plt.tight_layout()
    output_filename = "custom_data_comparison.png"
    plt.savefig(output_filename, dpi=200)
    print(f"\nPlot saved to: {output_filename}")

if __name__ == "__main__":
    main()
