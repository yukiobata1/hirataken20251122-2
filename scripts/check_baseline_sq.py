import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate
from io import StringIO

# ================= SETTINGS =================
EXP_FILE = 'data/g_exp_cleaned.dat'  # Experimental g(r)
RHO = 0.052  # Density of Ga (approx atoms/A^3)
# ============================================

def load_exp_data(filepath):
    """Load experimental g(r)"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None, None
    try:
        data = np.loadtxt(filepath)
        return data[:, 0], data[:, 1] # r, g(r)
    except Exception as e:
        print(f"Error reading exp data: {e}")
        return None, None

def calc_sq(r, g, rho):
    """Calculate S(Q) from g(r)"""
    Q = np.linspace(0.5, 20.0, 300)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    
    for i, q in enumerate(Q):
        if q < 1e-6: continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q
    return Q, S

def load_sim_data(filepath):
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

def main():
    print("Checking Baseline S(Q) from Experimental Data...")
    
    # 1. Load Experiment
    r_exp, g_exp = load_exp_data(EXP_FILE)
    if r_exp is None: return

    # 2. Calculate S(Q) from Exp g(r)
    print("Calculating S(Q) from Experimental g(r)...")
    Q_exp, S_exp = calc_sq(r_exp, g_exp, RHO)

    # 3. Load Simulation Baseline (Run 0 or equivalent)
    # We look for grid_outputs/out_s100_e100.rdf (Sigma 1.0, Eps 1.0)
    sim_file = "grid_outputs/out_s100_e100.rdf"
    
    Q_sim, S_sim, r_sim, g_sim = None, None, None, None

    if os.path.exists(sim_file):
        print(f"Loading Simulation Baseline: {sim_file}")
        try:
            sim_data = load_sim_data(sim_file)
            
            r_sim = sim_data[:, 1]
            # Run 0 is effectively pure. If binary output:
            # 1-1, 2-2, 1-2 are all identical.
            if sim_data.shape[1] >= 7:
                 g_sim = 0.25*sim_data[:, 2] + 0.25*sim_data[:, 4] + 0.5*sim_data[:, 6]
            else:
                 g_sim = sim_data[:, 2]
                 
            Q_sim, S_sim = calc_sq(r_sim, g_sim, RHO)
            
        except Exception as e:
            print(f"Error reading simulation data: {e}")
    else:
        print("Simulation baseline file not found. Run grid search with s100_e100 first.")

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # g(r)
    ax1.plot(r_exp, g_exp, 'k-', label='Experiment', lw=2)
    if r_sim is not None:
        ax1.plot(r_sim, g_sim, 'r--', label='Simulation (LJ)', lw=2)
    ax1.set_title('Radial Distribution Function g(r)')
    ax1.set_xlabel('r (Å)')
    ax1.set_ylabel('g(r)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)

    # S(Q)
    ax2.plot(Q_exp, S_exp, 'k-', label='Experiment', lw=2)
    if Q_sim is not None:
        ax2.plot(Q_sim, S_sim, 'r--', label='Simulation (LJ)', lw=2)
    
    # Highlight the famous Ga shoulder
    ax2.axvspan(2.5, 3.5, color='orange', alpha=0.2, label='Shoulder Region')
    
    ax2.set_title('Structure Factor S(Q)')
    ax2.set_xlabel('Q (Å⁻¹)')
    ax2.set_ylabel('S(Q)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(-1, 3.5)
    
    plt.tight_layout()
    plt.savefig('baseline_check.png', dpi=150)
    print("Saved comparison to 'baseline_check.png'")

if __name__ == "__main__":
    main()
