import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# Grid Search Parameters
SIGMA_RATIOS = [1.0, 0.95, 0.90, 0.85]  # Size ratio (Row)
EPSILON_RATIOS = [1.0, 1.25, 1.5]       # Energy ratio (Column)

# Base Parameters (Ga)
SIGMA_BASE = 2.70
EPSILON_BASE = 0.430
TEMP = 293.0

# LAMMPS Command (GPU)
# -sf kk: Enable Kokkos suffix
# -k on g 1: Use 1 GPU
LMP_CMD = "lmp -k on g 1 -sf kk"
# =============================================

def write_lammps_input(filename, output_prefix, sig_r, eps_r):
    """
    Create LAMMPS input file for 12,000 atom system.
    Uses 'replicate' to expand system and 'set' to randomize types.
    """
    
    # Calculate parameters
    s1, e1 = SIGMA_BASE, EPSILON_BASE
    s2 = s1 * sig_r
    e2 = e1 * eps_r
    
    # Mixing rules (Lorentz-Berthelot)
    s12 = (s1 + s2) / 2.0
    e12 = np.sqrt(e1 * e2)

    content = f"""
# Grid Search: Sigma_ratio={sig_r}, Epsilon_ratio={eps_r}
units           real
atom_style      atomic
boundary        p p p

# 1. Read base 1000 atom data
# Note: This file must have "2 atom types" defined in header
read_data       inputs/data.ga_base_2types

# 2. Expand system (1000 -> 12000 atoms)
replicate       2 2 3

# 3. Randomize types (50% Type 2)
#    set group all type/fraction 2 0.5 12345
set             group all type/fraction 2 0.5 12345

# Force Field
pair_style      lj/cut/kk 12.0
pair_coeff      1 1 {e1:.4f} {s1:.4f}
pair_coeff      2 2 {e2:.4f} {s2:.4f}
pair_coeff      1 2 {e12:.4f} {s12:.4f}

# Settings
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
velocity        all create {TEMP} {int(sig_r*1000 + eps_r*100)} dist gaussian

# Thermodynamic Output
thermo          1000
thermo_style    custom step temp press density

# Minimization (Important after randomization)
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Equilibration (5000 steps)
timestep        2.0
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             5000

# Production & RDF (10000 steps)
# compute rdf args: Nbin type1 type1 type2 type2 type1 type2
compute         myrdf all rdf 200 1 1 2 2 1 2 cutoff 12.0
fix             rdfout all ave/time 100 10 1000 c_myrdf[*] file {output_prefix}.rdf mode vector

run             10000
"""
    with open(filename, 'w') as f:
        f.write(content)

def prepare_base_data():
    """Create temporary base file with '2 atom types' from '1 atom type' file"""
    print("Preparing base data file...")
    if not os.path.exists('inputs/data.ga_1000'):
        raise FileNotFoundError("inputs/data.ga_1000 not found!")

    with open('inputs/data.ga_1000', 'r') as f:
        content = f.read()
    
    # Replace "1 atom types" with "2 atom types"
    content = content.replace("1 atom types", "2 atom types")
    
    # Add Type 2 mass if needed
    if "Masses" in content:
        # Simple replace: assumes standard format "1 69.723"
        # Adding Type 2 mass (same as Type 1)
        content = content.replace("1 69.723", "1 69.723\n2 69.723")
        
    with open('inputs/data.ga_base_2types', 'w') as f:
        f.write(content)
    print("Created temporary base file: inputs/data.ga_base_2types")

def calc_sq_simple(r, g, rho=0.05):
    """Calculate S(Q) from g(r)"""
    Q = np.linspace(0.5, 20.0, 200)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    
    for i, q in enumerate(Q):
        if q < 1e-6: continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q
    return Q, S

def main():
    # 0. Setup
    os.makedirs("grid_outputs", exist_ok=True)
    try:
        prepare_base_data()
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    results = {}

    print(f"Starting Large Scale Grid Search (12,000 atoms)...")
    print(f"Grid: {len(SIGMA_RATIOS)}x{len(EPSILON_RATIOS)} = {len(SIGMA_RATIOS)*len(EPSILON_RATIOS)} simulations")

    # 1. Simulation Loop
    for sig_r in SIGMA_RATIOS:
        for eps_r in EPSILON_RATIOS:
            label = f"s{int(sig_r*100)}_e{int(eps_r*100)}"
            print(f"Processing: Sigma={sig_r:.2f}, Epsilon={eps_r:.2f} ... ", end="", flush=True) 
            
            input_file = f"grid_outputs/in.{label}"
            rdf_file = f"grid_outputs/out_{label}"
            
            write_lammps_input(input_file, rdf_file, sig_r, eps_r)
            
            # Run LAMMPS
            cmd = f"{LMP_CMD} -in {input_file} -log grid_outputs/log.{label}"
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print("Done.")
            except subprocess.CalledProcessError as e:
                print("Failed!")
                print(f"  Error log: {e.stderr.decode()}")
                continue

            # Analyze
            try:
                data = np.loadtxt(f"{rdf_file}.rdf", skiprows=4)
                r = data[:, 1]
                # g_total approximation (assuming 50:50 mix)
                # format: Row, bin, r, g11, g22, g12
                g_total = 0.25*data[:, 2] + 0.25*data[:, 4] + 0.5*data[:, 6]
                Q, S = calc_sq_simple(r, g_total)
                results[(sig_r, eps_r)] = (Q, S)
            except Exception as e:
                print(f"  Analysis Error: {e}")

    # 2. Generate Plot
    print("Generating Summary Plot...")
    fig, axes = plt.subplots(len(SIGMA_RATIOS), len(EPSILON_RATIOS), 
                             figsize=(18, 14), sharex=True, sharey=True)
    
    # Titles
    cols = [f"Eps x{r}" for r in EPSILON_RATIOS]
    rows = [f"Sig x{r}" for r in SIGMA_RATIOS]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold')
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(f"{row}\nS(Q)", fontsize=16, fontweight='bold')

    # Plotting
    for i, sig_r in enumerate(SIGMA_RATIOS):
        for j, eps_r in enumerate(EPSILON_RATIOS):
            ax = axes[i, j]
            if (sig_r, eps_r) in results:
                Q, S = results[(sig_r, eps_r)]
                
                # Plot S(Q)
                ax.plot(Q, S, 'b-', lw=2, label='Sim')
                
                # Highlight shoulder region (approx 2.2 - 3.2 A^-1)
                ax.axvspan(2.2, 3.2, color='orange', alpha=0.15)
                
                # Axis settings
                ax.set_xlim(0, 12)
                ax.set_ylim(0, 3.5)
                
                # Annotate Max Peak
                max_idx = np.argmax(S)
                ax.text(0.95, 0.95, f"Peak: {Q[max_idx]:.2f}", 
                        transform=ax.transAxes, ha='right', va='top', fontsize=10)
            
            ax.grid(True, alpha=0.4)
            if i == len(SIGMA_RATIOS)-1:
                ax.set_xlabel("Q (A^-1)", fontsize=12)

    plt.tight_layout()
    plt.savefig("grid_search_large.png", dpi=150)
    print("All Done! Check 'grid_search_large.png'")

if __name__ == "__main__":
    main()
