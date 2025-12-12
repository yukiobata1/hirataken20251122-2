import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from io import StringIO

# ================= CONFIGURATION =================
# TEST RUN: Check if ratio=1.0 matches reference data
SIGMA_RATIOS = [1.0]  # Pure Ga test
EPSILON_RATIOS = [1.0]

# Base Parameters (Ga)
SIGMA_BASE = 2.70
EPSILON_BASE = 0.430
TEMP = 423.15  # 150°C (same as reference data)

# LAMMPS Command
LMP_CMD = "lmp -k on g 1 -sf kk"
# =============================================

def write_lammps_input(filename, output_prefix, sig_r, eps_r):
    s1, e1 = SIGMA_BASE, EPSILON_BASE
    s2 = s1 * sig_r
    e2 = e1 * eps_r

    # Lorentz-Berthelot Mixing
    s12 = (s1 + s2) / 2.0
    e12 = np.sqrt(e1 * e2)

    content = f"""
# Grid Search TEST: Sigma_ratio={sig_r}, Epsilon_ratio={eps_r}
# KOKKOS Initialization (CRITICAL: Must be first)
package         kokkos neigh full newton off

units           real
atom_style      atomic
boundary        p p p

read_data       inputs/data.ga_base_2types

# Randomize types (50% Type 2)
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

thermo          1000
thermo_style    custom step temp press density

# Minimization
minimize        1.0e-4 1.0e-6 10000 100000
reset_timestep  0

# Equilibration (50000 steps - same as reference)
timestep        2.0
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             50000
unfix           nvt
reset_timestep  0

# Production (100000 steps - same as reference)
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
compute         myrdf all rdf 200 1 1 2 2 1 2 cutoff 12.0
# Consistent averaging over full run (Nevery=100, Nrepeat=1000, Nfreq=100000)
fix             rdfout all ave/time 100 1000 100000 c_myrdf[*] file {output_prefix}.rdf mode vector

run             100000
"""
    with open(filename, 'w') as f:
        f.write(content)

def prepare_base_data():
    if not os.path.exists('inputs/data.ga_1000'):
        raise FileNotFoundError("inputs/data.ga_1000 not found!")
    with open('inputs/data.ga_1000', 'r') as f:
        content = f.read()
    content = content.replace("1 atom types", "2 atom types")
    if "Masses" in content:
        content = content.replace("1 69.723", "1 69.723\n2 69.723")
    with open('inputs/data.ga_base_2types', 'w') as f:
        f.write(content)

def calc_sq_simple(r, g, rho=0.0522):
    """Calculate S(Q) from g(r). rho=0.0522 atoms/Å³ for Ga at 150°C"""
    Q = np.linspace(0.5, 20.0, 200)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1
    for i, q in enumerate(Q):
        if q < 1e-6: continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q
    return Q, S

def load_rdf_robust(filepath):
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
    os.makedirs("grid_outputs", exist_ok=True)
    try:
        prepare_base_data()
    except Exception as e:
        print(f"Warning: {e}")
        pass

    results = {}
    print("=" * 60)
    print("TEST RUN: Checking if ratio=1.0 matches reference data")
    print("=" * 60)
    print(f"Temperature: {TEMP} K (150°C)")
    print(f"Density: 0.0522 atoms/Å³")
    print(f"Sigma ratios: {SIGMA_RATIOS}")
    print(f"Epsilon ratios: {EPSILON_RATIOS}")
    print("=" * 60)

    for sig_r in SIGMA_RATIOS:
        for eps_r in EPSILON_RATIOS:
            label = f"s{int(sig_r*100)}_e{int(eps_r*100)}_test"
            print(f"\nProcessing: Sigma={sig_r:.2f}, Epsilon={eps_r:.2f}")

            input_file = f"grid_outputs/in.{label}"
            rdf_file = f"grid_outputs/out_{label}"
            write_lammps_input(input_file, rdf_file, sig_r, eps_r)

            print(f"  Running LAMMPS... (this will take 10-30 minutes)")
            cmd = f"{LMP_CMD} -in {input_file} -log grid_outputs/log.{label}"
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print("  ✓ Simulation completed")
            except subprocess.CalledProcessError as e:
                print("  ✗ Simulation failed!")
                print(f"  Error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
                continue

            try:
                print("  Analyzing RDF and calculating S(Q)...")
                data = load_rdf_robust(f"{rdf_file}.rdf")
                r = data[:, 1]
                if data.shape[1] >= 7:
                    # 1-1, 2-2, 1-2
                    g_total = 0.25*data[:, 2] + 0.25*data[:, 4] + 0.5*data[:, 6]
                else:
                    g_total = data[:, 2]
                Q, S = calc_sq_simple(r, g_total)
                results[(sig_r, eps_r)] = (Q, S)

                # Save S(Q) data
                sq_file = f"grid_outputs/sq_{label}.txt"
                np.savetxt(sq_file, np.column_stack([Q, S]), header="Q (A^-1)  S(Q)", comments="# ")
                print(f"  ✓ S(Q) saved to {sq_file}")

            except Exception as e:
                print(f"  ✗ Analysis Error: {e}")

    # Plotting comparison with reference
    if results:
        print("\n" + "=" * 60)
        print("Generating comparison plot with reference data...")
        print("=" * 60)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Load reference data
        try:
            ref_data = np.loadtxt('outputs/ga_lj_sq.txt', comments='#')
            Q_ref = ref_data[:, 0]
            S_ref = ref_data[:, 1]
            ax.plot(Q_ref, S_ref, 'k-', lw=2, label='Reference (150°C, pure Ga)', alpha=0.7)
        except:
            print("  Warning: Could not load reference data from outputs/ga_lj_sq.txt")

        # Plot test result
        for (sig_r, eps_r), (Q, S) in results.items():
            ax.plot(Q, S, 'r-', lw=2, label=f'Test: σ×{sig_r}, ε×{eps_r} (150°C, 2-type)')

        ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
        ax.set_ylabel('S(Q)', fontsize=12)
        ax.set_title('Comparison: Test vs Reference Data', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)
        ax.set_ylim(-0.5, 3)
        ax.axvspan(2.2, 3.2, color='orange', alpha=0.1, label='Shoulder region')

        plt.tight_layout()
        plt.savefig("grid_search_test_comparison.png", dpi=150)
        print("✓ Plot saved to: grid_search_test_comparison.png")
    else:
        print("\nNo results to plot!")

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
