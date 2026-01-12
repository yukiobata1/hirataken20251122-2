#!/usr/bin/env python3
"""
Single-component LJ Ga simulation at 423.15 K (150°C)
For comparison with experimental data in thesis section 4.1 and 4.2

Parameters:
  - σ = 2.70 Å
  - ε = 0.430 kcal/mol (= 1.8 kJ/mol)
  - Temperature = 423.15 K (150°C)
  - N = 4000 atoms

Usage:
  Run on GPU instance:
    cd /path/to/experiments/01_single_component
    python scripts/run_single_component_423K.py
"""

import os
import subprocess
from pathlib import Path

# ================= CONFIGURATION =================
SIGMA = 2.70        # Angstrom
EPSILON = 0.430     # kcal/mol
TEMP = 423.15       # K (150°C)
N_ATOMS = 4000      # Number of atoms

# LAMMPS command (KOKKOS GPU)
LMP_CMD = "lmp -k on g 1 -sf kk"

# Paths (relative to experiments/01_single_component/)
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "423K"
INPUT_DIR = Path("/home/yuki/lammps_settings_obata/hirataken20251122-2/inputs")
# ================================================


def write_lammps_input(filepath, output_prefix):
    """Generate LAMMPS input file for single-component Ga at 423.15K"""

    content = f"""# Single-component Ga LJ simulation at {TEMP} K (150°C)
# For thesis comparison with experimental data
# Parameters: σ = {SIGMA} Å, ε = {EPSILON} kcal/mol

# ========== KOKKOS Initialization ==========
package         kokkos neigh full newton off

# ========== Initialization ==========
units           real
atom_style      atomic
boundary        p p p

# ========== Read structure ==========
read_data       {INPUT_DIR}/data.ga_1000

# ========== Force field ==========
# Single-component LJ: σ = {SIGMA} Å, ε = {EPSILON} kcal/mol
pair_style      lj/cut/kk 12.0
pair_coeff      1 1 {EPSILON:.4f} {SIGMA:.4f}

# ========== Settings ==========
mass            1 69.723    # Ga atomic mass

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# ========== Initial velocity ==========
velocity        all create {TEMP} 42315 dist gaussian

# ========== Thermodynamic output ==========
thermo          1000
thermo_style    custom step temp pe ke etotal press vol density

# ========== Minimization ==========
print "========== Energy Minimization =========="
minimize        1.0e-4 1.0e-6 10000 100000
reset_timestep  0

# ========== Equilibration (NVT) ==========
print "========== Equilibration =========="

# Start with small timestep
timestep        0.5
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             10000
unfix           nvt

# Increase timestep
timestep        2.0
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             40000
unfix           nvt
reset_timestep  0

# ========== Production (NVT) ==========
print "========== Production run =========="

fix             nvt all nvt temp {TEMP} {TEMP} 100.0

# RDF computation (200 bins, cutoff 12 Å)
compute         myrdf all rdf 200 1 1 cutoff 12.0
fix             rdfout all ave/time 100 1000 100000 c_myrdf[*] file {output_prefix}.rdf mode vector

# Trajectory output (optional, comment out to save space)
# dump            1 all custom 10000 {output_prefix}.lammpstrj id type x y z

run             100000
unfix           nvt

# ========== Finalize ==========
print "========== Simulation completed =========="

variable        density_final equal density
variable        temp_final equal temp
variable        press_final equal press

print "========================================="
print "RESULTS:"
print "  Temperature: ${{temp_final}} K"
print "  Density:     ${{density_final}} g/cm^3"
print "  Pressure:    ${{press_final}} bar"
print "  Sigma:       {SIGMA} A"
print "  Epsilon:     {EPSILON} kcal/mol"
print "========================================="

write_data      {output_prefix}_final.data
"""

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  Written: {filepath}")


def main():
    print("=" * 70)
    print("Single-component Ga LJ Simulation at 423.15 K (150°C)")
    print("=" * 70)
    print(f"  σ = {SIGMA} Å")
    print(f"  ε = {EPSILON} kcal/mol")
    print(f"  T = {TEMP} K")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate input file
    input_file = OUTPUT_DIR / "in.ga_single_423K"
    output_prefix = OUTPUT_DIR / "ga_single_423K"

    write_lammps_input(input_file, output_prefix)

    # Run LAMMPS
    log_file = OUTPUT_DIR / "log.ga_single_423K"
    cmd = f"{LMP_CMD} -in {input_file} -log {log_file}"

    print(f"\nRunning LAMMPS...")
    print(f"  Command: {cmd}")
    print("-" * 70)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=str(OUTPUT_DIR)
        )
        print("-" * 70)
        print("Simulation completed successfully!")
        print(f"  RDF output: {output_prefix}.rdf")
        print(f"  Log file: {log_file}")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: LAMMPS failed with return code {e.returncode}")
        return 1

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run analysis script to calculate S(Q)")
    print("  2. Compare with experimental data")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
