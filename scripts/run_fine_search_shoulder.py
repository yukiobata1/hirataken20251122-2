#!/usr/bin/env python3
"""
Fine grid search around Ga1=50%, sigma12=1.20x for shoulder reproduction.
"""

import os
import numpy as np
import subprocess
from io import StringIO

# Base Parameters
SIGMA_BASE = 2.70
EPSILON_BASE = 0.430
TEMP = 423.15

# Fixed sigma ratios for Ga1-Ga1 and Ga2-Ga2
SIGMA_RATIO_GA1 = 1.1  # Ga1-Ga1: 2.97 A
SIGMA_RATIO_GA2 = 0.9  # Ga2-Ga2: 2.43 A

# Fine search grid around sigma12=1.15, Ga1=50%
SIGMA12_RATIOS = [1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18]
GA1_FRACTIONS = [0.45, 0.48, 0.50, 0.52, 0.55]

LMP_CMD = "lmp -k on g 1 -sf kk"
OUTPUT_DIR = "outputs/fine_search_shoulder"


def write_lammps_input(filename, output_prefix, ga1_frac, sig12_ratio):
    sigma1 = SIGMA_BASE * SIGMA_RATIO_GA1
    sigma2 = SIGMA_BASE * SIGMA_RATIO_GA2
    sigma12 = SIGMA_BASE * sig12_ratio

    eps1 = EPSILON_BASE
    eps2 = EPSILON_BASE
    eps12 = np.sqrt(eps1 * eps2)

    seed = int(ga1_frac * 10000 + sig12_ratio * 1000) + 12345

    content = f"""# Fine Search: Ga1={ga1_frac:.2f}, sigma12={sig12_ratio:.2f}x
# Ga1-Ga1 sigma = {sigma1:.4f} A (x1.1)
# Ga2-Ga2 sigma = {sigma2:.4f} A (x0.9)
# Ga1-Ga2 sigma = {sigma12:.4f} A (x{sig12_ratio:.2f})
# Epsilon = {EPSILON_BASE:.4f} kcal/mol

package         kokkos neigh full newton off

units           real
atom_style      atomic
boundary        p p p

read_data       inputs/data.ga_base_2types

set             group all type 1
set             group all type/fraction 2 {1-ga1_frac:.2f} {seed}

pair_style      lj/cut/kk 12.0
pair_coeff      1 1 {eps1:.4f} {sigma1:.4f}
pair_coeff      2 2 {eps2:.4f} {sigma2:.4f}
pair_coeff      1 2 {eps12:.4f} {sigma12:.4f}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
velocity        all create {TEMP} {seed} dist gaussian

thermo          1000
thermo_style    custom step temp press density

minimize        1.0e-4 1.0e-6 10000 100000
reset_timestep  0

timestep        2.0
fix             nvt all nvt temp {TEMP} {TEMP} 100.0
run             50000
unfix           nvt
reset_timestep  0

fix             nvt all nvt temp {TEMP} {TEMP} 100.0
compute         myrdf all rdf 200 1 1 2 2 1 2 cutoff 12.0
fix             rdfout all ave/time 100 1000 100000 c_myrdf[*] file {output_prefix}.rdf mode vector

run             100000
"""
    with open(filename, 'w') as f:
        f.write(content)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Fine Grid Search for Shoulder Reproduction")
    print("=" * 70)
    print(f"sigma12 ratios: {SIGMA12_RATIOS}")
    print(f"Ga1 fractions: {GA1_FRACTIONS}")
    print(f"Total simulations: {len(SIGMA12_RATIOS) * len(GA1_FRACTIONS)}")
    print("=" * 70)

    count = 0
    total = len(SIGMA12_RATIOS) * len(GA1_FRACTIONS)

    for sig12_ratio in SIGMA12_RATIOS:
        for ga1_frac in GA1_FRACTIONS:
            count += 1
            sig12_label = round(sig12_ratio * 100)
            ga1_label = round(ga1_frac * 100)
            label = f"sig12_{sig12_label}_ga1_{ga1_label}"

            print(f"[{count}/{total}] sig12={sig12_ratio:.2f}x, Ga1={ga1_frac:.0%} ... ", end="", flush=True)

            input_file = f"{OUTPUT_DIR}/in.{label}"
            output_prefix = f"{OUTPUT_DIR}/out_{label}"

            write_lammps_input(input_file, output_prefix, ga1_frac, sig12_ratio)

            cmd = f"{LMP_CMD} -in {input_file} -log {OUTPUT_DIR}/log.{label}"
            try:
                subprocess.run(cmd, shell=True, check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             timeout=600)
                print("Done.")
            except subprocess.CalledProcessError as e:
                print(f"Failed! {e}")
            except subprocess.TimeoutExpired:
                print("Timeout!")

    print("\n" + "=" * 70)
    print("FINE SEARCH COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
