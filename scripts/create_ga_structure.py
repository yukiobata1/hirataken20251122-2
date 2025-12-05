#!/usr/bin/env python3
"""
Create initial random structure for Pure Gallium
Temperature: 150°C
"""

import numpy as np
import argparse
import os

def create_ga_structure(n_atoms=1000, T=150.0, output_file='inputs/data.ga_1000'):
    """
    Create initial random structure for Pure Gallium

    Parameters
    ----------
    n_atoms : int
        Total number of atoms (default: 1000)
    T : float
        Temperature in Celsius (default: 150°C)
    output_file : str
        Output LAMMPS data file name
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Creating Pure Ga structure:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Temperature: {T} °C")

    # Density of Ga at 150°C
    # Approximate linear dependence from MP (29.8°C, 6.095 g/cm^3)
    # Using rho ~ 6.0 g/cm^3 for 150°C
    rho = 6.04  # g/cm³ 

    # Atomic masses
    M_Ga = 69.723  # g/mol
    
    # Calculate box size from density
    # V = n_atoms * M_Ga / (rho * N_A)
    # where N_A = 6.022e23 mol^-1
    N_A = 6.022e23
    V = n_atoms * M_Ga / (rho * N_A) * 1e24  # Å³
    L = V ** (1/3)  # Cubic box side length

    print(f"  Density: {rho:.3f} g/cm³")
    print(f"  Box size: {L:.2f} Å")
    print(f"  Volume: {V:.2f} Å³")

    # Generate random atomic positions
    np.random.seed(42)  # For reproducibility
    positions = np.random.rand(n_atoms, 3) * L

    # Write LAMMPS data file
    with open(output_file, 'w') as f:
        f.write(f'Pure Gallium structure at {T}C\n\n')
        f.write(f'{n_atoms} atoms\n')
        f.write('1 atom types\n\n')
        f.write(f'0.0 {L:.10f} xlo xhi\n')
        f.write(f'0.0 {L:.10f} ylo yhi\n')
        f.write(f'0.0 {L:.10f} zlo zhi\n\n')
        f.write('Masses\n\n')
        f.write(f'1 {M_Ga:.6f}  # Ga\n\n')
        f.write('Atoms  # atomic\n\n')

        for i, pos in enumerate(positions):
            f.write(f'{i+1} 1 {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n')

    print(f"\nStructure written to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create initial structure for Pure Gallium')
    parser.add_argument('-n', '--natoms', type=int, default=1000,
                        help='Total number of atoms (default: 1000)')
    parser.add_argument('-T', '--temperature', type=float, default=150.0,
                        help='Temperature in Celsius (default: 150.0)')
    parser.add_argument('-o', '--output', type=str, default='inputs/data.ga_1000',
                        help='Output LAMMPS data file')

    args = parser.parse_args()

    create_ga_structure(
        n_atoms=args.natoms,
        T=args.temperature,
        output_file=args.output
    )
