#!/usr/bin/env python3
"""
Create initial random structure for Ga-In eutectic alloy
Composition: Ga0.858In0.142
Temperature: 150°C
"""

import numpy as np
import argparse


def create_egain_structure(n_atoms=1000, x_In=0.142, T=150.0, output_file='initial_structure.data'):
    """
    Create initial random structure for EGaIn alloy

    Parameters
    ----------
    n_atoms : int
        Total number of atoms (default: 1000)
    x_In : float
        Mole fraction of indium (default: 0.142 for eutectic)
    T : float
        Temperature in Celsius (default: 150°C)
    output_file : str
        Output LAMMPS data file name
    """

    # Calculate number of each atom type
    n_In = int(n_atoms * x_In)
    n_Ga = n_atoms - n_In

    print(f"Creating EGaIn structure:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Ga atoms: {n_Ga} ({n_Ga/n_atoms*100:.1f}%)")
    print(f"  In atoms: {n_In} ({n_In/n_atoms*100:.1f}%)")
    print(f"  Temperature: {T} °C")

    # Density at T=150°C from experimental data
    # For eutectic composition, interpolate between Ga and In densities
    # From paper: ρ(Ga, 150°C) ≈ 6.0 g/cm³, ρ(In, 150°C) ≈ 7.0 g/cm³
    # For eutectic: use experimental value
    if T == 150.0:
        rho = 6.28  # g/cm³ (experimental value at 150°C for eutectic)
    else:
        # Approximate density for other temperatures
        rho = 6.28 * (1 - 0.0001 * (T - 150))  # Rough temperature dependence

    # Atomic masses
    M_Ga = 69.723  # g/mol
    M_In = 114.818  # g/mol
    M_avg = (1 - x_In) * M_Ga + x_In * M_In

    # Calculate box size from density
    # V = n_atoms * M_avg / (rho * N_A)
    # where N_A = 6.022e23 mol^-1
    N_A = 6.022e23
    V = n_atoms * M_avg / (rho * N_A) * 1e24  # Å³
    L = V ** (1/3)  # Cubic box side length

    print(f"  Density: {rho:.3f} g/cm³")
    print(f"  Box size: {L:.2f} Å")
    print(f"  Volume: {V:.2f} Å³")

    # Generate random atomic positions
    np.random.seed(12345)  # For reproducibility
    positions = np.random.rand(n_atoms, 3) * L

    # Assign atom types (1=Ga, 2=In)
    atom_types = np.array([1] * n_Ga + [2] * n_In)

    # Shuffle to randomize distribution
    indices = np.arange(n_atoms)
    np.random.shuffle(indices)
    atom_types = atom_types[indices]

    # Write LAMMPS data file
    with open(output_file, 'w') as f:
        f.write(f'EGaIn eutectic structure Ga{1-x_In:.3f}In{x_In:.3f} at {T}C\n\n')
        f.write(f'{n_atoms} atoms\n')
        f.write('2 atom types\n\n')
        f.write(f'0.0 {L:.10f} xlo xhi\n')
        f.write(f'0.0 {L:.10f} ylo yhi\n')
        f.write(f'0.0 {L:.10f} zlo zhi\n\n')
        f.write('Masses\n\n')
        f.write(f'1 {M_Ga:.6f}  # Ga\n')
        f.write(f'2 {M_In:.6f}  # In\n\n')
        f.write('Atoms  # atomic\n\n')

        for i, (pos, atype) in enumerate(zip(positions, atom_types)):
            f.write(f'{i+1} {atype} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n')

    print(f"\nStructure written to {output_file}")

    # Print statistics
    n_Ga_actual = np.sum(atom_types == 1)
    n_In_actual = np.sum(atom_types == 2)
    print(f"\nFinal composition check:")
    print(f"  Ga: {n_Ga_actual} atoms ({n_Ga_actual/n_atoms*100:.2f}%)")
    print(f"  In: {n_In_actual} atoms ({n_In_actual/n_atoms*100:.2f}%)")

    return L, n_Ga, n_In


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create initial structure for EGaIn alloy')
    parser.add_argument('-n', '--natoms', type=int, default=1000,
                        help='Total number of atoms (default: 1000)')
    parser.add_argument('-x', '--xIn', type=float, default=0.142,
                        help='Mole fraction of In (default: 0.142 for eutectic)')
    parser.add_argument('-T', '--temperature', type=float, default=150.0,
                        help='Temperature in Celsius (default: 150.0)')
    parser.add_argument('-o', '--output', type=str, default='inputs/initial_structure.data',
                        help='Output LAMMPS data file (default: inputs/initial_structure.data)')

    args = parser.parse_args()

    create_egain_structure(
        n_atoms=args.natoms,
        x_In=args.xIn,
        T=args.temperature,
        output_file=args.output
    )
