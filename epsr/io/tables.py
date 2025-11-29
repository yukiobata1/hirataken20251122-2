"""
Writing LAMMPS table potential files.
"""

import numpy as np
import os
from typing import Dict


def write_lammps_table(
    filename: str,
    r: np.ndarray,
    U: np.ndarray,
    label: str,
    r_min: float,
    r_max: float,
    n_points: int
) -> None:
    """
    Write a LAMMPS table potential file.

    Format:
    # Comment lines
    LABEL
    N n_points

    index  r  U  F
    ...

    Parameters
    ----------
    filename : str
        Output filename
    r : np.ndarray
        Distance array (Å)
    U : np.ndarray
        Potential energy (kcal/mol)
    label : str
        Table label (e.g., 'EP_GAGA')
    r_min : float
        Minimum distance (Å)
    r_max : float
        Maximum distance (Å)
    n_points : int
        Number of table points
    """
    # Create uniform grid if needed
    if len(r) != n_points or not np.allclose(np.diff(r), r[1] - r[0]):
        r_uniform = np.linspace(r_min, r_max, n_points)
        U_uniform = np.interp(r_uniform, r, U)
    else:
        r_uniform = r
        U_uniform = U

    # Calculate force: F = -dU/dr
    F = -np.gradient(U_uniform, r_uniform)

    # Write table file
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f'# LAMMPS table potential for EPSR\n')
        f.write(f'# Generated empirical potential\n')
        f.write(f'# Units: distance (Å), energy (kcal/mol), force (kcal/mol/Å)\n\n')
        f.write(f'{label}\n')
        f.write(f'N {n_points}\n\n')

        for i, (ri, ui, fi) in enumerate(zip(r_uniform, U_uniform, F)):
            f.write(f'{i+1:6d} {ri:15.10f} {ui:15.10f} {fi:15.10f}\n')


def write_potential_tables(
    output_dir: str,
    r_grid: np.ndarray,
    potentials: Dict[str, np.ndarray],
    r_min: float,
    r_max: float,
    n_points: int
) -> None:
    """
    Write multiple LAMMPS table files for different pair interactions.

    Parameters
    ----------
    output_dir : str
        Output directory
    r_grid : np.ndarray
        Distance grid (Å)
    potentials : dict
        Dictionary of potentials, e.g.,
        {'GaGa': U_GaGa, 'InIn': U_InIn, 'GaIn': U_GaIn}
    r_min : float
        Minimum distance (Å)
    r_max : float
        Maximum distance (Å)
    n_points : int
        Number of table points

    Examples
    --------
    >>> write_potential_tables(
    ...     'data',
    ...     r_grid,
    ...     {'GaGa': U_GaGa, 'InIn': U_InIn, 'GaIn': U_GaIn},
    ...     r_min=2.0,
    ...     r_max=12.0,
    ...     n_points=200
    ... )
    """
    os.makedirs(output_dir, exist_ok=True)

    for pair_name, U in potentials.items():
        filename = os.path.join(output_dir, f'ep_{pair_name}.table')
        label = f'EP_{pair_name.upper()}'

        write_lammps_table(
            filename=filename,
            r=r_grid,
            U=U,
            label=label,
            r_min=r_min,
            r_max=r_max,
            n_points=n_points
        )

        print(f"Wrote {filename} (label: {label})")
