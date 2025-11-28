#!/usr/bin/env python3
"""
U_EP (Empirical Potential) update functions for EPSR
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def update_ep(r, g_sim, g_exp, U_ep_old, alpha=0.3, T=423.15, max_amp=1.0, sigma_smooth=2):
    """
    Update empirical potential U_EP based on difference between simulated and experimental g(r)

    Parameters
    ----------
    r : array
        Distance array (Å)
    g_sim : array
        Simulated g(r)
    g_exp : array
        Experimental g(r)
    U_ep_old : array
        Current U_EP (kcal/mol)
    alpha : float
        Learning rate (default: 0.3)
    T : float
        Temperature (K) (default: 423.15 K = 150°C)
    max_amp : float
        Maximum amplitude for U_EP (kcal/mol) (default: 1.0)
    sigma_smooth : float
        Gaussian smoothing sigma (default: 2)

    Returns
    -------
    U_ep_new : array
        Updated U_EP (kcal/mol)
    """
    # Boltzmann constant in kcal/(mol·K)
    kB = 0.001987  # kcal/(mol·K)
    kT = kB * T

    # Calculate difference
    delta_g = g_sim - g_exp

    # Update U_EP according to EPSR formula:
    # U_EP^(n+1) = U_EP^(n) + α * kT * [g_sim - g_exp]
    U_ep_new = U_ep_old + alpha * kT * delta_g

    # Apply amplitude clipping to prevent non-physical values
    U_ep_new = np.clip(U_ep_new, -max_amp, max_amp)

    # Apply Gaussian smoothing to reduce noise
    if sigma_smooth > 0:
        U_ep_new = gaussian_filter1d(U_ep_new, sigma=sigma_smooth)

    return U_ep_new


def calc_force_from_potential(r, U):
    """
    Calculate force from potential: F = -dU/dr

    Parameters
    ----------
    r : array
        Distance array (Å)
    U : array
        Potential energy (kcal/mol)

    Returns
    -------
    F : array
        Force (kcal/mol/Å)
    """
    # F = -dU/dr
    F = -np.gradient(U, r)
    return F


def write_lammps_table(filename, r, U_ep, label, r_min=None, r_max=None, N=None):
    """
    Write LAMMPS table potential file

    Parameters
    ----------
    filename : str
        Output filename
    r : array
        Distance array (Å)
    U_ep : array
        Empirical potential (kcal/mol)
    label : str
        Table label (e.g., 'EP_GAGA', 'EP_ININ', 'EP_GAIN')
    r_min : float, optional
        Minimum distance for table (default: min(r))
    r_max : float, optional
        Maximum distance for table (default: max(r))
    N : int, optional
        Number of table entries (default: len(r))
    """
    # Set defaults
    if r_min is None:
        r_min = r[0]
    if r_max is None:
        r_max = r[-1]
    if N is None:
        N = len(r)

    # Create uniform grid if needed
    if len(r) != N or not np.allclose(np.diff(r), r[1] - r[0]):
        r_uniform = np.linspace(r_min, r_max, N)
        U_ep_uniform = np.interp(r_uniform, r, U_ep)
    else:
        r_uniform = r
        U_ep_uniform = U_ep

    # Calculate force
    F = calc_force_from_potential(r_uniform, U_ep_uniform)

    # Write LAMMPS table file
    with open(filename, 'w') as f:
        f.write(f'# LAMMPS table potential for EPSR\n')
        f.write(f'# Generated empirical potential\n')
        f.write(f'# Units: distance (Å), energy (kcal/mol), force (kcal/mol/Å)\n\n')
        f.write(f'{label}\n')
        f.write(f'N {N}\n\n')

        for i, (ri, ui, fi) in enumerate(zip(r_uniform, U_ep_uniform, F)):
            # Format: index distance energy force
            f.write(f'{i+1:6d} {ri:15.10f} {ui:15.10f} {fi:15.10f}\n')

    print(f"LAMMPS table written to {filename}")
    print(f"  Label: {label}")
    print(f"  Range: {r_min:.2f} - {r_max:.2f} Å")
    print(f"  N points: {N}")


def initialize_ep_tables(r_min=2.0, r_max=12.0, N=200, output_dir='data'):
    """
    Initialize empirical potential tables (all zeros) for Ga-Ga, In-In, and Ga-In pairs

    Parameters
    ----------
    r_min : float
        Minimum distance (Å) (default: 2.0)
    r_max : float
        Maximum distance (Å) (default: 12.0)
    N : int
        Number of points (default: 200)
    output_dir : str
        Output directory (default: 'data')

    Returns
    -------
    r : array
        Distance array
    """
    # Create uniform distance grid
    r = np.linspace(r_min, r_max, N)

    # Initialize U_EP as zeros
    U_ep_zero = np.zeros_like(r)

    # Write table files
    import os
    os.makedirs(output_dir, exist_ok=True)

    write_lammps_table(f'{output_dir}/ep_GaGa.table', r, U_ep_zero, 'EP_GAGA', r_min, r_max, N)
    write_lammps_table(f'{output_dir}/ep_InIn.table', r, U_ep_zero, 'EP_ININ', r_min, r_max, N)
    write_lammps_table(f'{output_dir}/ep_GaIn.table', r, U_ep_zero, 'EP_GAIN', r_min, r_max, N)

    return r


def calc_chi_squared(g_sim, g_exp, sigma=0.01):
    """
    Calculate χ² between simulated and experimental g(r)

    Parameters
    ----------
    g_sim : array
        Simulated g(r)
    g_exp : array
        Experimental g(r)
    sigma : float or array
        Measurement uncertainty (default: 0.01)

    Returns
    -------
    chi2 : float
        χ² value
    """
    chi2 = np.sum((g_sim - g_exp)**2 / sigma**2)
    return chi2


def calc_r_factor(g_sim, g_exp):
    """
    Calculate R-factor (crystallography metric)

    Parameters
    ----------
    g_sim : array
        Simulated g(r)
    g_exp : array
        Experimental g(r)

    Returns
    -------
    r_factor : float
        R-factor value
    """
    r_factor = np.sum(np.abs(g_sim - g_exp)) / np.sum(np.abs(g_exp))
    return r_factor


if __name__ == '__main__':
    # Test: Initialize EP tables
    print("Initializing empirical potential tables...")
    r = initialize_ep_tables(r_min=2.0, r_max=12.0, N=200, output_dir='data')
    print(f"\nCreated initial EP tables with r range: {r[0]:.2f} - {r[-1]:.2f} Å")
