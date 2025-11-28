#!/usr/bin/env python3
"""
Main EPSR (Empirical Potential Structure Refinement) loop for EGaIn system
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import update functions
from update_ep import (
    update_ep, write_lammps_table, calc_chi_squared,
    calc_r_factor, initialize_ep_tables
)


def read_lammps_rdf(filename):
    """
    Read LAMMPS RDF output file

    Parameters
    ----------
    filename : str
        LAMMPS RDF output file

    Returns
    -------
    r : array
        Distance array (Å)
    g_total : array
        Total g(r)
    g_GaGa : array or None
        Ga-Ga partial g(r) (if available)
    g_GaIn : array or None
        Ga-In partial g(r) (if available)
    g_InIn : array or None
        In-In partial g(r) (if available)
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(x) for x in parts])
                except:
                    continue

    if len(data) == 0:
        raise ValueError(f"No data found in {filename}")

    data = np.array(data)

    # LAMMPS RDF output format:
    # Column 1: timestep
    # Column 2: r
    # Column 3: g_total (all pairs)
    # Columns 4+: coordination numbers and partial g(r) if requested

    r = data[:, 1]
    g_total = data[:, 2]

    # Check if partial g(r) are available
    # For "compute rdf all rdf 200 1 1 1 2 2 2":
    # Columns: timestep, r, g_total, coord_total, g_11, coord_11, g_12, coord_12, g_22, coord_22
    g_GaGa = None
    g_GaIn = None
    g_InIn = None

    if data.shape[1] >= 9:
        g_GaGa = data[:, 4]  # g_11 (Ga-Ga)
        g_GaIn = data[:, 6]  # g_12 (Ga-In)
        g_InIn = data[:, 8]  # g_22 (In-In)

    return r, g_total, g_GaGa, g_GaIn, g_InIn


def load_experimental_data(filename):
    """
    Load experimental g(r) data

    Parameters
    ----------
    filename : str
        Experimental data file (2 columns: r, g(r))

    Returns
    -------
    r_exp : array
        Distance array (Å)
    g_exp : array
        Experimental g(r)
    """
    data = np.loadtxt(filename)
    r_exp = data[:, 0]
    g_exp = data[:, 1]
    return r_exp, g_exp


def run_lammps(input_file, log_file='lammps.log', use_gpu=True, gpu_id=0):
    """
    Run LAMMPS simulation with optional GPU/Kokkos acceleration

    Parameters
    ----------
    input_file : str
        LAMMPS input file
    log_file : str
        LAMMPS log file (default: 'lammps.log')
    use_gpu : bool
        Use GPU acceleration with Kokkos (default: True)
    gpu_id : int
        GPU device ID (default: 0)
    """
    if use_gpu:
        # KOKKOS command for H100 GPU
        # -k on g 1 : Enable Kokkos, use 1 GPU
        # -sf kk : Apply Kokkos suffix to supported pair styles
        cmd = ['lmp', '-k', 'on', 'g', '1', '-sf', 'kk',
               '-in', input_file, '-log', log_file]
        print(f"Running LAMMPS with Kokkos/H100 GPU {gpu_id}: {input_file}")

        # Set environment variable to select specific GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        # CPU-only mode
        cmd = ['lmp', '-in', input_file, '-log', log_file]
        env = None
        print(f"Running LAMMPS (CPU only) with input: {input_file}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode != 0:
        print(f"LAMMPS STDOUT:\n{result.stdout}")
        print(f"LAMMPS STDERR:\n{result.stderr}")
        raise RuntimeError(f"LAMMPS failed with return code {result.returncode}")

    print("LAMMPS completed successfully")


def plot_results(r_exp, g_exp, r_sim, g_sim, r_grid, U_ep_GaGa, U_ep_InIn, U_ep_GaIn,
                 chi2_history, iteration, output_file='outputs/epsr_results.png'):
    """
    Plot EPSR results: g(r) comparison, U_EP, and convergence

    Parameters
    ----------
    r_exp, g_exp : arrays
        Experimental data
    r_sim, g_sim : arrays
        Simulated data
    r_grid : array
        Distance grid for U_EP
    U_ep_GaGa, U_ep_InIn, U_ep_GaIn : arrays
        Empirical potentials
    chi2_history : list
        χ² history
    iteration : int
        Current iteration number
    output_file : str
        Output figure filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: g(r) comparison
    axes[0].plot(r_exp, g_exp, 'k-', label='Experiment', linewidth=2)
    axes[0].plot(r_sim, g_sim, 'r--', label='Simulation', linewidth=2)
    axes[0].set_xlabel('r (Å)', fontsize=12)
    axes[0].set_ylabel('G(r)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_title(f'Pair Distribution Function (Iter {iteration})', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: U_EP
    axes[1].plot(r_grid, U_ep_GaGa, label='Ga-Ga', linewidth=2)
    axes[1].plot(r_grid, U_ep_InIn, label='In-In', linewidth=2)
    axes[1].plot(r_grid, U_ep_GaIn, label='Ga-In', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('r (Å)', fontsize=12)
    axes[1].set_ylabel('U_EP (kcal/mol)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_title('Empirical Potential', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Convergence
    if len(chi2_history) > 0:
        axes[2].plot(range(1, len(chi2_history) + 1), chi2_history, 'o-', linewidth=2)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('χ²', fontsize=12)
        axes[2].set_title('Convergence', fontsize=12)
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to {output_file}")
    plt.close()


def main():
    """
    Main EPSR loop
    """
    # ========== Parameters ==========
    max_iter = 50           # Maximum iterations
    alpha = 0.3             # Learning rate
    tol = 0.1               # Convergence tolerance (χ²)
    T = 423.15              # Temperature (K) - 150°C
    r_min = 2.0             # Minimum distance for U_EP grid (Å)
    r_max = 12.0            # Maximum distance for U_EP grid (Å)
    N_grid = 200            # Number of grid points
    max_amp = 1.0           # Maximum amplitude for U_EP (kcal/mol)
    sigma_smooth = 2        # Gaussian smoothing sigma

    # GPU/Kokkos settings for H100
    use_gpu = True          # Use H100 GPU with Kokkos
    gpu_id = 0              # GPU device ID

    # File paths
    exp_data_file = 'data/g_exp.dat'
    lammps_input = 'inputs/in.egain_epsr_H100' if use_gpu else 'inputs/in.egain_epsr'
    rdf_output = 'rdf.dat'

    print("=" * 60)
    print("EPSR for EGaIn System (H100 GPU-Accelerated)" if use_gpu else "EPSR for EGaIn System")
    print("=" * 60)
    print(f"GPU Mode: {'ENABLED (H100 via Kokkos)' if use_gpu else 'DISABLED (CPU only)'}")
    if use_gpu:
        print(f"GPU Device: {gpu_id}")
    print(f"Max iterations: {max_iter}")
    print(f"Learning rate α: {alpha}")
    print(f"Convergence tolerance: χ² < {tol}")
    print(f"Temperature: {T} K ({T-273.15:.1f}°C)")
    print(f"U_EP grid: {r_min} - {r_max} Å ({N_grid} points)")
    print(f"Max amplitude: {max_amp} kcal/mol")
    print("=" * 60)

    # ========== Load experimental data ==========
    print("\nLoading experimental data...")
    if not os.path.exists(exp_data_file):
        raise FileNotFoundError(f"Experimental data file not found: {exp_data_file}")

    r_exp, g_exp = load_experimental_data(exp_data_file)
    print(f"Loaded {len(r_exp)} experimental data points")
    print(f"r_exp range: {r_exp.min():.2f} - {r_exp.max():.2f} Å")
    print(f"g_exp range: {g_exp.min():.3f} - {g_exp.max():.3f}")

    # ========== Initialize U_EP ==========
    print("\nInitializing empirical potentials...")
    r_grid = np.linspace(r_min, r_max, N_grid)
    U_ep_GaGa = np.zeros_like(r_grid)
    U_ep_InIn = np.zeros_like(r_grid)
    U_ep_GaIn = np.zeros_like(r_grid)

    # Write initial (zero) tables
    write_lammps_table('data/ep_GaGa.table', r_grid, U_ep_GaGa, 'EP_GAGA', r_min, r_max, N_grid)
    write_lammps_table('data/ep_InIn.table', r_grid, U_ep_InIn, 'EP_ININ', r_min, r_max, N_grid)
    write_lammps_table('data/ep_GaIn.table', r_grid, U_ep_GaIn, 'EP_GAIN', r_min, r_max, N_grid)

    # ========== EPSR iterations ==========
    chi2_history = []

    for iteration in range(1, max_iter + 1):
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration}/{max_iter}")
        print("=" * 60)

        # 1. Write current U_EP tables
        write_lammps_table('data/ep_GaGa.table', r_grid, U_ep_GaGa, 'EP_GAGA', r_min, r_max, N_grid)
        write_lammps_table('data/ep_InIn.table', r_grid, U_ep_InIn, 'EP_ININ', r_min, r_max, N_grid)
        write_lammps_table('data/ep_GaIn.table', r_grid, U_ep_GaIn, 'EP_GAIN', r_min, r_max, N_grid)

        # 2. Run LAMMPS
        try:
            run_lammps(lammps_input,
                      log_file=f'outputs/lammps_iter{iteration:03d}.log',
                      use_gpu=use_gpu,
                      gpu_id=gpu_id)
        except RuntimeError as e:
            print(f"LAMMPS failed: {e}")
            print("Stopping EPSR loop.")
            break

        # 3. Read g(r) from LAMMPS output
        if not os.path.exists(rdf_output):
            raise FileNotFoundError(f"RDF output not found: {rdf_output}")

        r_sim, g_sim_total, g_GaGa, g_GaIn, g_InIn = read_lammps_rdf(rdf_output)
        print(f"Read {len(r_sim)} simulated RDF points")

        # 4. Interpolate simulated g(r) to experimental grid
        g_sim_interp = np.interp(r_exp, r_sim, g_sim_total)

        # 5. Calculate χ²
        chi2 = calc_chi_squared(g_sim_interp, g_exp, sigma=0.01)
        r_factor = calc_r_factor(g_sim_interp, g_exp)
        chi2_history.append(chi2)

        print(f"χ² = {chi2:.6f}")
        print(f"R-factor = {r_factor:.6f}")

        # 6. Plot current results
        plot_results(r_exp, g_exp, r_sim, g_sim_total, r_grid,
                     U_ep_GaGa, U_ep_InIn, U_ep_GaIn, chi2_history, iteration,
                     output_file=f'outputs/epsr_iter{iteration:03d}.png')

        # 7. Check convergence
        if chi2 < tol:
            print(f"\nConverged! χ² = {chi2:.6f} < {tol}")
            break

        if len(chi2_history) > 1:
            delta_chi2 = abs(chi2_history[-1] - chi2_history[-2])
            rel_change = delta_chi2 / chi2 if chi2 > 0 else 0
            print(f"Relative χ² change: {rel_change:.6f}")

            if rel_change < 0.01:
                print(f"\nConverged! Relative χ² change ({rel_change:.6f}) < 0.01")
                break

        # 8. Update U_EP
        print("\nUpdating empirical potentials...")

        # Interpolate experimental data to U_EP grid
        g_exp_on_grid = np.interp(r_grid, r_exp, g_exp)
        g_sim_on_grid = np.interp(r_grid, r_sim, g_sim_total)

        # Update U_EP for each pair (simplified: use total g(r))
        # In a more sophisticated approach, use partial g(r) for each pair
        U_ep_GaGa = update_ep(r_grid, g_sim_on_grid, g_exp_on_grid, U_ep_GaGa,
                              alpha=alpha, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)
        U_ep_InIn = update_ep(r_grid, g_sim_on_grid, g_exp_on_grid, U_ep_InIn,
                              alpha=alpha, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)
        U_ep_GaIn = update_ep(r_grid, g_sim_on_grid, g_exp_on_grid, U_ep_GaIn,
                              alpha=alpha, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)

        print(f"U_EP ranges: Ga-Ga [{U_ep_GaGa.min():.3f}, {U_ep_GaGa.max():.3f}], "
              f"In-In [{U_ep_InIn.min():.3f}, {U_ep_InIn.max():.3f}], "
              f"Ga-In [{U_ep_GaIn.min():.3f}, {U_ep_GaIn.max():.3f}] kcal/mol")

    # ========== Save final results ==========
    print("\n" + "=" * 60)
    print("EPSR completed")
    print("=" * 60)

    # Save final U_EP
    np.savez('outputs/final_ep.npz',
             r=r_grid,
             U_ep_GaGa=U_ep_GaGa,
             U_ep_InIn=U_ep_InIn,
             U_ep_GaIn=U_ep_GaIn,
             chi2_history=chi2_history)

    print(f"\nFinal results saved to outputs/final_ep.npz")
    print(f"Total iterations: {len(chi2_history)}")

    if len(chi2_history) > 0:
        print(f"Final χ²: {chi2_history[-1]:.6f}")

        # Final plot
        if os.path.exists(rdf_output):
            r_sim, g_sim_total, _, _, _ = read_lammps_rdf(rdf_output)
            plot_results(r_exp, g_exp, r_sim, g_sim_total, r_grid,
                         U_ep_GaGa, U_ep_InIn, U_ep_GaIn, chi2_history, len(chi2_history),
                         output_file='outputs/epsr_final.png')
        else:
            print(f"Warning: RDF output file not found: {rdf_output}")
    else:
        print("\nWarning: No iterations completed successfully.")
        print("Possible causes:")
        print("  - LAMMPS execution failed")
        print("  - RDF output file was not generated")
        print("  - Check LAMMPS input files and data files")
        print("  - Review LAMMPS log files on the cloud instance")


if __name__ == '__main__':
    main()
