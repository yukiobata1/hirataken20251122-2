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

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {log_file}")

    # Use Popen to show real-time progress from log file
    import time
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    # Monitor log file for thermo output
    last_size = 0
    total_steps = 70000  # 5000 + 15000 + 50000 from input file
    header_printed = False
    current_phase = ""

    while process.poll() is None:
        time.sleep(0.5)

        # Check if log file exists and has new content
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    last_size = f.tell()

                    for line in new_content.split('\n'):
                        # Detect phase changes
                        if "Energy Minimization" in line:
                            current_phase = "Minimization"
                            print(f"\n  [{current_phase}]")
                        elif "Equilibration" in line:
                            current_phase = "Equilibration"
                            print(f"\n  [{current_phase}] (20,000 steps)")
                        elif "Production run" in line:
                            current_phase = "Production"
                            print(f"\n  [{current_phase}] (50,000 steps)")

                        # Print thermo header
                        if line.strip().startswith("Step ") and not header_printed:
                            print(f"  {line.strip()}")
                            header_printed = True

                        # Print thermo data lines (start with step number)
                        parts = line.split()
                        if len(parts) >= 8:
                            try:
                                step = int(parts[0])
                                temp = float(parts[1])
                                pe = float(parts[2])
                                # Format: Step, Temp, PE, KE, Etotal, Press, Vol, Density, CPU
                                elapsed = time.time() - start_time
                                print(f"  Step {step:>6} | T={temp:>7.2f}K | PE={pe:>12.1f} | "
                                      f"elapsed: {elapsed:>6.1f}s", flush=True)
                            except (ValueError, IndexError):
                                pass
            except Exception:
                pass

    stdout, stderr = process.communicate()
    elapsed_time = time.time() - start_time

    if process.returncode != 0:
        print(f"\nLAMMPS STDOUT:\n{stdout}")
        print(f"LAMMPS STDERR:\n{stderr}")
        raise RuntimeError(f"LAMMPS failed with return code {process.returncode}")

    print(f"\n  LAMMPS completed (total elapsed: {elapsed_time:.1f}s)")


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
    alpha = 0.2             # Learning rate (reduced from 0.3 for stability)
    tol = 250.0             # Convergence tolerance (χ²) - expect ~179 for good fit
    T = 423.15              # Temperature (K) - 150°C
    r_min = 2.0             # Minimum distance for U_EP grid (Å)
    r_max = 12.0            # Maximum distance for U_EP grid (Å)
    N_grid = 200            # Number of grid points
    max_amp = 3.0           # Maximum amplitude for U_EP (kcal/mol) - increased from 1.0
    sigma_smooth = 1.5      # Gaussian smoothing sigma (reduced from 2 to preserve features)

    # GPU/Kokkos settings for H100
    use_gpu = True          # Use H100 GPU with Kokkos
    gpu_id = 0              # GPU device ID

    # File paths
    exp_data_file = 'data/g_exp_cleaned.dat'
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
    print(f"\nExpected χ² for good fit: ~{len(r_exp)} (number of data points)")
    print(f"Target χ² < {tol} is very strict - may need adjustment")

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
        # sigma = experimental uncertainty in g(r)
        # For neutron diffraction: typically 0.05-0.10
        # Adjust based on your data quality
        chi2 = calc_chi_squared(g_sim_interp, g_exp, sigma=0.05)
        r_factor = calc_r_factor(g_sim_interp, g_exp)
        chi2_history.append(chi2)

        print(f"χ² = {chi2:.6f}")
        print(f"R-factor = {r_factor:.6f}")

        # 6. Plot current results
        plot_results(r_exp, g_exp, r_sim, g_sim_total, r_grid,
                     U_ep_GaGa, U_ep_InIn, U_ep_GaIn, chi2_history, iteration,
                     output_file=f'outputs/epsr_iter{iteration:03d}.png')

        # 7. Check convergence
        # Primary criterion: absolute χ² must be below tolerance
        if chi2 < tol:
            print(f"\n✅ Converged! χ² = {chi2:.6f} < {tol}")
            break

        # Secondary criteria: check for stagnation or divergence
        if len(chi2_history) > 1:
            delta_chi2 = chi2_history[-1] - chi2_history[-2]  # Keep sign to detect increase
            abs_delta = abs(delta_chi2)
            rel_change = abs_delta / chi2_history[-2] if chi2_history[-2] > 0 else 0
            print(f"Δχ² = {delta_chi2:.6f} (relative: {rel_change:.6f})")

            # Warning: χ² is increasing (fit getting worse)
            if delta_chi2 > 0:
                print(f"⚠️  WARNING: χ² increased by {delta_chi2:.2f}")

                # If χ² increases significantly multiple times, stop
                if len(chi2_history) >= 3:
                    recent_increases = sum(1 for i in range(-3, 0)
                                         if i+1 < 0 and chi2_history[i+1] > chi2_history[i])
                    if recent_increases >= 2:
                        print(f"❌ Stopping: χ² has increased {recent_increases} times in last 3 iterations")
                        print(f"   Algorithm appears to be diverging.")
                        break

            # Only check relative change if:
            # 1. χ² is reasonably small (< 100 at least)
            # 2. Multiple consecutive small changes
            if chi2 < 100 and rel_change < 0.01:
                print(f"\n✅ Converged! Relative χ² change ({rel_change:.6f}) < 0.01 with χ²={chi2:.2f}")
                break
            elif chi2 >= 100 and rel_change < 0.01:
                print(f"⚠️  Small relative change detected, but χ²={chi2:.2f} is still too high")
                print(f"   (need χ² < 100 for relative convergence check)")

                # Check if we're stuck with high χ²
                if len(chi2_history) >= 5:
                    recent_chi2 = chi2_history[-5:]
                    if max(recent_chi2) - min(recent_chi2) < 1.0:
                        print(f"❌ Stopping: χ² stuck around {chi2:.2f} for 5 iterations")
                        print(f"   Possible issues:")
                        print(f"   - Initial structure far from experiment")
                        print(f"   - Learning rate α={alpha} may be too small or too large")
                        print(f"   - Experimental data may have systematic differences")
                        break

        # 8. Update U_EP
        print("\nUpdating empirical potentials...")

        # Interpolate experimental data to U_EP grid
        g_exp_on_grid = np.interp(r_grid, r_exp, g_exp)

        # CRITICAL FIX: Interpolate PARTIAL g(r) to U_EP grid
        # Each pair potential is updated using its own partial g(r)
        # This allows independent optimization of each interaction
        g_GaGa_on_grid = np.interp(r_grid, r_sim, g_GaGa)
        g_GaIn_on_grid = np.interp(r_grid, r_sim, g_GaIn)
        g_InIn_on_grid = np.interp(r_grid, r_sim, g_InIn)

        # Calculate weights based on neutron scattering lengths and composition
        # Composition: Ga0.858 In0.142
        c_Ga = 0.858
        c_In = 0.142
        # Neutron scattering lengths (fm)
        b_Ga = 7.288
        b_In = 4.061

        # Weight contributions to total structure factor S(Q) or g(r)
        # W_ij = c_i * c_j * b_i * b_j * (2 if i!=j else 1)
        w_GaGa = (c_Ga * b_Ga)**2
        w_InIn = (c_In * b_In)**2
        w_GaIn = 2 * (c_Ga * b_Ga) * (c_In * b_In)

        # Normalize weights so the strongest contribution uses the full alpha
        w_max = max(w_GaGa, w_InIn, w_GaIn)
        w_GaGa_norm = w_GaGa / w_max
        w_InIn_norm = w_InIn / w_max
        w_GaIn_norm = w_GaIn / w_max

        if iteration == 1:
            print(f"\nWeighting factors for potential updates:")
            print(f"  Ga-Ga: {w_GaGa:.2f} (norm: {w_GaGa_norm:.3f})")
            print(f"  In-In: {w_InIn:.2f} (norm: {w_InIn_norm:.3f})")
            print(f"  Ga-In: {w_GaIn:.2f} (norm: {w_GaIn_norm:.3f})")

        # Apply weighted updates using PARTIAL g(r)
        # Each partial g(r) is compared to g_exp (approximation)
        # The weighted combination will converge to match g_exp
        # This prevents "invisible" components (like In-In) from being
        # driven wildly by errors in the dominant component (Ga-Ga)

        U_ep_GaGa = update_ep(r_grid, g_GaGa_on_grid, g_exp_on_grid, U_ep_GaGa,
                              alpha=alpha * w_GaGa_norm, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)
        U_ep_InIn = update_ep(r_grid, g_InIn_on_grid, g_exp_on_grid, U_ep_InIn,
                              alpha=alpha * w_InIn_norm, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)
        U_ep_GaIn = update_ep(r_grid, g_GaIn_on_grid, g_exp_on_grid, U_ep_GaIn,
                              alpha=alpha * w_GaIn_norm, T=T, max_amp=max_amp, sigma_smooth=sigma_smooth)

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
