"""
Main EPSR Engine - coordinates the entire refinement workflow.

This module brings together all components:
- LAMMPS simulations
- Structure factor calculations
- Potential refinement
- Convergence monitoring

Implements the full EPSR algorithm as described by Soper (1996, 2005).
"""

import numpy as np
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from epsr.core.potential import EmpiricalPotential, calculate_scattering_weights
from epsr.core.structure_factor import StructureFactor
from epsr.io.experimental import load_experimental_data, interpolate_to_grid
from epsr.io.lammps import LAMMPSInterface
from epsr.io.tables import write_potential_tables
from epsr.analysis.metrics import calculate_chi_squared, calculate_r_factor
from epsr.visualization.plots import plot_epsr_summary, plot_structure_factor, plot_rdf_comparison


@dataclass
class EPSRConfig:
    """
    Configuration for EPSR simulation.

    Parameters
    ----------
    temperature : float
        System temperature (K)
    composition : dict
        Mole fractions, e.g., {'Ga': 0.858, 'In': 0.142}
    scattering_lengths : dict
        Neutron scattering lengths (fm), e.g., {'Ga': 7.288, 'In': 4.061}
    density : float
        Number density (atoms/Å³)
    r_min : float, optional
        Minimum distance for potential grid (Å), default: 2.0
    r_max : float, optional
        Maximum distance for potential grid (Å), default: 12.0
    n_grid : int, optional
        Number of grid points for potential, default: 200
    max_iterations : int, optional
        Maximum EPSR iterations, default: 50
    convergence_tol : float, optional
        Chi-squared convergence tolerance, default: 250.0
    learning_rate : float, optional
        EPSR feedback parameter (alpha), default: 0.3
    max_amplitude : float, optional
        Maximum potential amplitude (kcal/mol), default: 3.0
    smooth_sigma : float, optional
        Gaussian smoothing width (Å), default: 0.3
    use_momentum : bool, optional
        Use momentum-based optimization, default: True
    momentum_beta : float, optional
        Momentum coefficient, default: 0.9
    method : str, optional
        Update method: 'simple', 'momentum', 'nesterov', default: 'momentum'
    use_sq_update : bool, optional
        Use S(Q) Fourier method (True) or direct g(r) method (False), default: True
    sigma_exp : float, optional
        Experimental uncertainty for chi-squared, default: 0.05
    use_gpu : bool, optional
        Use GPU acceleration for LAMMPS, default: False
    gpu_id : int, optional
        GPU device ID, default: 0
    output_dir : str, optional
        Output directory for results, default: 'outputs'
    data_dir : str, optional
        Directory for potential tables, default: 'data'
    verbose : bool, optional
        Print detailed progress, default: True
    """

    # System properties
    temperature: float
    composition: Dict[str, float]
    scattering_lengths: Dict[str, float]
    density: float

    # Grid parameters
    r_min: float = 2.0
    r_max: float = 12.0
    n_grid: int = 200

    # EPSR parameters
    max_iterations: int = 50
    convergence_tol: float = 250.0
    learning_rate: float = 0.3
    max_amplitude: float = 3.0
    smooth_sigma: float = 0.3

    # Optimization
    use_momentum: bool = True
    momentum_beta: float = 0.9
    method: str = 'momentum'
    use_sq_update: bool = True

    # Experimental data
    sigma_exp: float = 0.05

    # LAMMPS
    use_gpu: bool = False
    gpu_id: int = 0

    # I/O
    output_dir: str = 'outputs'
    data_dir: str = 'data'
    verbose: bool = True

    def __post_init__(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


class EPSREngine:
    """
    Main EPSR Engine for structure refinement.

    Coordinates the iterative refinement process:
    1. Run LAMMPS with current potentials
    2. Calculate g(r) from simulation
    3. Compare with experimental g(r)
    4. Update empirical potentials
    5. Repeat until convergence

    Parameters
    ----------
    config : EPSRConfig
        EPSR configuration
    experimental_data_file : str
        Path to experimental g(r) file

    Examples
    --------
    >>> config = EPSRConfig(
    ...     temperature=423.15,
    ...     composition={'Ga': 0.858, 'In': 0.142},
    ...     scattering_lengths={'Ga': 7.288, 'In': 4.061},
    ...     density=0.042
    ... )
    >>> engine = EPSREngine(config, 'data/g_exp_cleaned.dat')
    >>> engine.run('inputs/in.egain_epsr_H100', 'rdf.dat')
    """

    def __init__(self, config: EPSRConfig, experimental_data_file: str):
        self.config = config
        self.experimental_data_file = experimental_data_file

        # Create distance grid
        self.r_grid = np.linspace(config.r_min, config.r_max, config.n_grid)

        # Load experimental data
        self.r_exp, self.g_exp = load_experimental_data(
            experimental_data_file,
            r_min=config.r_min,
            r_max=config.r_max
        )

        # Initialize empirical potentials for each pair
        self.potentials = {}
        species = sorted(config.composition.keys())
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                if j < i:
                    continue
                pair_name = f"{si}{sj}"
                self.potentials[pair_name] = EmpiricalPotential(
                    r_grid=self.r_grid,
                    temperature=config.temperature,
                    rho=config.density,
                    max_amplitude=config.max_amplitude,
                    smooth_sigma=config.smooth_sigma
                )

        # Calculate scattering weights
        self.weights = calculate_scattering_weights(
            config.composition,
            config.scattering_lengths
        )

        # Initialize LAMMPS interface
        self.lammps = LAMMPSInterface(
            use_gpu=config.use_gpu,
            gpu_id=config.gpu_id
        )

        # Convergence tracking
        self.chi2_history = []
        self.r_factor_history = []
        self.current_iteration = 0

        if config.verbose:
            self._print_header()

    def _print_header(self):
        """Print initialization information."""
        print("=" * 70)
        print("EPSR Engine - Empirical Potential Structure Refinement")
        print("=" * 70)
        print(f"Temperature: {self.config.temperature} K "
              f"({self.config.temperature - 273.15:.1f}°C)")
        print(f"Density: {self.config.density:.6f} atoms/Å³")
        print(f"Composition: {self.config.composition}")
        print(f"Scattering lengths: {self.config.scattering_lengths} fm")
        print(f"\nWeighting factors:")
        for pair, weight in self.weights.items():
            print(f"  {pair}: {weight:.4f}")
        print(f"\nPotential grid: {self.config.r_min} - {self.config.r_max} Å "
              f"({self.config.n_grid} points)")
        print(f"Update method: {self.config.method.upper()}")
        if self.config.use_momentum:
            print(f"Momentum β: {self.config.momentum_beta}")
        print(f"Learning rate α: {self.config.learning_rate}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Convergence tolerance: χ² < {self.config.convergence_tol}")
        print(f"\nExperimental data: {self.experimental_data_file}")
        print(f"  {len(self.r_exp)} data points, r ∈ [{self.r_exp.min():.2f}, "
              f"{self.r_exp.max():.2f}] Å")
        print("=" * 70)

    def run(
        self,
        lammps_input: str,
        rdf_output: str,
        save_interval: int = 1
    ) -> Dict:
        """
        Run the EPSR refinement loop.

        Parameters
        ----------
        lammps_input : str
            Path to LAMMPS input script
        rdf_output : str
            Path to RDF output file from LAMMPS
        save_interval : int, optional
            Save plots every N iterations (default: 1)

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'converged': bool
            - 'iterations': int
            - 'final_chi2': float
            - 'final_r_factor': float
            - 'potentials': dict of final potentials
            - 'chi2_history': list
            - 'r_factor_history': list
        """
        for iteration in range(1, self.config.max_iterations + 1):
            self.current_iteration = iteration

            if self.config.verbose:
                print(f"\n{'=' * 70}")
                print(f"ITERATION {iteration}/{self.config.max_iterations}")
                print("=" * 70)

            # Step 1: Write current potentials to LAMMPS tables
            self._write_potential_tables()

            # Step 2: Run LAMMPS simulation
            try:
                log_file = os.path.join(
                    self.config.output_dir,
                    f'lammps_iter{iteration:03d}.log'
                )
                self.lammps.run(lammps_input, log_file, verbose=self.config.verbose)
            except RuntimeError as e:
                print(f"LAMMPS failed: {e}")
                print("Stopping EPSR loop.")
                break

            # Step 3: Read RDF from LAMMPS
            try:
                r_sim, g_sim, g_partial = self.lammps.read_rdf(rdf_output)
            except (FileNotFoundError, ValueError) as e:
                print(f"Failed to read RDF: {e}")
                break

            # Step 4: Calculate goodness of fit
            g_sim_interp = interpolate_to_grid(r_sim, g_sim, self.r_exp)
            chi2 = calculate_chi_squared(g_sim_interp, self.g_exp, self.config.sigma_exp)
            r_factor = calculate_r_factor(g_sim_interp, self.g_exp)

            self.chi2_history.append(chi2)
            self.r_factor_history.append(r_factor)

            if self.config.verbose:
                print(f"\nχ² = {chi2:.6f}")
                print(f"R-factor = {r_factor:.6f}")

            # Step 5: Check convergence
            if chi2 < self.config.convergence_tol:
                if self.config.verbose:
                    print(f"\n✓ Converged! χ² = {chi2:.6f} < {self.config.convergence_tol}")
                break

            # Check for stagnation
            if self._check_stagnation():
                break

            # Step 6: Update empirical potentials
            self._update_potentials(r_sim, g_sim, g_partial)

            # Step 7: Save plots
            if iteration % save_interval == 0:
                self._save_plots(r_sim, g_sim, iteration)

        # Final results
        return self._compile_results()

    def _write_potential_tables(self):
        """Write current empirical potentials to LAMMPS table files."""
        potentials_dict = {
            name: ep.get_potential()
            for name, ep in self.potentials.items()
        }

        write_potential_tables(
            output_dir=self.config.data_dir,
            r_grid=self.r_grid,
            potentials=potentials_dict,
            r_min=self.config.r_min,
            r_max=self.config.r_max,
            n_points=self.config.n_grid
        )

    def _update_potentials(
        self,
        r_sim: np.ndarray,
        g_sim: np.ndarray,
        g_partial: Optional[Dict[str, np.ndarray]]
    ):
        """
        Update empirical potentials based on simulation results.

        Uses either S(Q) Fourier method or direct g(r) method.
        """
        if self.config.use_sq_update:
            self._update_potentials_sq(r_sim, g_sim, g_partial)
        else:
            self._update_potentials_gr(r_sim, g_sim, g_partial)

    def _update_potentials_sq(
        self,
        r_sim: np.ndarray,
        g_sim: np.ndarray,
        g_partial: Optional[Dict[str, np.ndarray]]
    ):
        """
        Update empirical potentials using S(Q) Fourier method.
        """
        if self.config.verbose:
            print("\nUpdating empirical potentials (S(Q) method)...")

        # Define Q-space grid for Fourier transforms
        Q = np.linspace(0.5, 20.0, 300)  # Å⁻¹

        # Create StructureFactor calculator
        sf = StructureFactor(rho=self.config.density)

        # Convert experimental g(r) to S(Q)
        S_exp = sf.g_to_S(self.r_exp, self.g_exp, Q)

        if self.config.verbose:
            print(f"  Converted g_exp(r) → S_exp(Q) ({len(Q)} Q points)")

        # Calculate structure factors from simulation
        if g_partial and len(g_partial) >= 3:
            # Use partial g(r) to calculate weighted S(Q)
            if self.config.verbose:
                print("  Using partial g(r) for S(Q) calculation")

            # Convert each partial g(r) to S(Q)
            S_partial = {}
            for pair_name in ['GaGa', 'InIn', 'GaIn']:
                if pair_name in g_partial:
                    S_partial[pair_name] = sf.g_to_S(r_sim, g_partial[pair_name], Q)

            # Calculate weighted total S(Q)
            # S_total = Σ w_ij * S_ij(Q)
            S_sim = np.zeros_like(Q)
            weights_sum = 0.0
            for pair_name, weight in self.weights.items():
                if pair_name in S_partial:
                    S_sim += weight * S_partial[pair_name]
                    weights_sum += weight

            # Normalize
            if weights_sum > 0:
                S_sim /= weights_sum

            if self.config.verbose:
                print(f"  Calculated weighted S_total(Q)")
        else:
            # Use total g(r) if partial not available
            if self.config.verbose:
                print("  Warning: Using total g(r) for S(Q) (partial RDF not available)")
            S_sim = sf.g_to_S(r_sim, g_sim, Q)

        # Update each pair potential using S(Q) difference
        for pair_name, ep in self.potentials.items():
            # Apply weighting
            alpha_weighted = self.config.learning_rate * self.weights[pair_name]

            # Update potential in Q-space
            ep.update_from_structure_factors(
                Q=Q,
                S_sim=S_sim,
                S_exp=S_exp,
                alpha=alpha_weighted,
                use_momentum=self.config.use_momentum,
                beta=self.config.momentum_beta
            )

            U = ep.get_potential()
            if self.config.verbose:
                print(f"  {pair_name}: U ∈ [{U.min():.3f}, {U.max():.3f}] kcal/mol")

    def _update_potentials_gr(
        self,
        r_sim: np.ndarray,
        g_sim: np.ndarray,
        g_partial: Optional[Dict[str, np.ndarray]]
    ):
        """
        Update empirical potentials using direct g(r) method (more stable).
        """
        if self.config.verbose:
            print("\nUpdating empirical potentials (direct g(r) method)...")

        # Interpolate simulation g(r) onto potential grid
        g_sim_interp = interpolate_to_grid(r_sim, g_sim, self.r_grid)
        g_exp_interp = interpolate_to_grid(self.r_exp, self.g_exp, self.r_grid)

        # Update each pair potential using g(r) difference
        for pair_name, ep in self.potentials.items():
            # Apply weighting
            alpha_weighted = self.config.learning_rate * self.weights[pair_name]

            # Update potential directly in r-space
            ep.update_from_rdf(
                g_sim=g_sim_interp,
                g_exp=g_exp_interp,
                alpha=alpha_weighted,
                use_momentum=self.config.use_momentum,
                beta=self.config.momentum_beta
            )

            U = ep.get_potential()
            if self.config.verbose:
                print(f"  {pair_name}: U ∈ [{U.min():.3f}, {U.max():.3f}] kcal/mol")

    def _check_stagnation(self) -> bool:
        """Check if EPSR has stagnated."""
        if len(self.chi2_history) < 5:
            return False

        recent_chi2 = self.chi2_history[-5:]

        # Check for divergence
        if all(recent_chi2[i+1] > recent_chi2[i] for i in range(4)):
            if self.config.verbose:
                print("\n✗ Stopping: χ² is consistently increasing (divergence)")
            return True

        # Check for stagnation at high χ²
        if max(recent_chi2) - min(recent_chi2) < 1.0:
            avg_chi2 = np.mean(recent_chi2)
            if avg_chi2 > self.config.convergence_tol:
                if self.config.verbose:
                    print(f"\n✗ Stopping: χ² stagnated at {avg_chi2:.2f} "
                          f"(target: < {self.config.convergence_tol})")
                return True

        return False

    def _save_plots(self, r_sim: np.ndarray, g_sim: np.ndarray, iteration: int):
        """Save summary plots and data tables."""
        potentials_dict = {
            name: ep.get_potential()
            for name, ep in self.potentials.items()
        }

        # 1. Save Summary Plot
        output_file = os.path.join(
            self.config.output_dir,
            f'epsr_iter{iteration:03d}.png'
        )
        plot_epsr_summary(
            r_exp=self.r_exp,
            g_exp=self.g_exp,
            r_sim=r_sim,
            g_sim=g_sim,
            r_grid=self.r_grid,
            potentials=potentials_dict,
            chi2_history=self.chi2_history,
            iteration=iteration,
            output_file=output_file
        )
        
        # 2. Save G(r) Plot explicitly
        gr_plot_file = os.path.join(
            self.config.output_dir,
            f'gr_iter{iteration:03d}.png'
        )
        plot_rdf_comparison(
            r_exp=self.r_exp,
            g_exp=self.g_exp,
            r_sim=r_sim,
            g_sim=g_sim,
            output_file=gr_plot_file,
            title=f"Pair Distribution Function (Iter {iteration})"
        )

        # 3. Calculate and Save S(Q)
        # Define Q-space grid
        Q = np.linspace(0.5, 20.0, 300)
        sf = StructureFactor(rho=self.config.density)
        
        # Experimental S(Q)
        S_exp = sf.g_to_S(self.r_exp, self.g_exp, Q)
        
        # Simulated S(Q)
        S_sim = sf.g_to_S(r_sim, g_sim, Q)
        
        # Save S(Q) data to file
        sq_file = os.path.join(
            self.config.output_dir,
            f'sq_iter{iteration:03d}.dat'
        )
        np.savetxt(sq_file, np.column_stack((Q, S_sim, S_exp)), 
                   header="Q(A^-1) S_sim(Q) S_exp(Q)")
        
        # Plot S(Q)
        sq_plot_file = os.path.join(
            self.config.output_dir,
            f'sq_iter{iteration:03d}.png'
        )
        plot_structure_factor(
            Q_exp=Q,
            S_exp=S_exp,
            Q_sim=Q,
            S_sim=S_sim,
            output_file=sq_plot_file,
            title=f"Structure Factor (Iter {iteration})"
        )

        # 4. Save Density Table
        log_file = os.path.join(
            self.config.output_dir,
            f'lammps_iter{iteration:03d}.log'
        )
        thermo_data = self.lammps.read_log_thermo(log_file)
        
        if 'Density' in thermo_data and 'Step' in thermo_data:
            density_file = os.path.join(
                self.config.output_dir,
                f'density_iter{iteration:03d}.dat'
            )
            np.savetxt(density_file, 
                       np.column_stack((thermo_data['Step'], thermo_data['Density'])),
                       header="Step Density(g/cm^3)")

    def _compile_results(self) -> Dict:
        """Compile final results."""
        potentials_dict = {
            name: ep.get_potential()
            for name, ep in self.potentials.items()
        }

        converged = (len(self.chi2_history) > 0 and
                     self.chi2_history[-1] < self.config.convergence_tol)

        results = {
            'converged': converged,
            'iterations': len(self.chi2_history),
            'final_chi2': self.chi2_history[-1] if self.chi2_history else np.inf,
            'final_r_factor': self.r_factor_history[-1] if self.r_factor_history else np.inf,
            'potentials': potentials_dict,
            'chi2_history': self.chi2_history,
            'r_factor_history': self.r_factor_history,
            'r_grid': self.r_grid
        }

        # Save final results
        output_file = os.path.join(self.config.output_dir, 'final_ep.npz')
        np.savez(
            output_file,
            r=self.r_grid,
            chi2_history=self.chi2_history,
            r_factor_history=self.r_factor_history,
            **potentials_dict
        )

        if self.config.verbose:
            print(f"\n{'=' * 70}")
            print("EPSR COMPLETED")
            print("=" * 70)
            print(f"Converged: {'YES' if converged else 'NO'}")
            print(f"Iterations: {results['iterations']}")
            print(f"Final χ²: {results['final_chi2']:.6f}")
            print(f"Final R-factor: {results['final_r_factor']:.6f}")
            print(f"Results saved to: {output_file}")
            print("=" * 70)

        return results
