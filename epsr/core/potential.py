"""
Empirical Potential (EP) calculation and refinement.

Implements the core EPSR algorithm for updating empirical potentials
based on differences between simulated and experimental structure factors.

The key innovation of EPSR is that the empirical potential correction
is calculated in Q-space (reciprocal space) and then Fourier transformed
to real space.

References
----------
Soper, A. K. (1996). Chem. Phys., 202, 295-306.
Soper, A. K. (2005). Phys. Rev. B, 72, 104204.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter1d


class EmpiricalPotential:
    """
    Empirical Potential handler for EPSR.

    Manages the calculation and update of empirical correction potentials
    using the proper EPSR algorithm based on structure factor differences.

    Parameters
    ----------
    r_grid : np.ndarray
        Distance grid for potential (Å)
    temperature : float
        System temperature (K)
    rho : float
        Number density (atoms/Å³)
    max_amplitude : float, optional
        Maximum allowed potential amplitude (kcal/mol), default: 3.0
    smooth_sigma : float, optional
        Gaussian smoothing width in real space (Å), default: 0.3
    """

    def __init__(
        self,
        r_grid: np.ndarray,
        temperature: float,
        rho: float,
        max_amplitude: float = 3.0,
        smooth_sigma: float = 0.3
    ):
        self.r_grid = r_grid
        self.T = temperature
        self.rho = rho
        self.max_amplitude = max_amplitude
        self.smooth_sigma = smooth_sigma

        # Boltzmann constant in kcal/(mol·K)
        self.kB = 0.001987
        self.kT = self.kB * temperature

        # Initialize potential to zero
        self.U_ep = np.zeros_like(r_grid)

        # Momentum for optimization (if using momentum-based updates)
        self.velocity = np.zeros_like(r_grid)

    def update_from_structure_factors(
        self,
        Q: np.ndarray,
        S_sim: np.ndarray,
        S_exp: np.ndarray,
        alpha: float = 0.1,
        use_momentum: bool = True,
        beta: float = 0.9
    ) -> np.ndarray:
        """
        Update empirical potential using structure factor difference.

        This is the proper EPSR method:
        1. Calculate ΔS(Q) = S_sim(Q) - S_exp(Q) in reciprocal space
        2. Calculate potential correction ΔU(Q) ∝ ΔS(Q)
        3. Fourier transform ΔU(Q) → ΔU(r)
        4. Update U_EP(r) ← U_EP(r) + α * ΔU(r)

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector array (Å⁻¹)
        S_sim : np.ndarray
            Simulated structure factor
        S_exp : np.ndarray
            Experimental structure factor
        alpha : float, optional
            Learning rate / feedback parameter (default: 0.1)
        use_momentum : bool, optional
            Use momentum-based optimization (default: True)
        beta : float, optional
            Momentum coefficient (default: 0.9)

        Returns
        -------
        U_ep_new : np.ndarray
            Updated empirical potential
        """
        # Calculate structure factor difference
        delta_S = S_sim - S_exp

        # In EPSR, the potential correction in Q-space is:
        # ΔU(Q) = -kT * ΔS(Q) / S(Q)
        # For small corrections, S(Q) ≈ 1, so:
        # ΔU(Q) ≈ -kT * ΔS(Q)
        #
        # However, we need to be careful with the sign:
        # - If S_sim > S_exp: too much structure → add repulsive potential
        # - If S_sim < S_exp: too little structure → add attractive potential

        # Calculate F(Q) = Q * ΔS(Q) for Fourier transform
        F_Q = Q * delta_S

        # Inverse Fourier transform to get ΔU(r)
        # ΔU(r) = (kT / 2π²ρr) ∫ F(Q) * sin(Qr) dQ
        delta_U = np.zeros_like(self.r_grid)

        for i, r in enumerate(self.r_grid):
            if r < 1e-6:
                delta_U[i] = 0.0
            else:
                integrand = F_Q * np.sin(Q * r)
                integral = np.trapz(integrand, Q)
                delta_U[i] = self.kT * integral / (2.0 * np.pi**2 * self.rho * r)

        # Apply learning rate and sign correction
        # Positive delta_S → positive delta_U → repulsive correction
        gradient = alpha * delta_U

        # Apply momentum if requested
        if use_momentum:
            self.velocity = beta * self.velocity + gradient
            self.U_ep = self.U_ep + self.velocity
        else:
            self.U_ep = self.U_ep + gradient

        # Apply amplitude clipping
        self.U_ep = np.clip(self.U_ep, -self.max_amplitude, self.max_amplitude)

        # Apply smoothing in real space
        if self.smooth_sigma > 0:
            dr = self.r_grid[1] - self.r_grid[0]
            sigma_points = self.smooth_sigma / dr
            self.U_ep = gaussian_filter1d(self.U_ep, sigma=sigma_points)
            if use_momentum:
                self.velocity = gaussian_filter1d(self.velocity, sigma=sigma_points)

        return self.U_ep

    def update_from_rdf(
        self,
        g_sim: np.ndarray,
        g_exp: np.ndarray,
        alpha: float = 0.3,
        use_momentum: bool = True,
        beta: float = 0.9
    ) -> np.ndarray:
        """
        Update empirical potential using g(r) difference (simplified method).

        This is a simplified version that works directly in real space.
        It's less rigorous than the structure factor method but can be
        more stable and faster.

        ΔU(r) ∝ kT * [g_sim(r) - g_exp(r)]

        Parameters
        ----------
        g_sim : np.ndarray
            Simulated g(r) on self.r_grid
        g_exp : np.ndarray
            Experimental g(r) on self.r_grid
        alpha : float, optional
            Learning rate (default: 0.3)
        use_momentum : bool, optional
            Use momentum-based optimization (default: True)
        beta : float, optional
            Momentum coefficient (default: 0.9)

        Returns
        -------
        U_ep_new : np.ndarray
            Updated empirical potential
        """
        # Calculate g(r) difference
        delta_g = g_sim - g_exp

        # Calculate gradient
        # If g_sim > g_exp: too many atoms → add repulsive potential (positive)
        # If g_sim < g_exp: too few atoms → add attractive potential (negative)
        gradient = alpha * self.kT * delta_g

        # Apply momentum if requested
        if use_momentum:
            self.velocity = beta * self.velocity + gradient
            self.U_ep = self.U_ep + self.velocity
        else:
            self.U_ep = self.U_ep + gradient

        # Apply amplitude clipping
        self.U_ep = np.clip(self.U_ep, -self.max_amplitude, self.max_amplitude)

        # Apply smoothing
        if self.smooth_sigma > 0:
            dr = self.r_grid[1] - self.r_grid[0]
            sigma_points = self.smooth_sigma / dr
            self.U_ep = gaussian_filter1d(self.U_ep, sigma=sigma_points)
            if use_momentum:
                self.velocity = gaussian_filter1d(self.velocity, sigma=sigma_points)

        return self.U_ep

    def get_potential(self) -> np.ndarray:
        """Return current empirical potential."""
        return self.U_ep.copy()

    def get_force(self) -> np.ndarray:
        """
        Calculate force from potential: F = -dU/dr

        Returns
        -------
        F : np.ndarray
            Force (kcal/mol/Å)
        """
        return -np.gradient(self.U_ep, self.r_grid)

    def reset(self):
        """Reset potential and velocity to zero."""
        self.U_ep = np.zeros_like(self.r_grid)
        self.velocity = np.zeros_like(self.r_grid)


def update_potential(
    r_grid: np.ndarray,
    g_sim: np.ndarray,
    g_exp: np.ndarray,
    U_ep_old: np.ndarray,
    temperature: float,
    alpha: float = 0.3,
    max_amplitude: float = 3.0,
    smooth_sigma: float = 0.3,
    velocity: Optional[np.ndarray] = None,
    beta: float = 0.9,
    method: str = 'momentum'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to update empirical potential (backward compatible).

    Parameters
    ----------
    r_grid : np.ndarray
        Distance grid (Å)
    g_sim : np.ndarray
        Simulated g(r)
    g_exp : np.ndarray
        Experimental g(r)
    U_ep_old : np.ndarray
        Current empirical potential (kcal/mol)
    temperature : float
        Temperature (K)
    alpha : float, optional
        Learning rate (default: 0.3)
    max_amplitude : float, optional
        Maximum potential amplitude (default: 3.0)
    smooth_sigma : float, optional
        Smoothing width in Å (default: 0.3)
    velocity : np.ndarray, optional
        Momentum velocity from previous iteration
    beta : float, optional
        Momentum coefficient (default: 0.9)
    method : str, optional
        Update method: 'simple', 'momentum', 'nesterov' (default: 'momentum')

    Returns
    -------
    U_ep_new : np.ndarray
        Updated empirical potential
    velocity_new : np.ndarray
        Updated velocity for next iteration
    """
    # Boltzmann constant
    kB = 0.001987  # kcal/(mol·K)
    kT = kB * temperature

    # Calculate gradient
    delta_g = g_sim - g_exp
    gradient = alpha * kT * delta_g

    # Initialize velocity if needed
    if velocity is None:
        velocity = np.zeros_like(gradient)

    # Apply update method
    if method == 'simple':
        U_ep_new = U_ep_old + gradient
        velocity_new = gradient
    elif method == 'momentum':
        velocity_new = beta * velocity + gradient
        U_ep_new = U_ep_old + velocity_new
    elif method == 'nesterov':
        velocity_new = beta * velocity + gradient
        U_ep_new = U_ep_old + beta * velocity_new + gradient
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply clipping
    U_ep_new = np.clip(U_ep_new, -max_amplitude, max_amplitude)

    # Apply smoothing
    if smooth_sigma > 0:
        dr = r_grid[1] - r_grid[0]
        sigma_points = smooth_sigma / dr
        U_ep_new = gaussian_filter1d(U_ep_new, sigma=sigma_points)
        velocity_new = gaussian_filter1d(velocity_new, sigma=sigma_points)

    return U_ep_new, velocity_new


def calculate_scattering_weights(
    composition: dict,
    scattering_lengths: dict
) -> dict:
    """
    Calculate weighting factors for partial structure factors.

    For neutron scattering, the total structure factor is a weighted
    sum of partial structure factors:

    S(Q) = Σᵢⱼ wᵢⱼ * Sᵢⱼ(Q)

    where wᵢⱼ = cᵢcⱼbᵢbⱼ / [Σₖcₖbₖ]²

    Parameters
    ----------
    composition : dict
        Mole fractions, e.g., {'Ga': 0.858, 'In': 0.142}
    scattering_lengths : dict
        Neutron scattering lengths (fm), e.g., {'Ga': 7.288, 'In': 4.061}

    Returns
    -------
    weights : dict
        Weighting factors for each pair, e.g., {'GaGa': 0.7, 'GaIn': 0.25, ...}
    """
    species = sorted(composition.keys())
    total_scattering = sum(composition[s] * scattering_lengths[s] for s in species)

    weights = {}
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            if j < i:
                continue

            ci = composition[si]
            cj = composition[sj]
            bi = scattering_lengths[si]
            bj = scattering_lengths[sj]

            # Factor of 2 for i≠j pairs (cross-terms counted twice)
            factor = 2 if i != j else 1
            pair_name = f"{si}{sj}"
            weights[pair_name] = factor * ci * cj * bi * bj / total_scattering**2

    return weights
