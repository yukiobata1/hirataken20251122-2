"""
Structure Factor S(Q) calculation and Fourier transform utilities.

This module implements the proper EPSR algorithm using structure factors
in reciprocal space, as described by Soper (1996, 2005).

Key concepts:
- S(Q) = Structure factor in reciprocal space (Q-space)
- g(r) = Radial distribution function in real space (r-space)
- Relationship: S(Q) - 1 = 4πρ ∫[g(r) - 1] * r² * sin(Qr)/(Qr) dr

References
----------
Soper, A. K. (1996). Chem. Phys., 202, 295-306.
Soper, A. K. (2005). Phys. Rev. B, 72, 104204.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate


class StructureFactor:
    """
    Structure factor calculator with Fourier transform utilities.

    Handles conversion between g(r) and S(Q), and calculation of
    partial structure factors for multi-component systems.

    Parameters
    ----------
    rho : float
        Number density (atoms/Å³)
    r_max : float, optional
        Maximum r for integration (default: 30.0 Å)
    n_points : int, optional
        Number of points for integration (default: 1000)
    """

    def __init__(self, rho: float, r_max: float = 30.0, n_points: int = 1000):
        self.rho = rho
        self.r_max = r_max
        self.n_points = n_points

    def g_to_S(self, r: np.ndarray, g: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Convert g(r) to S(Q) using Fourier transform.

        S(Q) - 1 = 4πρ ∫[g(r) - 1] * r² * sin(Qr)/(Qr) dr

        Parameters
        ----------
        r : np.ndarray
            Distance array (Å)
        g : np.ndarray
            Radial distribution function
        Q : np.ndarray
            Scattering vector array (Å⁻¹)

        Returns
        -------
        S : np.ndarray
            Structure factor
        """
        S = np.ones_like(Q)

        for i, q in enumerate(Q):
            if q < 1e-6:
                # For Q → 0, use limiting behavior
                # lim(Q→0) S(Q) = S(0) = 1 + 4πρ ∫[g(r) - 1] * r² dr
                integrand = (g - 1.0) * r**2
            else:
                # General case
                integrand = (g - 1.0) * r**2 * np.sin(q * r) / (q * r)

            # Numerical integration using trapezoidal rule
            integral = np.trapz(integrand, r)
            S[i] = 1.0 + 4.0 * np.pi * self.rho * integral

        return S

    def S_to_g(self, Q: np.ndarray, S: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Convert S(Q) to g(r) using inverse Fourier transform.

        g(r) - 1 = 1/(2π²ρr) ∫[S(Q) - 1] * Q * sin(Qr) dQ

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector array (Å⁻¹)
        S : np.ndarray
            Structure factor
        r : np.ndarray
            Distance array (Å)

        Returns
        -------
        g : np.ndarray
            Radial distribution function
        """
        g = np.ones_like(r)

        for i, ri in enumerate(r):
            if ri < 1e-6:
                # For r → 0, g(r) → 0 (excluded volume)
                g[i] = 0.0
            else:
                # General case
                integrand = (S - 1.0) * Q * np.sin(Q * ri)
                integral = np.trapz(integrand, Q)
                g[i] = 1.0 + integral / (2.0 * np.pi**2 * self.rho * ri)

        return g

    def calculate_partial_S(
        self,
        r: np.ndarray,
        g_partial: dict,
        Q: np.ndarray,
        composition: dict,
        scattering_lengths: dict
    ) -> Tuple[np.ndarray, dict]:
        """
        Calculate total and partial structure factors for a multi-component system.

        For a binary system (A, B):
        S(Q) = Σᵢⱼ cᵢcⱼbᵢbⱼSᵢⱼ(Q) / [Σᵢcᵢbᵢ]²

        where:
        - cᵢ = concentration of species i
        - bᵢ = neutron scattering length of species i
        - Sᵢⱼ(Q) = partial structure factor for pair ij

        Parameters
        ----------
        r : np.ndarray
            Distance array (Å)
        g_partial : dict
            Partial g(r) functions, keys like 'GaGa', 'GaIn', 'InIn'
        Q : np.ndarray
            Scattering vector array (Å⁻¹)
        composition : dict
            Mole fractions, e.g., {'Ga': 0.858, 'In': 0.142}
        scattering_lengths : dict
            Neutron scattering lengths (fm), e.g., {'Ga': 7.288, 'In': 4.061}

        Returns
        -------
        S_total : np.ndarray
            Total weighted structure factor
        S_partial : dict
            Partial structure factors
        """
        # Calculate partial S(Q) from partial g(r)
        S_partial = {}
        for pair_name, g in g_partial.items():
            S_partial[pair_name] = self.g_to_S(r, g, Q)

        # Calculate weighting factors
        species = list(composition.keys())
        total_scattering = sum(composition[s] * scattering_lengths[s] for s in species)

        weights = {}
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                if j < i:
                    continue
                pair_name = f"{si}{sj}"
                ci = composition[si]
                cj = composition[sj]
                bi = scattering_lengths[si]
                bj = scattering_lengths[sj]

                # Factor of 2 for i≠j pairs (double counting)
                factor = 2 if i != j else 1
                weights[pair_name] = factor * ci * cj * bi * bj / total_scattering**2

        # Calculate total S(Q) as weighted sum
        S_total = np.zeros_like(Q)
        for pair_name, weight in weights.items():
            if pair_name in S_partial:
                S_total += weight * S_partial[pair_name]

        return S_total, S_partial

    def smooth_S(self, Q: np.ndarray, S: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Apply Gaussian smoothing to structure factor.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector array
        S : np.ndarray
            Structure factor
        sigma : float
            Smoothing width (in Q units)

        Returns
        -------
        S_smooth : np.ndarray
            Smoothed structure factor
        """
        from scipy.ndimage import gaussian_filter1d

        # Convert sigma from Q units to array index units
        dQ = Q[1] - Q[0] if len(Q) > 1 else 1.0
        sigma_points = sigma / dQ

        return gaussian_filter1d(S, sigma=sigma_points)


def calculate_structure_factor(
    r: np.ndarray,
    g: np.ndarray,
    Q: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Convenience function to calculate S(Q) from g(r).

    Parameters
    ----------
    r : np.ndarray
        Distance array (Å)
    g : np.ndarray
        Radial distribution function
    Q : np.ndarray
        Scattering vector array (Å⁻¹)
    rho : float
        Number density (atoms/Å³)

    Returns
    -------
    S : np.ndarray
        Structure factor
    """
    sf = StructureFactor(rho=rho)
    return sf.g_to_S(r, g, Q)


def calculate_F_Q(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Calculate F(Q) = Q[S(Q) - 1] for Fourier transform.

    This is the quantity that is directly related to the pair
    distribution function via inverse Fourier transform.

    Parameters
    ----------
    Q : np.ndarray
        Scattering vector array (Å⁻¹)
    S : np.ndarray
        Structure factor

    Returns
    -------
    F : np.ndarray
        F(Q) = Q[S(Q) - 1]
    """
    return Q * (S - 1.0)


def calculate_T_r(r: np.ndarray, g: np.ndarray, rho: float) -> np.ndarray:
    """
    Calculate T(r) = 4πρr[g(r) - 1] for Fourier transform.

    This is the quantity that is directly related to F(Q).

    Parameters
    ----------
    r : np.ndarray
        Distance array (Å)
    g : np.ndarray
        Radial distribution function
    rho : float
        Number density (atoms/Å³)

    Returns
    -------
    T : np.ndarray
        T(r) = 4πρr[g(r) - 1]
    """
    return 4.0 * np.pi * rho * r * (g - 1.0)
