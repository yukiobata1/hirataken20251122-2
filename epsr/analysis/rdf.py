"""
Radial Distribution Function (RDF) analysis tools.
"""

import numpy as np
from typing import Tuple


class RadialDistributionFunction:
    """
    Radial Distribution Function handler.

    Provides utilities for analyzing and manipulating g(r) data.

    Parameters
    ----------
    r : np.ndarray
        Distance array (Å)
    g : np.ndarray
        RDF values
    rho : float, optional
        Number density (atoms/Å³)
    """

    def __init__(self, r: np.ndarray, g: np.ndarray, rho: float = None):
        self.r = r
        self.g = g
        self.rho = rho

    def coordination_number(
        self,
        r_max: float,
        r_min: float = 0.0
    ) -> float:
        """
        Calculate coordination number in a shell [r_min, r_max].

        N = 4πρ ∫[r_min to r_max] g(r) r² dr

        Parameters
        ----------
        r_max : float
            Maximum distance (Å)
        r_min : float, optional
            Minimum distance (Å), default: 0.0

        Returns
        -------
        N : float
            Coordination number

        Raises
        ------
        ValueError
            If density is not set
        """
        if self.rho is None:
            raise ValueError("Density must be set to calculate coordination number")

        mask = (self.r >= r_min) & (self.r <= r_max)
        r_shell = self.r[mask]
        g_shell = self.g[mask]

        integrand = g_shell * r_shell**2
        integral = np.trapz(integrand, r_shell)

        return 4.0 * np.pi * self.rho * integral

    def first_peak_position(self, r_min: float = 2.0, r_max: float = 4.0) -> float:
        """
        Find position of first peak in g(r).

        Parameters
        ----------
        r_min : float, optional
            Minimum r to search (default: 2.0 Å)
        r_max : float, optional
            Maximum r to search (default: 4.0 Å)

        Returns
        -------
        r_peak : float
            Position of first peak (Å)
        """
        mask = (self.r >= r_min) & (self.r <= r_max)
        r_region = self.r[mask]
        g_region = self.g[mask]

        if len(g_region) == 0:
            raise ValueError(f"No data in range [{r_min}, {r_max}]")

        peak_idx = np.argmax(g_region)
        return r_region[peak_idx]

    def first_minimum(self, r_min: float = 2.0, r_max: float = 5.0) -> float:
        """
        Find position of first minimum after first peak.

        Parameters
        ----------
        r_min : float, optional
            Minimum r to search (default: 2.0 Å)
        r_max : float, optional
            Maximum r to search (default: 5.0 Å)

        Returns
        -------
        r_min : float
            Position of first minimum (Å)
        """
        # First find the peak
        r_peak = self.first_peak_position(r_min, r_max)

        # Search for minimum after peak
        mask = (self.r > r_peak) & (self.r <= r_max)
        r_region = self.r[mask]
        g_region = self.g[mask]

        if len(g_region) == 0:
            raise ValueError(f"No data after peak at r={r_peak:.2f}")

        min_idx = np.argmin(g_region)
        return r_region[min_idx]

    def interpolate(self, r_new: np.ndarray) -> np.ndarray:
        """
        Interpolate g(r) onto a new grid.

        Parameters
        ----------
        r_new : np.ndarray
            New distance grid (Å)

        Returns
        -------
        g_new : np.ndarray
            Interpolated g(r)
        """
        return np.interp(r_new, self.r, self.g)


def calculate_coordination_number(
    r: np.ndarray,
    g: np.ndarray,
    rho: float,
    r_min: float = 0.0,
    r_max: float = 3.5
) -> float:
    """
    Calculate coordination number from g(r).

    N = 4πρ ∫[r_min to r_max] g(r) r² dr

    Parameters
    ----------
    r : np.ndarray
        Distance array (Å)
    g : np.ndarray
        RDF
    rho : float
        Number density (atoms/Å³)
    r_min : float, optional
        Minimum distance (default: 0.0 Å)
    r_max : float, optional
        Maximum distance (default: 3.5 Å, typical first shell)

    Returns
    -------
    N : float
        Coordination number
    """
    rdf = RadialDistributionFunction(r, g, rho)
    return rdf.coordination_number(r_max, r_min)
