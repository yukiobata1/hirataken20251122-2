"""
Goodness-of-fit metrics for comparing simulated and experimental data.
"""

import numpy as np
from typing import Union


def calculate_chi_squared(
    y_sim: np.ndarray,
    y_exp: np.ndarray,
    sigma: Union[float, np.ndarray] = 0.05
) -> float:
    """
    Calculate chi-squared (χ²) statistic.

    χ² = Σ[(y_sim - y_exp)² / σ²]

    For a good fit, χ² ≈ N (number of data points).

    Parameters
    ----------
    y_sim : np.ndarray
        Simulated data
    y_exp : np.ndarray
        Experimental data
    sigma : float or np.ndarray, optional
        Measurement uncertainty (default: 0.05)
        For neutron diffraction g(r): typically 0.03-0.10

    Returns
    -------
    chi2 : float
        Chi-squared value
    """
    residuals = y_sim - y_exp
    chi2 = np.sum(residuals**2 / sigma**2)
    return chi2


def calculate_r_factor(
    y_sim: np.ndarray,
    y_exp: np.ndarray
) -> float:
    """
    Calculate R-factor (crystallographic metric).

    R = Σ|y_sim - y_exp| / Σ|y_exp|

    Typical values:
    - R < 0.05: Excellent agreement
    - R < 0.10: Good agreement
    - R < 0.20: Fair agreement
    - R > 0.20: Poor agreement

    Parameters
    ----------
    y_sim : np.ndarray
        Simulated data
    y_exp : np.ndarray
        Experimental data

    Returns
    -------
    R : float
        R-factor
    """
    numerator = np.sum(np.abs(y_sim - y_exp))
    denominator = np.sum(np.abs(y_exp))

    if denominator == 0:
        return np.inf

    return numerator / denominator


def calculate_residuals(
    y_sim: np.ndarray,
    y_exp: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Calculate residuals between simulated and experimental data.

    Parameters
    ----------
    y_sim : np.ndarray
        Simulated data
    y_exp : np.ndarray
        Experimental data
    normalize : bool, optional
        Normalize by experimental values (default: False)

    Returns
    -------
    residuals : np.ndarray
        Residuals (y_sim - y_exp) or (y_sim - y_exp) / y_exp
    """
    residuals = y_sim - y_exp

    if normalize:
        # Avoid division by zero
        mask = np.abs(y_exp) > 1e-10
        residuals[mask] /= y_exp[mask]
        residuals[~mask] = 0.0

    return residuals


def calculate_weighted_chi_squared(
    y_sim: np.ndarray,
    y_exp: np.ndarray,
    weights: np.ndarray,
    sigma: Union[float, np.ndarray] = 0.05
) -> float:
    """
    Calculate weighted chi-squared.

    χ² = Σ[w_i * (y_sim - y_exp)² / σ²]

    Parameters
    ----------
    y_sim : np.ndarray
        Simulated data
    y_exp : np.ndarray
        Experimental data
    weights : np.ndarray
        Weight for each data point
    sigma : float or np.ndarray, optional
        Measurement uncertainty (default: 0.05)

    Returns
    -------
    chi2 : float
        Weighted chi-squared value
    """
    residuals = y_sim - y_exp
    chi2 = np.sum(weights * residuals**2 / sigma**2)
    return chi2


def calculate_rmsd(
    y_sim: np.ndarray,
    y_exp: np.ndarray
) -> float:
    """
    Calculate Root Mean Square Deviation (RMSD).

    RMSD = sqrt(Σ(y_sim - y_exp)² / N)

    Parameters
    ----------
    y_sim : np.ndarray
        Simulated data
    y_exp : np.ndarray
        Experimental data

    Returns
    -------
    rmsd : float
        Root mean square deviation
    """
    residuals = y_sim - y_exp
    return np.sqrt(np.mean(residuals**2))
