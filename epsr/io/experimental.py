"""
Loading and preprocessing experimental scattering data.
"""

import numpy as np
from typing import Tuple, Optional
import os


def load_experimental_data(
    filename: str,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load experimental g(r) data from a two-column text file.

    File format:
    # Optional header lines starting with #
    r1  g1
    r2  g2
    ...

    Parameters
    ----------
    filename : str
        Path to experimental data file
    r_min : float, optional
        Minimum r value to include (default: None, use all)
    r_max : float, optional
        Maximum r value to include (default: None, use all)

    Returns
    -------
    r : np.ndarray
        Distance array (Å)
    g : np.ndarray
        Pair distribution function

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is invalid
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Experimental data file not found: {filename}")

    try:
        data = np.loadtxt(filename)
    except Exception as e:
        raise ValueError(f"Failed to load data from {filename}: {e}")

    if data.shape[1] < 2:
        raise ValueError(f"Data file must have at least 2 columns (r, g), got {data.shape[1]}")

    r = data[:, 0]
    g = data[:, 1]

    # Apply r range filter if specified
    if r_min is not None or r_max is not None:
        mask = np.ones(len(r), dtype=bool)
        if r_min is not None:
            mask &= (r >= r_min)
        if r_max is not None:
            mask &= (r <= r_max)

        r = r[mask]
        g = g[mask]

    if len(r) == 0:
        raise ValueError(f"No data points in range [{r_min}, {r_max}]")

    return r, g


def load_experimental_sq(
    filename: str,
    Q_min: Optional[float] = None,
    Q_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load experimental S(Q) data from a two-column text file.

    File format:
    # Optional header lines starting with #
    Q1  S1
    Q2  S2
    ...

    Parameters
    ----------
    filename : str
        Path to experimental S(Q) file
    Q_min : float, optional
        Minimum Q value to include (default: None, use all)
    Q_max : float, optional
        Maximum Q value to include (default: None, use all)

    Returns
    -------
    Q : np.ndarray
        Scattering vector array (Å⁻¹)
    S : np.ndarray
        Structure factor

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is invalid
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Experimental S(Q) file not found: {filename}")

    try:
        data = np.loadtxt(filename)
    except Exception as e:
        raise ValueError(f"Failed to load S(Q) from {filename}: {e}")

    if data.shape[1] < 2:
        raise ValueError(f"S(Q) file must have at least 2 columns (Q, S), got {data.shape[1]}")

    Q = data[:, 0]
    S = data[:, 1]

    # Apply Q range filter if specified
    if Q_min is not None or Q_max is not None:
        mask = np.ones(len(Q), dtype=bool)
        if Q_min is not None:
            mask &= (Q >= Q_min)
        if Q_max is not None:
            mask &= (Q <= Q_max)

        Q = Q[mask]
        S = S[mask]

    if len(Q) == 0:
        raise ValueError(f"No data points in Q range [{Q_min}, {Q_max}]")

    return Q, S


def save_data(filename: str, x: np.ndarray, y: np.ndarray, header: str = ""):
    """
    Save two-column data to a text file.

    Parameters
    ----------
    filename : str
        Output filename
    x : np.ndarray
        First column (e.g., r or Q)
    y : np.ndarray
        Second column (e.g., g or S)
    header : str, optional
        Header comment line (without # prefix)
    """
    with open(filename, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:15.8f}  {yi:15.8f}\n")


def interpolate_to_grid(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate data onto a new grid.

    Parameters
    ----------
    x_data : np.ndarray
        Original x values
    y_data : np.ndarray
        Original y values
    x_grid : np.ndarray
        Target grid

    Returns
    -------
    y_grid : np.ndarray
        Interpolated y values on target grid
    """
    return np.interp(x_grid, x_data, y_data)
