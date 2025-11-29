"""
Plotting functions for EPSR results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import os


def plot_rdf_comparison(
    r_exp: np.ndarray,
    g_exp: np.ndarray,
    r_sim: np.ndarray,
    g_sim: np.ndarray,
    output_file: Optional[str] = None,
    title: str = "Radial Distribution Function",
    show_residuals: bool = True
) -> plt.Figure:
    """
    Plot comparison of experimental and simulated g(r).

    Parameters
    ----------
    r_exp : np.ndarray
        Experimental distance array
    g_exp : np.ndarray
        Experimental g(r)
    r_sim : np.ndarray
        Simulated distance array
    g_sim : np.ndarray
        Simulated g(r)
    output_file : str, optional
        Save figure to this file
    title : str, optional
        Plot title
    show_residuals : bool, optional
        Show residuals panel (default: True)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Main plot
    ax1.plot(r_exp, g_exp, 'k-', label='Experiment', linewidth=2)
    ax1.plot(r_sim, g_sim, 'r--', label='Simulation', linewidth=2)
    ax1.set_xlabel('r (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_title(title, fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Residuals
    if show_residuals:
        g_sim_interp = np.interp(r_exp, r_sim, g_sim)
        residuals = g_sim_interp - g_exp

        ax2.plot(r_exp, residuals, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('r (Å)', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    return fig


def plot_structure_factor(
    Q_exp: np.ndarray,
    S_exp: np.ndarray,
    Q_sim: Optional[np.ndarray] = None,
    S_sim: Optional[np.ndarray] = None,
    output_file: Optional[str] = None,
    title: str = "Structure Factor"
) -> plt.Figure:
    """
    Plot structure factor S(Q).

    Parameters
    ----------
    Q_exp : np.ndarray
        Experimental Q array
    S_exp : np.ndarray
        Experimental S(Q)
    Q_sim : np.ndarray, optional
        Simulated Q array
    S_sim : np.ndarray, optional
        Simulated S(Q)
    output_file : str, optional
        Save figure to this file
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(Q_exp, S_exp, 'k-', label='Experiment', linewidth=2)

    if Q_sim is not None and S_sim is not None:
        ax.plot(Q_sim, S_sim, 'r--', label='Simulation', linewidth=2)
        ax.legend(fontsize=11)

    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('S(Q)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    return fig


def plot_potentials(
    r: np.ndarray,
    potentials: Dict[str, np.ndarray],
    output_file: Optional[str] = None,
    title: str = "Empirical Potentials"
) -> plt.Figure:
    """
    Plot empirical potentials.

    Parameters
    ----------
    r : np.ndarray
        Distance array
    potentials : dict
        Dictionary of potentials, e.g., {'GaGa': U_GaGa, 'InIn': U_InIn, ...}
    output_file : str, optional
        Save figure to this file
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'GaGa': 'blue', 'InIn': 'red', 'GaIn': 'green'}

    for name, U in potentials.items():
        color = colors.get(name, 'black')
        ax.plot(r, U, label=name, linewidth=2, color=color)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('r (Å)', fontsize=12)
    ax.set_ylabel('U_EP (kcal/mol)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    return fig


def plot_convergence(
    chi2_history: list,
    output_file: Optional[str] = None,
    title: str = "EPSR Convergence"
) -> plt.Figure:
    """
    Plot convergence history.

    Parameters
    ----------
    chi2_history : list
        Chi-squared values for each iteration
    output_file : str, optional
        Save figure to this file
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(chi2_history) + 1)
    ax.plot(iterations, chi2_history, 'o-', linewidth=2, markersize=6)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('χ²', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    return fig


def plot_epsr_summary(
    r_exp: np.ndarray,
    g_exp: np.ndarray,
    r_sim: np.ndarray,
    g_sim: np.ndarray,
    r_grid: np.ndarray,
    potentials: Dict[str, np.ndarray],
    chi2_history: list,
    iteration: int,
    output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive summary plot with g(r), potentials, and convergence.

    Parameters
    ----------
    r_exp, g_exp : np.ndarray
        Experimental g(r)
    r_sim, g_sim : np.ndarray
        Simulated g(r)
    r_grid : np.ndarray
        Distance grid for potentials
    potentials : dict
        Empirical potentials
    chi2_history : list
        Convergence history
    iteration : int
        Current iteration number
    output_file : str, optional
        Save figure to this file

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(16, 5))

    # Panel 1: g(r) comparison
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(r_exp, g_exp, 'k-', label='Experiment', linewidth=2)
    ax1.plot(r_sim, g_sim, 'r--', label='Simulation', linewidth=2)
    ax1.set_xlabel('r (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_title(f'Pair Distribution Function (Iter {iteration})', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Empirical potentials
    ax2 = plt.subplot(1, 3, 2)
    colors = {'GaGa': 'blue', 'InIn': 'red', 'GaIn': 'green'}
    for name, U in potentials.items():
        color = colors.get(name, 'black')
        ax2.plot(r_grid, U, label=name, linewidth=2, color=color)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('r (Å)', fontsize=12)
    ax2.set_ylabel('U_EP (kcal/mol)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_title('Empirical Potentials', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Convergence
    ax3 = plt.subplot(1, 3, 3)
    if len(chi2_history) > 0:
        iterations = range(1, len(chi2_history) + 1)
        ax3.plot(iterations, chi2_history, 'o-', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('χ²', fontsize=12)
        ax3.set_title('Convergence', fontsize=12)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    return fig
