#!/usr/bin/env python3
"""
Test and compare different EPSR update algorithms

This script creates a synthetic test case to demonstrate the behavior of:
1. Simple gradient descent (original)
2. Momentum method
3. Nesterov accelerated gradient
4. Adaptive learning rate

The test uses a simple 1D optimization problem that mimics the EPSR g(r) fitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from update_ep import update_ep
import os


def create_synthetic_problem():
    """
    Create a synthetic g(r) fitting problem

    Returns
    -------
    r : array
        Distance grid
    g_target : array
        Target g(r) (like experimental data)
    g_initial : array
        Initial g(r) (like simulation with poor potential)
    """
    # Distance grid
    r = np.linspace(2.0, 12.0, 200)

    # Target g(r): oscillating function similar to real RDF
    g_target = 1.0 + 0.8 * np.exp(-((r - 2.8) / 0.3)**2) + \
                     0.5 * np.exp(-((r - 5.5) / 0.5)**2) + \
                     0.3 * np.exp(-((r - 8.2) / 0.6)**2)

    # Initial g(r): poor starting point (offset and different peaks)
    g_initial = 1.0 + 0.3 * np.exp(-((r - 3.2) / 0.4)**2) + \
                      0.2 * np.exp(-((r - 6.0) / 0.6)**2) + \
                      0.1 * np.exp(-((r - 9.0) / 0.8)**2)

    return r, g_target, g_initial


def run_optimization(method, alpha, beta=0.9, adaptive_lr=False,
                     alpha_min=0.05, alpha_max=0.5, max_iter=50):
    """
    Run optimization with specified method

    Parameters
    ----------
    method : str
        'simple', 'momentum', or 'nesterov'
    alpha : float
        Initial learning rate
    beta : float
        Momentum coefficient
    adaptive_lr : bool
        Enable adaptive learning rate
    alpha_min, alpha_max : float
        Learning rate bounds for adaptive method
    max_iter : int
        Maximum iterations

    Returns
    -------
    chi2_history : list
        χ² values at each iteration
    alpha_history : list
        Learning rate at each iteration
    U_ep_final : array
        Final empirical potential
    """
    r, g_target, g_initial = create_synthetic_problem()

    # Initialize
    U_ep = np.zeros_like(r)
    velocity = None
    chi2_history = []
    alpha_history = []

    # Simulate g(r) as a function of U_ep
    # In real EPSR, we run LAMMPS; here we use a simple model:
    # g_sim = g_initial + f(U_ep), where f is some response function

    T = 423.15  # Temperature (K)
    kB = 0.001987  # Boltzmann constant (kcal/mol/K)

    for iteration in range(max_iter):
        # Simulate g(r) response to current U_ep
        # Use a more realistic model that mimics LAMMPS response:
        # g(r) is affected by the potential through Boltzmann factor
        # g_sim(r) ∝ g_initial(r) * exp(-U_ep(r) / kT)
        # This creates a nonlinear but physically motivated relationship

        # Normalize U_ep effect to be small perturbation
        U_normalized = U_ep / (kB * T)
        # Apply response: g changes based on potential
        # Use first-order approximation: exp(-U/kT) ≈ 1 - U/kT for small U
        g_sim = g_initial * (1.0 - 0.5 * U_normalized)  # 0.5 is coupling strength

        # Add noise to make it more realistic (like simulation noise)
        if iteration > 0:
            noise = np.random.normal(0, 0.01, size=g_sim.shape)
            g_sim = g_sim + noise

        # Calculate chi-squared
        chi2 = np.sum((g_sim - g_target)**2)
        chi2_history.append(chi2)
        alpha_history.append(alpha)

        # Adaptive learning rate
        if adaptive_lr and len(chi2_history) > 1:
            delta_chi2 = chi2_history[-1] - chi2_history[-2]
            if delta_chi2 > 0:
                # χ² increased - reduce learning rate
                alpha = max(alpha * 0.8, alpha_min)
            elif delta_chi2 < -0.1:
                # χ² decreased significantly
                alpha = min(alpha * 1.05, alpha_max)

        # Update U_ep using EPSR-like formula
        U_ep, velocity = update_ep(
            r, g_sim, g_target, U_ep,
            alpha=alpha, T=T, max_amp=3.0, sigma_smooth=1.5,
            momentum=velocity, beta=beta, method=method
        )

        # Early stopping
        if chi2 < 0.01:
            break

    return chi2_history, alpha_history, U_ep


def plot_comparison(results, save_path='outputs/algorithm_comparison.png'):
    """
    Plot comparison of different methods

    Parameters
    ----------
    results : dict
        Dictionary with method names as keys and (chi2_history, alpha_history) as values
    save_path : str
        Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define colors for each method
    colors = {
        'Simple (original)': '#d62728',
        'Momentum': '#2ca02c',
        'Nesterov': '#ff7f0e',
        'Simple + Adaptive LR': '#9467bd',
        'Momentum + Adaptive LR': '#1f77b4'
    }

    # Plot 1: Chi-squared vs iteration (linear scale)
    ax = axes[0, 0]
    for method_name, (chi2_hist, alpha_hist) in results.items():
        ax.plot(range(1, len(chi2_hist) + 1), chi2_hist,
                'o-', label=method_name, linewidth=2, markersize=4,
                color=colors.get(method_name, None))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('χ²', fontsize=12)
    ax.set_title('Convergence Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Chi-squared vs iteration (log scale)
    ax = axes[0, 1]
    for method_name, (chi2_hist, alpha_hist) in results.items():
        ax.semilogy(range(1, len(chi2_hist) + 1), chi2_hist,
                    'o-', label=method_name, linewidth=2, markersize=4,
                    color=colors.get(method_name, None))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('χ² (log scale)', fontsize=12)
    ax.set_title('Convergence Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 3: Learning rate vs iteration
    ax = axes[1, 0]
    for method_name, (chi2_hist, alpha_hist) in results.items():
        if 'Adaptive' in method_name or method_name == 'Momentum + Adaptive LR':
            ax.plot(range(1, len(alpha_hist) + 1), alpha_hist,
                    'o-', label=method_name, linewidth=2, markersize=4,
                    color=colors.get(method_name, None))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate α', fontsize=12)
    ax.set_title('Adaptive Learning Rate Behavior', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary statistics
    table_data = []
    headers = ['Method', 'Initial χ²', 'Final χ²', 'Iters', 'Reduction']

    for method_name, (chi2_hist, alpha_hist) in results.items():
        initial_chi2 = chi2_hist[0]
        final_chi2 = chi2_hist[-1]
        n_iters = len(chi2_hist)
        reduction = (initial_chi2 - final_chi2) / initial_chi2 * 100

        table_data.append([
            method_name,
            f'{initial_chi2:.2f}',
            f'{final_chi2:.4f}',
            f'{n_iters}',
            f'{reduction:.1f}%'
        ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='left', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.1, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")

    return fig


def main():
    """
    Run comparison tests
    """
    print("=" * 70)
    print("EPSR Algorithm Comparison Test")
    print("=" * 70)
    print("\nThis test compares different optimization algorithms on a")
    print("synthetic g(r) fitting problem that mimics EPSR behavior.\n")

    # Test parameters
    max_iter = 50
    alpha_init = 0.3
    beta = 0.9

    results = {}

    # Test 1: Simple gradient descent (original)
    print("Testing: Simple gradient descent (original method)...")
    chi2_hist, alpha_hist, _ = run_optimization(
        method='simple', alpha=alpha_init, beta=beta,
        adaptive_lr=False, max_iter=max_iter
    )
    results['Simple (original)'] = (chi2_hist, alpha_hist)
    print(f"  Final χ² = {chi2_hist[-1]:.6f} after {len(chi2_hist)} iterations")

    # Test 2: Momentum method
    print("\nTesting: Momentum method...")
    chi2_hist, alpha_hist, _ = run_optimization(
        method='momentum', alpha=alpha_init, beta=beta,
        adaptive_lr=False, max_iter=max_iter
    )
    results['Momentum'] = (chi2_hist, alpha_hist)
    print(f"  Final χ² = {chi2_hist[-1]:.6f} after {len(chi2_hist)} iterations")

    # Test 3: Nesterov accelerated gradient
    print("\nTesting: Nesterov accelerated gradient...")
    chi2_hist, alpha_hist, _ = run_optimization(
        method='nesterov', alpha=alpha_init, beta=beta,
        adaptive_lr=False, max_iter=max_iter
    )
    results['Nesterov'] = (chi2_hist, alpha_hist)
    print(f"  Final χ² = {chi2_hist[-1]:.6f} after {len(chi2_hist)} iterations")

    # Test 4: Simple + Adaptive LR
    print("\nTesting: Simple gradient descent + Adaptive learning rate...")
    chi2_hist, alpha_hist, _ = run_optimization(
        method='simple', alpha=alpha_init, beta=beta,
        adaptive_lr=True, alpha_min=0.05, alpha_max=0.5, max_iter=max_iter
    )
    results['Simple + Adaptive LR'] = (chi2_hist, alpha_hist)
    print(f"  Final χ² = {chi2_hist[-1]:.6f} after {len(chi2_hist)} iterations")
    print(f"  Final learning rate α = {alpha_hist[-1]:.4f}")

    # Test 5: Momentum + Adaptive LR (recommended combination)
    print("\nTesting: Momentum + Adaptive learning rate (RECOMMENDED)...")
    chi2_hist, alpha_hist, _ = run_optimization(
        method='momentum', alpha=alpha_init, beta=beta,
        adaptive_lr=True, alpha_min=0.05, alpha_max=0.5, max_iter=max_iter
    )
    results['Momentum + Adaptive LR'] = (chi2_hist, alpha_hist)
    print(f"  Final χ² = {chi2_hist[-1]:.6f} after {len(chi2_hist)} iterations")
    print(f"  Final learning rate α = {alpha_hist[-1]:.4f}")

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_method = min(results.items(), key=lambda x: x[1][0][-1])
    print(f"\nBest performing method: {best_method[0]}")
    print(f"  Final χ² = {best_method[1][0][-1]:.6f}")
    print(f"  Iterations = {len(best_method[1][0])}")

    print("\nRecommendation:")
    print("  Use 'Momentum + Adaptive LR' for stable and fast convergence.")
    print("  This is the default in the updated main_epsr.py")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
