#!/usr/bin/env python3
"""
Analyze why χ² is increasing while g(r) visually looks better.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load experimental data
exp_data = np.loadtxt('data/g_exp_cleaned.dat')
r_exp = exp_data[:, 0]
g_exp = exp_data[:, 1]

# Filter to r >= 2.0 (what EPSR uses)
mask = r_exp >= 2.0
r_exp = r_exp[mask]
g_exp = g_exp[mask]

# Read final RDF from last iteration
with open('rdf.dat', 'r') as f:
    lines = [line for line in f if not line.startswith('#') and line.strip()]

# Parse RDF data (skip timestep lines)
data_lines = []
for line in lines:
    parts = line.split()
    if len(parts) > 3:  # Data line (not timestep header)
        data_lines.append([float(x) for x in parts])

rdf_data = np.array(data_lines)
r_sim = rdf_data[:, 1]
g_sim_total = rdf_data[:, 2]

# Interpolate simulation to experimental grid
g_sim_interp = np.interp(r_exp, r_sim, g_sim_total)

# Calculate chi-squared
sigma = 0.05
residuals = g_sim_interp - g_exp
chi2_total = np.sum(residuals**2 / sigma**2)

print(f"Total χ² = {chi2_total:.1f}")
print(f"Number of points: {len(r_exp)}")
print(f"Average squared residual: {np.mean(residuals**2):.4f}")
print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.4f}")
print()

# Analyze by region
regions = [
    ("First peak (2.0-3.0Å)", 2.0, 3.0),
    ("Valley (3.0-4.5Å)", 3.0, 4.5),
    ("Second peak (4.5-6.5Å)", 4.5, 6.5),
    ("Tail (6.5-12Å)", 6.5, 12.0)
]

for name, r_min, r_max in regions:
    mask = (r_exp >= r_min) & (r_exp < r_max)
    if mask.sum() == 0:
        continue

    res_region = residuals[mask]
    chi2_region = np.sum(res_region**2 / sigma**2)
    n_points = mask.sum()

    print(f"{name}:")
    print(f"  Points: {n_points}")
    print(f"  χ² contribution: {chi2_region:.1f} ({100*chi2_region/chi2_total:.1f}%)")
    print(f"  RMSD: {np.sqrt(np.mean(res_region**2)):.4f}")
    print(f"  Max |error|: {np.abs(res_region).max():.4f}")
    print()

# Plot residuals
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Top: g(r) comparison
axes[0].plot(r_exp, g_exp, 'k-', label='Experiment', linewidth=2)
axes[0].plot(r_exp, g_sim_interp, 'r--', label='Simulation (final)', linewidth=1.5)
axes[0].set_xlabel('r (Å)')
axes[0].set_ylabel('g(r)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f'Final Iteration: χ² = {chi2_total:.1f}')

# Bottom: Residuals
axes[1].plot(r_exp, residuals, 'b-', linewidth=1.5)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].fill_between(r_exp, -sigma, sigma, alpha=0.2, color='gray', label=f'±σ = ±{sigma}')
axes[1].set_xlabel('r (Å)')
axes[1].set_ylabel('Residual (sim - exp)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Residuals by Region')

plt.tight_layout()
plt.savefig('outputs/residual_analysis.png', dpi=150)
print("Saved: outputs/residual_analysis.png")
