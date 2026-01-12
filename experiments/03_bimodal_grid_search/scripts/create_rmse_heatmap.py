#!/usr/bin/env python3
"""
Create RMSE heatmap from metrics_summary.csv
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load data
data_path = Path(__file__).parent.parent / 'outputs' / 'analysis' / 'metrics_summary.csv'
df = pd.read_csv(data_path)

# Get unique values
sig12_vals = sorted(df['sig12'].unique())
ga1_vals = sorted(df['ga1_frac'].unique())

# Create RMSE matrix
rmse_matrix = np.full((len(sig12_vals), len(ga1_vals)), np.nan)
for _, row in df.iterrows():
    i = sig12_vals.index(row['sig12'])
    j = ga1_vals.index(row['ga1_frac'])
    rmse_matrix[i, j] = row['rmse']

# Find best (minimum RMSE)
best_idx = df['rmse'].idxmin()
best_row = df.loc[best_idx]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(rmse_matrix, aspect='auto', origin='lower', cmap='viridis_r')
ax.set_xticks(range(len(ga1_vals)))
ax.set_xticklabels([f'{v*100:.0f}%' for v in ga1_vals])
ax.set_yticks(range(len(sig12_vals)))
ax.set_yticklabels([f'{v:.2f}x' for v in sig12_vals])
ax.set_xlabel('Ga1 fraction', fontsize=12)
ax.set_ylabel(r'$\sigma_{12}$ ratio', fontsize=12)
ax.set_title('RMSE Heatmap: S(Q) vs Experimental', fontsize=14)
plt.colorbar(im, label='RMSE')

# Mark best point
best_i = sig12_vals.index(best_row['sig12'])
best_j = ga1_vals.index(best_row['ga1_frac'])
ax.plot(best_j, best_i, 'r*', markersize=15, markeredgecolor='white')

plt.tight_layout()

# Save
output_path = Path(__file__).parent.parent / 'outputs' / 'analysis' / 'rmse_heatmap.png'
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

# Also copy to thesis
thesis_path = Path(__file__).parent.parent.parent.parent / 'thesis' / 'figures' / 'rmse_heatmap.png'
plt.savefig(thesis_path, dpi=150)
print(f"Saved: {thesis_path}")
