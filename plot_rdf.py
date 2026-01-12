#!/usr/bin/env python3
"""
Plot and compare RDF for multiple temperatures
"""

import numpy as np
import matplotlib.pyplot as plt

def read_rdf(filename):
    """Read LAMMPS RDF file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                float(parts[0])
                if len(parts) >= 3:
                    data.append(parts)
            except ValueError:
                continue
    
    data = np.array(data, dtype=float)
    r = data[:, 1]
    g_r = data[:, 2]
    return r, g_r

# Read all RDF files
temps = [293, 600, 1000]
colors = ['blue', 'orange', 'red']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: RDF comparison
ax1 = axes[0]
for temp, color in zip(temps, colors):
    filename = f'rdf_{temp}K.dat'
    r, g_r = read_rdf(filename)
    ax1.plot(r, g_r, linewidth=2, label=f'{temp}K', color=color)

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('r (Å)', fontsize=14)
ax1.set_ylabel('g(r)', fontsize=14)
ax1.set_title('Radial Distribution Function', fontsize=16)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# Plot 2: RDF peak region (zoomed)
ax2 = axes[1]
for temp, color in zip(temps, colors):
    filename = f'rdf_{temp}K.dat'
    r, g_r = read_rdf(filename)
    ax2.plot(r, g_r, linewidth=2, label=f'{temp}K', color=color)

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('r (Å)', fontsize=14)
ax2.set_ylabel('g(r)', fontsize=14)
ax2.set_title('First Peak Region (Zoomed)', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2, 5)
ax2.set_ylim(0, 3.5)

plt.tight_layout()
plt.savefig('rdf_comparison.png', dpi=300)
print("RDF comparison saved: rdf_comparison.png")

# Analyze first peak
print("\nFirst Peak Analysis:")
print("-" * 60)
print(f"{'Temp (K)':<12} {'Peak r (Å)':<15} {'Peak g(r)':<15}")
print("-" * 60)

for temp in temps:
    filename = f'rdf_{temp}K.dat'
    r, g_r = read_rdf(filename)
    
    # Find first peak (between 2-4 Å)
    mask = (r > 2.0) & (r < 4.0)
    r_peak = r[mask]
    g_peak = g_r[mask]
    
    peak_idx = np.argmax(g_peak)
    peak_r = r_peak[peak_idx]
    peak_height = g_peak[peak_idx]
    
    print(f"{temp:<12} {peak_r:<15.3f} {peak_height:<15.3f}")

print("-" * 60)
print("\nExpected for liquid Ga at 300K:")
print("  First peak position: ~2.7-2.8 Å")
print("  First peak height:   ~2.5-3.0")
plt.show()
