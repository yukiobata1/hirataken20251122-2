#!/usr/bin/env python3
"""
Calculate diffusion coefficient from MSD data
D = MSD / (6t)
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_diffusion(filename, temp):
    """Calculate diffusion coefficient from MSD file"""
    
    print(f"\n{'='*60}")
    print(f"Temperature: {temp}K")
    print(f"File: {filename}")
    print('='*60)
    
    # Read MSD data
    data = np.loadtxt(filename, comments='#')
    
    # Columns: timestep, MSD
    timestep = data[:, 0]
    msd = data[:, 1]  # Å²
    
    # Convert timestep to time (ps)
    # timestep × dt = timestep × 0.001 ps
    time_ps = timestep * 0.001
    
    # Use second half for linear fitting
    n_start = len(time_ps) // 2
    
    # Linear fit: MSD = 6Dt
    coeffs = np.polyfit(time_ps[n_start:], msd[n_start:], 1)
    slope = coeffs[0]  # Å²/ps
    
    # Diffusion coefficient
    D_A2ps = slope / 6.0  # Å²/ps
    D_cm2s = D_A2ps * 1e-4  # cm²/s
    
    print(f"  Time range used: {time_ps[n_start]:.1f} - {time_ps[-1]:.1f} ps")
    print(f"  MSD slope: {slope:.4f} Å²/ps")
    print(f"  Diffusion coefficient: {D_cm2s:.3e} cm²/s")
    
    # Expected value for Ga at different temperatures
    expected = {
        293: 1.5e-5,
        300: 1.5e-5,
        600: 5e-5,
        1000: 1.5e-4,
    }
    
    if temp in expected:
        exp_val = expected[temp]
        ratio = D_cm2s / exp_val
        print(f"  Expected (literature): ~{exp_val:.2e} cm²/s")
        print(f"  Ratio (calc/exp): {ratio:.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_ps, msd, 'b-', linewidth=2, label='MSD data')
    plt.plot(time_ps[n_start:], slope*time_ps[n_start:]+coeffs[1], 
             'r--', linewidth=2, label=f'Linear fit: D={D_cm2s:.2e} cm²/s')
    
    plt.xlabel('Time (ps)', fontsize=14)
    plt.ylabel('MSD (Å²)', fontsize=14)
    plt.title(f'Mean Square Displacement at {temp}K', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'msd_{temp}K_analysis.png', dpi=300)
    print(f"  Plot saved: msd_{temp}K_analysis.png")
    
    return D_cm2s

# Process all temperatures
temps = [293, 600, 1000]
diffusion_coeffs = {}

for temp in temps:
    filename = f'msd_{temp}K.dat'
    D = calculate_diffusion(filename, temp)
    diffusion_coeffs[temp] = D

# Summary plot
print(f"\n{'='*60}")
print("SUMMARY: Temperature-dependent Diffusion")
print('='*60)

plt.figure(figsize=(10, 6))
temps_list = sorted(diffusion_coeffs.keys())
D_list = [diffusion_coeffs[T] for T in temps_list]

plt.semilogy(temps_list, D_list, 'o-', linewidth=2, markersize=10, label='Simulation')

# Add expected values
exp_temps = [293, 600, 1000]
exp_D = [1.5e-5, 5e-5, 1.5e-4]
plt.semilogy(exp_temps, exp_D, 's--', linewidth=2, markersize=8, 
             alpha=0.7, label='Expected (literature)')

plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('Diffusion Coefficient (cm²/s)', fontsize=14)
plt.title('Temperature-dependent Diffusion Coefficient', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('diffusion_vs_temp.png', dpi=300)
print("\nComparison plot saved: diffusion_vs_temp.png")

# Print summary table
print("\nSummary Table:")
print("-" * 60)
print(f"{'Temp (K)':<12} {'D (cm²/s)':<15} {'Expected':<15} {'Ratio':<10}")
print("-" * 60)
for temp in temps_list:
    D_calc = diffusion_coeffs[temp]
    if temp == 293:
        D_exp = 1.5e-5
    elif temp == 600:
        D_exp = 5e-5
    else:
        D_exp = 1.5e-4
    ratio = D_calc / D_exp
    print(f"{temp:<12} {D_calc:<15.3e} {D_exp:<15.3e} {ratio:<10.2f}")
print("-" * 60)
