import numpy as np
import matplotlib.pyplot as plt
import os

# File paths
experimental_file = 'outputs/ga_lj_sq.txt'
simulation_file = 'user_provided_sq_data.txt'
output_plot = 'comparison_sq_plot.png'

# Load Experimental Data
print(f"Loading experimental data from {experimental_file}...")
try:
    exp_data = np.loadtxt(experimental_file, comments='#')
    q_exp = exp_data[:, 0]
    sq_exp = exp_data[:, 1]
except Exception as e:
    print(f"Error reading {experimental_file}: {e}")
    exit(1)

# Load Simulation Data (User provided)
print(f"Loading simulation data from {simulation_file}...")
try:
    sim_data = np.loadtxt(simulation_file, delimiter=',')
    q_sim = sim_data[:, 0]
    sq_sim = sim_data[:, 1]
except Exception as e:
    print(f"Error reading {simulation_file} with comma delimiter: {e}")
    # Try reading without delimiter if it fails
    try:
         sim_data = np.loadtxt(simulation_file)
         q_sim = sim_data[:, 0]
         sq_sim = sim_data[:, 1]
    except Exception as e2:
         print(f"Error reading {simulation_file} with whitespace: {e2}")
         exit(1)

# Plotting
plt.figure(figsize=(10, 6))

# Plot experimental data
plt.plot(q_exp, sq_exp, label='Reference Data (outputs/ga_lj_sq.txt)', 
         linestyle='-', linewidth=2, color='black', alpha=0.6)

# Plot simulation data
plt.plot(q_sim, sq_sim, label='Simulation Result', 
         linestyle='none', marker='o', markersize=5, color='red', markerfacecolor='none', markeredgewidth=1.5)

plt.xlabel('Q ($\AA^{-1}$)', fontsize=14)
plt.ylabel('S(Q)', fontsize=14)
plt.title('Comparison of Structure Factor S(Q)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save plot
plt.savefig(output_plot, dpi=300)
print(f"Plot saved successfully to {output_plot}")
