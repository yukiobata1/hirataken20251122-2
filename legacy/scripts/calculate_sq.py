#!/usr/bin/env python3
"""
Calculate S(Q) (structure factor) from g(r) (radial distribution function)
for liquid Ga at different temperatures.

S(Q) = 1 + (4πρ/Q) ∫ r[g(r)-1]sin(Qr) dr

where:
- g(r) is the radial distribution function
- ρ is the number density (atoms/Å³)
- Q is the wavevector magnitude
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_density(temp_K, atoms=108000):
    """
    Calculate number density from the thermo data.

    Parameters:
    -----------
    temp_K : int
        Temperature in Kelvin
    atoms : int
        Number of atoms in simulation

    Returns:
    --------
    float : number density in atoms/Å³
    """
    # Read the last line of thermo data to get final volume
    thermo_file = f'thermo_{temp_K}K.dat'
    try:
        data = np.loadtxt(thermo_file, comments='#')
        volume = data[-1, 6]  # Column 6 is volume in Å³
        density = atoms / volume
        print(f"{temp_K}K: Volume = {volume:.2f} Å³, Density = {density:.6f} atoms/Å³")
        return density
    except Exception as e:
        print(f"Error reading {thermo_file}: {e}")
        # Fallback: estimate from mass density (liquid Ga ~6.0 g/cm³ at room temp)
        # ρ_number = ρ_mass * N_A / M
        # For Ga: M = 69.723 g/mol
        if temp_K == 293:
            rho_mass = 6.0  # g/cm³
        elif temp_K == 600:
            rho_mass = 5.8
        elif temp_K == 1000:
            rho_mass = 5.6
        else:
            rho_mass = 6.0

        density = (rho_mass * 6.022e23) / (69.723 * 1e24)  # atoms/Å³
        print(f"{temp_K}K: Using estimated density = {density:.6f} atoms/Å³")
        return density

def load_rdf(filename):
    """
    Load RDF data from LAMMPS output.

    Returns:
    --------
    r : array of distances
    gr : array of g(r) values
    """
    data = np.loadtxt(filename, comments='#')
    # LAMMPS rdf output: column 1 = bin, column 2 = r, column 3 = g(r), column 4 = coordination
    r = data[:, 1]
    gr = data[:, 2]
    return r, gr

def calculate_sq(r, gr, rho, Q_max=20.0, Q_points=200):
    """
    Calculate S(Q) from g(r) using Fourier transform.

    S(Q) = 1 + (4πρ/Q) ∫ r[g(r)-1]sin(Qr) dr

    Parameters:
    -----------
    r : array
        Radial distances
    gr : array
        Radial distribution function g(r)
    rho : float
        Number density (atoms/Å³)
    Q_max : float
        Maximum Q value (Å⁻¹)
    Q_points : int
        Number of Q points

    Returns:
    --------
    Q : array of wavevector magnitudes
    SQ : array of structure factor values
    """
    # Create Q array
    Q = np.linspace(0.1, Q_max, Q_points)
    SQ = np.zeros_like(Q)

    # Calculate S(Q) for each Q value
    for i, q in enumerate(Q):
        # Integrand: r * [g(r) - 1] * sin(Qr)
        integrand = r * (gr - 1.0) * np.sin(q * r)

        # Numerical integration using trapezoidal rule
        integral = np.trapz(integrand, r)

        # S(Q) = 1 + (4πρ/Q) * integral
        SQ[i] = 1.0 + (4.0 * np.pi * rho / q) * integral

    return Q, SQ

def main():
    """Calculate S(Q) for all three temperatures."""

    temperatures = [293, 600, 1000]
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(10, 6))

    for temp, color in zip(temperatures, colors):
        print(f"\nProcessing {temp}K...")

        # Load RDF data
        rdf_file = f'rdf_{temp}K.dat'
        r, gr = load_rdf(rdf_file)
        print(f"  Loaded RDF: {len(r)} points, r_max = {r[-1]:.2f} Å")

        # Calculate density
        rho = calculate_density(temp)

        # Calculate S(Q)
        Q, SQ = calculate_sq(r, gr, rho, Q_max=20.0, Q_points=200)
        print(f"  Calculated S(Q): {len(Q)} points, Q_max = {Q[-1]:.2f} Å⁻¹")
        print(f"  S(Q=0) extrapolation would give compressibility info")
        print(f"  First peak in S(Q) at Q ≈ {Q[np.argmax(SQ)]:.2f} Å⁻¹")

        # Save to file
        output_file = f'sq_{temp}K.dat'
        header = f"# Q(A^-1)  S(Q)\n# Calculated from g(r) using Fourier transform\n"
        header += f"# Temperature: {temp} K\n"
        header += f"# Number density: {rho:.6f} atoms/Å³"

        np.savetxt(output_file, np.column_stack([Q, SQ]),
                   fmt='%f  %f', header=header, comments='')
        print(f"  Saved to {output_file}")

        # Plot
        plt.plot(Q, SQ, color=color, linewidth=2, label=f'{temp}K')

    # Format plot
    plt.xlabel('Q (Å⁻¹)', fontsize=14)
    plt.ylabel('S(Q)', fontsize=14)
    plt.title('Structure Factor S(Q) for Liquid Ga at Different Temperatures', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlim(0, 20)

    # Save plot
    plt.tight_layout()
    plt.savefig('sq_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to sq_comparison.png")
    plt.close()

    print("\n" + "="*60)
    print("S(Q) calculation completed for all temperatures!")
    print("="*60)

if __name__ == '__main__':
    main()
