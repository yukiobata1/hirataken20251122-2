import numpy as np
import datetime

def generate_ga_eam():
    """
    Generates a LAMMPS setfl EAM file for Liquid Gallium based on 
    D.K. Belashchenko, Russ. J. Phys. Chem. A, 2012, Vol. 86, No. 5, pp. 779-790.
    """
    
    # --- 1. PARAMETERS FROM PAPER ---
    
    # General
    atomic_number = 31
    atomic_mass = 69.723
    lattice_constant = 4.10  # FCC parameter from paper text
    lattice_type = 'fcc'
    
    # Grid settings for EAM file
    Nr = 20000
    Nrho = 20000
    rmax = 8.30  # Cutoff from paper
    rhomax = 4.0 # Sufficiently high density (paper covers up to 1.4, we extrapolate Eq 7)
    dr = rmax / Nr
    drho = rhomax / Nrho

    # --- 2. PAIR POTENTIAL PHI(r) ---
    # Knots for Pair Potential (Angstroms)
    # Paper: r_i = 2.15, 2.75, 3.35, 4.00, 6.50, 8.30
    # Intervals:
    # 0: 0 < r < 2.15 (Eq 4)
    # 1: 2.15 < r < 2.75 (Poly 1) -> Exp around 2.75
    # 2: 2.75 < r < 3.35 (Poly 2) -> Exp around 3.35
    # 3: 3.35 < r < 4.00 (Poly 3) -> Exp around 4.00
    # 4: 4.00 < r < 6.50 (Poly 4) -> Exp around 6.50
    # 5: 6.50 < r < 8.30 (Poly 5) -> Exp around 8.30
    
    r_knots = [2.15, 2.75, 3.35, 4.00, 6.50, 8.30]
    
    # Table 1 Coefficients (a_im)
    # Structure: [a_i0, a_i1, ..., a_i8]
    # i=1 (index 0 in python) to i=5 (index 4)
    # Note: Coefficients are for (r - r_{i+1})^m
    
    a_coeffs = [
        # i=1 (Interval 2.15 - 2.75), Expansion around 2.75
        [-0.65052509307861e-01, -0.32728102803230e+00, 0.51590444127493e+01, 
         0.90195221829217e+02, 0.72322004859499e+03, 0.27788989409594e+04, 
         0.56037895713613e+04, 0.57428084950480e+04, 0.23685488320885e+04],
         
        # i=2 (Interval 2.75 - 3.35), Expansion around 3.35
        [-0.15576396882534e+00, -0.16365580260754e-01, 0.20955204046244e+00, 
         -0.97550604734748e+00, -0.11625479189815e+02, -0.58549935696765e+02, 
         -0.15186293377510e+03, -0.19622924502226e+03, -0.98789413798382e+02],
         
        # i=3 (Interval 3.35 - 4.00), Expansion around 4.00
        [-0.13794735074043e+00, 0.78778542578220e-01, -0.83622260891495e-01, 
         -0.44410858010987e+01, -0.36415106938231e+02, -0.13414583419234e+03, 
         -0.25239146992011e+03, -0.23858760191913e+03, -0.90270667293646e+02],
         
        # i=4 (Interval 4.00 - 6.50), Expansion around 6.50
        [0.13303710147738e-01, 0.59769893996418e-02, 0.57411338894840e-01, 
         0.19517888219051e+00, 0.32162310059276e+00, 0.30195698240893e+00, 
         0.14850603977640e+00, 0.36233874262589e-01, 0.34984220138018e-02],
         
        # i=5 (Interval 6.50 - 8.30), Expansion around 8.30
        [0.00000000000000e+00, 0.00000000000000e+00, -0.60454444423660e-02, 
         -0.13258585494287e+00, -0.34988482891053e+00, -0.45183606796559e+00, 
         -0.31733856650298e+00, -0.11493645479281e+00, -0.16768950999376e-01]
    ]

    # --- 3. ELECTRON DENSITY RHO(r) ---
    # Equation 6: psi(r) = p1 * exp(-p2 * r)
    p1 = 2.24450
    p2 = 1.200

    # --- 4. EMBEDDING FUNCTION F(rho) ---
    # Knots: rho_5 < rho_4 < rho_3 < rho_2 < rho_1 < rho_0 < rho_6
    rho0 = 1.0
    rho1 = 0.92
    rho2 = 0.87
    rho3 = 0.80
    rho4 = 0.75
    rho5 = 0.65
    rho6 = 1.40
    
    # Coefficients from text
    # Region 1 (0.92 <= rho <= 1.40): Eq 7
    # F(rho) = a1 + c1(rho - rho0)^2
    F_a1 = -1.91235
    F_c1 = 1.3000
    
    # Region 2 (0.87 <= rho <= 0.92): Eq 8 (i=2)
    # F(rho) = a2 + b2(rho - rho1) + c2(rho - rho1)^2
    F_a2 = -1.904030
    F_b2 = -0.208000
    F_c2 = -1.500
    
    # Region 3 (0.80 <= rho <= 0.87): Eq 8 (i=3)
    F_a3 = -1.897380
    F_b3 = -0.058000
    F_c3 = 2.000
    
    # Region 4 (0.75 <= rho <= 0.80): Eq 8 (i=4)
    F_a4 = -1.883520
    F_b4 = -0.338000
    F_c4 = 5.600
    
    # Region 5 (0.65 <= rho <= 0.75): Eq 8 (i=5)
    F_a5 = -1.852620
    F_b5 = -0.898000
    F_c5 = -6.000
    
    # Region 6 (rho <= 0.65): Eq 9
    # F(rho) = [a6 + b6(rho - rho5) + c6(rho - rho5)^2] * [2(rho/rho5) - (rho/rho5)^2]
    F_a6 = -1.822820
    F_b6 = 0.302000
    F_c6 = 2.000

    # --- CALCULATION FUNCTIONS ---

    def calc_phi(r):
        if r >= 8.30:
            return 0.0
        elif r < 2.15:
            # Equation 4
            # 0.619588 - 51.86268*(2.15 - r) + 27.8*{exp[1.96*(2.15-r)] - 1}
            term1 = 0.619588
            term2 = -51.86268 * (2.15 - r)
            term3 = 27.8 * (np.exp(1.96 * (2.15 - r)) - 1.0)
            return term1 + term2 + term3
        else:
            # Spline regions
            if r < 2.75:
                idx = 0; r_next = 2.75
            elif r < 3.35:
                idx = 1; r_next = 3.35
            elif r < 4.00:
                idx = 2; r_next = 4.00
            elif r < 6.50:
                idx = 3; r_next = 6.50
            else:
                idx = 4; r_next = 8.30
            
            coeffs = a_coeffs[idx]
            delta = r - r_next
            val = 0.0
            for m, c in enumerate(coeffs):
                val += c * (delta**m)
            return val

    def calc_rho(r):
        if r >= 8.30:
            return 0.0
        return p1 * np.exp(-p2 * r)

    def calc_F(rho):
        # Piecewise definitions
        if rho > rho6:
            # Paper doesn't define rho > 1.40 (Shock region).
            # We extrapolate Eq 7 (harmonic) to ensure stability.
            # Since c1=1.3 is positive, this provides repulsion at high density.
            return F_a1 + F_c1 * (rho - rho0)**2
            
        elif rho >= rho1: # 0.92 to 1.40
            # Eq 7
            return F_a1 + F_c1 * (rho - rho0)**2
            
        elif rho >= rho2: # 0.87 to 0.92
            # Eq 8 (i=2) expansion around rho1
            return F_a2 + F_b2 * (rho - rho1) + F_c2 * (rho - rho1)**2
            
        elif rho >= rho3: # 0.80 to 0.87
            # Eq 8 (i=3) expansion around rho2
            return F_a3 + F_b3 * (rho - rho2) + F_c3 * (rho - rho2)**2
            
        elif rho >= rho4: # 0.75 to 0.80
            # Eq 8 (i=4) expansion around rho3
            return F_a4 + F_b4 * (rho - rho3) + F_c4 * (rho - rho3)**2
            
        elif rho >= rho5: # 0.65 to 0.75
            # Eq 8 (i=5) expansion around rho4
            return F_a5 + F_b5 * (rho - rho4) + F_c5 * (rho - rho4)**2
            
        else: # rho < 0.65
            # Eq 9
            # F(rho) = [Poly around rho5] * ScalingFactor
            poly = F_a6 + F_b6 * (rho - rho5) + F_c6 * (rho - rho5)**2
            scale = 2.0 * (rho / rho5) - (rho / rho5)**2
            return poly * scale

    # --- GENERATE ARRAYS ---
    
    # 1. Embedding F(rho)
    # LAMMPS needs Nrho values. Index i corresponds to rho = (i-1)*drho? 
    # No, usually i=1 is drho. setfl format implies rho goes from 0 to rhomax? 
    # LAMMPS: "The file lists ... Nrho values ... from 0 to rhomax"
    F_vals = []
    for i in range(Nrho):
        r_val = (i + 1) * drho # Avoid rho=0 if possible, or handle 0
        # Actually LAMMPS includes 0? Typically yes. Let's generate Nrho values.
        # Standard practice: Nrho values, rho = i * drho + drho/2 ? Or just grid.
        # Let's assume grid [0, drho, 2drho ...]. But array length is Nrho.
        # Let's align r_val such that rhomax is the end.
        rho_val = i * drho
        if rho_val == 0:
            F_vals.append(0.0) # F(0) should be 0 based on Eq 9 scale factor
        else:
            F_vals.append(calc_F(rho_val))
            
    # 2. Z(r)
    # LAMMPS setfl requires Z(r) where Z(r) = sqrt(r * phi(r)) ? NO.
    # LAMMPS setfl requires: F(rho), Z(r), rho(r).
    # Where phi(r) = Z(r) * Z(r) / r.
    # However, calculating Z(r) from phi(r) is tricky if phi is negative (which it is).
    # Fortunately, LAMMPS eam/alloy setfl format actually stores:
    #   F(rho) array
    #   rho(r) array
    #   phi(r)*r array  <-- This is specific to eam/alloy setfl
    # Wait, let me verify standard.
    # Standard funcfl: Z(r).
    # Standard setfl (eam/alloy): 
    #   Line 1-6: Header
    #   Section 1: F(rho)
    #   Section 2: rho(r)
    #   Section 3: phi(r)*r
    
    rho_r_vals = []
    phir_vals = []
    
    for i in range(Nr):
        r_dist = i * dr
        # For r=0, handle singularity
        if r_dist == 0:
            rho_r_vals.append(calc_rho(0.0)) # p1
            # limit of phi(r)*r as r->0
            # Eq 4 dominates: 27.8 * exp(...) -> large positive.
            # It is finite but large. Let's compute at very small r
            phir_vals.append(0.0) # LAMMPS usually ignores index 0 or handles it
        else:
            rho_r_vals.append(calc_rho(r_dist))
            phir_vals.append(calc_phi(r_dist) * r_dist)

    # --- WRITE FILE ---
    filename = "Ga_belashchenko2012.eam.alloy"
    with open(filename, 'w') as f:
        # Header
        f.write(f"Date: {datetime.date.today()} EAM for Ga (Belashchenko 2012) generated by Python script\n")
        f.write("# Ref: D.K. Belashchenko, Russ. J. Phys. Chem. A, 86, 5, 779 (2012)\n")
        f.write("# Equations 4, 6-9 used. Extrapolated > 1.40 rho.\n")
        f.write("1 Ga\n")
        f.write(f"{Nrho} {drho:.6f} {Nr} {dr:.6f} {rmax:.4f}\n")
        f.write(f"{atomic_number} {atomic_mass} {lattice_constant} {lattice_type}\n")
        
        # Function to chunk data into 5 columns
        def write_chunk(data):
            for k, val in enumerate(data):
                f.write(f"{val:20.12e}")
                if (k + 1) % 5 == 0:
                    f.write("\n")
            if len(data) % 5 != 0:
                f.write("\n")

        # 1. Embedding F(rho)
        write_chunk(F_vals)
        
        # 2. Density rho(r)
        write_chunk(rho_r_vals)
        
        # 3. Pair potential r*phi(r)
        write_chunk(phir_vals)

    print(f"File {filename} generated successfully.")

if __name__ == "__main__":
    generate_ga_eam()
