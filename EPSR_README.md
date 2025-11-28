# EPSR for EGaIn System - Quick Start Guide

EPSR (Empirical Potential Structure Refinement) implementation for liquid Ga-In eutectic alloy.

## System Information

- **Composition**: Ga₀.₈₅₈In₀.₁₄₂ (eutectic)
- **Temperature**: 150°C (423.15 K)
- **Number of atoms**: 1000 (858 Ga + 142 In)
- **Experimental data**: Neutron diffraction g(r) from Amon et al. (2023)

## Prerequisites

1. **LAMMPS** installed and accessible via `lmp` command
2. **Python 3** with required packages:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Quick Start

### Step 1: Create Initial Structure

```bash
cd /home/yuki/lammps_settings_obata/hirataken20251122-2
python3 scripts/create_initial_structure.py
```

This creates `inputs/initial_structure.data` with random atomic positions for Ga₀.₈₅₈In₀.₁₄₂.

**Options**:
- `-n, --natoms`: Number of atoms (default: 1000)
- `-x, --xIn`: Mole fraction of In (default: 0.142)
- `-T, --temperature`: Temperature in °C (default: 150.0)
- `-o, --output`: Output file path

### Step 2: (Optional) Test LJ-only Simulation

Before running EPSR, you can test the baseline Lennard-Jones simulation:

```bash
lmp -in inputs/in.egain_lj
```

This runs a pure LJ simulation and outputs `rdf.dat`. Compare with experimental data to see the initial discrepancy.

### Step 3: Initialize Empirical Potentials

```bash
python3 scripts/update_ep.py
```

This creates initial (zero) empirical potential tables in `data/`:
- `ep_GaGa.table`
- `ep_InIn.table`
- `ep_GaIn.table`

### Step 4: Run EPSR

```bash
python3 scripts/main_epsr.py
```

This starts the EPSR iterative refinement loop:
1. Runs LAMMPS with current U_EP
2. Calculates g(r) from simulation
3. Compares with experimental g(r)
4. Updates U_EP based on the difference
5. Repeats until convergence

**Parameters** (edit in `scripts/main_epsr.py`):
- `max_iter = 50`: Maximum iterations
- `alpha = 0.3`: Learning rate
- `tol = 0.1`: Convergence tolerance (χ²)
- `max_amp = 1.0`: Maximum U_EP amplitude (kcal/mol)

**Output files** (in `outputs/`):
- `epsr_iter###.png`: Results plot for each iteration
- `epsr_final.png`: Final results
- `final_ep.npz`: Final empirical potentials
- `lammps_iter###.log`: LAMMPS log files

### Step 5: Analyze Results

The EPSR process generates plots showing:
1. **g(r) comparison**: Simulated vs experimental
2. **U_EP**: Empirical potentials for Ga-Ga, In-In, Ga-In pairs
3. **Convergence**: χ² vs iteration number

Load final results:
```python
import numpy as np
data = np.load('outputs/final_ep.npz')
r = data['r']
U_ep_GaGa = data['U_ep_GaGa']
U_ep_InIn = data['U_ep_InIn']
U_ep_GaIn = data['U_ep_GaIn']
chi2_history = data['chi2_history']
```

## File Structure

```
.
├── inputs/
│   ├── initial_structure.data   # Initial atomic configuration
│   ├── in.egain_lj              # LJ-only LAMMPS input
│   └── in.egain_epsr            # EPSR LAMMPS input
├── scripts/
│   ├── create_initial_structure.py  # Generate initial structure
│   ├── update_ep.py                 # U_EP update functions
│   ├── main_epsr.py                 # Main EPSR loop
│   └── extract_gr_from_si.py        # Data extraction utilities
├── data/
│   ├── g_exp.dat                # Experimental g(r) (150°C)
│   ├── g_exp_200C.dat           # Experimental g(r) (200°C, for reference)
│   ├── ep_GaGa.table            # Ga-Ga empirical potential
│   ├── ep_InIn.table            # In-In empirical potential
│   └── ep_GaIn.table            # Ga-In empirical potential
└── outputs/
    ├── epsr_iter###.png         # Iteration plots
    ├── epsr_final.png           # Final results
    ├── final_ep.npz             # Final U_EP data
    └── lammps_iter###.log       # LAMMPS logs
```

## LJ Parameters

From Amon et al. (2023):

| Pair  | σ (Å) | ε (kcal/mol) | ε (kJ/mol) |
|-------|-------|--------------|------------|
| Ga-Ga | 2.70  | 0.430        | 1.80       |
| In-In | 3.11  | 0.430        | 1.80       |
| Ga-In | 2.905 | 0.430        | 1.80       |

Ga-In parameters use Lorentz-Berthelot mixing rules:
- σ_GaIn = (σ_Ga + σ_In)/2 = 2.905 Å
- ε_GaIn = √(ε_Ga × ε_In) = 1.80 kJ/mol

## Troubleshooting

### LAMMPS Errors

1. **"Could not find pair table file"**
   - Make sure you ran `python3 scripts/update_ep.py` first
   - Check that `data/ep_*.table` files exist

2. **"Atoms too close"**
   - Initial structure may have overlapping atoms
   - Try regenerating: `python3 scripts/create_initial_structure.py`

3. **Simulation unstable**
   - Reduce timestep in LAMMPS input (default: 2.0 fs)
   - Reduce learning rate `alpha` in `main_epsr.py`
   - Reduce `max_amp` for U_EP

### EPSR Not Converging

1. **χ² increasing**
   - Reduce learning rate `alpha` (try 0.1-0.2)
   - Increase smoothing: `sigma_smooth` in `update_ep.py`

2. **χ² oscillating**
   - Reduce `alpha`
   - Reduce `max_amp` to limit U_EP amplitude

3. **χ² stuck at high value**
   - Increase `alpha` (try 0.4-0.5)
   - Check experimental data quality
   - Verify temperature matches between simulation and experiment

## Advanced Options

### Using Different Temperature Data

To use 200°C data instead:

```bash
cp data/g_exp_200C.dat data/g_exp.dat
```

And update temperature in `main_epsr.py`:
```python
T = 473.15  # 200°C
```

### Changing EPSR Grid

Edit in `main_epsr.py`:
```python
r_min = 2.0      # Minimum distance (Å)
r_max = 12.0     # Maximum distance (Å)
N_grid = 200     # Number of grid points
```

### Using Partial g(r)

For more accurate results, update U_EP using partial g(r) instead of total g(r). This requires:
1. Experimental partial g(r) for Ga-Ga, In-In, Ga-In
2. Modifications to `main_epsr.py` to use partial RDF from LAMMPS

## References

1. Amon, A. et al. (2023). "Local Order in Liquid Gallium−Indium Alloys."
   *J. Phys. Chem. C*, 127(33), 16687-16694.
   https://doi.org/10.1021/acs.jpcc.3c03857

2. Soper, A. K. (1996). "Empirical potential Monte Carlo simulation of fluid structure."
   *Chem. Phys.*, 202, 295-306.

3. Soper, A. K. (2005). "Partial structure factors from disordered materials diffraction data:
   An approach using empirical potential structure refinement."
   *Phys. Rev. B*, 72, 104204.

## Support

For detailed methodology, see `plans/espr_lammps_guide.md`.

For issues or questions, refer to:
- LAMMPS documentation: https://docs.lammps.org/
- EPSR methodology: Soper (1996, 2005)
