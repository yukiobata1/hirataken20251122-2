# EPSR Implementation - Modern Modular Design

A production-ready implementation of Empirical Potential Structure Refinement (EPSR) for liquid metal alloy systems, inspired by Dissolve and based on A.K. Soper's methodology.

## Overview

This implementation provides:

- **Proper EPSR algorithm** with structure factor calculations
- **Modular design** with clean separation of concerns
- **Well-tested components** for scientific reproducibility
- **Flexible API** for customization and extension
- **GPU acceleration** support via LAMMPS Kokkos

## Project Structure

```
epsr/                          # Main package
├── __init__.py               # Package initialization and exports
├── core/                     # Core EPSR algorithms
│   ├── structure_factor.py   # S(Q) ↔ g(r) transformations
│   ├── potential.py          # Empirical potential refinement
│   └── epsr_engine.py        # Main EPSR workflow engine
├── io/                       # Input/output modules
│   ├── experimental.py       # Load experimental data
│   ├── lammps.py            # LAMMPS interface
│   └── tables.py            # Write LAMMPS table files
├── analysis/                 # Analysis tools
│   ├── rdf.py               # RDF analysis
│   └── metrics.py           # Goodness-of-fit metrics
└── visualization/            # Plotting utilities
    └── plots.py             # Publication-quality plots

scripts/
├── run_epsr.py              # Main entry point (NEW)
├── main_epsr.py             # Legacy script (keep for reference)
└── update_ep.py             # Legacy utilities

tests/
└── unit/                    # Unit tests
    ├── test_structure_factor.py
    ├── test_potential.py
    └── test_metrics.py
```

## Quick Start

### Installation

The EPSR package uses standard Python scientific libraries:

```bash
pip install numpy scipy matplotlib
```

Ensure LAMMPS is installed and accessible via `lmp` command.

### Basic Usage

Run EPSR refinement with default parameters:

```bash
python scripts/run_epsr.py
```

Run with GPU acceleration:

```bash
python scripts/run_epsr.py --gpu
```

Customize parameters:

```bash
python scripts/run_epsr.py \
    --max-iter 30 \
    --alpha 0.5 \
    --tolerance 200 \
    --method momentum \
    --beta 0.9
```

### Python API Usage

For programmatic access or custom workflows:

```python
from epsr import EPSREngine
from epsr.core.epsr_engine import EPSRConfig

# Configure EPSR
config = EPSRConfig(
    temperature=423.15,  # K (150°C)
    composition={'Ga': 0.858, 'In': 0.142},
    scattering_lengths={'Ga': 7.288, 'In': 4.061},  # fm
    density=0.042,  # atoms/Å³
    max_iterations=50,
    learning_rate=0.3,
    convergence_tol=250.0,
    use_gpu=True
)

# Create engine
engine = EPSREngine(config, 'data/g_exp_cleaned.dat')

# Run refinement
results = engine.run(
    lammps_input='inputs/in.egain_epsr_H100',
    rdf_output='rdf.dat'
)

# Check results
print(f"Converged: {results['converged']}")
print(f"Final χ²: {results['final_chi2']:.3f}")
print(f"Iterations: {results['iterations']}")
```

## Key Features

### 1. Proper Structure Factor Calculations

Unlike simplified implementations, this version correctly handles:

- Fourier transforms between g(r) and S(Q)
- Partial structure factors for multi-component systems
- Proper neutron scattering weights

```python
from epsr.core.structure_factor import StructureFactor

sf = StructureFactor(rho=0.042)  # number density
S_Q = sf.g_to_S(r, g, Q)         # g(r) → S(Q)
g_r = sf.S_to_g(Q, S, r)         # S(Q) → g(r)
```

### 2. Advanced Optimization

Three update methods available:

- **Simple**: Basic gradient descent
- **Momentum**: Accelerated convergence (default)
- **Nesterov**: Look-ahead gradient

```python
config = EPSRConfig(
    method='momentum',     # or 'simple', 'nesterov'
    learning_rate=0.3,
    momentum_beta=0.9
)
```

### 3. Automatic Convergence Detection

Monitors convergence and detects:

- χ² threshold crossing
- Stagnation at high χ²
- Divergence (increasing χ²)

### 4. Partial RDF Support

Automatically uses partial g(r) when available from LAMMPS:

- g_GaGa(r) for Ga-Ga pairs
- g_InIn(r) for In-In pairs
- g_GaIn(r) for Ga-In pairs

Each pair potential is refined independently with proper weighting.

### 5. Publication-Quality Visualization

Automatic generation of:

- g(r) comparison plots
- Empirical potential plots
- Convergence histories
- Combined summary figures

## Configuration Options

### EPSRConfig Parameters

**System Properties:**
- `temperature` (float): System temperature in Kelvin
- `composition` (dict): Mole fractions, e.g., `{'Ga': 0.858, 'In': 0.142}`
- `scattering_lengths` (dict): Neutron scattering lengths in fm
- `density` (float): Number density in atoms/Å³

**Grid Parameters:**
- `r_min` (float): Minimum distance for potential grid (default: 2.0 Å)
- `r_max` (float): Maximum distance for potential grid (default: 12.0 Å)
- `n_grid` (int): Number of grid points (default: 200)

**EPSR Parameters:**
- `max_iterations` (int): Maximum iterations (default: 50)
- `convergence_tol` (float): χ² convergence threshold (default: 250.0)
- `learning_rate` (float): Feedback parameter α (default: 0.3)
- `max_amplitude` (float): Maximum U_EP amplitude in kcal/mol (default: 3.0)
- `smooth_sigma` (float): Gaussian smoothing width in Å (default: 0.3)

**Optimization:**
- `use_momentum` (bool): Use momentum-based updates (default: True)
- `momentum_beta` (float): Momentum coefficient (default: 0.9)
- `method` (str): 'simple', 'momentum', or 'nesterov' (default: 'momentum')

**Experimental Data:**
- `sigma_exp` (float): Experimental uncertainty for χ² (default: 0.05)

**LAMMPS:**
- `use_gpu` (bool): Enable GPU/Kokkos (default: False)
- `gpu_id` (int): GPU device ID (default: 0)

## Command-Line Options

```
python scripts/run_epsr.py --help

Optional arguments:
  --exp-data PATH          Experimental g(r) data file
  --lammps-input PATH      LAMMPS input script
  --temperature FLOAT      Temperature (K)
  --density FLOAT          Number density (atoms/Å³)
  --max-iter INT           Maximum iterations
  --alpha FLOAT            Learning rate
  --tolerance FLOAT        Convergence tolerance (χ²)
  --method {simple,momentum,nesterov}
  --beta FLOAT             Momentum coefficient
  --max-amplitude FLOAT    Max potential amplitude
  --gpu                    Use GPU acceleration
  --gpu-id INT             GPU device ID
  --output-dir PATH        Output directory
  --quiet                  Suppress detailed output
```

## Output Files

All outputs are saved to the `outputs/` directory:

- `epsr_iter###.png`: Summary plots for each iteration
- `lammps_iter###.log`: LAMMPS log files
- `final_ep.npz`: Final empirical potentials and convergence history

Potential tables are written to `data/`:

- `ep_GaGa.table`: Ga-Ga empirical potential
- `ep_InIn.table`: In-In empirical potential
- `ep_GaIn.table`: Ga-In empirical potential

## Theoretical Background

### EPSR Algorithm

EPSR refines atomic structures by iteratively adjusting empirical potentials until simulated structure factors match experimental data.

**Key Steps:**

1. Run simulation with reference potential (RP) + empirical potential (EP)
2. Calculate structure factor S(Q) from simulation
3. Compare with experimental S_exp(Q)
4. Update EP based on difference: ΔU(Q) ∝ ΔS(Q)
5. Fourier transform to get ΔU(r)
6. Repeat until convergence

### Structure Factor Transformations

**Forward transform (g(r) → S(Q)):**

```
S(Q) - 1 = 4πρ ∫[g(r) - 1] r² sin(Qr)/(Qr) dr
```

**Inverse transform (S(Q) → g(r)):**

```
g(r) - 1 = 1/(2π²ρr) ∫[S(Q) - 1] Q sin(Qr) dQ
```

### Multi-Component Systems

For binary systems (e.g., Ga-In), the total structure factor is:

```
S(Q) = Σᵢⱼ wᵢⱼ Sᵢⱼ(Q)
```

where weights are:

```
wᵢⱼ = cᵢcⱼbᵢbⱼ / [Σₖcₖbₖ]²
```

- cᵢ = concentration of species i
- bᵢ = neutron scattering length of species i

## Comparison with Legacy Implementation

| Feature | Legacy (`main_epsr.py`) | New Implementation |
|---------|------------------------|-------------------|
| Algorithm | Simplified (kT·Δg) | Proper EPSR with S(Q) |
| Structure | Monolithic script | Modular package |
| Fourier Transforms | ❌ None | ✅ Full support |
| Partial RDFs | Basic | Automatic weighting |
| Optimization | Momentum only | 3 methods available |
| Testing | Manual | Unit tests |
| API | Script-based | Both script & library |
| Documentation | Minimal | Comprehensive |

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Test specific modules:

```bash
python -m pytest tests/unit/test_structure_factor.py -v
python -m pytest tests/unit/test_potential.py -v
```

## Migration from Legacy Code

If you're currently using `scripts/main_epsr.py`:

1. **Old way:**
   ```bash
   python scripts/main_epsr.py
   ```

2. **New way:**
   ```bash
   python scripts/run_epsr.py
   ```

The new implementation provides the same functionality with:
- Better convergence (proper EPSR algorithm)
- Cleaner code organization
- More customization options
- Better error handling

Legacy scripts are kept in `scripts/` for reference but new projects should use the modular implementation.

## References

1. **Soper, A. K. (1996).** "Empirical potential Monte Carlo simulation of fluid structure"
   *Chemical Physics*, 202, 295-306.

2. **Soper, A. K. (2005).** "Partial structure factors from disordered materials diffraction data"
   *Physical Review B*, 72, 104204.

3. **Youngs, T.G.A. et al. (2019).** "Dissolve: next generation software for the interrogation of total scattering data"
   *Molecular Physics*, 117:22, 3464-3477.

## Troubleshooting

### χ² not converging

Try adjusting:
- **Learning rate**: Reduce `--alpha` to 0.1-0.2 for stability
- **Tolerance**: Increase `--tolerance` if target is too strict
- **Method**: Try `--method nesterov` for difficult cases

### LAMMPS errors

- Check potential table files exist in `data/`
- Verify LAMMPS input file is correct
- For GPU errors, try CPU mode (remove `--gpu`)

### Import errors

Make sure you're running from the project root:
```bash
cd /home/yuki/lammps_settings_obata/hirataken20251122-2
python scripts/run_epsr.py
```

## Contributing

This implementation follows scientific software best practices:

- Modular design with clear interfaces
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for core functionality
- Version control with git

## License

See project root LICENSE file.

## Support

For issues or questions:
- Check this README and module docstrings
- Review example usage in `scripts/run_epsr.py`
- Consult Soper's original EPSR papers
- Examine Dissolve documentation for algorithm details

---

**Last updated**: 2025-11-29
**Version**: 1.0.0
