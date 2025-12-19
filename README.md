# EPSR for EGaIn Liquid Metal Alloy

Modern implementation of Empirical Potential Structure Refinement (EPSR) for liquid Ga-In eutectic alloy system, based on neutron scattering data.

## 🚀 Quick Start

### New Implementation (Recommended)

```bash
# Basic run with default parameters
python scripts/run_epsr.py

# With GPU acceleration (if available)
python scripts/run_epsr.py --gpu

# Custom parameters
python scripts/run_epsr.py --max-iter 30 --alpha 0.5 --method momentum
```

### Legacy Implementation

```bash
# Old script (still functional)
python scripts/main_epsr.py
```

## 📚 Documentation

- **[docs/NEW_EPSR_README.md](docs/NEW_EPSR_README.md)** - Complete guide to new implementation
- **[docs/QUICK_MIGRATION_GUIDE.md](docs/QUICK_MIGRATION_GUIDE.md)** - Migrate from old to new
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick reference
- **[docs/SQ_METHOD_EXPLANATION.md](docs/SQ_METHOD_EXPLANATION.md)** - S(Q) method details
- **[legacy/docs/](legacy/docs/)** - Legacy documentation

## 🏗️ Project Structure

```
.
├── epsr/                      # Main EPSR package (production code)
│   ├── core/                  # Core algorithms
│   ├── io/                    # Input/output handlers
│   ├── analysis/              # RDF and metrics
│   └── visualization/         # Plotting tools
├── scripts/                   # All executable scripts
│   ├── run_epsr.py           # Main CLI entry point ⭐
│   └── build_lammps_kokkos.sh # LAMMPS build script
├── data/                      # Data files
│   ├── g_exp_cleaned.dat     # Experimental g(r) (150°C)
│   └── ep_*.table            # Empirical potential tables
├── inputs/                    # LAMMPS input files
├── outputs/                   # Results (auto-generated)
│   └── grid_search/          # Grid search results
├── tests/                     # Test suite
├── docs/                      # Documentation
└── legacy/                    # Old implementation (reference only)
    └── outputs/              # Old outputs
```

## ✨ Key Features

### New Implementation

- ✅ **Proper EPSR algorithm** with S(Q) structure factors (Soper 1996, 2005)
- ✅ **Fourier transforms** between g(r) and S(Q) spaces
- ✅ **Modular design** - clean, testable, extensible
- ✅ **Multiple optimization methods** - Simple, Momentum, Nesterov
- ✅ **Automatic convergence detection**
- ✅ **Partial RDF support** with proper neutron scattering weights
- ✅ **Q-space refinement** - theoretically correct multi-component handling
- ✅ **Python API** for custom workflows
- ✅ **Command-line interface** for easy use
- ✅ **Unit tests** for reliability

### System

- **Composition**: Ga₀.₈₅₈In₀.₁₄₂ (eutectic)
- **Temperature**: 150°C (423.15 K)
- **Atoms**: 1000 (858 Ga + 142 In)
- **Experimental data**: Neutron diffraction from Amon et al. (2023)

## 🔧 Installation

### Prerequisites

```bash
# Python packages
pip install numpy scipy matplotlib

# LAMMPS (must be in PATH)
lmp --version  # Verify installation
```

### Optional: GPU Support

For H100 GPU acceleration, ensure LAMMPS is built with Kokkos/CUDA support.

## 📖 Usage Examples

### Command Line

```bash
# See all options
python scripts/run_epsr.py --help

# Basic run
python scripts/run_epsr.py

# GPU mode with custom parameters
python scripts/run_epsr.py \
    --gpu \
    --max-iter 30 \
    --alpha 0.3 \
    --tolerance 200 \
    --method momentum

# Quiet mode
python scripts/run_epsr.py --quiet
```

### Python API

```python
from epsr import EPSREngine
from epsr.core.epsr_engine import EPSRConfig

# Configure
config = EPSRConfig(
    temperature=423.15,
    composition={'Ga': 0.858, 'In': 0.142},
    scattering_lengths={'Ga': 7.288, 'In': 4.061},
    density=0.042,
    max_iterations=50,
    learning_rate=0.3,
    use_gpu=True
)

# Run
engine = EPSREngine(config, 'data/g_exp_cleaned.dat')
results = engine.run(
    lammps_input='inputs/in.egain_epsr_H100',
    rdf_output='rdf.dat'
)

# Results
print(f"Converged: {results['converged']}")
print(f"Final χ²: {results['final_chi2']:.2f}")
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/unit/test_structure_factor.py -v
```

## 📊 Output Files

All results in `outputs/`:

- `epsr_iter###.png` - Summary plots for each iteration
- `lammps_iter###.log` - LAMMPS log files
- `final_ep.npz` - Final empirical potentials
- `epsr_final.png` - Final results plot

Potential tables in `data/`:

- `ep_GaGa.table` - Ga-Ga empirical potential
- `ep_InIn.table` - In-In empirical potential
- `ep_GaIn.table` - Ga-In empirical potential

## 🔬 Scientific Background

### EPSR Algorithm

EPSR iteratively refines atomic structures by adjusting empirical potentials until simulated structure factors match experimental data.

**Key equation:**
```
U_total(r) = U_reference(r) + U_empirical(r)
```

Where U_empirical is refined using:
```
ΔU(Q) ∝ [S_sim(Q) - S_exp(Q)]
```

### References

1. **Soper, A. K. (1996)** - Original EPSR methodology
   *Chemical Physics*, 202, 295-306

2. **Soper, A. K. (2005)** - Multi-component EPSR
   *Physical Review B*, 72, 104204

3. **Amon, A. et al. (2023)** - EGaIn experimental data
   *J. Phys. Chem. C*, 127(33), 16687-16694

4. **Youngs, T.G.A. et al. (2019)** - Dissolve (modern EPSR)
   *Molecular Physics*, 117:22, 3464-3477

## 🆚 Implementation Comparison

| Feature | Legacy | New |
|---------|--------|-----|
| Algorithm | Simplified kT·Δg | Proper EPSR with S(Q) Fourier transforms |
| Multi-component | Approximate | Correct weighted S(Q) |
| Theoretical basis | Ad-hoc | Soper (1996, 2005) |
| Structure | Monolithic script | Modular package |
| API | Script only | Script + Library |
| Testing | Manual | Automated |
| Optimization | Momentum | Simple/Momentum/Nesterov |
| Documentation | Minimal | Comprehensive |

## 🐛 Troubleshooting

### LAMMPS not found

```bash
# Check LAMMPS installation
which lmp

# If not found, install or add to PATH
export PATH=/path/to/lammps/bin:$PATH
```

### GPU errors

```bash
# Try CPU mode
python scripts/run_epsr.py  # without --gpu

# Check GPU availability
nvidia-smi
```

### Convergence issues

```bash
# Reduce learning rate
python scripts/run_epsr.py --alpha 0.1

# Try different method
python scripts/run_epsr.py --method nesterov

# Increase tolerance
python scripts/run_epsr.py --tolerance 300
```

### Import errors

```bash
# Run from project root
cd /home/yuki/lammps_settings_obata/hirataken20251122-2
python scripts/run_epsr.py
```

## 📈 Performance

**Typical performance on H100 GPU:**
- Iteration time: ~2-3 minutes
- Convergence: 15-25 iterations
- Total time: ~30-60 minutes
- Final χ²: ~200-300

**CPU mode:**
- Iteration time: ~10-15 minutes
- Total time: ~3-5 hours

## 🤝 Contributing

This implementation follows scientific software best practices:

- Modular architecture
- Type hints
- Comprehensive docstrings
- Unit tests
- Version control

## 📝 License

See LICENSE file.

## 🙋 Support

- **Documentation**: See `docs/NEW_EPSR_README.md` for details
- **Migration**: See `docs/QUICK_MIGRATION_GUIDE.md` to switch from old version
- **Quick Reference**: See `docs/QUICK_REFERENCE.md`

## 🎯 Next Steps

1. **New users**: Start with `python scripts/run_epsr.py`
2. **Existing users**: Read `docs/QUICK_MIGRATION_GUIDE.md`
3. **Developers**: See `docs/NEW_EPSR_README.md` for API details
4. **Advanced**: Explore `epsr/` package for customization

---

**Version**: 1.0.0
**Last updated**: 2025-11-29
**Status**: Production-ready ✅
