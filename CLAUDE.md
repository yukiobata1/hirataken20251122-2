# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Empirical Potential Structure Refinement (EPSR) implementation for liquid Ga-In eutectic alloy (EGaIn), based on neutron scattering data. Uses LAMMPS for MD simulations and iteratively refines empirical potentials to match experimental structure factors.

**Target system**: Ga₀.₈₅₈In₀.₁₄₂ at 150°C (423.15 K), 1000 atoms

## Directory Structure

```
.
├── epsr/                    # Main EPSR package (production code)
│   ├── core/               # Core algorithms (epsr_engine, structure_factor, potential)
│   ├── io/                 # I/O handlers (lammps, experimental, tables)
│   ├── analysis/           # Analysis tools (rdf, metrics)
│   └── visualization/      # Plotting (plots)
├── scripts/                 # All executable scripts
│   ├── run_epsr.py         # Main CLI entry point ⭐
│   ├── build_lammps_kokkos.sh  # LAMMPS build script
│   └── ...                 # Other utility scripts
├── data/                    # Data files
│   ├── g_exp_cleaned.dat   # Experimental g(r) data
│   └── ep_*.table          # Empirical potential tables
├── inputs/                  # LAMMPS input files
│   ├── in.egain_epsr_H100  # GPU input
│   └── in.egain_epsr       # CPU input
├── outputs/                 # Generated results (auto-generated)
│   └── grid_search/        # Grid search results
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── NEW_EPSR_README.md  # Complete guide
│   ├── QUICK_REFERENCE.md  # Quick reference
│   └── ...
└── legacy/                  # Old implementation (reference only)
    └── outputs/            # Old outputs
```

## Common Commands

```bash
# Run EPSR simulation
python scripts/run_epsr.py

# Run with GPU acceleration
python scripts/run_epsr.py --gpu

# Custom parameters
python scripts/run_epsr.py --max-iter 30 --alpha 0.5 --method momentum

# Run tests
python tests/test_basic.py
python -m pytest tests/ -v

# Build LAMMPS with Kokkos
./scripts/build_lammps_kokkos.sh

# Verify package
python -c "from epsr import EPSREngine; print('OK')"
```

## Architecture

**Core workflow**: `scripts/run_epsr.py` → `EPSREngine.run()` → runs LAMMPS, calculates S(Q), updates potentials until convergence.

Key classes:
- `EPSREngine`: Main workflow orchestrator
- `EPSRConfig`: Configuration dataclass
- `StructureFactor`: S(Q) ↔ g(r) Fourier transforms
- `EmpiricalPotential`: Potential management

## Dependencies

- Python 3.11+
- numpy, scipy, matplotlib
- LAMMPS (must be in PATH as `lmp`)
- Optional: freud-analysis, mdanalysis

## Notes

- `legacy/` contains old implementation for reference only; do not modify
- Algorithm uses Soper's S(Q)-based EPSR method with Fourier transforms
- Run scripts from project root directory
