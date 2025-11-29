# Project Structure

Clean, organized structure for EPSR implementation.

```
.
├── epsr/                      # 📦 Main EPSR package
│   ├── __init__.py
│   ├── core/                  # Core algorithms
│   │   ├── structure_factor.py
│   │   ├── potential.py
│   │   └── epsr_engine.py
│   ├── io/                    # Input/output
│   │   ├── experimental.py
│   │   ├── lammps.py
│   │   └── tables.py
│   ├── analysis/              # Analysis tools
│   │   ├── rdf.py
│   │   └── metrics.py
│   └── visualization/         # Plotting
│       └── plots.py
│
├── scripts/                   # 🚀 Entry points
│   └── run_epsr.py           # Main CLI script
│
├── data/                      # 📊 Data files
│   ├── g_exp_cleaned.dat     # Experimental g(r)
│   └── ep_*.table            # Empirical potentials (generated)
│
├── inputs/                    # ⚙️ LAMMPS inputs
│   ├── in.egain_epsr_H100    # GPU input
│   ├── in.egain_epsr         # CPU input
│   └── initial_structure.data
│
├── outputs/                   # 📈 Results (generated)
│   ├── epsr_iter###.png
│   ├── lammps_iter###.log
│   └── final_ep.npz
│
├── tests/                     # 🧪 Tests
│   ├── test_basic.py         # Basic tests (no pytest)
│   └── unit/                 # Unit tests (requires pytest)
│
├── legacy/                    # 🗄️ Old files (reference only)
│   ├── README.md
│   ├── docs/
│   ├── scripts/
│   └── ...
│
├── README.md                  # 📖 Main documentation
├── NEW_EPSR_README.md        # 📚 Detailed guide
├── QUICK_MIGRATION_GUIDE.md  # 🔄 Migration guide
└── pyproject.toml            # 📦 Project config
```

## Directory Purposes

### Production (Use these)

- **epsr/** - Production-ready EPSR package with proper algorithms
- **scripts/run_epsr.py** - Main entry point for running EPSR
- **data/** - Experimental data and generated potentials
- **inputs/** - LAMMPS input files
- **outputs/** - Generated results and plots
- **tests/** - Test suite

### Reference (Don't modify)

- **legacy/** - Old implementation and experiments, kept for reference

## Quick Commands

```bash
# Run EPSR
python scripts/run_epsr.py

# Run tests
python tests/test_basic.py

# View structure
ls -R epsr/

# Clean outputs
rm -rf outputs/*.png outputs/*.log
```

## File Counts

- **14 Python modules** in epsr/ package
- **~2,400 lines** of clean code
- **1 main script** (run_epsr.py)
- **3 documentation files** (README.md, NEW_EPSR_README.md, QUICK_MIGRATION_GUIDE.md)

## What's Where

| What | Where |
|------|-------|
| EPSR algorithms | `epsr/core/` |
| Data I/O | `epsr/io/` |
| Analysis tools | `epsr/analysis/` |
| Plotting | `epsr/visualization/` |
| Main script | `scripts/run_epsr.py` |
| Tests | `tests/` |
| Experimental data | `data/g_exp_cleaned.dat` |
| LAMMPS inputs | `inputs/in.egain_epsr*` |
| Results | `outputs/` |
| Old code | `legacy/` |

## Keep or Delete?

### Keep
- Everything except `legacy/`
- `outputs/` can be cleaned periodically

### Can Delete (if needed)
- `legacy/` - after confirming you don't need old code
- `outputs/*` - generated files, can regenerate
- `__pycache__/` - Python cache, auto-generated

### Never Delete
- `epsr/` - the main package
- `scripts/run_epsr.py` - main entry point
- `data/g_exp_cleaned.dat` - experimental data
- `inputs/` - LAMMPS input files
- `tests/` - test suite
- Documentation files
