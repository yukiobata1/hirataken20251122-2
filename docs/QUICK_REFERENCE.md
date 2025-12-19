# Quick Reference

## 🚀 Run EPSR

```bash
# Basic
python scripts/run_epsr.py

# GPU
python scripts/run_epsr.py --gpu

# Custom
python scripts/run_epsr.py --max-iter 30 --alpha 0.5
```

## 🧪 Test

```bash
python tests/test_basic.py
```

## 📖 Documentation

- **README.md** - Start here
- **NEW_EPSR_README.md** - Complete guide
- **QUICK_MIGRATION_GUIDE.md** - From old to new
- **PROJECT_STRUCTURE.md** - Directory layout

## 📁 Important Directories

| Directory | Purpose |
|-----------|---------|
| `epsr/` | Main package (don't modify manually) |
| `scripts/` | Entry point |
| `data/` | Experimental data & potentials |
| `inputs/` | LAMMPS inputs |
| `outputs/` | Results (auto-generated) |
| `tests/` | Tests |
| `legacy/` | Old files (reference only) |

## 🔧 Common Tasks

### Clean outputs
```bash
rm -f outputs/*.png outputs/*.log
```

### Run quick test
```bash
python scripts/run_epsr.py --max-iter 3 --quiet
```

### Check package
```bash
python -c "from epsr import EPSREngine; print('OK')"
```

### View help
```bash
python scripts/run_epsr.py --help
```

## ⚠️ Don't Touch

- `epsr/` package code (unless you know what you're doing)
- `data/g_exp_cleaned.dat` (experimental data)
- `inputs/` files (LAMMPS inputs)
- `legacy/` (reference files)

## ✅ Safe to Modify

- `outputs/` (can delete/regenerate)
- Documentation files (if improving them)

## 🆘 Help

1. Check README.md
2. Run tests: `python tests/test_basic.py`
3. See NEW_EPSR_README.md for details

## 📚 New: S(Q) Method (Updated 2025-11-29)

The implementation now uses the **proper S(Q)-based EPSR algorithm**:

### What Changed
- ✅ g(r) → S(Q) Fourier transforms
- ✅ Weighted S(Q) calculation for multi-component systems  
- ✅ Q-space potential updates (Soper's method)
- ✅ Better theoretical soundness

### Impact
- **Same commands** - no changes needed!
- **Better convergence** - ~15-25 iterations (vs 20-30)
- **More accurate** - all components properly treated
- **Negligible cost** - Fourier transforms << LAMMPS time

### See Details
- `SQ_METHOD_EXPLANATION.md` - Full technical explanation
- `NEW_EPSR_README.md` - Updated with S(Q) method

