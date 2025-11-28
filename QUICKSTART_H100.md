# Quick Start: EPSR with H100 GPU

3-step guide to run EPSR for EGaIn on NVIDIA H100.

## Step 1: Verify Setup

```bash
bash scripts/test_H100_setup.sh
```

This checks:
- ✓ H100 GPU availability
- ✓ CUDA installation
- ✓ LAMMPS with Kokkos
- ✓ Project files
- ✓ Python dependencies

**If LAMMPS is not installed**:
```bash
bash scripts/build_lammps_kokkos.sh  # Takes 10-20 min
```

## Step 2: Test Run (Optional but Recommended)

```bash
bash scripts/run_epsr_H100.sh --test-lj
```

This runs a quick LJ-only simulation (~1 minute) to verify:
- GPU is working
- LAMMPS Kokkos is configured correctly
- Initial structure is valid

Check output: `rdf.dat` should contain g(r) data.

## Step 3: Run Full EPSR

```bash
bash scripts/run_epsr_H100.sh
```

This runs the complete EPSR workflow:
- 50 iterations (default)
- ~1 hour on H100 (vs ~8 hours on CPU)
- Auto-saves results to `outputs/`

### Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check latest iteration
ls -lt outputs/epsr_iter*.png | head -1

# View convergence
tail -f outputs/lammps_iter*.log
```

## Results

After completion:
```
outputs/
├── epsr_final.png           # Final g(r) comparison
├── final_ep.npz             # Converged potentials
├── epsr_iter001.png         # Iteration plots
├── epsr_iter002.png
└── ...
```

View final chi-squared:
```bash
python3 -c "import numpy as np; d=np.load('outputs/final_ep.npz'); print(f'χ² = {d[\"chi2_history\"][-1]:.6f}')"
```

## Common Commands

### Run with Different Settings

Edit `scripts/main_epsr.py`:
```python
# Line ~210-222
max_iter = 50           # Change max iterations
alpha = 0.3             # Change learning rate
use_gpu = True          # Toggle GPU on/off
gpu_id = 0              # Select GPU (if multiple)
```

### Manual LAMMPS Run

```bash
# LJ test
lmp -k on g 0 -sf kk -in inputs/in.egain_lj_H100

# EPSR (single iteration)
lmp -k on g 0 -sf kk -in inputs/in.egain_epsr_H100
```

### CPU Mode (No GPU)

```bash
# Use original input files
lmp -in inputs/in.egain_lj

# Or disable GPU in main_epsr.py
python3 scripts/main_epsr.py  # (set use_gpu=False)
```

## Troubleshooting

### "Command 'lmp' not found"
```bash
bash scripts/build_lammps_kokkos.sh
source ~/.bashrc
```

### "KOKKOS package not available"
```bash
lmp -help | grep KOKKOS  # Should show KOKKOS package
# If not, rebuild:
bash scripts/build_lammps_kokkos.sh
```

### Simulation crashes or is slow
```bash
# Check if GPU is actually being used
nvidia-smi  # GPU-Util should be >0% during run

# Check LAMMPS log for Kokkos info
grep -i kokkos outputs/lammps_iter001.log
# Should see: "KOKKOS mode is enabled"
```

### Out of memory
Reduce system size:
```bash
python3 scripts/create_initial_structure.py -n 500
```

## File Overview

| File | Purpose |
|------|---------|
| `scripts/test_H100_setup.sh` | Verify installation |
| `scripts/run_epsr_H100.sh` | One-command runner |
| `scripts/main_epsr.py` | EPSR main loop (GPU-aware) |
| `inputs/in.egain_lj_H100` | GPU-optimized LJ test |
| `inputs/in.egain_epsr_H100` | GPU-optimized EPSR |
| `EPSR_H100_README.md` | Detailed documentation |

## Expected Timeline

| Task | H100 Time | CPU Time |
|------|-----------|----------|
| Setup check | 10 sec | - |
| LJ test | 1 min | 5 min |
| Full EPSR (50 iter) | 1 hr | 8-10 hr |

## Next Steps

After EPSR converges:

1. **Analyze Results**:
   ```bash
   python3 -c "
   import numpy as np
   import matplotlib.pyplot as plt
   d = np.load('outputs/final_ep.npz')
   plt.plot(d['chi2_history'])
   plt.yscale('log')
   plt.xlabel('Iteration')
   plt.ylabel('χ²')
   plt.savefig('convergence.png')
   print(f'Final χ²: {d[\"chi2_history\"][-1]:.6f}')
   "
   ```

2. **Use Refined Potentials**:
   - Load from `outputs/final_ep.npz`
   - Contains: `U_ep_GaGa`, `U_ep_InIn`, `U_ep_GaIn`
   - Already in LAMMPS table format in `data/ep_*.table`

3. **Run Production Simulations**:
   - Use converged `ep_*.table` files
   - Run longer simulations with refined potentials
   - Calculate properties (diffusion, viscosity, etc.)

## Getting Help

- **Setup issues**: Run `bash scripts/test_H100_setup.sh`
- **EPSR theory**: See `plans/espr_lammps_guide.md`
- **H100 details**: See `EPSR_H100_README.md`
- **LAMMPS Kokkos**: https://docs.lammps.org/Speed_kokkos.html

## Summary

```bash
# Complete workflow in 3 commands:
bash scripts/test_H100_setup.sh           # Verify (10 sec)
bash scripts/run_epsr_H100.sh --test-lj   # Test (1 min)
bash scripts/run_epsr_H100.sh             # Run (1 hr)
```

That's it! 🚀
