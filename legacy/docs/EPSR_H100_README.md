# EPSR for EGaIn with H100 GPU Acceleration

H100 GPU-optimized EPSR (Empirical Potential Structure Refinement) for liquid Ga-In eutectic alloy using LAMMPS with Kokkos.

## Quick Start

### 1. Test GPU Setup

```bash
# Test if H100 is available
nvidia-smi

# Verify LAMMPS has Kokkos support
lmp -help | grep KOKKOS
```

### 2. Run LJ-only Test (Recommended First)

```bash
bash scripts/run_epsr_H100.sh --test-lj
```

This runs a quick baseline simulation to verify your setup.

### 3. Run Full EPSR

```bash
bash scripts/run_epsr_H100.sh
```

## System Specifications

- **GPU**: NVIDIA H100 (Hopper architecture, sm_90)
- **Backend**: LAMMPS + Kokkos
- **System**: Ga₀.₈₅₈In₀.₁₄₂ (1000 atoms)
- **Temperature**: 150°C (423.15 K)

## Performance Optimization

### H100-Specific Settings

The simulation is optimized for H100 with:

1. **Kokkos Package**: `package kokkos neigh full newton off`
   - `newton off` is CRITICAL for H100 stability

2. **GPU-Accelerated Pair Styles**:
   - LJ potential: `lj/cut/kk` (runs on GPU)
   - Table potential: standard `table` (runs on CPU - no Kokkos support)

3. **Optimized Neighbor Lists**:
   - `binsize 4.0` for better GPU utilization
   - `neigh_modify delay 0 every 1 check yes`

### Execution Command

The script automatically runs LAMMPS with:
```bash
lmp -k on g 1 -sf kk -in inputs/in.egain_epsr_H100
```

Flags explained:
- `-k on g 1`: Enable Kokkos, use 1 GPU
- `-sf kk`: Apply Kokkos suffix to supported pair styles
- LJ potential automatically uses `lj/cut/kk` (GPU)
- Table potential remains on CPU (no /kk version available)

## File Structure

### H100-Optimized Files

```
inputs/
├── in.egain_lj_H100        # LJ-only test simulation (GPU)
└── in.egain_epsr_H100      # Full EPSR with GPU acceleration

scripts/
├── main_epsr.py            # Modified for GPU support
├── run_epsr_H100.sh        # One-command runner
└── build_lammps_kokkos.sh  # LAMMPS build script (if needed)
```

### Original CPU Files (still available)

```
inputs/
├── in.egain_lj            # CPU-only LJ simulation
└── in.egain_epsr          # CPU-only EPSR
```

## GPU vs CPU Mode

### Using GPU (Default)

```python
# In main_epsr.py (line ~221)
use_gpu = True
gpu_id = 0
```

### Using CPU Only

```python
# In main_epsr.py (line ~221)
use_gpu = False
```

Or run original files:
```bash
lmp -in inputs/in.egain_lj
```

## Installation

If LAMMPS with Kokkos is not installed:

```bash
bash scripts/build_lammps_kokkos.sh
```

This will:
1. Clone LAMMPS source
2. Configure with Kokkos for H100 (Hopper architecture)
3. Build and install to `~/.local/bin`
4. Takes ~10-20 minutes

### CUDA Requirements

- CUDA Toolkit 12.x or later
- NVIDIA driver supporting H100
- Check: `nvcc --version`

## Expected Performance

### GPU (H100) vs CPU

For 1000-atom EGaIn system:

| Stage              | CPU Time | H100 Time | Speedup |
|--------------------|----------|-----------|---------|
| Equilibration      | ~5 min   | ~30 sec   | ~10x    |
| Production (50k)   | ~10 min  | ~1 min    | ~10x    |
| Full EPSR (50 iter)| ~8-10 hr | ~1 hr     | ~8-10x  |

*Times are approximate and depend on convergence*

## Monitoring

### GPU Usage

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi
```

### Check LAMMPS Progress

```bash
# Tail latest log
tail -f outputs/lammps_iter*.log | tail -20
```

## Troubleshooting

### KOKKOS Errors

**Error**: "KOKKOS package is not available in this executable"
```bash
# Rebuild LAMMPS with Kokkos
bash scripts/build_lammps_kokkos.sh
```

**Error**: "CUDA error ... invalid configuration"
```bash
# Make sure newton is off in LAMMPS input
package kokkos neigh full newton off
```

### GPU Not Used

Check if LJ potential uses `/kk` suffix:
```bash
grep "pair_style" outputs/lammps_iter001.log
# Should see: pair_style lj/cut/kk
```

### Out of Memory

Reduce system size or use CPU mode:
```python
# In scripts/create_initial_structure.py
python3 scripts/create_initial_structure.py -n 500  # Smaller system
```

## Advanced: Multi-GPU

To use multiple H100s (if available):

```bash
# Use GPU 0 and 1
lmp -k on g 2 -sf kk -in inputs/in.egain_epsr_H100
```

Modify `main_epsr.py`:
```python
use_gpu = True
gpu_id = 1  # Or 0, 1, 2, ... for multi-GPU systems
```

## Benchmarking

Run benchmark:
```bash
# Time LJ simulation
time lmp -k on g 1 -sf kk -in inputs/in.egain_lj_H100 -log outputs/bench_gpu.log

# Compare with CPU
time lmp -in inputs/in.egain_lj -log outputs/bench_cpu.log
```

## Parameters

All parameters in `scripts/main_epsr.py`:

```python
# EPSR parameters
max_iter = 50           # Maximum iterations
alpha = 0.3             # Learning rate
tol = 0.1               # Convergence (χ²)
max_amp = 1.0           # U_EP limit (kcal/mol)

# GPU settings
use_gpu = True          # Enable H100
gpu_id = 0              # Which GPU to use
```

## References

1. **LAMMPS Kokkos Package**:
   https://docs.lammps.org/Speed_kokkos.html

2. **H100 (Hopper) Architecture**:
   - Compute Capability: 9.0
   - CUDA Arch: sm_90

3. **Amon et al. (2023)**:
   "Local Order in Liquid Gallium−Indium Alloys"
   *J. Phys. Chem. C*, 127, 16687-16694

## Getting Help

For issues specific to:
- **EPSR methodology**: See `plans/espr_lammps_guide.md`
- **LAMMPS Kokkos**: https://docs.lammps.org/Packages_details.html#pkg-kokkos
- **H100 setup**: Check `scripts/build_lammps_kokkos.sh`

## Summary of Changes from CPU Version

1. Added `package kokkos` directive (CRITICAL: `newton off`)
2. Input files: `in.egain_*_H100` variants
3. Execution: `lmp -k on g 1 -sf kk` instead of just `lmp`
4. LJ potential: Auto-converted to `lj/cut/kk`
5. Table potential: Remains CPU (no Kokkos version)
6. Performance: ~8-10x faster for full EPSR workflow
