# GPU Optimization Guide for EPSR on H100

## System Size Recommendations

### For H100 GPU

| Atoms | Box Size | GPU Utilization | Time/Iteration | Recommendation |
|-------|----------|-----------------|----------------|----------------|
| 1,000 | ~27 Å | ~10% | Very fast | ❌ Too small - GPU underutilized |
| 5,000 | ~47 Å | ~30% | Fast | ⚠️ Still small |
| **10,000** | ~59 Å | ~50% | Medium | ✅ Good minimum |
| **20,000** | ~74 Å | ~80% | Balanced | ✅ **Recommended** |
| **50,000** | ~108 Å | ~95% | Slower | ✅ Best GPU utilization |
| 100,000 | ~136 Å | ~98% | Slow | ⚠️ May be overkill for EPSR |

### Current Setup

Default configuration is now **20,000 atoms** - optimal balance between:
- Good GPU utilization (~80%)
- Reasonable iteration time
- Sufficient statistics for EPSR

## Creating Different Sizes

```bash
# Small test (quick iterations, low GPU usage)
python3 scripts/create_initial_structure.py -n 5000

# Recommended (default)
python3 scripts/create_initial_structure.py -n 20000

# Maximum performance (slower but best GPU utilization)
python3 scripts/create_initial_structure.py -n 50000
```

## Performance Comparison

### 20,000 atoms (Recommended)

**Pros:**
- 80% GPU utilization
- Good statistics for g(r)
- Balanced iteration time
- EPSR converges in ~1 hour

**Cons:**
- Still room for GPU optimization

### 50,000 atoms (Maximum)

**Pros:**
- 95%+ GPU utilization
- Excellent statistics
- Maximum accuracy

**Cons:**
- Slower iterations
- EPSR may take 2-3 hours
- Higher memory usage

### 1,000 atoms (Original - NOT recommended for GPU)

**Pros:**
- Very fast iterations
- Low memory

**Cons:**
- ❌ Only ~10% GPU utilization
- ❌ Wastes H100 resources
- ❌ Poor statistics
- ⚠️ Only use for CPU testing

## Memory Requirements

| Atoms | RAM | GPU Memory | Notes |
|-------|-----|------------|-------|
| 10,000 | ~1 GB | ~2 GB | Safe for all GPUs |
| 20,000 | ~2 GB | ~4 GB | H100 (80GB) - no problem |
| 50,000 | ~5 GB | ~10 GB | H100 (80GB) - plenty of room |
| 100,000 | ~10 GB | ~20 GB | Still fine on H100 |

## LAMMPS Settings for Large Systems

For systems >20,000 atoms, optimize neighbor list:

```lammps
# Optimized for large systems on H100
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes binsize 5.0  # Increase binsize
```

## When to Use Each Size

### Development/Testing: 5,000-10,000 atoms
```bash
python3 scripts/create_initial_structure.py -n 5000
```
Fast iterations for code testing.

### Production EPSR: 20,000 atoms (Default)
```bash
python3 scripts/create_initial_structure.py -n 20000
```
Best balance for actual research.

### High-Accuracy: 50,000 atoms
```bash
python3 scripts/create_initial_structure.py -n 50000
```
When you need the best statistics and have time.

## Verifying GPU Usage

During simulation, check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Good utilization:
- GPU-Util: >70%
- Memory-Usage: Several GB
- Power: >200W (H100)

Poor utilization (system too small):
- GPU-Util: <30%
- Memory-Usage: <1 GB
- Power: <100W

## Rule of Thumb

**For H100:** Aim for **at least 10,000 atoms**, preferably **20,000+**

The extra atoms pay off in:
1. Better GPU utilization (80%+ vs 10%)
2. Better statistics for g(r)
3. More physical realism
4. Actually using the H100's capabilities

## Updated Defaults

All scripts now default to **20,000 atoms**:
- `create_initial_structure.py`: `-n 20000` (default)
- Optimized for H100
- Can override with `-n` flag if needed

---

**Bottom Line:** Don't use 1,000 atoms on H100. It's like using a Ferrari to go to the corner store. Use at least 20,000! 🚀
