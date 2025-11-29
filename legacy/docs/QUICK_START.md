# Quick Start: Improved EPSR Algorithm

## What's Changed

The EPSR algorithm has been improved with **Momentum optimization** to prevent χ² divergence.

### Test Results

We tested 5 different optimization methods on a synthetic g(r) fitting problem:

| Method | χ² Reduction | Speed |
|--------|--------------|-------|
| **Momentum** ⭐ | **99.3%** | Fast |
| **Nesterov** ⭐ | **99.3%** | Fast |
| Simple (original) | 95.6% | Medium |
| Momentum + Adaptive LR | 95.5% | Slow |
| Simple + Adaptive LR | 78.5% | Very Slow |

**Recommendation**: Use **Momentum** method (default in `main_epsr.py`)

## Running EPSR

### Basic Usage

```bash
cd /home/yuki/lammps_settings_obata/hirataken20251122-2
python3 scripts/main_epsr.py
```

### Expected Output

```
============================================================
EPSR for EGaIn System (H100 GPU-Accelerated)
============================================================
GPU Mode: ENABLED (H100 via Kokkos)
GPU Device: 0
Max iterations: 50
Update method: MOMENTUM
Momentum coefficient β: 0.9
Learning rate α: 0.3 (fixed)
Adaptive learning rate: DISABLED (momentum provides stability)
============================================================
```

### Monitor Progress

During each iteration, you'll see:

```
ITERATION 1/50
============================================================
  Running LAMMPS...
  Step 70000 | T= 423.15K | PE=-1234.5 | elapsed: 45.2s
  LAMMPS completed
χ² = 850.123456
R-factor = 0.123456
U_EP ranges: Ga-Ga [-0.123, 0.456], In-In [...], Ga-In [...]
```

**Key metrics**:
- **χ²**: Should **decrease** over iterations
- **R-factor**: Relative error (lower is better)
- **U_EP ranges**: Empirical potential values

## Configuration Options

Edit `scripts/main_epsr.py` (lines 320-329) to tune the optimizer:

### Option 1: Default (Recommended)

```python
method = 'momentum'
beta = 0.9
alpha = 0.3
adaptive_lr = False
```

**Use when**: You want fast, stable convergence (99.3% χ² reduction in tests)

### Option 2: More Aggressive

```python
method = 'nesterov'    # Slightly faster than momentum
beta = 0.95            # More momentum
alpha = 0.4            # Higher learning rate
adaptive_lr = False
```

**Use when**: Convergence is too slow, and you want to take risks

### Option 3: Conservative (if χ² oscillates)

```python
method = 'momentum'
beta = 0.9
alpha = 0.2            # Lower learning rate
adaptive_lr = True     # Enable safety net
```

**Use when**: You see χ² increasing or wild oscillations

### Option 4: Original (not recommended)

```python
method = 'simple'
beta = 0.9             # (unused)
alpha = 0.2
adaptive_lr = False
```

**Use when**: Debugging or comparing with old results

## Testing the Algorithm

To verify the optimization algorithm works correctly:

```bash
python3 scripts/test_algorithms.py
```

This runs a synthetic test comparing all methods and generates:
- `outputs/algorithm_comparison.png`: Convergence plots
- Console output with performance summary

## Troubleshooting

### χ² is increasing

**Solution 1**: Enable adaptive learning rate
```python
adaptive_lr = True
```

**Solution 2**: Reduce learning rate
```python
alpha = 0.1  # or even 0.05
```

**Solution 3**: Check experimental data quality
```bash
cat data/g_exp_cleaned.dat
```

### Convergence too slow

**Solution 1**: Increase learning rate
```python
alpha = 0.5
```

**Solution 2**: Try Nesterov method
```python
method = 'nesterov'
```

### LAMMPS crashes

Check LAMMPS log files:
```bash
tail -100 outputs/lammps_iter001.log
```

Common issues:
- Atoms too close (reduce `max_amp`)
- Simulation unstable (reduce `alpha`)

## Files Generated

### During EPSR

- `outputs/epsr_iter###.png`: Results plot for each iteration
- `outputs/lammps_iter###.log`: LAMMPS log files
- `data/ep_*.table`: Current empirical potentials

### Final Results

- `outputs/epsr_final.png`: Final convergence plot
- `outputs/final_ep.npz`: Final empirical potentials (numpy format)

Load final results in Python:
```python
import numpy as np
data = np.load('outputs/final_ep.npz')
r = data['r']
U_ep_GaGa = data['U_ep_GaGa']
chi2_history = data['chi2_history']

import matplotlib.pyplot as plt
plt.plot(chi2_history)
plt.xlabel('Iteration')
plt.ylabel('χ²')
plt.yscale('log')
plt.show()
```

## Understanding the Algorithm

### What is Momentum?

Momentum optimization adds "inertia" to the update:

```python
# Without momentum (original):
U_ep = U_ep_old + alpha * gradient

# With momentum:
velocity = beta * velocity_old + alpha * gradient
U_ep = U_ep_old + velocity
```

**Analogy**: Like a ball rolling down a hill
- `gradient`: Current slope
- `velocity`: Ball's speed and direction
- `beta`: Friction (0.9 = low friction, keeps rolling)

**Benefits**:
- Reduces oscillations
- Accelerates convergence in consistent directions
- Dampens noise from simulation fluctuations

### Why not Adaptive LR?

Adaptive learning rate adjusts `α` based on χ² changes:
- χ² increases → reduce α
- χ² decreases → increase α

**Problem**: In noisy optimization (like EPSR with LAMMPS), χ² can temporarily increase due to:
- Statistical fluctuations in simulation
- RDF sampling noise
- Temperature fluctuations

Adaptive LR interprets this as "going wrong direction" and reduces α too aggressively, slowing convergence.

**Momentum handles noise better** by averaging over multiple steps.

## Advanced Usage

### Run multiple EPSR experiments

```bash
# Experiment 1: Momentum
python3 scripts/main_epsr.py
mv outputs/final_ep.npz outputs/final_ep_momentum.npz

# Experiment 2: Nesterov
# (edit main_epsr.py: method = 'nesterov')
python3 scripts/main_epsr.py
mv outputs/final_ep.npz outputs/final_ep_nesterov.npz

# Compare results
python3 -c "
import numpy as np
import matplotlib.pyplot as plt

d1 = np.load('outputs/final_ep_momentum.npz')
d2 = np.load('outputs/final_ep_nesterov.npz')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(d1['chi2_history'], label='Momentum')
plt.plot(d2['chi2_history'], label='Nesterov')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('χ²')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(d1['r'], d1['U_ep_GaGa'], label='Momentum')
plt.plot(d2['r'], d2['U_ep_GaGa'], '--', label='Nesterov')
plt.xlabel('r (Å)')
plt.ylabel('U_EP Ga-Ga (kcal/mol)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/method_comparison.png')
print('Saved to outputs/method_comparison.png')
"
```

## References

- **Algorithm improvements**: `ALGORITHM_IMPROVEMENTS.md`
- **Test code**: `scripts/test_algorithms.py`
- **Original EPSR method**: Soper, A. K. (2005). Phys. Rev. B, 72, 104204
- **Momentum optimization**: Qian, N. (1999). Neural Networks, 12(1), 145-151

## Summary

✅ **Momentum method** provides 99.3% χ² reduction (best performance)
✅ **No adaptive LR needed** (momentum handles stability)
✅ **Fixed α = 0.3** works well for most cases
✅ **Test before use**: Run `test_algorithms.py` to verify

**Next steps**: Run `python3 scripts/main_epsr.py` and watch χ² decrease!
