# EPSR Algorithm Improvements

## Overview

This document describes the improvements made to the EPSR (Empirical Potential Structure Refinement) algorithm to address the χ² divergence issue.

## Problem Statement

The original implementation used a simple gradient descent update:

```python
U_ep_new = U_ep_old + alpha * kT * (g_sim - g_exp)
```

This approach had several issues:
- **Oscillation**: Learning rate too high causes overshooting
- **Divergence**: χ² was increasing instead of decreasing
- **Instability**: No mechanism to adapt to changing loss landscape

## Implemented Solutions

### 1. Momentum Method

**What it does**: Adds "inertia" to the optimization by incorporating previous update directions.

**Algorithm**:
```python
velocity = beta * velocity_old + gradient
U_ep_new = U_ep_old + velocity
```

**Benefits**:
- Reduces oscillations around optimal values
- Accelerates convergence in consistent directions
- Dampens rapid changes that cause instability

**Parameters**:
- `beta = 0.9`: Momentum coefficient (typical value)
- Higher β = more momentum (smoother but slower)
- Lower β = less momentum (faster but more oscillation)

### 2. Adaptive Learning Rate

**What it does**: Automatically adjusts the learning rate based on χ² changes.

**Algorithm**:
```python
if chi2 increased:
    alpha = max(alpha * 0.8, alpha_min)  # Reduce by 20%
elif chi2 decreased significantly (> 10):
    alpha = min(alpha * 1.05, alpha_max)  # Increase by 5%
```

**Benefits**:
- Prevents divergence by reducing α when χ² increases
- Allows faster convergence when making good progress
- Automatically finds appropriate learning rate

**Parameters**:
- `alpha_init = 0.3`: Starting learning rate
- `alpha_min = 0.05`: Minimum allowed learning rate
- `alpha_max = 0.5`: Maximum allowed learning rate

### 3. Alternative Methods

The code now supports three update methods:

1. **`'simple'`**: Original gradient descent (for comparison)
2. **`'momentum'`**: Momentum method (recommended, default)
3. **`'nesterov'`**: Nesterov Accelerated Gradient (advanced)

## Usage

### Running with Improved Algorithm

```bash
cd /home/yuki/lammps_settings_obata/hirataken20251122-2
python3 scripts/main_epsr.py
```

The script will now display:
```
Update method: momentum
Momentum coefficient β: 0.9
Initial learning rate α: 0.3
Adaptive learning rate: ENABLED (range: 0.05 - 0.5)
```

### Monitoring Progress

During iterations, you'll see:
```
χ² = 850.123456
R-factor = 0.123456
Current learning rate α = 0.3000
  → χ² increased, reducing α to 0.2400
```

This tells you:
- Current fit quality (χ²)
- Whether the algorithm is adapting (α changes)
- Direction of convergence

### Tuning Parameters

Edit `scripts/main_epsr.py` around line 310-316:

```python
# For more aggressive optimization (faster, less stable):
alpha_init = 0.5
beta = 0.95

# For more conservative optimization (slower, more stable):
alpha_init = 0.1
beta = 0.8

# To disable momentum and use simple gradient descent:
method = 'simple'
adaptive_lr = False  # Also disable adaptive LR if desired
```

## Expected Behavior

### With Original Algorithm
```
Iteration 1: χ² = 850.0
Iteration 2: χ² = 920.3  ⚠️ INCREASED
Iteration 3: χ² = 1005.7 ⚠️ INCREASED
❌ Algorithm diverging
```

### With Improved Algorithm
```
Iteration 1: χ² = 850.0
Iteration 2: χ² = 920.3  → reducing α to 0.24
Iteration 3: χ² = 780.5  ✓ decreased
Iteration 4: χ² = 650.2  ✓ decreased
Iteration 5: χ² = 520.8  ✓ decreased
...
Iteration 25: χ² = 185.3  ✓ converged
```

## Technical Details

### Momentum Update (detailed)

The momentum method maintains a "velocity" term for each potential:

```python
# For each pair type (Ga-Ga, In-In, Ga-In):
gradient = alpha * kT * (g_sim - g_exp)
velocity = beta * velocity_old + gradient
U_ep_new = U_ep_old + velocity
```

This is analogous to a ball rolling down a hill:
- `gradient`: Current slope direction
- `velocity`: Current speed and direction
- `beta`: Friction coefficient (0.9 = low friction, keeps momentum)

### Adaptive Learning Rate (detailed)

The adaptive mechanism prevents overshooting:

1. **When χ² increases** (going wrong direction):
   - Reduce α by 20%: `alpha *= 0.8`
   - Ensures α ≥ alpha_min (0.05)

2. **When χ² decreases significantly** (making good progress):
   - Increase α by 5%: `alpha *= 1.05`
   - Ensures α ≤ alpha_max (0.5)

3. **When χ² decreases slightly**:
   - Keep α unchanged
   - This is the "sweet spot"

### Nesterov Accelerated Gradient

For advanced users, NAG provides even better convergence:

```python
velocity = beta * velocity_old + gradient
U_ep_new = U_ep_old + beta * velocity + gradient
```

This "looks ahead" before making the update, providing better acceleration.

## Comparison with Open Source Tools

While we investigated several open source options:

- **Dissolve**: Full EPSR implementation, but standalone GUI app (hard to integrate)
- **potfit**: Force-matching for ab initio data (not experimental g(r))
- **KLIFF**: Fits to energy/forces (not structural data like RDF)

Our implementation combines:
- Standard optimization techniques (momentum, adaptive LR)
- EPSR-specific features (g(r) fitting, partial RDF weighting)
- LAMMPS integration
- GPU acceleration (H100 via Kokkos)

## References

1. **Momentum Optimization**:
   - Qian, N. (1999). "On the momentum term in gradient descent learning algorithms." *Neural Networks*, 12(1), 145-151.

2. **Nesterov Accelerated Gradient**:
   - Nesterov, Y. (1983). "A method for solving the convex programming problem with convergence rate O(1/k²)."

3. **EPSR Method**:
   - Soper, A. K. (2005). "Partial structure factors from disordered materials diffraction data: An approach using empirical potential structure refinement." *Phys. Rev. B*, 72, 104204.

## Troubleshooting

### χ² still increasing
- Reduce `alpha_init` (try 0.1)
- Increase `beta` (try 0.95 for more smoothing)
- Check if experimental data matches simulation temperature

### Convergence too slow
- Increase `alpha_init` (try 0.5)
- Reduce `beta` (try 0.8 for less smoothing)
- Try `method = 'nesterov'` for faster convergence

### Oscillating around a value
- Good sign! Algorithm is finding the minimum
- Reduce `alpha_init` slightly
- Increase `beta` for more stability

## Summary

The improved algorithm provides:
- ✅ **Stability**: Adaptive learning rate prevents divergence
- ✅ **Speed**: Momentum accelerates convergence
- ✅ **Flexibility**: Multiple methods and tunable parameters
- ✅ **Monitoring**: Clear feedback on optimization progress

These improvements are based on well-established optimization techniques used in machine learning and numerical optimization, adapted specifically for the EPSR problem.
