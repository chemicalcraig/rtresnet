# Data Format Specification

This document describes the data formats used by RT-ResNet.

## Input Data Files

### density_series.npy

Time series of density matrices from RT-TDDFT simulation.

**Shape**: `(N_steps, N_spin, N_bf, N_bf)`

| Dimension | Description |
|-----------|-------------|
| N_steps | Number of timesteps in the trajectory |
| N_spin | Number of spin channels (1=closed-shell, 2=open-shell) |
| N_bf | Number of atomic orbital basis functions |

**Data Type**: `complex128` or `complex64`

**Example**:
```python
import numpy as np

# Load density series
density = np.load('density_series.npy')
print(f"Shape: {density.shape}")  # (5000, 2, 4, 4) for H2+ with 5000 steps

# Access specific timestep and spin
rho_alpha_t0 = density[0, 0]  # Alpha density at t=0
rho_beta_t100 = density[100, 1]  # Beta density at t=100
```

**Properties**:
- Each `density[t, s]` is a Hermitian matrix: `rho = rho^dagger`
- Trace with overlap: `Tr(density[t, s] @ overlap) = N_electrons[s]`

### overlap.npy

Atomic orbital overlap matrix S.

**Shape**: `(N_bf, N_bf)`

**Data Type**: `float64` or `float32`

**Properties**:
- Symmetric: `S = S.T`
- Positive definite: all eigenvalues > 0
- Diagonal elements = 1.0 (normalized basis functions)

**Example**:
```python
import numpy as np

S = np.load('overlap.npy')
print(f"Shape: {S.shape}")  # (4, 4) for H2+

# Verify properties
print(f"Symmetric: {np.allclose(S, S.T)}")
print(f"Trace: {np.trace(S)}")  # Should equal N_bf
print(f"Min eigenvalue: {np.linalg.eigvalsh(S).min()}")  # Should be > 0
```

### field.npy

External electric field time series.

**Shape**: `(N_steps, 3)`

| Dimension | Description |
|-----------|-------------|
| N_steps | Number of timesteps (must match density_series) |
| 3 | Field components (Ex, Ey, Ez) |

**Data Type**: `float32`

**Units**: Atomic units (Hartree/bohr/e)

**Example**:
```python
import numpy as np

field = np.load('field.npy')
print(f"Shape: {field.shape}")  # (5000, 3)

# Access field at specific time
E_t50 = field[50]  # (Ex, Ey, Ez) at timestep 50

# For field-free propagation
print(f"Max field: {np.max(np.abs(field))}")  # Should be 0.0
```

## Data Preparation

### From NWChem RT-TDDFT

Use the provided preparation script:

```bash
python scripts/prepare_data.py \
    --restart-dir path/to/perm/ \
    --nwchem-out path/to/output.out \
    --field-file path/to/field.dat \
    --output-dir data/
```

This creates:
- `data/density_series.npy`
- `data/density_series_times.npy` (timestamps in a.u.)
- `data/overlap.npy`
- `data/field.npy`

### NWChem Restart File Format

The script parses NWChem restart files with format:

```
nbf_ao 4
nmats 2
t 0.200000
checksum 12345678
0.5 0.0 0.2 0.1 ...  (interleaved real/imag values)
```

Fields:
- `nbf_ao`: Number of basis functions
- `nmats`: Number of spin matrices (2 for open-shell)
- `t`: Simulation time in atomic units
- Data: Interleaved real/imaginary values for each matrix

### Custom Data Sources

If using a different quantum chemistry code, create the numpy arrays directly:

```python
import numpy as np

# Assuming you have density matrices from your simulation
# density_list: list of complex matrices, each (n_spin, nbf, nbf)
# times: list of timestamps
# overlap: (nbf, nbf) overlap matrix
# field: (n_steps, 3) field values

density_series = np.stack(density_list, axis=0)
np.save('density_series.npy', density_series)

np.save('overlap.npy', overlap)
np.save('field.npy', field.astype(np.float32))
```

## Normalization

The training pipeline optionally normalizes density matrices.

### Frobenius Normalization

Default mode. Scales all density matrices by a global factor:

```python
scale = mean(||rho||_F)
density_normalized = density / scale
```

The scale factor is saved in the config and checkpoint for denormalization at inference.

### Per-Element Normalization

Alternative mode. Normalizes each matrix element to zero mean, unit variance:

```python
mean = density.mean(axis=0)  # Per-element mean
std = density.std(axis=0)    # Per-element std
density_normalized = (density - mean) / std
```

## Data Splits

The dataset is split temporally (not randomly) to prevent data leakage:

```
|------ Train (70%) ------||-- Val (15%) --||-- Test (15%) --|
      t=0                                                  t=T
```

This ensures the model is tested on future timesteps it hasn't seen.

## Memory Considerations

For large trajectories:

| Trajectory | Memory (complex128) |
|------------|---------------------|
| 5000 steps, 2 spin, 4 bf | 1.3 MB |
| 10000 steps, 2 spin, 10 bf | 32 MB |
| 50000 steps, 2 spin, 50 bf | 4 GB |
| 100000 steps, 2 spin, 100 bf | 32 GB |

For very large systems, consider:
- Using `complex64` instead of `complex128`
- Memory-mapped files with `np.memmap`
- Chunked loading during training

## Validation

Use the preprocessing module to validate data:

```python
from data.preprocessing import validate_density_data

issues = validate_density_data(density_series, overlap, field)
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
```

Checks performed:
- Shape consistency
- Complex dtype for density
- Hermiticity of density matrices
- Symmetry of overlap matrix
- Finite values (no NaN/Inf)
- Field shape matching density
