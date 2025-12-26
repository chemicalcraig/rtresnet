# Model Architecture

This document provides a detailed description of the DensityResNet architecture.

## Overview

DensityResNet is a physics-informed neural network designed to predict time-dependent density matrix evolution. The architecture consists of three main encoder pathways that are fused and processed through a ResNet backbone.

## Input Representation

The model takes three inputs:

1. **Density History** `(batch, history_length, n_spin, nbf, nbf)` - Complex tensor
   - Previous `history_length` density matrices
   - `n_spin`: Number of spin channels (1 for closed-shell, 2 for open-shell)
   - `nbf`: Number of basis functions

2. **Overlap Matrix** `(batch, nbf, nbf)` - Real tensor
   - The atomic orbital overlap matrix S
   - Encodes molecular geometry information

3. **Field History** `(batch, history_length, 3)` - Real tensor
   - External electric field components (Ex, Ey, Ez)
   - History matching density time window

## Architecture Components

### 1. Density History Encoder

```
DensityHistoryEncoder:
    Input: (batch, history, n_spin, nbf, nbf) complex

    1. Flatten: (batch, history * n_spin * nbf * nbf * 2)
       - Complex -> real/imag concatenation

    2. Linear: hidden_dim * 2
    3. LayerNorm
    4. GELU
    5. Linear: hidden_dim
    6. LayerNorm

    Output: (batch, hidden_dim) real
```

The density history encoder flattens the complex-valued density matrix history and projects it to a fixed-size hidden representation.

### 2. Overlap Encoder

Three encoding modes are available:

**Spectral Mode (Recommended)**
```
OverlapEncoder (spectral):
    Input: (batch, nbf, nbf) real

    1. Eigendecomposition: eigenvalues, eigenvectors
    2. Flatten eigenvalues
    3. MLP: eigenvalues -> hidden_dim

    Output: (batch, hidden_dim) real
```

The spectral mode uses eigendecomposition to extract geometry-invariant features from the overlap matrix. This is recommended for future generalization to different molecules.

**Direct Mode**
```
OverlapEncoder (direct):
    Input: (batch, nbf, nbf) real

    1. Flatten: (batch, nbf * nbf)
    2. MLP: -> hidden_dim

    Output: (batch, hidden_dim) real
```

**Cholesky Mode**
```
OverlapEncoder (cholesky):
    Input: (batch, nbf, nbf) real

    1. Cholesky decomposition: L
    2. Flatten lower triangle
    3. MLP: -> hidden_dim

    Output: (batch, hidden_dim) real
```

### 3. Field Encoder

```
FieldEncoderSimple:
    Input: (batch, history_length, 3) real

    1. Flatten: (batch, history_length * 3)
    2. Linear: hidden_dim
    3. LayerNorm
    4. GELU

    Output: (batch, hidden_dim) real
```

For time-varying fields, a more sophisticated LSTM-based encoder is also available.

### 4. Fusion Layer

```
Fusion:
    Inputs: density_enc, overlap_enc, field_enc  (each: batch, hidden_dim)

    1. Concatenate: (batch, hidden_dim * 3)
    2. Linear: hidden_dim * 2
    3. LayerNorm
    4. GELU
    5. Linear: hidden_dim

    Output: (batch, hidden_dim) real
```

### 5. ResNet Backbone

```
ResNetStack:
    Input: (batch, hidden_dim) real

    For each of num_resnet_blocks:
        DensityResBlock:
            1. LayerNorm
            2. Linear: hidden_dim * 2
            3. GELU
            4. Dropout
            5. Linear: hidden_dim
            6. Dropout
            7. Residual: x + block(x)

    Final LayerNorm

    Output: (batch, hidden_dim) real
```

### 6. Output Projection

```
OutputProjection:
    Input: (batch, hidden_dim) real

    1. Linear: n_spin * nbf * nbf * 2
    2. Reshape: (batch, n_spin, nbf, nbf, 2)
    3. Convert to complex: (batch, n_spin, nbf, nbf)

    Output: (batch, n_spin, nbf, nbf) complex
```

### 7. Residual Prediction

The model predicts the **change** in density matrix rather than the absolute value:

```
rho_predicted = rho(t) + delta_rho
```

where `rho(t)` is the most recent density matrix in the history and `delta_rho` is the network output.

### 8. Physics Projections (Inference Only)

During inference, optional physics projections are applied:

```
if apply_hermitian:
    rho = (rho + rho.conj().T) / 2

if apply_trace_scaling:
    rho = rho * (n_electrons / trace(rho @ S))

if apply_mcweeney:
    rho = 3*rho@S@rho - 2*rho@S@rho@S@rho
```

## Parameter Count

For typical configurations:

| Config | Parameters |
|--------|------------|
| hidden_dim=128, blocks=4 | ~350K |
| hidden_dim=256, blocks=6 | ~1.4M |
| hidden_dim=512, blocks=8 | ~5.6M |

## Design Rationale

### Why Flatten + MLP?

While convolutional or attention-based architectures could capture spatial correlations in the density matrix, the flatten + MLP approach was chosen for:

1. **Simplicity**: Easier to implement and debug
2. **Flexibility**: Works with any matrix size (with padding)
3. **Efficiency**: Fast training and inference
4. **Effectiveness**: Sufficient for small molecules where all basis functions interact

### Why Residual Prediction?

Predicting `delta_rho` instead of `rho` directly:

1. **Easier learning**: Changes between timesteps are small
2. **Identity initialization**: Model starts by predicting no change
3. **Stability**: Gradients flow more easily through identity + small delta

### Why Spectral Overlap Encoding?

The spectral decomposition of S:

1. **Invariance**: Eigenvalues are invariant to basis rotation
2. **Physical meaning**: Eigenvalues relate to linear dependence
3. **Generalization**: Same representation works for different molecules
