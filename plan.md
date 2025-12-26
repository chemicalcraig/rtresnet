# Implementation Plan: Physics-Informed ResNet for RT-TDDFT Density Matrix Prediction

## Overview

Build a ResNet-based neural network to predict time-dependent evolution of complex-valued electronic density matrices from RT-TDDFT simulations. The model incorporates physics constraints (Hermiticity, idempotency, trace conservation, positivity) into both the architecture and loss function.

**Existing Infrastructure:**
- `src/rtparse.py` - Parses NWChem output files
- `src/aggregate_data.py` - Creates density_series.npy from restart files
- `src/extract_densities.py` - Extracts bootstrap densities
- `src/sync_field.py` - Synchronizes field with density timestamps
- `src/timestep_sync.py` - Timestep synchronization utilities
- `src/get_overlap.py` - Extracts AO overlap matrix S
- `test/h2p/` - H2+ test case with 5013 timesteps, 4 basis functions, open-shell

---

## File Structure

```
rtresnet/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # DensityMatrixDataset class
│   │   └── preprocessing.py     # Normalization, train/val/test splits
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── complex_layers.py    # ComplexLinear, ComplexBatchNorm, ComplexActivation
│   │   ├── resnet_blocks.py     # DensityResBlock
│   │   ├── encoders.py          # OverlapEncoder, FieldEncoder
│   │   ├── density_resnet.py    # Main DensityResNet model
│   │   └── graph_encoder.py     # Graph NN for generalization (future)
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── projections.py       # Hermitian, McWeeney, trace projections
│   │   ├── losses.py            # PhysicsInformedLoss
│   │   └── validation.py        # Physics property validation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop with checkpointing
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py         # Prediction and multi-step rollout
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py            # JSON config loading/validation
│   │   └── visualization.py     # Plotting utilities
│   │
│   └── [existing utility files]
│
├── configs/
│   ├── default.json             # Default configuration template
│   └── h2p_training.json        # H2+ specific config
│
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── predict.py               # Prediction entry point
│   └── prepare_data.py          # Data preparation pipeline
│
└── tests/
    ├── test_complex_layers.py
    ├── test_physics.py
    └── test_model.py
```

---

## Implementation Steps

### Phase 1: Complex-Valued Neural Network Layers

**File: `src/models/complex_layers.py`**

1. **ComplexLinear** - Linear layer for complex tensors
   - Use PyTorch native `torch.complex64/128`
   - Separate real/imaginary weight parameters
   - Complex bias support

2. **ComplexBatchNorm** - Batch normalization for complex tensors
   - Covariance-based normalization (Trabelsi et al., 2018)
   - Learnable 2x2 scale matrix (gamma_rr, gamma_ii, gamma_ri)
   - Complex beta shift

3. **ComplexActivation** - Activation functions
   - ModReLU: ReLU on magnitude, preserve phase
   - Cardioid activation alternative
   - zReLU: separate ReLU on real/imag parts

---

### Phase 2: Physics Constraint Implementations

**File: `src/physics/projections.py`**

1. **hermitian_projection(rho)**
   ```
   rho_hermitian = (rho + rho^dagger) / 2
   ```

2. **mcweeney_purification(rho, S, n_iterations=1)**
   ```
   rho' = 3*rho*S*rho - 2*rho*S*rho*S*rho
   ```
   - Enforces idempotency: rho*S*rho = rho
   - Apply iteratively for stronger enforcement

3. **trace_scaling(rho, S, n_electrons)**
   ```
   scale = n_electrons / Tr(rho @ S)
   rho_scaled = rho * scale
   ```

4. **positivity_projection(rho)** (optional)
   - Eigendecompose, clamp negative eigenvalues to zero
   - Reconstruct from modified spectrum

**File: `src/physics/losses.py`**

**PhysicsInformedLoss** class with weighted penalties:
- `weight_mse`: MSE between predicted and target (real + imag)
- `weight_hermitian`: ||rho - rho^dagger||_F^2
- `weight_idempotency`: ||rho*S*rho - rho||_F^2
- `weight_trace`: (Tr(rho*S) - N_electrons)^2
- `weight_positivity`: sum(ReLU(-eigenvalues))^2

---

### Phase 3: Encoder Modules

**File: `src/models/encoders.py`**

1. **OverlapEncoder** - Encode overlap matrix S
   - Modes: `spectral` (eigendecomposition), `direct` (flatten), `cholesky`
   - Output: conditioning vector (hidden_dim,)
   - Spectral mode recommended for generalization

2. **FieldEncoder** - Encode external field E(t) history
   - Conv1D + LSTM architecture
   - Input: (history_length, 3) field values
   - Output: conditioning vector (hidden_dim,)

---

### Phase 4: ResNet Architecture

**File: `src/models/resnet_blocks.py`**

**DensityResBlock**:
- ComplexBatchNorm -> ComplexLinear -> ModReLU -> ComplexLinear -> residual add
- Dropout on both real/imag parts

**File: `src/models/density_resnet.py`**

**DensityResNet** architecture:
```
Input: rho_history (batch, history_len, n_spin, nbf, nbf) complex
       S (batch, nbf, nbf) real
       E_history (batch, history_len, 3) real

1. Flatten rho_history -> embed to hidden_dim
2. Encode S via OverlapEncoder -> hidden_dim
3. Encode E via FieldEncoder -> hidden_dim
4. Concatenate [rho_embed, S_embed, E_embed] -> fusion layer
5. Pass through N ResNet blocks
6. Project to output shape (n_spin, nbf, nbf, 2)
7. Reshape to complex, add residual from rho(t)
8. Apply physics projections at inference only (Hermitian, McWeeney, trace)

Output: rho_predicted (batch, n_spin, nbf, nbf) complex
```

**Key parameters:**
- `max_nbf`: Maximum basis function count (for padding)
- `hidden_dim`: Hidden layer size (256 typical)
- `num_resnet_blocks`: Number of ResNet blocks (4-8)
- `history_length`: Time history window (5 recommended)
- `n_spin`: 1 (closed-shell) or 2 (open-shell)

---

### Phase 5: Dataset and Data Loading

**File: `src/data/dataset.py`**

**DensityMatrixDataset**:
- Load `density_series.npy`, `overlap.npy`, `field_synced.npy`
- Create sliding windows of history_length
- Return: `{density_history, target, overlap, field}`
- Handle complex dtype properly

**File: `src/data/preprocessing.py`**

- Train/val/test split (70/15/15 default)
- Optional Frobenius norm normalization
- Padding for variable-size molecules

---

### Phase 6: Training Infrastructure

**File: `src/training/trainer.py`**

**DensityTrainer** class:
- AdamW optimizer with weight decay
- Cosine annealing LR scheduler with warmup
- Gradient clipping
- Early stopping on validation loss
- Checkpoint saving (best model, periodic)
- Physics metrics logging (trace error, Hermiticity, eigenvalue stats)

**File: `src/utils/config.py`**

- Load JSON configuration
- Validate required fields
- Merge with defaults

---

### Phase 7: Prediction/Inference

**File: `src/inference/predictor.py`**

**DensityPredictor** class:
- Load checkpoint
- Bootstrap from initial densities
- Multi-step rollout with optional physics projections
- Save predicted trajectory

---

### Phase 8: Entry Points

**File: `scripts/train.py`**
```
python scripts/train.py --config configs/h2p_training.json
```

**File: `scripts/predict.py`**
```
python scripts/predict.py --config configs/h2p_predict.json --checkpoint best_model.pt
```

**File: `scripts/prepare_data.py`**
- Run aggregate_data.py
- Run sync_field.py
- Run get_overlap.py
- Validate outputs

---

## Configuration Schema

```json
{
  "model": {
    "type": "DensityResNet",
    "max_nbf": 16,
    "hidden_dim": 256,
    "num_resnet_blocks": 6,
    "history_length": 5,
    "n_spin": 2,
    "dropout": 0.1,
    "overlap_mode": "spectral",
    "apply_hermitian_projection": true,
    "apply_mcweeney_projection": false,
    "apply_trace_scaling": false
  },
  "physics": {
    "n_electrons": [1.0, 0.0],
    "comment": "Per-spin electron counts: [alpha, beta]"
  },
  "loss": {
    "weight_mse": 1.0,
    "weight_hermitian": 0.1,
    "weight_idempotency": 0.05,
    "weight_trace": 0.1,
    "weight_positivity": 0.01
  },
  "data": {
    "density_file": "density_series.npy",
    "overlap_file": "overlap.npy",
    "field_file": "field_synced.npy",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "batch_size": 32
  },
  "training": {
    "epochs": 500,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "early_stopping_patience": 50,
    "checkpoint_dir": "checkpoints/"
  },
  "device": "cuda",
  "seed": 42
}
```

---

## Generalization Strategy (Future)

For handling different molecules with varying sizes:

1. **Graph Neural Network Encoder**
   - Treat basis functions as graph nodes
   - Use overlap matrix S_ij as edge weights
   - Node features: atomic number, angular momentum
   - Produces molecule-agnostic conditioning vector

2. **Tokenized Transformer**
   - Each matrix element rho_ij as a token
   - Learnable position encodings for (i,j) indices
   - Handles variable matrix sizes naturally

3. **Multi-Molecule Training**
   - Curriculum learning: start with similar molecules
   - Shared ResNet core, molecule-specific conditioning

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Complex numbers | Native PyTorch complex | Cleaner gradients, matches math |
| Architecture | Flatten + MLP | User preference; simpler, effective |
| History length | 5 timesteps | Captures dynamics, manageable memory |
| Physics enforcement | Loss penalties only | User preference; projections at inference only |
| GNN generalization | Deferred | Focus on core ResNet with H2+ first |
| Overlap encoding | Spectral mode | Best for future generalization |
| Activation | ModReLU | Preserves phase information |

---

## Testing Strategy

1. **Unit tests** for complex layers (gradient flow, shapes)
2. **Physics tests** for projections (verify constraints satisfied)
3. **Integration test** on H2+ (overfit small dataset)
4. **Validation** of physics metrics during training
5. **Rollout stability** over 100+ timesteps

---

## H2+ Test Case Specifics

- **Basis functions**: 4 (nbf_ao = 4)
- **Spin channels**: 2 (open-shell format, but beta is empty)
- **Electrons**: 1 (all in alpha channel; beta channel is zero)
- **Timesteps**: ~5000
- **dt**: 0.2 a.u.
- **N_electrons for trace**: 1.0 for alpha, 0.0 for beta (or handle per-spin)
- **External field**: Zero (field-free propagation)

**Spin handling**: Keep n_spin=2 format. The model will learn that beta channel should remain zero. This prepares the architecture for multi-electron open-shell systems.

Data preparation:
```bash
cd test/h2p
python ../../src/aggregate_data.py perm/ densities/
python ../../src/get_overlap.py h2_plus_rttddft.out -o data/overlap
python ../../src/sync_field.py perm/ rt_data/field.dat -o data/
```
