# Configuration Reference

This document provides a complete reference for all configuration options.

## Configuration File Format

Configuration files use JSON format. Example:

```json
{
  "model": { ... },
  "physics": { ... },
  "loss": { ... },
  "data": { ... },
  "training": { ... },
  "device": "cuda",
  "seed": 42
}
```

## Model Configuration

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
    "field_encoder_type": "simple",
    "apply_hermitian_projection": true,
    "apply_mcweeney_projection": false,
    "apply_trace_scaling": false
  }
}
```

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | "DensityResNet" | Model architecture type |
| `max_nbf` | int | 16 | Maximum basis functions (for padding) |
| `hidden_dim` | int | 256 | Hidden layer dimension |
| `num_resnet_blocks` | int | 6 | Number of residual blocks |
| `history_length` | int | 5 | Number of past timesteps as input |
| `n_spin` | int | 2 | Spin channels (1=closed, 2=open shell) |
| `dropout` | float | 0.1 | Dropout probability |
| `overlap_mode` | string | "spectral" | Overlap encoding: "spectral", "direct", "cholesky" |
| `field_encoder_type` | string | "simple" | Field encoder: "simple" or "lstm" |
| `apply_hermitian_projection` | bool | true | Apply Hermitian projection at inference |
| `apply_mcweeney_projection` | bool | false | Apply McWeeney purification at inference |
| `apply_trace_scaling` | bool | false | Apply trace scaling at inference |

### Choosing Parameters

**max_nbf**: Set to the largest basis set you plan to use. Smaller values reduce memory and compute.

**hidden_dim**: Larger values increase model capacity. Recommended:
- Small molecules (< 10 bf): 128
- Medium molecules (10-50 bf): 256
- Large molecules (> 50 bf): 512

**num_resnet_blocks**: More blocks increase depth. Recommended: 4-8

**history_length**: How much temporal context to use. More history captures slower dynamics but increases memory. Recommended: 5-10

**overlap_mode**:
- `spectral`: Best for generalization, uses eigendecomposition
- `direct`: Fastest, flattens matrix directly
- `cholesky`: Uses Cholesky decomposition

## Physics Configuration

```json
{
  "physics": {
    "n_electrons": [1.0, 0.0]
  }
}
```

### Physics Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_electrons` | list[float] | [1.0, 0.0] | Electrons per spin [alpha, beta] |

For closed-shell systems: `n_electrons = [N/2, N/2]`
For open-shell systems: Specify each channel separately

## Loss Configuration

```json
{
  "loss": {
    "weight_mse": 1.0,
    "weight_hermitian": 0.1,
    "weight_idempotency": 0.05,
    "weight_trace": 0.1,
    "weight_positivity": 0.0
  }
}
```

### Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_mse` | float | 1.0 | MSE loss weight |
| `weight_hermitian` | float | 0.1 | Hermiticity penalty weight |
| `weight_idempotency` | float | 0.05 | Idempotency penalty weight |
| `weight_trace` | float | 0.1 | Trace conservation weight |
| `weight_positivity` | float | 0.0 | Positivity penalty weight |

### Weight Recommendations

**Baseline**: Start with defaults, adjust based on validation metrics.

**Strong physics enforcement**:
```json
{
  "weight_mse": 1.0,
  "weight_hermitian": 0.5,
  "weight_idempotency": 0.2,
  "weight_trace": 0.5
}
```

**Minimal physics** (for debugging):
```json
{
  "weight_mse": 1.0,
  "weight_hermitian": 0.0,
  "weight_idempotency": 0.0,
  "weight_trace": 0.0
}
```

## Data Configuration

```json
{
  "data": {
    "density_file": "data/density_series.npy",
    "overlap_file": "data/overlap.npy",
    "field_file": "data/field.npy",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "batch_size": 32,
    "num_workers": 0,
    "normalize": true,
    "normalization_mode": "frobenius"
  }
}
```

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `density_file` | string | required | Path to density series |
| `overlap_file` | string | required | Path to overlap matrix |
| `field_file` | string | null | Path to field data (optional) |
| `train_ratio` | float | 0.7 | Training set fraction |
| `val_ratio` | float | 0.15 | Validation set fraction |
| `batch_size` | int | 32 | Training batch size |
| `num_workers` | int | 0 | DataLoader workers |
| `normalize` | bool | true | Apply normalization |
| `normalization_mode` | string | "frobenius" | Normalization type |

### Path Resolution

Paths can be:
- Absolute: `/home/user/data/density.npy`
- Relative to config file: `data/density.npy`
- Relative to working directory

### Data Splits

Test ratio is computed as: `1 - train_ratio - val_ratio`

Default splits:
- Train: 70%
- Validation: 15%
- Test: 15%

## Training Configuration

```json
{
  "training": {
    "epochs": 500,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "warmup_epochs": 10,
    "early_stopping_patience": 50,
    "checkpoint_dir": "checkpoints",
    "save_every": 50
  }
}
```

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 500 | Maximum training epochs |
| `learning_rate` | float | 1e-4 | Initial learning rate |
| `weight_decay` | float | 1e-5 | AdamW weight decay |
| `gradient_clip` | float | 1.0 | Gradient clipping norm |
| `warmup_epochs` | int | 10 | LR warmup period |
| `early_stopping_patience` | int | 50 | Epochs without improvement before stopping |
| `checkpoint_dir` | string | "checkpoints" | Where to save models |
| `save_every` | int | 50 | Save checkpoint every N epochs |

### Learning Rate Schedule

Uses cosine annealing with linear warmup:

```
LR(epoch) =
  if epoch < warmup_epochs:
    lr * epoch / warmup_epochs  # Linear warmup
  else:
    lr * 0.5 * (1 + cos(pi * (epoch - warmup) / (total - warmup)))  # Cosine decay
```

## Global Configuration

```json
{
  "device": "cuda",
  "seed": 42
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | "cuda" | Compute device: "cuda", "cpu", "mps" |
| `seed` | int | 42 | Random seed for reproducibility |

## Command-Line Overrides

Many parameters can be overridden via command line:

```bash
python scripts/train.py \
    --config configs/default.json \
    --epochs 1000 \
    --lr 5e-5 \
    --batch-size 64 \
    --device cpu \
    --seed 123 \
    --output-dir checkpoints/experiment1
```

Override precedence: command-line > config file > defaults

## Example Configurations

### Small Molecule (H2+)

```json
{
  "model": {
    "max_nbf": 4,
    "hidden_dim": 128,
    "num_resnet_blocks": 4,
    "history_length": 5,
    "n_spin": 2
  },
  "physics": {
    "n_electrons": [1.0, 0.0]
  },
  "training": {
    "epochs": 200,
    "learning_rate": 1e-4
  }
}
```

### Medium Molecule

```json
{
  "model": {
    "max_nbf": 50,
    "hidden_dim": 256,
    "num_resnet_blocks": 6,
    "history_length": 5,
    "n_spin": 2
  },
  "physics": {
    "n_electrons": [10.0, 10.0]
  },
  "training": {
    "epochs": 500,
    "learning_rate": 5e-5,
    "batch_size": 16
  }
}
```

### Large Molecule (GPU)

```json
{
  "model": {
    "max_nbf": 200,
    "hidden_dim": 512,
    "num_resnet_blocks": 8,
    "history_length": 10,
    "n_spin": 2,
    "dropout": 0.2
  },
  "training": {
    "epochs": 1000,
    "learning_rate": 1e-5,
    "batch_size": 8,
    "gradient_clip": 0.5
  },
  "device": "cuda"
}
```
