# RT-ResNet: Physics-Informed Neural Network for RT-TDDFT Density Matrix Prediction

A ResNet-based deep learning framework for predicting time-dependent evolution of complex-valued electronic density matrices from Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations.

## Overview

RT-ResNet learns to predict the next density matrix in a time series given a history of previous density matrices, the atomic orbital (AO) overlap matrix, and optional external field data. The model incorporates quantum mechanical constraints directly into the loss function and applies physics projections at inference time.

### Key Features

- **Complex-valued neural networks** with native PyTorch complex tensor support
- **Physics-informed loss function** enforcing:
  - Hermiticity: rho = rho^dagger
  - Idempotency: rho * S * rho = rho (via McWeeney purification)
  - Trace conservation: Tr(rho * S) = N_electrons
  - Positivity: non-negative eigenvalues
- **Autoregressive rollout** for multi-step trajectory prediction
- **Flexible architecture** supporting different molecular systems
- **NWChem integration** for parsing RT-TDDFT restart files

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rtresnet.git
cd rtresnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy matplotlib
```

## Quick Start

### 1. Prepare Data

Convert NWChem RT-TDDFT output to training format:

```bash
python scripts/prepare_data.py \
    --restart-dir path/to/restart/files \
    --nwchem-out path/to/nwchem.out \
    --output-dir data/ \
    --n-electrons 1.0 0.0  # alpha, beta electrons
```

This creates:
- `density_series.npy` - Time series of density matrices (N_steps, N_spin, N_bf, N_bf)
- `overlap.npy` - AO overlap matrix (N_bf, N_bf)
- `field.npy` - External field data (N_steps, 3)

### 2. Train Model

```bash
python scripts/train.py --config configs/h2p_training.json
```

Override config options via command line:
```bash
python scripts/train.py \
    --config configs/default.json \
    --epochs 500 \
    --lr 1e-4 \
    --batch-size 32 \
    --device cuda
```

### 3. Run Predictions

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/h2p_training.json \
    --n-steps 1000 \
    --compare-reference
```

## Project Structure

```
rtresnet/
├── src/
│   ├── data/                   # Dataset and preprocessing
│   │   ├── dataset.py          # DensityMatrixDataset class
│   │   └── preprocessing.py    # Normalization, validation
│   │
│   ├── models/                 # Neural network components
│   │   ├── complex_layers.py   # ComplexLinear, ComplexBatchNorm, ModReLU
│   │   ├── encoders.py         # Overlap/Field/Density encoders
│   │   ├── resnet_blocks.py    # ResNet blocks
│   │   └── density_resnet.py   # Main DensityResNet model
│   │
│   ├── physics/                # Physics constraints
│   │   ├── projections.py      # Hermitian, McWeeney, trace projections
│   │   └── losses.py           # PhysicsInformedLoss
│   │
│   ├── training/               # Training infrastructure
│   │   └── trainer.py          # DensityTrainer with checkpointing
│   │
│   ├── inference/              # Prediction and rollout
│   │   └── predictor.py        # DensityPredictor class
│   │
│   └── utils/                  # Utilities
│       └── config.py           # Configuration management
│
├── configs/                    # Configuration files
│   ├── default.json            # Default template
│   └── h2p_training.json       # H2+ specific config
│
├── scripts/                    # Entry points
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction script
│   └── prepare_data.py         # Data preparation
│
└── test/                       # Test cases
    └── h2p/                    # H2+ molecule test data
```

## Configuration

Configuration is specified via JSON files. See `configs/default.json` for all options:

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
    "overlap_mode": "spectral"
  },
  "physics": {
    "n_electrons": [1.0, 0.0]
  },
  "loss": {
    "weight_mse": 1.0,
    "weight_hermitian": 0.1,
    "weight_idempotency": 0.05,
    "weight_trace": 0.1
  },
  "data": {
    "density_file": "data/density_series.npy",
    "overlap_file": "data/overlap.npy",
    "batch_size": 32,
    "train_ratio": 0.7,
    "val_ratio": 0.15
  },
  "training": {
    "epochs": 500,
    "learning_rate": 1e-4,
    "early_stopping_patience": 50
  }
}
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_nbf` | Maximum basis functions (for padding) | 16 |
| `hidden_dim` | Hidden layer dimension | 256 |
| `num_resnet_blocks` | Number of ResNet blocks | 6 |
| `history_length` | Time steps in input window | 5 |
| `n_spin` | Spin channels (1=closed, 2=open shell) | 2 |
| `overlap_mode` | How to encode S matrix (spectral/direct/cholesky) | spectral |

## Model Architecture

```
Input: rho_history (batch, history_len, n_spin, nbf, nbf) complex
       S (batch, nbf, nbf) real
       E_history (batch, history_len, 3) real

Pipeline:
1. DensityHistoryEncoder: Flatten + MLP -> hidden_dim
2. OverlapEncoder: Spectral decomposition + MLP -> hidden_dim
3. FieldEncoder: Simple MLP -> hidden_dim
4. Fusion: Concatenate + MLP
5. ResNet Blocks: N layers of residual blocks
6. Output Projection: -> (n_spin, nbf, nbf) complex
7. Residual Connection: Add rho(t) for delta prediction
8. Physics Projections (inference only): Hermitian, McWeeney, trace

Output: rho_predicted (batch, n_spin, nbf, nbf) complex
```

## Physics Constraints

### Loss Function Components

The `PhysicsInformedLoss` combines multiple terms:

```
L = w_mse * L_mse + w_herm * L_hermitian + w_idemp * L_idempotency + w_trace * L_trace
```

- **MSE Loss**: Mean squared error on real and imaginary parts
- **Hermiticity Loss**: ||rho - rho^dagger||_F^2
- **Idempotency Loss**: ||rho*S*rho - rho||_F^2
- **Trace Loss**: (Tr(rho*S) - N_electrons)^2

### Inference Projections

At inference time, optional projections enforce exact constraints:

1. **Hermitian Projection**: rho = (rho + rho^dagger) / 2
2. **McWeeney Purification**: rho' = 3*rho*S*rho - 2*rho*S*rho*S*rho
3. **Trace Scaling**: rho *= N_electrons / Tr(rho*S)

## API Usage

### Training Programmatically

```python
from models import create_model_from_config
from physics import create_loss_from_config
from training import DensityTrainer
from data import create_dataloaders
from utils import load_config

# Load configuration
config = load_config('configs/h2p_training.json')

# Create model and loss
model = create_model_from_config(config)
criterion = create_loss_from_config(config)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    density_series, overlap_matrix, field_data,
    history_length=5, batch_size=32
)

# Train
trainer = DensityTrainer(model, train_loader, val_loader, criterion, config)
history = trainer.train()
```

### Inference Programmatically

```python
from inference import DensityPredictor
import numpy as np

# Load predictor
predictor = DensityPredictor(
    checkpoint_path='checkpoints/best_model.pt',
    apply_hermitian=True,
    apply_mcweeney=False,
    n_electrons=[1.0, 0.0]
)

# Load initial data
initial_densities = np.load('data/density_series.npy')[:5]  # First 5 steps
overlap = np.load('data/overlap.npy')

# Run rollout
result = predictor.rollout(
    initial_densities=initial_densities,
    overlap_matrix=overlap,
    n_steps=1000
)

# Access predictions
predictions = result.predictions  # (1000, 2, 4, 4) complex
print(f"Physics metrics: {result.physics_metrics[-1]}")
```

## Test Case: H2+ Molecule

The repository includes a test case for the hydrogen molecular ion (H2+):

- **Basis functions**: 4
- **Electrons**: 1 (alpha channel only)
- **Timesteps**: ~5000
- **Time step**: 0.2 a.u.

```bash
# Prepare H2+ data
python scripts/prepare_data.py \
    --restart-dir test/h2p/perm \
    --nwchem-out test/h2p/h2_plus_rttddft.out \
    --output-dir test/h2p/data \
    --n-electrons 1.0 0.0

# Train on H2+
python scripts/train.py --config configs/h2p_training.json --epochs 200

# Predict
python scripts/predict.py \
    --checkpoint checkpoints/h2p/best_model.pt \
    --config configs/h2p_training.json \
    --n-steps 500 \
    --compare-reference
```

## Future Development

- **Graph Neural Network Encoder**: For handling variable-size molecules
- **Transformer Architecture**: Token-based approach for scalability
- **Multi-molecule Training**: Transfer learning across chemical space
- **Active Learning**: Efficient sampling of training trajectories

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rtresnet2024,
  title={RT-ResNet: Physics-Informed Neural Network for RT-TDDFT},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rtresnet}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Designed for [NWChem](https://nwchemgit.github.io/) RT-TDDFT simulations
- Complex batch normalization based on [Deep Complex Networks](https://arxiv.org/abs/1705.09792)
