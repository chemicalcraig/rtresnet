# Tutorial: Training Your First Model

This tutorial walks through training a density matrix prediction model on the H2+ test case.

## Prerequisites

1. Python 3.10+ with PyTorch installed
2. Clone the repository and activate the virtual environment

```bash
git clone https://github.com/yourusername/rtresnet.git
cd rtresnet
source venv/bin/activate  # or your virtual environment
```

## Step 1: Explore the Data

First, let's look at the H2+ test case data:

```bash
ls test/h2p/
```

You should see:
- `perm/` - NWChem restart files
- `h2_plus_rttddft.out` - NWChem output file
- `data/` - Processed data (after preparation)

## Step 2: Prepare the Data

Run the data preparation script:

```bash
python scripts/prepare_data.py \
    --restart-dir test/h2p/perm \
    --nwchem-out test/h2p/h2_plus_rttddft.out \
    --output-dir test/h2p/data \
    --n-electrons 1.0 0.0
```

This will:
1. Parse 5000 restart files into `density_series.npy`
2. Extract the overlap matrix to `overlap.npy`
3. Create zero field data in `field.npy`
4. Validate the data

Expected output:
```
============================================================
DensityResNet Data Preparation
============================================================

Step 1: Aggregating density matrices...
Found 5000 restart files. Processing...
  Shape: torch.Size([5000, 2, 4, 4])

Step 2: Extracting overlap matrix...
  Shape: (4, 4)
  Trace: 4.000000

Step 3: Synchronizing field data...
  No external field data found. Using zero field.

Step 4: Validating data...
  Trace(rho*S) statistics:
    Alpha: 1.000000
    Beta:  0.000000

Data Preparation Complete!
```

## Step 3: Explore the Data in Python

```python
import numpy as np

# Load the prepared data
density = np.load('test/h2p/data/density_series.npy')
overlap = np.load('test/h2p/data/overlap.npy')
field = np.load('test/h2p/data/field.npy')

print(f"Density shape: {density.shape}")  # (5000, 2, 4, 4)
print(f"Overlap shape: {overlap.shape}")   # (4, 4)
print(f"Field shape: {field.shape}")       # (5000, 3)

# Check a density matrix
rho = density[0, 0]  # Alpha density at t=0
print(f"\nAlpha density at t=0:")
print(rho)

# Verify Hermiticity
print(f"\nHermitian error: {np.max(np.abs(rho - rho.conj().T))}")

# Compute trace
trace = np.trace(rho @ overlap).real
print(f"Tr(rho @ S) = {trace:.6f}")  # Should be 1.0
```

## Step 4: Configure the Training

Look at the H2+ configuration:

```bash
cat configs/h2p_training.json
```

Key settings for H2+:
- `max_nbf: 4` - 4 basis functions
- `hidden_dim: 128` - Smaller model for small system
- `n_electrons: [1.0, 0.0]` - 1 alpha electron, 0 beta

## Step 5: Train the Model

Start training:

```bash
python scripts/train.py --config configs/h2p_training.json --epochs 200
```

Monitor the output:
```
============================================================
DensityResNet Training
============================================================

Loading config from: configs/h2p_training.json
Using device: cuda

Loading data...
Density series shape: (5000, 2, 4, 4)

Creating dataloaders...
Train batches: 110
Val batches: 24
Test batches: 24

Creating model...
Total parameters: 1,001,732

Starting training...
  -> Saved best model (val_loss: 0.357226)
Epoch    0 | Train: 0.376547 | Val: 0.357226 | LR: 1.09e-05
         Physics: Herm=0.00e+00, Trace MAE=0.0528

Epoch   50 | Train: 0.001234 | Val: 0.001456 | LR: 8.5e-05
         Physics: Herm=0.00e+00, Trace MAE=0.0012

Training Complete!
Best validation loss: 0.000892 at epoch 142
Checkpoints saved to: checkpoints/h2p
```

### Understanding the Metrics

- **Train/Val Loss**: Total loss including physics penalties
- **Herm**: Hermiticity error (should be ~0 with projection)
- **Trace MAE**: Mean absolute error in Tr(rho*S) vs N_electrons
- **Frob**: Frobenius norm of predictions

## Step 6: Run Predictions

After training, run a prediction rollout:

```bash
python scripts/predict.py \
    --checkpoint checkpoints/h2p/best_model.pt \
    --config configs/h2p_training.json \
    --n-steps 500 \
    --compare-reference
```

Output:
```
============================================================
DensityResNet Prediction
============================================================

Running prediction...
Starting rollout for 500 steps...
  Step 50/500
  Step 100/500
  ...
Rollout complete. Output shape: (500, 2, 4, 4)

Physics metrics (mean over trajectory):
  hermiticity_error: 0.000000
  trace_alpha: 0.998234
  trace_beta: 0.000123

Error metrics:
  Mean Frobenius error: 0.00234
  Final Frobenius error: 0.00567

Predictions saved to: predictions/
```

## Step 7: Analyze Results

Load and analyze the predictions:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
predictions = np.load('predictions/predictions.npy')
reference = np.load('test/h2p/data/density_series.npy')

# Compare trajectories
history_length = 5
ref_slice = reference[history_length:history_length+500]

# Compute errors per timestep
errors = np.sqrt(np.sum(np.abs(predictions - ref_slice)**2, axis=(-3,-2,-1)))

plt.figure(figsize=(10, 4))
plt.plot(errors)
plt.xlabel('Timestep')
plt.ylabel('Frobenius Error')
plt.title('Prediction Error Over Time')
plt.savefig('error_plot.png')
plt.show()

# Check trace conservation
traces = []
overlap = np.load('test/h2p/data/overlap.npy')
for t in range(len(predictions)):
    trace = np.trace(predictions[t, 0] @ overlap).real
    traces.append(trace)

plt.figure(figsize=(10, 4))
plt.plot(traces)
plt.axhline(y=1.0, color='r', linestyle='--', label='Expected')
plt.xlabel('Timestep')
plt.ylabel('Tr(rho_alpha @ S)')
plt.legend()
plt.savefig('trace_plot.png')
plt.show()
```

## Step 8: Improve the Model

If results aren't satisfactory, try:

### More Training
```bash
python scripts/train.py --config configs/h2p_training.json --epochs 500
```

### Different Learning Rate
```bash
python scripts/train.py --config configs/h2p_training.json --lr 5e-5
```

### Stronger Physics Penalties

Create a new config with higher physics weights:

```json
{
  "loss": {
    "weight_mse": 1.0,
    "weight_hermitian": 0.5,
    "weight_trace": 0.5,
    "weight_idempotency": 0.1
  }
}
```

### Larger Model
```json
{
  "model": {
    "hidden_dim": 256,
    "num_resnet_blocks": 8
  }
}
```

## Step 9: Use Physics Projections

For stricter physics enforcement at inference:

```bash
python scripts/predict.py \
    --checkpoint checkpoints/h2p/best_model.pt \
    --config configs/h2p_training.json \
    --n-steps 500 \
    --apply-hermitian \
    --apply-trace-scaling \
    --n-electrons 1.0 0.0
```

## Next Steps

1. **Try different molecules**: Prepare data from your own RT-TDDFT simulations
2. **Experiment with architectures**: Modify hidden_dim, num_blocks, etc.
3. **Tune physics weights**: Balance prediction accuracy vs constraint satisfaction
4. **Long rollouts**: Test stability over 1000+ timesteps
5. **Transfer learning**: Train on one molecule, fine-tune on another

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `hidden_dim`
- Use `--device cpu` for testing

### Training Not Converging
- Lower learning rate
- Increase `warmup_epochs`
- Check data normalization

### Poor Prediction Quality
- Train longer
- Increase model capacity
- Add more training data
- Tune physics loss weights

### NaN in Loss
- Enable gradient clipping: `"gradient_clip": 0.5`
- Lower learning rate
- Check input data for NaN/Inf values
