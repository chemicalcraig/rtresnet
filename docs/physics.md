# Physics Constraints

This document explains the quantum mechanical constraints enforced by RT-ResNet and how they are implemented.

## Background: Density Matrix Properties

In quantum chemistry, the one-electron reduced density matrix rho describes the electronic state of a system. For a valid density matrix in the atomic orbital (AO) basis, the following properties must hold:

### 1. Hermiticity

The density matrix must be Hermitian (equal to its conjugate transpose):

```
rho = rho^dagger
```

This ensures that all observables computed from rho are real-valued.

### 2. Idempotency

For a single-determinant wavefunction (e.g., Hartree-Fock or Kohn-Sham DFT), the density matrix satisfies:

```
rho * S * rho = rho
```

where S is the AO overlap matrix. This property is related to the density matrix being a projector onto the occupied orbital space.

### 3. Trace Conservation

The trace of rho with respect to S gives the number of electrons:

```
Tr(rho * S) = N_electrons
```

This ensures charge conservation throughout the dynamics.

### 4. Positivity

The density matrix must be positive semidefinite (all eigenvalues >= 0). In the S-orthonormal basis, this means:

```
eigenvalues(S^{-1/2} * rho * S^{-1/2}) >= 0
```

## Implementation

### Physics-Informed Loss Function

The `PhysicsInformedLoss` class combines prediction accuracy with physics penalties:

```python
L_total = w_mse * L_mse
        + w_herm * L_hermitian
        + w_idemp * L_idempotency
        + w_trace * L_trace
        + w_pos * L_positivity
```

#### MSE Loss

Standard mean squared error between predicted and target density matrices:

```python
L_mse = mean(|rho_pred - rho_target|^2)
```

Applied separately to real and imaginary parts, then averaged.

#### Hermiticity Loss

```python
L_hermitian = mean(|rho - rho^dagger|_F^2)
```

Measures deviation from Hermitian symmetry using the Frobenius norm.

#### Idempotency Loss

```python
L_idempotency = mean(|rho * S * rho - rho|_F^2)
```

Penalizes deviation from the idempotency condition.

#### Trace Loss

```python
L_trace = mean((Tr(rho * S) - N_electrons)^2)
```

Penalizes deviation from the correct electron count. Computed per spin channel.

#### Positivity Loss (Optional)

```python
eigenvalues = eigenvalues(rho)
L_positivity = mean(sum(ReLU(-eigenvalues))^2)
```

Penalizes negative eigenvalues.

### Inference-Time Projections

While the loss function encourages constraint satisfaction, exact enforcement is achieved via projections at inference time.

#### Hermitian Projection

```python
def hermitian_projection(rho):
    return (rho + rho.conj().transpose(-2, -1)) / 2
```

Projects to the nearest Hermitian matrix. This is an exact projection.

#### McWeeney Purification

```python
def mcweeney_purification(rho, S, n_iterations=1):
    for _ in range(n_iterations):
        rho_S = rho @ S
        rho_S_rho = rho_S @ rho
        rho = 3 * rho_S_rho - 2 * rho_S @ rho_S_rho
    return rho
```

The McWeeney purification iteratively enforces idempotency. Each iteration:
- Reduces idempotency error
- Preserves Hermiticity (if input is Hermitian)
- May slightly change trace

For near-idempotent inputs (trained models), 1-2 iterations are sufficient.

#### Trace Scaling

```python
def trace_scaling(rho, S, n_electrons):
    current_trace = torch.einsum('...ij,...ji->...', rho, S).real
    scale = n_electrons / current_trace
    return rho * scale.unsqueeze(-1).unsqueeze(-1)
```

Scales the density matrix to achieve exactly the target electron count. Per-spin scaling:
- `n_electrons = [n_alpha, n_beta]` for open-shell systems

#### Positivity Projection

```python
def positivity_projection(rho):
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.conj().T
```

Projects to the nearest positive semidefinite matrix by clamping negative eigenvalues.

## Recommended Settings

### Training

For training, use loss penalties only:

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

Notes:
- `weight_mse` should be largest (primary objective)
- `weight_hermitian` is cheap to compute, use moderate weight
- `weight_idempotency` is expensive, use smaller weight
- `weight_trace` helps maintain correct electron count
- `weight_positivity` is expensive and often not needed

### Inference

For inference, apply projections in order:

```python
predictor = DensityPredictor(
    checkpoint_path='model.pt',
    apply_hermitian=True,      # Always recommended
    apply_trace_scaling=True,  # If charge conservation is critical
    apply_mcweeney=False       # Only if idempotency needed
)
```

The projection order matters:
1. Hermitian (preserves symmetry)
2. Trace scaling (fixes electron count)
3. McWeeney (enforces idempotency, may change trace)

If using McWeeney, consider applying trace scaling afterward.

## Physical Interpretation

### Why These Constraints Matter

1. **Hermiticity**: Required for all observables to be real-valued. The dipole moment, for example, is computed as Tr(rho * mu) and must be real.

2. **Idempotency**: Ensures the density represents a valid single-determinant state. Violation indicates mixing with excited states or numerical artifacts.

3. **Trace Conservation**: Guarantees charge conservation. Violation would mean electrons are being created or destroyed.

4. **Positivity**: Ensures occupation numbers are physical (between 0 and 1 for normalized orbitals).

### RT-TDDFT Context

In RT-TDDFT, the density matrix evolves according to the Liouville-von Neumann equation:

```
i * hbar * d(rho)/dt = [H(rho), rho]
```

This evolution:
- Preserves Hermiticity (if H is Hermitian)
- Preserves trace (number of electrons)
- Preserves idempotency (for single-determinant states)
- Preserves positivity

The neural network learns to approximate this evolution while respecting these constraints.
