"""
Physics-based projection operators for density matrices.

These projections enforce physical constraints on the predicted density matrix:
- Hermiticity: rho = rho^dagger
- Idempotency: rho @ S @ rho = rho (in non-orthonormal AO basis)
- Trace conservation: Tr(rho @ S) = N_electrons
- Positivity: rho is positive semi-definite

These can be applied as hard projections at inference time or used
to compute penalty terms during training.
"""

import torch
from typing import Optional, Union, List


def hermitian_projection(rho: torch.Tensor) -> torch.Tensor:
    """Project density matrix to Hermitian form.

    Computes rho_hermitian = (rho + rho^dagger) / 2

    This is the simplest and most fundamental constraint for a density matrix.

    Args:
        rho: Complex density matrix of shape (..., n, n)

    Returns:
        Hermitian density matrix of same shape
    """
    return (rho + rho.conj().transpose(-2, -1)) / 2


def mcweeney_purification(
    rho: torch.Tensor,
    S: torch.Tensor,
    n_iterations: int = 1
) -> torch.Tensor:
    """McWeeney purification to enforce idempotency.

    For a density matrix in a non-orthonormal AO basis, the idempotency
    condition is: rho @ S @ rho = rho

    The McWeeney purification iteration is:
        rho' = 3 * rho @ S @ rho - 2 * rho @ S @ rho @ S @ rho

    This converges quadratically to an idempotent matrix.

    Note: This projection is typically only applied at inference time,
    not during training, as the gradients can be problematic.

    Args:
        rho: Complex density matrix of shape (batch, n_spin, nbf, nbf)
        S: Real overlap matrix of shape (batch, nbf, nbf) or (nbf, nbf)
        n_iterations: Number of purification iterations. Default: 1

    Returns:
        Purified density matrix of same shape as rho
    """
    # Ensure S has the right shape for broadcasting
    if S.dim() == 2:
        # (nbf, nbf) -> (1, 1, nbf, nbf)
        S = S.unsqueeze(0).unsqueeze(0)
    elif S.dim() == 3:
        # (batch, nbf, nbf) -> (batch, 1, nbf, nbf)
        S = S.unsqueeze(1)

    # Convert S to complex for matrix multiplication with complex rho
    S_complex = S.to(rho.dtype)

    for _ in range(n_iterations):
        # rho @ S
        rho_S = torch.matmul(rho, S_complex)
        # rho @ S @ rho
        rho_S_rho = torch.matmul(rho_S, rho)
        # rho @ S @ rho @ S @ rho
        rho_S_rho_S_rho = torch.matmul(torch.matmul(rho_S_rho, S_complex), rho)
        # McWeeney update
        rho = 3 * rho_S_rho - 2 * rho_S_rho_S_rho

    return rho


def trace_scaling(
    rho: torch.Tensor,
    S: torch.Tensor,
    n_electrons: Union[float, List[float], torch.Tensor],
    eps: float = 1e-8
) -> torch.Tensor:
    """Scale density matrix to conserve electron number.

    Enforces Tr(rho @ S) = N_electrons by scaling:
        rho_scaled = rho * (N_electrons / Tr(rho @ S))

    For open-shell systems, n_electrons can be a list [n_alpha, n_beta]
    to scale each spin channel independently.

    Args:
        rho: Complex density matrix of shape (batch, n_spin, nbf, nbf)
        S: Real overlap matrix of shape (batch, nbf, nbf) or (nbf, nbf)
        n_electrons: Target number of electrons. Can be:
            - float: Same target for all spin channels
            - List[float]: Per-spin targets [n_alpha, n_beta]
            - Tensor: Shape (n_spin,) for per-spin targets
        eps: Small constant to avoid division by zero

    Returns:
        Scaled density matrix with correct trace
    """
    # Ensure S has the right shape
    if S.dim() == 2:
        # (nbf, nbf) -> (1, nbf, nbf)
        S = S.unsqueeze(0)

    # Compute trace: Tr(rho @ S) = sum_ij rho_ij * S_ji
    # Using einsum for clarity
    # rho: (batch, n_spin, nbf, nbf), S: (batch, nbf, nbf)
    trace_rhoS = torch.einsum('bsij,bji->bs', rho, S.to(rho.dtype)).real

    # Handle n_electrons input
    if isinstance(n_electrons, (int, float)):
        target = torch.tensor(n_electrons, device=rho.device, dtype=trace_rhoS.dtype)
        target = target.expand_as(trace_rhoS)
    elif isinstance(n_electrons, list):
        target = torch.tensor(n_electrons, device=rho.device, dtype=trace_rhoS.dtype)
        # Shape: (n_spin,) -> (1, n_spin) for broadcasting
        target = target.unsqueeze(0).expand_as(trace_rhoS)
    else:
        target = n_electrons.to(trace_rhoS.dtype)
        if target.dim() == 1:
            target = target.unsqueeze(0).expand_as(trace_rhoS)

    # Compute scaling factor
    scale = target / (trace_rhoS + eps)

    # Apply scaling: (batch, n_spin) -> (batch, n_spin, 1, 1)
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    return rho * scale


def positivity_projection(
    rho: torch.Tensor,
    eps: float = 0.0
) -> torch.Tensor:
    """Project density matrix to positive semi-definite form.

    Eigendecomposes the Hermitian part of rho, clamps negative eigenvalues
    to eps (default 0), and reconstructs the matrix.

    This is a relatively expensive operation due to eigendecomposition.

    Args:
        rho: Complex density matrix of shape (..., n, n)
        eps: Minimum eigenvalue (clamp negatives to this). Default: 0.0

    Returns:
        Positive semi-definite density matrix
    """
    # First ensure Hermiticity
    rho_herm = hermitian_projection(rho)

    # Get original shape
    original_shape = rho_herm.shape
    nbf = original_shape[-1]

    # Flatten batch dimensions for eigendecomposition
    rho_flat = rho_herm.reshape(-1, nbf, nbf)

    # Eigendecomposition of Hermitian matrix
    # eigenvalues are real, eigenvectors are complex
    eigenvalues, eigenvectors = torch.linalg.eigh(rho_flat)

    # Clamp negative eigenvalues
    eigenvalues_clamped = torch.clamp(eigenvalues, min=eps)

    # Reconstruct: V @ diag(lambda) @ V^H
    # eigenvalues_clamped: (batch, nbf)
    # eigenvectors: (batch, nbf, nbf)
    rho_positive = torch.matmul(
        eigenvectors * eigenvalues_clamped.unsqueeze(-2),
        eigenvectors.conj().transpose(-2, -1)
    )

    # Reshape back to original shape
    return rho_positive.reshape(original_shape)


def compute_idempotency_error(
    rho: torch.Tensor,
    S: torch.Tensor
) -> torch.Tensor:
    """Compute the idempotency error ||rho @ S @ rho - rho||_F.

    For a density matrix in non-orthonormal basis, the idempotency
    condition is rho @ S @ rho = rho.

    Args:
        rho: Complex density matrix of shape (batch, n_spin, nbf, nbf)
        S: Real overlap matrix of shape (batch, nbf, nbf) or (nbf, nbf)

    Returns:
        Frobenius norm of the idempotency error, shape (batch, n_spin)
    """
    if S.dim() == 2:
        S = S.unsqueeze(0).unsqueeze(0)
    elif S.dim() == 3:
        S = S.unsqueeze(1)

    S_complex = S.to(rho.dtype)
    rho_S_rho = torch.matmul(torch.matmul(rho, S_complex), rho)
    error = rho_S_rho - rho

    # Frobenius norm per (batch, spin)
    return torch.sqrt((error.abs() ** 2).sum(dim=(-2, -1)))


def compute_hermiticity_error(rho: torch.Tensor) -> torch.Tensor:
    """Compute the Hermiticity error ||rho - rho^dagger||_F.

    Args:
        rho: Complex density matrix of shape (..., n, n)

    Returns:
        Frobenius norm of the Hermiticity error
    """
    error = rho - rho.conj().transpose(-2, -1)
    # Sum over matrix dimensions, keep batch dimensions
    return torch.sqrt((error.abs() ** 2).sum(dim=(-2, -1)))


def compute_trace(
    rho: torch.Tensor,
    S: torch.Tensor
) -> torch.Tensor:
    """Compute Tr(rho @ S) for density matrix in non-orthonormal basis.

    Args:
        rho: Complex density matrix of shape (batch, n_spin, nbf, nbf)
        S: Real overlap matrix of shape (batch, nbf, nbf) or (nbf, nbf)

    Returns:
        Trace values of shape (batch, n_spin), real-valued
    """
    if S.dim() == 2:
        S = S.unsqueeze(0)

    return torch.einsum('bsij,bji->bs', rho, S.to(rho.dtype)).real


def validate_density_matrix(
    rho: torch.Tensor,
    S: torch.Tensor,
    n_electrons: Optional[Union[float, List[float]]] = None,
    verbose: bool = True
) -> dict:
    """Validate physical properties of a density matrix.

    Checks:
    - Hermiticity error
    - Idempotency error
    - Trace (electron number)
    - Eigenvalue spectrum (positivity)

    Args:
        rho: Complex density matrix of shape (batch, n_spin, nbf, nbf)
        S: Real overlap matrix
        n_electrons: Expected number of electrons (optional)
        verbose: If True, print validation results

    Returns:
        Dictionary with validation metrics
    """
    results = {}

    # Hermiticity
    herm_error = compute_hermiticity_error(rho)
    results['hermiticity_error'] = herm_error.mean().item()
    results['hermiticity_max'] = herm_error.max().item()

    # Idempotency
    idemp_error = compute_idempotency_error(rho, S)
    results['idempotency_error'] = idemp_error.mean().item()
    results['idempotency_max'] = idemp_error.max().item()

    # Trace
    trace = compute_trace(rho, S)
    results['trace_mean'] = trace.mean().item()
    results['trace_std'] = trace.std().item()
    if n_electrons is not None:
        if isinstance(n_electrons, list):
            target = torch.tensor(n_electrons, device=trace.device)
        else:
            target = n_electrons
        results['trace_error'] = (trace - target).abs().mean().item()

    # Eigenvalues (positivity check)
    rho_herm = hermitian_projection(rho)
    rho_flat = rho_herm.reshape(-1, rho.shape[-1], rho.shape[-1])
    eigenvalues = torch.linalg.eigvalsh(rho_flat)
    results['min_eigenvalue'] = eigenvalues.min().item()
    results['max_eigenvalue'] = eigenvalues.max().item()
    results['negative_eigenvalue_count'] = (eigenvalues < -1e-10).sum().item()

    if verbose:
        print("Density Matrix Validation:")
        print(f"  Hermiticity error: {results['hermiticity_error']:.2e} (max: {results['hermiticity_max']:.2e})")
        print(f"  Idempotency error: {results['idempotency_error']:.2e} (max: {results['idempotency_max']:.2e})")
        print(f"  Trace: {results['trace_mean']:.4f} +/- {results['trace_std']:.4f}")
        if 'trace_error' in results:
            print(f"  Trace error: {results['trace_error']:.2e}")
        print(f"  Eigenvalues: [{results['min_eigenvalue']:.4f}, {results['max_eigenvalue']:.4f}]")
        if results['negative_eigenvalue_count'] > 0:
            print(f"  WARNING: {results['negative_eigenvalue_count']} negative eigenvalues detected")

    return results
