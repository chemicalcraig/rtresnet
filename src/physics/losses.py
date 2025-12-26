"""
Physics-informed loss functions for density matrix prediction.

The loss function combines:
- MSE reconstruction loss between predicted and target density matrices
- Physics constraint penalties (Hermiticity, idempotency, trace, positivity)

Physics constraints are applied as soft penalties during training,
while hard projections can be applied at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from .projections import (
    compute_hermiticity_error,
    compute_idempotency_error,
    compute_trace,
)


class PhysicsInformedLoss(nn.Module):
    """Combined loss with physics constraint penalties.

    The total loss is:
        L = w_mse * L_mse
          + w_hermitian * L_hermitian
          + w_idempotency * L_idempotency
          + w_trace * L_trace
          + w_positivity * L_positivity

    Where:
        L_mse = MSE(pred.real, target.real) + MSE(pred.imag, target.imag)
        L_hermitian = ||rho - rho^dagger||_F^2
        L_idempotency = ||rho @ S @ rho - rho||_F^2
        L_trace = (Tr(rho @ S) - N_electrons)^2
        L_positivity = sum(ReLU(-eigenvalues))^2

    Args:
        weight_mse: Weight for MSE loss. Default: 1.0
        weight_hermitian: Weight for Hermiticity penalty. Default: 0.1
        weight_idempotency: Weight for idempotency penalty. Default: 0.05
        weight_trace: Weight for trace penalty. Default: 0.1
        weight_positivity: Weight for positivity penalty. Default: 0.01
        n_electrons: Target electron count per spin channel. Can be:
            - float: Same for all spin channels
            - List[float]: Per-spin targets [n_alpha, n_beta]
            Default: None (trace loss disabled)
        reduction: How to reduce batch dimension. Default: 'mean'
    """

    def __init__(
        self,
        weight_mse: float = 1.0,
        weight_hermitian: float = 0.1,
        weight_idempotency: float = 0.05,
        weight_trace: float = 0.1,
        weight_positivity: float = 0.01,
        n_electrons: Optional[Union[float, List[float]]] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_hermitian = weight_hermitian
        self.weight_idempotency = weight_idempotency
        self.weight_trace = weight_trace
        self.weight_positivity = weight_positivity
        self.n_electrons = n_electrons
        self.reduction = reduction

        # Register n_electrons as buffer if it's a list/tensor
        if isinstance(n_electrons, list):
            self.register_buffer('_n_electrons', torch.tensor(n_electrons))
        elif isinstance(n_electrons, torch.Tensor):
            self.register_buffer('_n_electrons', n_electrons)
        else:
            self._n_electrons = n_electrons

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        overlap_matrix: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """Compute the physics-informed loss.

        Args:
            pred: Predicted density matrix (batch, n_spin, nbf, nbf) complex
            target: Ground truth density matrix (batch, n_spin, nbf, nbf) complex
            overlap_matrix: AO overlap matrix (batch, nbf, nbf) or (nbf, nbf) real
            return_components: If True, also return dict of individual loss components

        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: (optional) Dictionary with individual loss values
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.real.dtype)

        # 1. MSE Loss on complex density matrix
        if self.weight_mse > 0:
            mse_real = F.mse_loss(pred.real, target.real, reduction=self.reduction)
            mse_imag = F.mse_loss(pred.imag, target.imag, reduction=self.reduction)
            mse_loss = mse_real + mse_imag
            loss_dict['mse'] = mse_loss
            total_loss = total_loss + self.weight_mse * mse_loss

        # 2. Hermiticity penalty: ||rho - rho^dagger||_F^2
        if self.weight_hermitian > 0:
            hermitian_error = pred - pred.conj().transpose(-2, -1)
            hermitian_loss = (hermitian_error.abs() ** 2).mean()
            loss_dict['hermitian'] = hermitian_loss
            total_loss = total_loss + self.weight_hermitian * hermitian_loss

        # 3. Idempotency penalty: ||rho @ S @ rho - rho||_F^2
        if self.weight_idempotency > 0:
            idempotency_loss = self._compute_idempotency_loss(pred, overlap_matrix)
            loss_dict['idempotency'] = idempotency_loss
            total_loss = total_loss + self.weight_idempotency * idempotency_loss

        # 4. Trace penalty: (Tr(rho @ S) - N_electrons)^2
        if self.weight_trace > 0 and self._n_electrons is not None:
            trace_loss = self._compute_trace_loss(pred, overlap_matrix)
            loss_dict['trace'] = trace_loss
            total_loss = total_loss + self.weight_trace * trace_loss

        # 5. Positivity penalty: sum(ReLU(-eigenvalues))^2
        if self.weight_positivity > 0:
            positivity_loss = self._compute_positivity_loss(pred)
            loss_dict['positivity'] = positivity_loss
            total_loss = total_loss + self.weight_positivity * positivity_loss

        loss_dict['total'] = total_loss

        if return_components:
            return total_loss, loss_dict
        return total_loss

    def _compute_idempotency_loss(
        self,
        rho: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """Compute idempotency loss ||rho @ S @ rho - rho||_F^2."""
        # Ensure S has correct shape for broadcasting
        if S.dim() == 2:
            S = S.unsqueeze(0).unsqueeze(0)
        elif S.dim() == 3:
            S = S.unsqueeze(1)

        S_complex = S.to(rho.dtype)

        # rho @ S @ rho
        rho_S = torch.matmul(rho, S_complex)
        rho_S_rho = torch.matmul(rho_S, rho)

        # Error
        error = rho_S_rho - rho

        # Mean squared error
        if self.reduction == 'mean':
            return (error.abs() ** 2).mean()
        elif self.reduction == 'sum':
            return (error.abs() ** 2).sum()
        else:
            return (error.abs() ** 2).mean(dim=(-2, -1))

    def _compute_trace_loss(
        self,
        rho: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """Compute trace loss (Tr(rho @ S) - N_electrons)^2."""
        trace = compute_trace(rho, S)  # (batch, n_spin)

        # Get target
        if isinstance(self._n_electrons, torch.Tensor):
            target = self._n_electrons.to(trace.device)
            if target.dim() == 1:
                target = target.unsqueeze(0)  # (1, n_spin)
        else:
            target = self._n_electrons

        # Squared error
        trace_error = (trace - target) ** 2

        if self.reduction == 'mean':
            return trace_error.mean()
        elif self.reduction == 'sum':
            return trace_error.sum()
        else:
            return trace_error

    def _compute_positivity_loss(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute positivity loss: penalize negative eigenvalues."""
        # First make Hermitian for eigenvalue computation
        rho_herm = (rho + rho.conj().transpose(-2, -1)) / 2

        # Flatten batch and spin dimensions
        batch_size, n_spin, nbf, _ = rho.shape
        rho_flat = rho_herm.reshape(-1, nbf, nbf)

        # Compute eigenvalues (real for Hermitian matrices)
        eigenvalues = torch.linalg.eigvalsh(rho_flat)

        # Penalize negative eigenvalues: sum(ReLU(-lambda))^2
        negative_penalty = F.relu(-eigenvalues) ** 2

        if self.reduction == 'mean':
            return negative_penalty.mean()
        elif self.reduction == 'sum':
            return negative_penalty.sum()
        else:
            return negative_penalty.mean(dim=-1).reshape(batch_size, n_spin)

    def extra_repr(self) -> str:
        return (f'w_mse={self.weight_mse}, w_hermitian={self.weight_hermitian}, '
                f'w_idempotency={self.weight_idempotency}, w_trace={self.weight_trace}, '
                f'w_positivity={self.weight_positivity}, n_electrons={self._n_electrons}')


class MSEComplexLoss(nn.Module):
    """Simple MSE loss for complex tensors.

    Computes MSE separately on real and imaginary parts and sums them.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_real = F.mse_loss(pred.real, target.real, reduction=self.reduction)
        mse_imag = F.mse_loss(pred.imag, target.imag, reduction=self.reduction)
        return mse_real + mse_imag


class FrobeniusComplexLoss(nn.Module):
    """Frobenius norm loss for complex matrices.

    Computes ||pred - target||_F for complex matrices.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
        squared: If True, return squared Frobenius norm. Default: True
    """

    def __init__(self, reduction: str = 'mean', squared: bool = True):
        super().__init__()
        self.reduction = reduction
        self.squared = squared

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        frobenius_sq = (diff.abs() ** 2).sum(dim=(-2, -1))

        if not self.squared:
            frobenius = torch.sqrt(frobenius_sq)
            values = frobenius
        else:
            values = frobenius_sq

        if self.reduction == 'mean':
            return values.mean()
        elif self.reduction == 'sum':
            return values.sum()
        else:
            return values


def create_loss_from_config(config: dict) -> PhysicsInformedLoss:
    """Create a PhysicsInformedLoss from a configuration dictionary.

    Expected config structure:
    {
        "loss": {
            "weight_mse": 1.0,
            "weight_hermitian": 0.1,
            "weight_idempotency": 0.05,
            "weight_trace": 0.1,
            "weight_positivity": 0.01
        },
        "physics": {
            "n_electrons": [1.0, 0.0]  # or single float
        }
    }

    Args:
        config: Configuration dictionary

    Returns:
        Configured PhysicsInformedLoss instance
    """
    loss_config = config.get('loss', {})
    physics_config = config.get('physics', {})

    return PhysicsInformedLoss(
        weight_mse=loss_config.get('weight_mse', 1.0),
        weight_hermitian=loss_config.get('weight_hermitian', 0.1),
        weight_idempotency=loss_config.get('weight_idempotency', 0.05),
        weight_trace=loss_config.get('weight_trace', 0.1),
        weight_positivity=loss_config.get('weight_positivity', 0.01),
        n_electrons=physics_config.get('n_electrons', None),
    )
