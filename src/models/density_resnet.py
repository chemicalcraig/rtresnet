"""
DensityResNet: Physics-informed ResNet for density matrix evolution.

This is the main model that predicts the next density matrix given:
- History of density matrices
- Overlap matrix S (molecular geometry/basis)
- External field history E(t)

The model incorporates physics constraints through:
- Optional projections at inference time (Hermitian, McWeeney, trace scaling)
- Physics-informed loss penalties during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any

from .encoders import OverlapEncoder, FieldEncoder, DensityHistoryEncoder
from .resnet_blocks import ResNetStack, DensityResBlock


class DensityResNet(nn.Module):
    """Physics-informed ResNet for density matrix evolution prediction.

    Architecture:
    1. Encode density history -> hidden_dim
    2. Encode overlap matrix S -> hidden_dim
    3. Encode field history E(t) -> hidden_dim
    4. Fuse encodings via concatenation + MLP
    5. Process through ResNet blocks
    6. Project to output density matrix shape
    7. Add residual connection from most recent input
    8. Apply physics projections (inference only)

    Args:
        max_nbf: Maximum number of basis functions (for padding)
        hidden_dim: Hidden dimension for all internal representations
        num_resnet_blocks: Number of ResNet blocks. Default: 6
        history_length: Number of timesteps in input history. Default: 5
        n_spin: Number of spin channels (1=closed, 2=open). Default: 2
        dropout: Dropout probability. Default: 0.1
        overlap_mode: Overlap encoding mode ('spectral', 'direct', 'cholesky'). Default: 'spectral'
        field_dim: Dimension of field vector (Ex, Ey, Ez). Default: 3
        apply_hermitian_projection: Apply Hermitian projection at inference. Default: True
        apply_mcweeney_projection: Apply McWeeney purification at inference. Default: False
        apply_trace_scaling: Apply trace scaling at inference. Default: False
        n_electrons: Target electron count per spin [n_alpha, n_beta]. Default: None

    Shape:
        - density_history: (batch, history_length, n_spin, nbf, nbf) complex
        - overlap_matrix: (batch, nbf, nbf) real
        - field_history: (batch, history_length, field_dim) real
        - Output: (batch, n_spin, nbf, nbf) complex
    """

    def __init__(
        self,
        max_nbf: int,
        hidden_dim: int = 256,
        num_resnet_blocks: int = 6,
        history_length: int = 5,
        n_spin: int = 2,
        dropout: float = 0.1,
        overlap_mode: str = 'spectral',
        field_dim: int = 3,
        apply_hermitian_projection: bool = True,
        apply_mcweeney_projection: bool = False,
        apply_trace_scaling: bool = False,
        n_electrons: Optional[List[float]] = None
    ):
        super().__init__()

        # Store configuration
        self.max_nbf = max_nbf
        self.hidden_dim = hidden_dim
        self.num_resnet_blocks = num_resnet_blocks
        self.history_length = history_length
        self.n_spin = n_spin
        self.dropout_rate = dropout
        self.overlap_mode = overlap_mode
        self.field_dim = field_dim

        # Physics projection settings (inference only)
        self.apply_hermitian_projection = apply_hermitian_projection
        self.apply_mcweeney_projection = apply_mcweeney_projection
        self.apply_trace_scaling = apply_trace_scaling

        if n_electrons is not None:
            self.register_buffer('n_electrons', torch.tensor(n_electrons))
        else:
            self.n_electrons = None

        # === Encoders ===
        # Density history encoder
        self.density_encoder = DensityHistoryEncoder(
            max_nbf=max_nbf,
            n_spin=n_spin,
            hidden_dim=hidden_dim,
            history_length=history_length
        )

        # Overlap matrix encoder
        self.overlap_encoder = OverlapEncoder(
            max_nbf=max_nbf,
            hidden_dim=hidden_dim,
            mode=overlap_mode
        )

        # Field encoder
        self.field_encoder = FieldEncoder(
            hidden_dim=hidden_dim,
            history_length=history_length,
            field_dim=field_dim
        )

        # === Fusion Layer ===
        # Combine three encodings: density + overlap + field
        fusion_input_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # === ResNet Blocks ===
        self.resnet_stack = ResNetStack(
            hidden_dim=hidden_dim,
            num_blocks=num_resnet_blocks,
            expansion=4,
            dropout=dropout,
            block_type='density'
        )

        # === Output Projection ===
        # Project back to density matrix shape
        # Output: n_spin * nbf * nbf * 2 (for real and imag parts)
        output_size = n_spin * max_nbf * max_nbf * 2
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_size)
        )

        self._init_output_weights()

    def _init_output_weights(self):
        """Initialize output projection for small initial predictions."""
        # Small initialization helps with residual learning
        for module in self.output_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        density_history: torch.Tensor,
        overlap_matrix: torch.Tensor,
        field_history: torch.Tensor,
        apply_projections: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Args:
            density_history: (batch, history_length, n_spin, nbf, nbf) complex
            overlap_matrix: (batch, nbf, nbf) real
            field_history: (batch, history_length, field_dim) real
            apply_projections: Override for applying physics projections.
                              If None, uses self.training to decide (projections in eval only)

        Returns:
            Predicted density matrix (batch, n_spin, nbf, nbf) complex
        """
        batch_size = density_history.shape[0]
        nbf = density_history.shape[-1]

        # === Encode inputs ===
        # Density history encoding
        density_encoding = self.density_encoder(density_history)

        # Overlap matrix encoding
        overlap_encoding = self.overlap_encoder(overlap_matrix)

        # Field history encoding
        field_encoding = self.field_encoder(field_history)

        # === Fusion ===
        fused = torch.cat([density_encoding, overlap_encoding, field_encoding], dim=-1)
        fused = self.fusion(fused)

        # === ResNet processing ===
        features = self.resnet_stack(fused)

        # === Output projection ===
        output = self.output_proj(features)

        # Reshape to (batch, n_spin, max_nbf, max_nbf, 2)
        output = output.view(batch_size, self.n_spin, self.max_nbf, self.max_nbf, 2)

        # Convert to complex
        output_complex = torch.complex(output[..., 0], output[..., 1])

        # Crop to actual size if needed
        if nbf < self.max_nbf:
            output_complex = output_complex[..., :nbf, :nbf]

        # === Residual connection ===
        # Add prediction to most recent density matrix
        most_recent = density_history[:, -1]  # (batch, n_spin, nbf, nbf)
        output_complex = output_complex + most_recent

        # === Physics projections (inference only) ===
        should_apply = apply_projections if apply_projections is not None else (not self.training)

        if should_apply:
            output_complex = self._apply_physics_projections(output_complex, overlap_matrix)

        return output_complex

    def _apply_physics_projections(
        self,
        rho: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """Apply physics projections at inference time.

        Args:
            rho: Predicted density matrix (batch, n_spin, nbf, nbf) complex
            S: Overlap matrix (batch, nbf, nbf) real

        Returns:
            Projected density matrix
        """
        # Import here to avoid circular imports
        # Use try/except to handle different import scenarios
        try:
            from physics.projections import (
                hermitian_projection,
                mcweeney_purification,
                trace_scaling
            )
        except ImportError:
            from ..physics.projections import (
                hermitian_projection,
                mcweeney_purification,
                trace_scaling
            )

        # Hermitian projection (always cheap and important)
        if self.apply_hermitian_projection:
            rho = hermitian_projection(rho)

        # Trace scaling
        if self.apply_trace_scaling and self.n_electrons is not None:
            rho = trace_scaling(rho, S, self.n_electrons)

        # McWeeney purification (expensive, use sparingly)
        if self.apply_mcweeney_projection:
            rho = mcweeney_purification(rho, S, n_iterations=1)

        return rho

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as dictionary."""
        return {
            'max_nbf': self.max_nbf,
            'hidden_dim': self.hidden_dim,
            'num_resnet_blocks': self.num_resnet_blocks,
            'history_length': self.history_length,
            'n_spin': self.n_spin,
            'dropout': self.dropout_rate,
            'overlap_mode': self.overlap_mode,
            'field_dim': self.field_dim,
            'apply_hermitian_projection': self.apply_hermitian_projection,
            'apply_mcweeney_projection': self.apply_mcweeney_projection,
            'apply_trace_scaling': self.apply_trace_scaling,
            'n_electrons': self.n_electrons.tolist() if self.n_electrons is not None else None,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DensityResNet':
        """Create model from configuration dictionary."""
        return cls(**config)

    def extra_repr(self) -> str:
        return (f'max_nbf={self.max_nbf}, hidden_dim={self.hidden_dim}, '
                f'num_blocks={self.num_resnet_blocks}, history={self.history_length}, '
                f'n_spin={self.n_spin}')


def create_model_from_config(config: Dict[str, Any]) -> DensityResNet:
    """Create a DensityResNet model from a full configuration dictionary.

    Expected config structure:
    {
        "model": {
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
            "n_electrons": [1.0, 0.0]
        }
    }

    Args:
        config: Full configuration dictionary

    Returns:
        Configured DensityResNet model
    """
    model_config = config.get('model', {})
    physics_config = config.get('physics', {})

    return DensityResNet(
        max_nbf=model_config.get('max_nbf', 16),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_resnet_blocks=model_config.get('num_resnet_blocks', 6),
        history_length=model_config.get('history_length', 5),
        n_spin=model_config.get('n_spin', 2),
        dropout=model_config.get('dropout', 0.1),
        overlap_mode=model_config.get('overlap_mode', 'spectral'),
        field_dim=model_config.get('field_dim', 3),
        apply_hermitian_projection=model_config.get('apply_hermitian_projection', True),
        apply_mcweeney_projection=model_config.get('apply_mcweeney_projection', False),
        apply_trace_scaling=model_config.get('apply_trace_scaling', False),
        n_electrons=physics_config.get('n_electrons', None),
    )
