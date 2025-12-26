"""
Encoder modules for conditioning information.

This module provides encoders for:
- Overlap matrix S: Encodes molecular geometry and basis set information
- External field E(t): Encodes time-dependent driving field

These encoders produce conditioning vectors that are fused with the
density matrix representation in the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional


class OverlapEncoder(nn.Module):
    """Encode the AO overlap matrix S into a conditioning vector.

    The overlap matrix S encodes:
    - Molecular geometry (via basis function overlaps)
    - Basis set characteristics

    For fixed-molecule training, S is constant and provides structural info.
    For generalization across molecules, S provides molecule-specific conditioning.

    Encoding modes:
    - 'spectral': Use eigenvalue decomposition (recommended for generalization)
    - 'direct': Flatten upper triangle and encode directly
    - 'cholesky': Use Cholesky factor L where S = L @ L.T

    Args:
        max_nbf: Maximum number of basis functions (for padding)
        hidden_dim: Output dimension of the encoding
        mode: Encoding mode ('spectral', 'direct', 'cholesky'). Default: 'spectral'

    Shape:
        - Input: (batch, nbf, nbf) real overlap matrix
        - Output: (batch, hidden_dim) conditioning vector
    """

    def __init__(
        self,
        max_nbf: int,
        hidden_dim: int,
        mode: Literal['spectral', 'direct', 'cholesky'] = 'spectral'
    ):
        super().__init__()
        self.max_nbf = max_nbf
        self.hidden_dim = hidden_dim
        self.mode = mode

        if mode == 'spectral':
            # Encode eigenvalues and eigenvectors separately
            self.eigenvalue_encoder = nn.Sequential(
                nn.Linear(max_nbf, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            self.eigenvector_encoder = nn.Sequential(
                nn.Linear(max_nbf * max_nbf, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        elif mode == 'direct':
            # Flatten upper triangle (including diagonal)
            # Number of elements: n*(n+1)/2
            n_elements = max_nbf * (max_nbf + 1) // 2
            self.encoder = nn.Sequential(
                nn.Linear(n_elements, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        elif mode == 'cholesky':
            # Cholesky factor L is lower triangular
            # Number of elements: n*(n+1)/2
            n_elements = max_nbf * (max_nbf + 1) // 2
            self.encoder = nn.Sequential(
                nn.Linear(n_elements, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: Overlap matrix of shape (batch, nbf, nbf)

        Returns:
            Encoding vector of shape (batch, hidden_dim)
        """
        batch_size, nbf, _ = S.shape

        if self.mode == 'spectral':
            return self._spectral_encode(S, batch_size, nbf)
        elif self.mode == 'direct':
            return self._direct_encode(S, batch_size, nbf)
        elif self.mode == 'cholesky':
            return self._cholesky_encode(S, batch_size, nbf)

    def _spectral_encode(
        self,
        S: torch.Tensor,
        batch_size: int,
        nbf: int
    ) -> torch.Tensor:
        """Encode using eigendecomposition."""
        # Compute eigendecomposition (S is symmetric positive definite)
        eigenvalues, eigenvectors = torch.linalg.eigh(S)

        # Pad eigenvalues if needed
        if nbf < self.max_nbf:
            pad_size = self.max_nbf - nbf
            eigenvalues = F.pad(eigenvalues, (0, pad_size), value=0.0)
            # Pad eigenvectors: (batch, nbf, nbf) -> (batch, max_nbf, max_nbf)
            eigenvectors = F.pad(eigenvectors, (0, pad_size, 0, pad_size), value=0.0)

        # Flatten eigenvectors
        eigenvectors_flat = eigenvectors.reshape(batch_size, -1)

        # Encode separately
        ev_encoding = self.eigenvalue_encoder(eigenvalues)
        vec_encoding = self.eigenvector_encoder(eigenvectors_flat)

        # Combine
        combined = torch.cat([ev_encoding, vec_encoding], dim=-1)
        return self.output_proj(combined)

    def _direct_encode(
        self,
        S: torch.Tensor,
        batch_size: int,
        nbf: int
    ) -> torch.Tensor:
        """Encode by flattening upper triangle."""
        # Extract upper triangular elements (including diagonal)
        # Use indices for the current size
        triu_indices = torch.triu_indices(nbf, nbf, device=S.device)
        S_triu = S[:, triu_indices[0], triu_indices[1]]

        # Pad if needed
        n_elements_current = nbf * (nbf + 1) // 2
        n_elements_max = self.max_nbf * (self.max_nbf + 1) // 2

        if n_elements_current < n_elements_max:
            S_triu = F.pad(S_triu, (0, n_elements_max - n_elements_current), value=0.0)

        return self.encoder(S_triu)

    def _cholesky_encode(
        self,
        S: torch.Tensor,
        batch_size: int,
        nbf: int
    ) -> torch.Tensor:
        """Encode using Cholesky decomposition."""
        # Compute Cholesky factor: S = L @ L.T
        L = torch.linalg.cholesky(S)

        # Extract lower triangular elements
        tril_indices = torch.tril_indices(nbf, nbf, device=S.device)
        L_tril = L[:, tril_indices[0], tril_indices[1]]

        # Pad if needed
        n_elements_current = nbf * (nbf + 1) // 2
        n_elements_max = self.max_nbf * (self.max_nbf + 1) // 2

        if n_elements_current < n_elements_max:
            L_tril = F.pad(L_tril, (0, n_elements_max - n_elements_current), value=0.0)

        return self.encoder(L_tril)

    def extra_repr(self) -> str:
        return f'max_nbf={self.max_nbf}, hidden_dim={self.hidden_dim}, mode={self.mode}'


class FieldEncoder(nn.Module):
    """Encode the external electric field E(t) history.

    The external field drives the electron dynamics. This encoder captures
    the temporal pattern of the applied field to condition the density
    matrix prediction.

    Architecture:
    1. Conv1D to extract local temporal features
    2. LSTM to capture sequential dependencies
    3. Linear projection to output dimension

    For field-free propagation (all zeros), the encoder should produce
    a near-zero conditioning vector.

    Args:
        hidden_dim: Output dimension of the encoding
        history_length: Number of time steps in the input history
        field_dim: Dimension of field at each timestep (default: 3 for Ex, Ey, Ez)
        num_lstm_layers: Number of LSTM layers. Default: 1

    Shape:
        - Input: (batch, history_length, field_dim) real tensor
        - Output: (batch, hidden_dim) conditioning vector
    """

    def __init__(
        self,
        hidden_dim: int,
        history_length: int,
        field_dim: int = 3,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.field_dim = field_dim

        # Intermediate dimension for Conv1D output
        conv_out_dim = hidden_dim // 2

        # Temporal convolution to extract local features
        # Input: (batch, field_dim, history_length)
        # Output: (batch, conv_out_dim, history_length)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(field_dim, conv_out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_out_dim, conv_out_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM for sequential dependencies
        # Input: (batch, history_length, conv_out_dim)
        # Output: hidden state (batch, hidden_dim // 2)
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field: External field history of shape (batch, history_length, field_dim)

        Returns:
            Encoding vector of shape (batch, hidden_dim)
        """
        batch_size = field.shape[0]

        # Check for zero field (common case for field-free propagation)
        # This helps the model learn that zero field means no driving
        field_magnitude = field.abs().sum()

        # Conv1D expects (batch, channels, length)
        x = field.transpose(1, 2)  # (batch, field_dim, history_length)

        # Temporal convolution
        x = self.temporal_conv(x)  # (batch, conv_out_dim, history_length)

        # Back to (batch, length, features) for LSTM
        x = x.transpose(1, 2)  # (batch, history_length, conv_out_dim)

        # LSTM - use final hidden state
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim//2)
        h_n = h_n[-1]  # Take last layer: (batch, hidden_dim//2)

        # Project to output dimension
        output = self.output_proj(h_n)

        # Layer normalization
        output = self.layer_norm(output)

        return output

    def extra_repr(self) -> str:
        return (f'hidden_dim={self.hidden_dim}, history_length={self.history_length}, '
                f'field_dim={self.field_dim}')


class FieldEncoderSimple(nn.Module):
    """Simpler field encoder using just MLPs.

    Alternative to the Conv+LSTM encoder for cases where the field
    pattern is simple or when computational efficiency is important.

    Args:
        hidden_dim: Output dimension of the encoding
        history_length: Number of time steps in the input history
        field_dim: Dimension of field at each timestep (default: 3)

    Shape:
        - Input: (batch, history_length, field_dim) real tensor
        - Output: (batch, hidden_dim) conditioning vector
    """

    def __init__(
        self,
        hidden_dim: int,
        history_length: int,
        field_dim: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.field_dim = field_dim

        input_dim = history_length * field_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field: External field history of shape (batch, history_length, field_dim)

        Returns:
            Encoding vector of shape (batch, hidden_dim)
        """
        # Flatten time and field dimensions
        x = field.reshape(field.shape[0], -1)
        return self.encoder(x)


class DensityHistoryEncoder(nn.Module):
    """Encode the history of density matrices.

    Takes a sequence of density matrices and produces a fixed-size
    encoding that captures the temporal evolution.

    Args:
        max_nbf: Maximum number of basis functions
        n_spin: Number of spin channels (1 or 2)
        hidden_dim: Output dimension
        history_length: Number of time steps in history

    Shape:
        - Input: (batch, history_length, n_spin, nbf, nbf) complex
        - Output: (batch, hidden_dim) real
    """

    def __init__(
        self,
        max_nbf: int,
        n_spin: int,
        hidden_dim: int,
        history_length: int
    ):
        super().__init__()
        self.max_nbf = max_nbf
        self.n_spin = n_spin
        self.hidden_dim = hidden_dim
        self.history_length = history_length

        # Input size: flattened complex matrix (real + imag)
        # Shape: history_length * n_spin * nbf * nbf * 2
        input_size = history_length * n_spin * max_nbf * max_nbf * 2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, rho_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho_history: Density matrix history (batch, history, n_spin, nbf, nbf) complex

        Returns:
            Encoding vector (batch, hidden_dim) real
        """
        batch_size = rho_history.shape[0]
        nbf = rho_history.shape[-1]

        # Pad if necessary
        if nbf < self.max_nbf:
            pad_size = self.max_nbf - nbf
            rho_history = F.pad(rho_history, (0, pad_size, 0, pad_size))

        # Flatten and separate real/imag
        rho_flat = rho_history.reshape(batch_size, -1)
        x = torch.cat([rho_flat.real, rho_flat.imag], dim=-1)

        return self.encoder(x)

    def extra_repr(self) -> str:
        return (f'max_nbf={self.max_nbf}, n_spin={self.n_spin}, '
                f'hidden_dim={self.hidden_dim}, history_length={self.history_length}')
