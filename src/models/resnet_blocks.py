"""
ResNet blocks for density matrix prediction.

These blocks operate on real-valued hidden representations (after the
density matrices have been flattened and embedded). The complex structure
is handled at the input/output level of the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResNetBlock(nn.Module):
    """Standard ResNet block for real-valued features.

    Architecture:
        x -> LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> + x
                                                                  |
                                                            (residual)

    Args:
        hidden_dim: Dimension of input and output features
        expansion: Expansion factor for intermediate layer. Default: 4
        dropout: Dropout probability. Default: 0.1

    Shape:
        - Input: (batch, hidden_dim)
        - Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        intermediate_dim = hidden_dim * expansion

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize output layer to small values for stable training
        nn.init.zeros_(self.linear2.bias)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, hidden_dim)

        Returns:
            Output tensor of shape (batch, hidden_dim)
        """
        residual = x

        x = self.norm1(x)
        x = self.linear1(x)
        x = F.gelu(x)  # GELU often works better than ReLU for transformers/resnets
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x + residual


class ResNetBlockPreNorm(nn.Module):
    """Pre-normalization ResNet block.

    Uses pre-normalization (norm before each sub-layer) which often
    provides more stable training for deep networks.

    Architecture:
        x -> [LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout] -> + x

    Args:
        hidden_dim: Dimension of input and output features
        expansion: Expansion factor for intermediate layer. Default: 4
        dropout: Dropout probability. Default: 0.1

    Shape:
        - Input: (batch, hidden_dim)
        - Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        intermediate_dim = hidden_dim * expansion

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Scale initialization for residual path
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return residual + self.scale * x


class DensityResBlock(nn.Module):
    """ResNet block designed for density matrix prediction.

    This block includes:
    - Pre-normalization for stable deep training
    - GELU activation (smooth, works well for scientific data)
    - Scaled residual connection
    - Optional conditioning input

    Args:
        hidden_dim: Dimension of features
        expansion: Expansion factor for FFN. Default: 4
        dropout: Dropout probability. Default: 0.1
        use_conditioning: If True, accepts additional conditioning input. Default: False
        conditioning_dim: Dimension of conditioning vector (if used). Default: None

    Shape:
        - Input: (batch, hidden_dim)
        - Conditioning (optional): (batch, conditioning_dim)
        - Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.1,
        use_conditioning: bool = False,
        conditioning_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_conditioning = use_conditioning
        intermediate_dim = hidden_dim * expansion

        # Main pathway
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm2 = nn.LayerNorm(intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Conditioning pathway (FiLM-style modulation)
        if use_conditioning and conditioning_dim is not None:
            self.cond_proj = nn.Linear(conditioning_dim, hidden_dim * 2)
        else:
            self.cond_proj = None

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, std=0.02 / (2 ** 0.5))
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, hidden_dim)
            conditioning: Optional conditioning vector (batch, conditioning_dim)

        Returns:
            Output features (batch, hidden_dim)
        """
        residual = x

        # First sub-layer
        x = self.norm1(x)

        # Apply FiLM conditioning if provided
        if self.use_conditioning and conditioning is not None and self.cond_proj is not None:
            cond = self.cond_proj(conditioning)
            gamma, beta = cond.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Second sub-layer
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return residual + self.residual_scale * x


class ResNetStack(nn.Module):
    """Stack of ResNet blocks.

    Args:
        hidden_dim: Dimension of features
        num_blocks: Number of ResNet blocks
        expansion: Expansion factor for FFN. Default: 4
        dropout: Dropout probability. Default: 0.1
        block_type: Type of block ('standard', 'prenorm', 'density'). Default: 'density'

    Shape:
        - Input: (batch, hidden_dim)
        - Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
        dropout: float = 0.1,
        block_type: str = 'density'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        if block_type == 'standard':
            block_cls = ResNetBlock
        elif block_type == 'prenorm':
            block_cls = ResNetBlockPreNorm
        elif block_type == 'density':
            block_cls = DensityResBlock
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = nn.ModuleList([
            block_cls(hidden_dim, expansion, dropout)
            for _ in range(num_blocks)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, hidden_dim)

        Returns:
            Output features (batch, hidden_dim)
        """
        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)

    def extra_repr(self) -> str:
        return f'hidden_dim={self.hidden_dim}, num_blocks={self.num_blocks}'
