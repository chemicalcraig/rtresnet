"""
Complex-valued neural network layers for density matrix prediction.

This module provides PyTorch layers that operate on complex tensors using
native torch.complex64/complex128 support.

References:
    Trabelsi et al., "Deep Complex Networks" (2018)
    https://arxiv.org/abs/1705.09792
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal


class ComplexLinear(nn.Module):
    """Linear layer for complex-valued inputs.

    Implements complex matrix multiplication using separate real and imaginary
    weight matrices. For complex weight W = W_r + i*W_i and input x = x_r + i*x_i:

        y = W @ x = (W_r @ x_r - W_i @ x_i) + i*(W_r @ x_i + W_i @ x_r)

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable complex bias. Default: True

    Shape:
        - Input: (*, in_features) complex tensor
        - Output: (*, out_features) complex tensor
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Separate real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot initialization scaled for complex."""
        # Scale by 1/sqrt(2) to account for complex variance
        scale = 1.0 / np.sqrt(2.0)
        nn.init.kaiming_uniform_(self.weight_real, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=np.sqrt(5))
        self.weight_real.data *= scale
        self.weight_imag.data *= scale

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (*, in_features)

        Returns:
            Complex tensor of shape (*, out_features)
        """
        # Extract real and imaginary parts
        x_real = x.real
        x_imag = x.imag

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        out_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)

        # Add bias
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        return torch.complex(out_real, out_imag)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_real is not None}'


class ComplexBatchNorm(nn.Module):
    """Batch normalization for complex-valued tensors.

    Uses the covariance-based normalization approach from "Deep Complex Networks"
    (Trabelsi et al., 2018). The complex input is treated as a 2D random variable
    (real, imag) and normalized using the 2x2 covariance matrix.

    The learnable parameters are:
        - gamma: 2x2 scale matrix (gamma_rr, gamma_ri, gamma_ir, gamma_ii)
        - beta: complex shift

    Args:
        num_features: Number of features (channels)
        eps: Small constant for numerical stability. Default: 1e-5
        momentum: Momentum for running stats. Default: 0.1
        affine: If True, use learnable scale and shift. Default: True
        track_running_stats: If True, track running mean and covariance. Default: True

    Shape:
        - Input: (N, num_features, *) or (N, *, num_features) complex tensor
        - Output: Same shape as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # 2x2 scale matrix per feature (initialized to identity / sqrt(2))
            self.gamma_rr = nn.Parameter(torch.ones(num_features) / np.sqrt(2))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ir = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(torch.ones(num_features) / np.sqrt(2))
            # Complex shift
            self.beta_real = nn.Parameter(torch.zeros(num_features))
            self.beta_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ri', None)
            self.register_parameter('gamma_ir', None)
            self.register_parameter('gamma_ii', None)
            self.register_parameter('beta_real', None)
            self.register_parameter('beta_imag', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            # Covariance matrix elements (Vrr, Vri, Vii)
            self.register_buffer('running_Vrr', torch.ones(num_features) / 2)
            self.register_buffer('running_Vri', torch.zeros(num_features))
            self.register_buffer('running_Vii', torch.ones(num_features) / 2)
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean_real', None)
            self.register_buffer('running_mean_imag', None)
            self.register_buffer('running_Vrr', None)
            self.register_buffer('running_Vri', None)
            self.register_buffer('running_Vii', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (N, num_features, *) or (N, *, num_features)

        Returns:
            Normalized complex tensor of same shape
        """
        # Determine which axis is the feature axis
        # Assume features are in the last dimension if not 2D or 3D with channels first
        if x.dim() == 2:
            # (N, features)
            reduce_dims = (0,)
            view_shape = (1, -1)
        else:
            # Assume (N, features, ...) format
            reduce_dims = tuple([0] + list(range(2, x.dim())))
            view_shape = (1, -1) + (1,) * (x.dim() - 2)

        x_real = x.real
        x_imag = x.imag

        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean_real = x_real.mean(dim=reduce_dims)
            mean_imag = x_imag.mean(dim=reduce_dims)

            # Center the data
            x_real_centered = x_real - mean_real.view(view_shape)
            x_imag_centered = x_imag - mean_imag.view(view_shape)

            # Compute covariance matrix elements
            n = x_real.numel() // self.num_features
            Vrr = (x_real_centered ** 2).mean(dim=reduce_dims)
            Vii = (x_imag_centered ** 2).mean(dim=reduce_dims)
            Vri = (x_real_centered * x_imag_centered).mean(dim=reduce_dims)

            # Update running stats
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        # Cumulative moving average
                        exp_avg_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exp_avg_factor = self.momentum

                    self.running_mean_real.mul_(1 - exp_avg_factor).add_(mean_real, alpha=exp_avg_factor)
                    self.running_mean_imag.mul_(1 - exp_avg_factor).add_(mean_imag, alpha=exp_avg_factor)
                    self.running_Vrr.mul_(1 - exp_avg_factor).add_(Vrr, alpha=exp_avg_factor)
                    self.running_Vri.mul_(1 - exp_avg_factor).add_(Vri, alpha=exp_avg_factor)
                    self.running_Vii.mul_(1 - exp_avg_factor).add_(Vii, alpha=exp_avg_factor)
        else:
            # Use running stats
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            Vrr = self.running_Vrr
            Vri = self.running_Vri
            Vii = self.running_Vii

            x_real_centered = x_real - mean_real.view(view_shape)
            x_imag_centered = x_imag - mean_imag.view(view_shape)

        # Compute inverse square root of covariance matrix
        # V = [[Vrr, Vri], [Vri, Vii]]
        # We need V^{-1/2}
        # For 2x2 positive definite matrix, use analytical formula

        # Determinant
        det = Vrr * Vii - Vri * Vri + self.eps

        # Trace
        trace = Vrr + Vii

        # s = sqrt(det)
        s = torch.sqrt(det)

        # t = sqrt(trace + 2*s)
        t = torch.sqrt(trace + 2 * s + self.eps)

        # V^{-1/2} = (V + s*I) / (t * s)
        # But we can compute it more directly
        # Using the formula for 2x2 matrix square root inverse

        # For numerical stability, we use a different approach:
        # Normalize by the magnitude: scale = 1/sqrt(Vrr + Vii)
        # This is a simplified whitening that works well in practice

        # Full whitening using Cholesky-like decomposition
        # V^{-1/2} = [[a, 0], [b, c]] where we solve for a, b, c

        # Simpler approach: use the inverse square root formula
        inv_sqrt_det = 1.0 / (s + self.eps)

        # Components of V^{-1/2}
        # Using the formula: V^{-1/2} = (1/(t*s)) * (V + s*I) for positive definite V
        denom = t * s + self.eps

        Wrr = (Vii + s) / denom
        Wri = -Vri / denom
        Wii = (Vrr + s) / denom

        # Apply whitening transformation
        Wrr = Wrr.view(view_shape)
        Wri = Wri.view(view_shape)
        Wii = Wii.view(view_shape)

        y_real = Wrr * x_real_centered + Wri * x_imag_centered
        y_imag = Wri * x_real_centered + Wii * x_imag_centered

        # Apply affine transformation
        if self.affine:
            gamma_rr = self.gamma_rr.view(view_shape)
            gamma_ri = self.gamma_ri.view(view_shape)
            gamma_ir = self.gamma_ir.view(view_shape)
            gamma_ii = self.gamma_ii.view(view_shape)
            beta_real = self.beta_real.view(view_shape)
            beta_imag = self.beta_imag.view(view_shape)

            out_real = gamma_rr * y_real + gamma_ri * y_imag + beta_real
            out_imag = gamma_ir * y_real + gamma_ii * y_imag + beta_imag
        else:
            out_real = y_real
            out_imag = y_imag

        return torch.complex(out_real, out_imag)

    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')


class ComplexActivation(nn.Module):
    """Activation functions for complex-valued tensors.

    Supported activation types:
        - 'modrelu': ModReLU - applies ReLU to magnitude with learnable bias,
                     preserves phase. f(z) = ReLU(|z| + b) * z/|z|
        - 'cardioid': Cardioid activation - f(z) = 0.5 * (1 + cos(angle(z))) * z
        - 'zrelu': Applies ReLU separately to real and imaginary parts
        - 'modtanh': Applies tanh to magnitude, preserves phase

    Args:
        activation_type: Type of activation function. Default: 'modrelu'
        num_features: Number of features (for learnable parameters). Default: None
                     If None, uses a single shared parameter for ModReLU.

    Shape:
        - Input: (*) complex tensor
        - Output: Same shape as input
    """

    def __init__(
        self,
        activation_type: Literal['modrelu', 'cardioid', 'zrelu', 'modtanh'] = 'modrelu',
        num_features: Optional[int] = None
    ):
        super().__init__()
        self.activation_type = activation_type
        self.num_features = num_features

        if activation_type == 'modrelu':
            # Learnable bias for ModReLU
            if num_features is not None:
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of any shape

        Returns:
            Activated complex tensor of same shape
        """
        if self.activation_type == 'modrelu':
            return self._modrelu(x)
        elif self.activation_type == 'cardioid':
            return self._cardioid(x)
        elif self.activation_type == 'zrelu':
            return self._zrelu(x)
        elif self.activation_type == 'modtanh':
            return self._modtanh(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def _modrelu(self, x: torch.Tensor) -> torch.Tensor:
        """ModReLU: ReLU on magnitude with bias, preserve phase.

        f(z) = ReLU(|z| + b) * exp(i * angle(z))
             = ReLU(|z| + b) * z / |z|
        """
        magnitude = torch.abs(x)

        # Reshape bias for broadcasting if needed
        if self.num_features is not None and x.dim() >= 2:
            # Assume features in last dimension or second dimension
            if x.shape[-1] == self.num_features:
                bias = self.bias
            elif x.dim() >= 2 and x.shape[1] == self.num_features:
                view_shape = (1, -1) + (1,) * (x.dim() - 2)
                bias = self.bias.view(view_shape)
            else:
                bias = self.bias.view(-1)
        else:
            bias = self.bias

        # Apply ReLU to (magnitude + bias)
        activated_magnitude = F.relu(magnitude + bias)

        # Compute phase (avoid division by zero)
        phase = x / (magnitude + 1e-8)

        return activated_magnitude * phase

    def _cardioid(self, x: torch.Tensor) -> torch.Tensor:
        """Cardioid activation: f(z) = 0.5 * (1 + cos(angle(z))) * z"""
        phase = torch.angle(x)
        scale = 0.5 * (1.0 + torch.cos(phase))
        return scale * x

    def _zrelu(self, x: torch.Tensor) -> torch.Tensor:
        """zReLU: Apply ReLU separately to real and imaginary parts."""
        return torch.complex(F.relu(x.real), F.relu(x.imag))

    def _modtanh(self, x: torch.Tensor) -> torch.Tensor:
        """ModTanh: Apply tanh to magnitude, preserve phase."""
        magnitude = torch.abs(x)
        activated_magnitude = torch.tanh(magnitude)
        phase = x / (magnitude + 1e-8)
        return activated_magnitude * phase

    def extra_repr(self) -> str:
        s = f"activation_type='{self.activation_type}'"
        if self.num_features is not None:
            s += f", num_features={self.num_features}"
        return s


class ComplexDropout(nn.Module):
    """Dropout for complex-valued tensors.

    Applies the same dropout mask to both real and imaginary parts
    to maintain complex structure.

    Args:
        p: Probability of an element to be zeroed. Default: 0.5

    Shape:
        - Input: (*) complex tensor
        - Output: Same shape as input
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        # Create mask on real part shape, apply to both
        mask = torch.ones_like(x.real)
        mask = F.dropout(mask, p=self.p, training=True)

        # Scale is already applied by F.dropout
        return torch.complex(x.real * mask, x.imag * mask)

    def extra_repr(self) -> str:
        return f'p={self.p}'
