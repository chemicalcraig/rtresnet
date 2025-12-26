"""
Neural network models for density matrix prediction.
"""

from .complex_layers import (
    ComplexLinear,
    ComplexBatchNorm,
    ComplexActivation,
    ComplexDropout,
)
from .encoders import (
    OverlapEncoder,
    FieldEncoder,
    FieldEncoderSimple,
    DensityHistoryEncoder,
)
from .resnet_blocks import (
    ResNetBlock,
    ResNetBlockPreNorm,
    DensityResBlock,
    ResNetStack,
)
from .density_resnet import (
    DensityResNet,
    create_model_from_config,
)

__all__ = [
    # Complex layers
    'ComplexLinear',
    'ComplexBatchNorm',
    'ComplexActivation',
    'ComplexDropout',
    # Encoders
    'OverlapEncoder',
    'FieldEncoder',
    'FieldEncoderSimple',
    'DensityHistoryEncoder',
    # ResNet blocks
    'ResNetBlock',
    'ResNetBlockPreNorm',
    'DensityResBlock',
    'ResNetStack',
    # Main model
    'DensityResNet',
    'create_model_from_config',
]
