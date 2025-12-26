"""
Physics constraints and loss functions for density matrix prediction.
"""

from .projections import (
    hermitian_projection,
    mcweeney_purification,
    trace_scaling,
    positivity_projection,
    compute_hermiticity_error,
    compute_idempotency_error,
    compute_trace,
    validate_density_matrix,
)
from .losses import (
    PhysicsInformedLoss,
    MSEComplexLoss,
    FrobeniusComplexLoss,
    create_loss_from_config,
)

__all__ = [
    # Projections
    'hermitian_projection',
    'mcweeney_purification',
    'trace_scaling',
    'positivity_projection',
    # Metrics
    'compute_hermiticity_error',
    'compute_idempotency_error',
    'compute_trace',
    'validate_density_matrix',
    # Losses
    'PhysicsInformedLoss',
    'MSEComplexLoss',
    'FrobeniusComplexLoss',
    'create_loss_from_config',
]
