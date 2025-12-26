"""
Preprocessing utilities for density matrix data.

Includes:
- Train/validation/test splitting
- Normalization (Frobenius norm, element-wise)
- Data validation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class NormalizationStats:
    """Statistics for normalization/denormalization.

    Attributes:
        mode: Normalization mode ('frobenius', 'elementwise', 'trace')
        mean_real: Mean of real parts
        mean_imag: Mean of imaginary parts
        std_real: Std of real parts (or Frobenius norm scale)
        std_imag: Std of imaginary parts (or Frobenius norm scale)
        scale: Overall scale factor
    """
    mode: str
    mean_real: Optional[np.ndarray] = None
    mean_imag: Optional[np.ndarray] = None
    std_real: Optional[np.ndarray] = None
    std_imag: Optional[np.ndarray] = None
    scale: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'mode': self.mode,
            'mean_real': self.mean_real.tolist() if self.mean_real is not None else None,
            'mean_imag': self.mean_imag.tolist() if self.mean_imag is not None else None,
            'std_real': self.std_real.tolist() if self.std_real is not None else None,
            'std_imag': self.std_imag.tolist() if self.std_imag is not None else None,
            'scale': self.scale,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'NormalizationStats':
        """Create from dictionary."""
        return cls(
            mode=d['mode'],
            mean_real=np.array(d['mean_real']) if d['mean_real'] is not None else None,
            mean_imag=np.array(d['mean_imag']) if d['mean_imag'] is not None else None,
            std_real=np.array(d['std_real']) if d['std_real'] is not None else None,
            std_imag=np.array(d['std_imag']) if d['std_imag'] is not None else None,
            scale=d['scale'],
        )


def train_val_test_split(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: Optional[float] = None
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Calculate train/val/test split indices.

    For time series data, splits are temporal (not shuffled).

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training. Default: 0.7
        val_ratio: Fraction for validation. Default: 0.15
        test_ratio: Fraction for test. Default: remaining

    Returns:
        Tuple of ((train_start, train_end), (val_start, val_end), (test_start, test_end))
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    assert train_ratio + val_ratio + test_ratio <= 1.0 + 1e-6, \
        f"Ratios sum to {train_ratio + val_ratio + test_ratio}, must be <= 1.0"

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    return (
        (0, train_end),
        (train_end, val_end),
        (val_end, n_samples)
    )


def compute_normalization_stats(
    density_series: np.ndarray,
    mode: str = 'frobenius'
) -> NormalizationStats:
    """Compute normalization statistics from density matrix series.

    Args:
        density_series: Complex array of shape (N_steps, N_spin, N_bf, N_bf)
        mode: Normalization mode:
            - 'frobenius': Scale by average Frobenius norm
            - 'elementwise': Z-score normalization per element
            - 'trace': Scale by average trace magnitude
            - 'none': No normalization (returns identity stats)

    Returns:
        NormalizationStats object
    """
    if mode == 'none':
        return NormalizationStats(mode='none', scale=1.0)

    elif mode == 'frobenius':
        # Compute average Frobenius norm across all matrices
        # ||A||_F = sqrt(sum(|a_ij|^2))
        norms = np.sqrt(np.sum(np.abs(density_series) ** 2, axis=(-2, -1)))
        avg_norm = np.mean(norms)

        return NormalizationStats(
            mode='frobenius',
            scale=avg_norm if avg_norm > 1e-10 else 1.0
        )

    elif mode == 'elementwise':
        # Compute mean and std for each element position
        mean_real = np.mean(density_series.real, axis=0)
        mean_imag = np.mean(density_series.imag, axis=0)
        std_real = np.std(density_series.real, axis=0)
        std_imag = np.std(density_series.imag, axis=0)

        # Avoid division by zero
        std_real = np.maximum(std_real, 1e-10)
        std_imag = np.maximum(std_imag, 1e-10)

        return NormalizationStats(
            mode='elementwise',
            mean_real=mean_real,
            mean_imag=mean_imag,
            std_real=std_real,
            std_imag=std_imag
        )

    elif mode == 'trace':
        # Scale by average trace magnitude
        traces = np.trace(density_series, axis1=-2, axis2=-1)  # (N_steps, N_spin)
        avg_trace = np.mean(np.abs(traces))

        return NormalizationStats(
            mode='trace',
            scale=avg_trace if avg_trace > 1e-10 else 1.0
        )

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def normalize_densities(
    density_series: Union[np.ndarray, torch.Tensor],
    stats: NormalizationStats
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize density matrices using precomputed statistics.

    Args:
        density_series: Complex array of shape (..., N_spin, N_bf, N_bf)
        stats: NormalizationStats from compute_normalization_stats

    Returns:
        Normalized density series (same type as input)
    """
    is_tensor = isinstance(density_series, torch.Tensor)

    if stats.mode == 'none':
        return density_series

    elif stats.mode == 'frobenius' or stats.mode == 'trace':
        return density_series / stats.scale

    elif stats.mode == 'elementwise':
        if is_tensor:
            mean_real = torch.from_numpy(stats.mean_real).to(density_series.device)
            mean_imag = torch.from_numpy(stats.mean_imag).to(density_series.device)
            std_real = torch.from_numpy(stats.std_real).to(density_series.device)
            std_imag = torch.from_numpy(stats.std_imag).to(density_series.device)

            real_norm = (density_series.real - mean_real) / std_real
            imag_norm = (density_series.imag - mean_imag) / std_imag
            return torch.complex(real_norm, imag_norm)
        else:
            real_norm = (density_series.real - stats.mean_real) / stats.std_real
            imag_norm = (density_series.imag - stats.mean_imag) / stats.std_imag
            return real_norm + 1j * imag_norm

    else:
        raise ValueError(f"Unknown normalization mode: {stats.mode}")


def denormalize_densities(
    density_series: Union[np.ndarray, torch.Tensor],
    stats: NormalizationStats
) -> Union[np.ndarray, torch.Tensor]:
    """Denormalize density matrices (inverse of normalize_densities).

    Args:
        density_series: Normalized complex array
        stats: NormalizationStats used for normalization

    Returns:
        Denormalized density series
    """
    is_tensor = isinstance(density_series, torch.Tensor)

    if stats.mode == 'none':
        return density_series

    elif stats.mode == 'frobenius' or stats.mode == 'trace':
        return density_series * stats.scale

    elif stats.mode == 'elementwise':
        if is_tensor:
            mean_real = torch.from_numpy(stats.mean_real).to(density_series.device)
            mean_imag = torch.from_numpy(stats.mean_imag).to(density_series.device)
            std_real = torch.from_numpy(stats.std_real).to(density_series.device)
            std_imag = torch.from_numpy(stats.std_imag).to(density_series.device)

            real_denorm = density_series.real * std_real + mean_real
            imag_denorm = density_series.imag * std_imag + mean_imag
            return torch.complex(real_denorm, imag_denorm)
        else:
            real_denorm = density_series.real * stats.std_real + stats.mean_real
            imag_denorm = density_series.imag * stats.std_imag + stats.mean_imag
            return real_denorm + 1j * imag_denorm

    else:
        raise ValueError(f"Unknown normalization mode: {stats.mode}")


def validate_density_data(
    density_series: np.ndarray,
    overlap_matrix: np.ndarray,
    field_data: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, bool]:
    """Validate density matrix data for training.

    Checks:
    - Shape consistency
    - Complex dtype for densities
    - Symmetry of overlap matrix
    - NaN/Inf values
    - Approximate Hermiticity of densities

    Args:
        density_series: Density matrices (N_steps, N_spin, N_bf, N_bf)
        overlap_matrix: Overlap matrix (N_bf, N_bf)
        field_data: Optional field data (N_steps, 3)
        verbose: Print validation results

    Returns:
        Dictionary of validation results
    """
    results = {}

    # Check shapes
    n_steps, n_spin, nbf1, nbf2 = density_series.shape
    results['density_shape_valid'] = (nbf1 == nbf2)

    if verbose:
        print(f"Density series shape: {density_series.shape}")
        print(f"  N_steps: {n_steps}")
        print(f"  N_spin: {n_spin}")
        print(f"  N_bf: {nbf1}")

    # Check complex dtype
    results['density_is_complex'] = np.iscomplexobj(density_series)
    if verbose:
        print(f"Density is complex: {results['density_is_complex']}")

    # Check overlap matrix
    results['overlap_shape_valid'] = (overlap_matrix.shape == (nbf1, nbf1))
    results['overlap_is_symmetric'] = np.allclose(
        overlap_matrix, overlap_matrix.T, atol=1e-10
    )
    if verbose:
        print(f"Overlap shape valid: {results['overlap_shape_valid']}")
        print(f"Overlap is symmetric: {results['overlap_is_symmetric']}")

    # Check for NaN/Inf
    results['density_finite'] = np.all(np.isfinite(density_series))
    results['overlap_finite'] = np.all(np.isfinite(overlap_matrix))
    if verbose:
        print(f"Density values finite: {results['density_finite']}")
        print(f"Overlap values finite: {results['overlap_finite']}")

    # Check approximate Hermiticity of densities
    hermiticity_error = np.abs(density_series - np.conj(density_series.transpose(0, 1, 3, 2)))
    max_herm_error = np.max(hermiticity_error)
    results['density_hermitian'] = max_herm_error < 1e-6
    if verbose:
        print(f"Max Hermiticity error: {max_herm_error:.2e}")

    # Check field data if provided
    if field_data is not None:
        results['field_shape_valid'] = (field_data.shape[0] == n_steps)
        results['field_finite'] = np.all(np.isfinite(field_data))
        if verbose:
            print(f"Field shape valid: {results['field_shape_valid']}")
            print(f"Field values finite: {results['field_finite']}")

    # Overall validity
    results['all_valid'] = all(v for v in results.values())
    if verbose:
        print(f"\nAll validations passed: {results['all_valid']}")

    return results


def prepare_data_for_training(
    density_file: str,
    overlap_file: str,
    field_file: Optional[str] = None,
    normalize: bool = True,
    normalization_mode: str = 'frobenius',
    validate: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[NormalizationStats]]:
    """Load and prepare data for training.

    Convenience function that:
    1. Loads numpy files
    2. Validates data
    3. Computes normalization statistics
    4. Normalizes data

    Args:
        density_file: Path to density_series.npy
        overlap_file: Path to overlap.npy
        field_file: Path to field.npy (optional)
        normalize: Whether to normalize data
        normalization_mode: Normalization mode
        validate: Whether to validate data

    Returns:
        Tuple of (density_series, overlap_matrix, field_data, normalization_stats)
    """
    # Load data
    density_series = np.load(density_file)
    overlap_matrix = np.load(overlap_file)

    field_data = None
    if field_file is not None:
        field_data = np.load(field_file)

    # Validate
    if validate:
        validation_results = validate_density_data(
            density_series, overlap_matrix, field_data, verbose=True
        )
        if not validation_results['all_valid']:
            raise ValueError("Data validation failed")

    # Normalize
    norm_stats = None
    if normalize:
        norm_stats = compute_normalization_stats(density_series, mode=normalization_mode)
        density_series = normalize_densities(density_series, norm_stats)
        print(f"\nNormalization applied: mode={normalization_mode}, scale={norm_stats.scale:.4f}")

    return density_series, overlap_matrix, field_data, norm_stats
