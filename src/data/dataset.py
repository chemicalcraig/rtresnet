"""
PyTorch Dataset for density matrix time series.

Loads density matrices from RT-TDDFT simulations and creates
sliding window samples for training the prediction model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any


class DensityMatrixDataset(Dataset):
    """Dataset for density matrix time series prediction.

    Creates sliding window samples from a time series of density matrices.
    Each sample contains:
    - density_history: History window of density matrices
    - target: The next density matrix to predict
    - overlap: The AO overlap matrix S
    - field: History of external field values

    Args:
        density_series: Density matrices of shape (N_steps, N_spin, N_bf, N_bf) complex
        overlap_matrix: AO overlap matrix of shape (N_bf, N_bf) real
        field_data: External field of shape (N_steps, 3) real. If None, uses zeros.
        history_length: Number of timesteps in history window. Default: 5
        stride: Step size between consecutive samples. Default: 1
        start_idx: Starting index in the time series. Default: None (use 0)
        end_idx: Ending index in the time series. Default: None (use all)
        transform: Optional transform to apply to samples. Default: None

    Shape:
        - density_history: (history_length, n_spin, nbf, nbf) complex
        - target: (n_spin, nbf, nbf) complex
        - overlap: (nbf, nbf) real
        - field: (history_length, 3) real
    """

    def __init__(
        self,
        density_series: np.ndarray,
        overlap_matrix: np.ndarray,
        field_data: Optional[np.ndarray] = None,
        history_length: int = 5,
        stride: int = 1,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        transform: Optional[callable] = None
    ):
        self.history_length = history_length
        self.stride = stride
        self.transform = transform

        # Validate and store density series
        assert density_series.ndim == 4, \
            f"Expected density_series shape (N_steps, N_spin, N_bf, N_bf), got {density_series.shape}"

        # Apply index slicing
        start_idx = start_idx or 0
        end_idx = end_idx or density_series.shape[0]

        self.density_series = density_series[start_idx:end_idx]
        self.n_steps, self.n_spin, self.nbf, _ = self.density_series.shape

        # Validate and store overlap matrix
        assert overlap_matrix.ndim == 2, \
            f"Expected overlap_matrix shape (N_bf, N_bf), got {overlap_matrix.shape}"
        assert overlap_matrix.shape[0] == self.nbf, \
            f"Overlap matrix size {overlap_matrix.shape[0]} != density matrix size {self.nbf}"
        self.overlap_matrix = overlap_matrix

        # Handle field data
        if field_data is not None:
            # Slice field data to match density series
            self.field_data = field_data[start_idx:end_idx]
            assert self.field_data.shape[0] == self.n_steps, \
                f"Field data length {self.field_data.shape[0]} != density series length {self.n_steps}"
        else:
            # Create zero field if not provided
            self.field_data = np.zeros((self.n_steps, 3), dtype=np.float32)

        # Calculate number of valid samples
        # Need history_length steps for input + 1 for target
        min_required = history_length + 1
        if self.n_steps < min_required:
            raise ValueError(
                f"Time series too short: {self.n_steps} steps, but need at least "
                f"{min_required} (history_length={history_length} + 1 for target)"
            )

        # Number of samples with given stride
        self.n_samples = (self.n_steps - min_required) // stride + 1

        # Convert to torch tensors
        self._density_tensor = torch.from_numpy(self.density_series)
        self._overlap_tensor = torch.from_numpy(self.overlap_matrix)
        self._field_tensor = torch.from_numpy(self.field_data)

        # Ensure correct dtypes
        if not self._density_tensor.is_complex():
            # If loaded as float, assume it's stacked real/imag
            raise ValueError("Density series must be complex-valued")

        self._overlap_tensor = self._overlap_tensor.float()
        self._field_tensor = self._field_tensor.float()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - 'density_history': (history_length, n_spin, nbf, nbf) complex
                - 'target': (n_spin, nbf, nbf) complex
                - 'overlap': (nbf, nbf) real
                - 'field': (history_length, 3) real
        """
        if idx < 0:
            idx = self.n_samples + idx
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_samples} samples")

        # Calculate time indices
        t_start = idx * self.stride
        t_end = t_start + self.history_length
        t_target = t_end  # Target is the step right after history

        # Extract data
        density_history = self._density_tensor[t_start:t_end]  # (history_length, n_spin, nbf, nbf)
        target = self._density_tensor[t_target]  # (n_spin, nbf, nbf)
        field_history = self._field_tensor[t_start:t_end]  # (history_length, 3)

        sample = {
            'density_history': density_history,
            'target': target,
            'overlap': self._overlap_tensor,
            'field': field_history,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'n_samples': self.n_samples,
            'n_steps': self.n_steps,
            'n_spin': self.n_spin,
            'nbf': self.nbf,
            'history_length': self.history_length,
            'stride': self.stride,
        }

    @classmethod
    def from_files(
        cls,
        density_file: Union[str, Path],
        overlap_file: Union[str, Path],
        field_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> 'DensityMatrixDataset':
        """Create dataset from numpy files.

        Args:
            density_file: Path to density_series.npy
            overlap_file: Path to overlap.npy
            field_file: Path to field.npy (optional)
            **kwargs: Additional arguments passed to __init__

        Returns:
            DensityMatrixDataset instance
        """
        density_series = np.load(density_file)
        overlap_matrix = np.load(overlap_file)

        field_data = None
        if field_file is not None:
            field_file = Path(field_file)
            if field_file.exists():
                field_data = np.load(field_file)

        return cls(
            density_series=density_series,
            overlap_matrix=overlap_matrix,
            field_data=field_data,
            **kwargs
        )


class MultiStepDataset(Dataset):
    """Dataset for multi-step rollout training.

    Instead of single-step prediction, provides sequences for
    training with multiple prediction steps.

    Args:
        density_series: Density matrices (N_steps, N_spin, N_bf, N_bf) complex
        overlap_matrix: AO overlap matrix (N_bf, N_bf) real
        field_data: External field (N_steps, 3) real or None
        history_length: Input history window size. Default: 5
        rollout_length: Number of steps to predict. Default: 5
        stride: Step between samples. Default: 1
    """

    def __init__(
        self,
        density_series: np.ndarray,
        overlap_matrix: np.ndarray,
        field_data: Optional[np.ndarray] = None,
        history_length: int = 5,
        rollout_length: int = 5,
        stride: int = 1
    ):
        self.history_length = history_length
        self.rollout_length = rollout_length
        self.stride = stride

        self.density_series = density_series
        self.overlap_matrix = overlap_matrix
        self.n_steps = density_series.shape[0]
        self.n_spin = density_series.shape[1]
        self.nbf = density_series.shape[2]

        if field_data is not None:
            self.field_data = field_data
        else:
            self.field_data = np.zeros((self.n_steps, 3), dtype=np.float32)

        # Need history + rollout steps
        min_required = history_length + rollout_length
        if self.n_steps < min_required:
            raise ValueError(f"Need at least {min_required} steps")

        self.n_samples = (self.n_steps - min_required) // stride + 1

        # Convert to tensors
        self._density_tensor = torch.from_numpy(self.density_series)
        self._overlap_tensor = torch.from_numpy(self.overlap_matrix).float()
        self._field_tensor = torch.from_numpy(self.field_data).float()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_start = idx * self.stride
        t_history_end = t_start + self.history_length
        t_rollout_end = t_history_end + self.rollout_length

        return {
            'density_history': self._density_tensor[t_start:t_history_end],
            'target_sequence': self._density_tensor[t_history_end:t_rollout_end],
            'overlap': self._overlap_tensor,
            'field_sequence': self._field_tensor[t_start:t_rollout_end],
        }


def create_dataloaders(
    density_series: np.ndarray,
    overlap_matrix: np.ndarray,
    field_data: Optional[np.ndarray] = None,
    history_length: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Splits the time series temporally (not shuffled) to maintain
    temporal ordering and avoid data leakage.

    Args:
        density_series: Full density matrix time series
        overlap_matrix: AO overlap matrix
        field_data: External field data (optional)
        history_length: History window size
        batch_size: Batch size for all loaders
        train_ratio: Fraction for training. Default: 0.7
        val_ratio: Fraction for validation. Default: 0.15
        num_workers: DataLoader workers. Default: 0
        pin_memory: Pin memory for GPU transfer. Default: True
        seed: Random seed for reproducibility. Default: None

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_steps = density_series.shape[0]

    # Calculate split indices
    train_end = int(n_steps * train_ratio)
    val_end = int(n_steps * (train_ratio + val_ratio))

    # Create datasets for each split
    train_dataset = DensityMatrixDataset(
        density_series=density_series,
        overlap_matrix=overlap_matrix,
        field_data=field_data,
        history_length=history_length,
        start_idx=0,
        end_idx=train_end
    )

    val_dataset = DensityMatrixDataset(
        density_series=density_series,
        overlap_matrix=overlap_matrix,
        field_data=field_data,
        history_length=history_length,
        start_idx=train_end,
        end_idx=val_end
    )

    test_dataset = DensityMatrixDataset(
        density_series=density_series,
        overlap_matrix=overlap_matrix,
        field_data=field_data,
        history_length=history_length,
        start_idx=val_end,
        end_idx=n_steps
    )

    # Create generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
