"""
Prediction and rollout for density matrix evolution.

Provides:
- DensityPredictor: Main inference class
- Single-step and multi-step (rollout) prediction
- Physics projection options
- Trajectory saving and analysis
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json


@dataclass
class RolloutResult:
    """Results from a multi-step rollout prediction.

    Attributes:
        predictions: Predicted density matrices (n_steps, n_spin, nbf, nbf) complex
        timestamps: Time indices or values for each prediction
        physics_metrics: Physics metrics at each step
        initial_densities: Bootstrap densities used to start rollout
    """
    predictions: np.ndarray
    timestamps: np.ndarray
    physics_metrics: List[Dict[str, float]] = field(default_factory=list)
    initial_densities: Optional[np.ndarray] = None

    def save(self, path: Union[str, Path]) -> None:
        """Save rollout results to files.

        Saves:
        - predictions.npy: Predicted density matrices
        - timestamps.npy: Time indices
        - metrics.json: Physics metrics
        - initial_densities.npy: Bootstrap densities (if available)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / 'predictions.npy', self.predictions)
        np.save(path / 'timestamps.npy', self.timestamps)

        if self.physics_metrics:
            with open(path / 'metrics.json', 'w') as f:
                json.dump(self.physics_metrics, f, indent=2)

        if self.initial_densities is not None:
            np.save(path / 'initial_densities.npy', self.initial_densities)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'RolloutResult':
        """Load rollout results from files."""
        path = Path(path)

        predictions = np.load(path / 'predictions.npy')
        timestamps = np.load(path / 'timestamps.npy')

        physics_metrics = []
        metrics_file = path / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                physics_metrics = json.load(f)

        initial_densities = None
        initial_file = path / 'initial_densities.npy'
        if initial_file.exists():
            initial_densities = np.load(initial_file)

        return cls(
            predictions=predictions,
            timestamps=timestamps,
            physics_metrics=physics_metrics,
            initial_densities=initial_densities
        )


class DensityPredictor:
    """Inference class for density matrix prediction.

    Handles:
    - Loading trained models from checkpoints
    - Single-step prediction
    - Multi-step rollout prediction
    - Physics projections at inference time
    - Trajectory analysis

    Args:
        model: Trained DensityResNet model (or None to load from checkpoint)
        checkpoint_path: Path to checkpoint file (optional if model provided)
        device: Device for inference. Default: auto-detect
        apply_hermitian: Apply Hermitian projection. Default: True
        apply_mcweeney: Apply McWeeney purification. Default: False
        apply_trace_scaling: Apply trace scaling. Default: False
        n_electrons: Electron counts per spin (for trace scaling)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        apply_hermitian: bool = True,
        apply_mcweeney: bool = False,
        apply_trace_scaling: bool = False,
        n_electrons: Optional[List[float]] = None
    ):
        # Device setup
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Load model
        if model is not None:
            self.model = model.to(device)
            self.config = model.get_config() if hasattr(model, 'get_config') else {}
        elif checkpoint_path is not None:
            self.model, self.config = self._load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError("Must provide either model or checkpoint_path")

        self.model.eval()

        # Physics projection settings
        self.apply_hermitian = apply_hermitian
        self.apply_mcweeney = apply_mcweeney
        self.apply_trace_scaling = apply_trace_scaling

        if n_electrons is not None:
            self.n_electrons = torch.tensor(n_electrons, device=device)
        else:
            self.n_electrons = None

        # Get model parameters
        self.history_length = self.config.get('history_length', 5)
        self.n_spin = self.config.get('n_spin', 2)

    def _load_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Tuple[nn.Module, Dict]:
        """Load model from checkpoint file."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        config = checkpoint['config']

        # Import model factory
        try:
            from models.density_resnet import create_model_from_config
        except ImportError:
            from ..models.density_resnet import create_model_from_config

        model = create_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model, config

    @torch.no_grad()
    def predict_single_step(
        self,
        density_history: torch.Tensor,
        overlap_matrix: torch.Tensor,
        field_history: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next density matrix (single step).

        Args:
            density_history: (batch, history_length, n_spin, nbf, nbf) complex
            overlap_matrix: (batch, nbf, nbf) real
            field_history: (batch, history_length, 3) real

        Returns:
            Predicted density matrix (batch, n_spin, nbf, nbf) complex
        """
        # Ensure on correct device and dtype
        density_history = density_history.to(self.device)
        overlap_matrix = overlap_matrix.to(self.device).float()
        field_history = field_history.to(self.device).float()

        # Convert complex tensors to complex64 if they're complex128
        if density_history.is_complex() and density_history.dtype == torch.complex128:
            density_history = density_history.to(torch.complex64)

        # Model prediction
        pred = self.model(density_history, overlap_matrix, field_history)

        # Apply additional physics projections if requested
        pred = self._apply_projections(pred, overlap_matrix)

        return pred

    def _apply_projections(
        self,
        rho: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """Apply physics projections to predicted density."""
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

        if self.apply_hermitian:
            rho = hermitian_projection(rho)

        if self.apply_trace_scaling and self.n_electrons is not None:
            rho = trace_scaling(rho, S, self.n_electrons)

        if self.apply_mcweeney:
            rho = mcweeney_purification(rho, S, n_iterations=1)

        return rho

    @torch.no_grad()
    def rollout(
        self,
        initial_densities: Union[np.ndarray, torch.Tensor],
        overlap_matrix: Union[np.ndarray, torch.Tensor],
        n_steps: int,
        field_sequence: Optional[Union[np.ndarray, torch.Tensor]] = None,
        compute_metrics: bool = True,
        verbose: bool = True
    ) -> RolloutResult:
        """Perform multi-step rollout prediction.

        Starting from initial density matrices, autoregressively predicts
        future density matrices.

        Args:
            initial_densities: Bootstrap densities (history_length, n_spin, nbf, nbf)
            overlap_matrix: AO overlap matrix (nbf, nbf)
            n_steps: Number of steps to predict
            field_sequence: External field for all steps (n_steps + history_length, 3)
                           If None, uses zero field
            compute_metrics: Compute physics metrics at each step. Default: True
            verbose: Print progress. Default: True

        Returns:
            RolloutResult with predictions and metrics
        """
        # Convert to tensors if needed
        if isinstance(initial_densities, np.ndarray):
            initial_densities = torch.from_numpy(initial_densities)
        if isinstance(overlap_matrix, np.ndarray):
            overlap_matrix = torch.from_numpy(overlap_matrix)
        if field_sequence is not None and isinstance(field_sequence, np.ndarray):
            field_sequence = torch.from_numpy(field_sequence)

        # Move to device and add batch dimension
        initial_densities = initial_densities.to(self.device)
        overlap_matrix = overlap_matrix.to(self.device).float()

        # Handle dimensions
        if initial_densities.dim() == 4:
            # (history, n_spin, nbf, nbf) -> (1, history, n_spin, nbf, nbf)
            initial_densities = initial_densities.unsqueeze(0)
        if overlap_matrix.dim() == 2:
            overlap_matrix = overlap_matrix.unsqueeze(0)

        batch_size = initial_densities.shape[0]
        nbf = initial_densities.shape[-1]

        # Create field sequence if not provided
        total_steps = n_steps + self.history_length
        if field_sequence is None:
            field_sequence = torch.zeros(batch_size, total_steps, 3, device=self.device)
        else:
            field_sequence = field_sequence.to(self.device).float()
            if field_sequence.dim() == 2:
                field_sequence = field_sequence.unsqueeze(0)

        # Initialize history buffer with initial densities
        history_buffer = initial_densities.clone()

        # Storage for predictions
        predictions = []
        physics_metrics = []

        if verbose:
            print(f"Starting rollout for {n_steps} steps...")

        for step in range(n_steps):
            # Get current field history
            field_start = step
            field_end = step + self.history_length
            field_history = field_sequence[:, field_start:field_end]

            # Predict next step
            pred = self.predict_single_step(
                history_buffer,
                overlap_matrix,
                field_history
            )

            # Store prediction
            predictions.append(pred.cpu())

            # Compute physics metrics if requested
            if compute_metrics:
                metrics = self._compute_step_metrics(pred, overlap_matrix)
                physics_metrics.append(metrics)

            # Update history buffer (shift and append new prediction)
            history_buffer = torch.cat([
                history_buffer[:, 1:],
                pred.unsqueeze(1)
            ], dim=1)

            # Progress logging
            if verbose and (step + 1) % max(1, n_steps // 10) == 0:
                print(f"  Step {step + 1}/{n_steps}")

        # Stack predictions
        predictions_tensor = torch.stack(predictions, dim=1)  # (batch, n_steps, n_spin, nbf, nbf)

        # Remove batch dimension if it was added
        if batch_size == 1:
            predictions_tensor = predictions_tensor.squeeze(0)

        # Convert to numpy
        predictions_np = predictions_tensor.numpy()
        if np.iscomplexobj(predictions_np) is False:
            # Handle case where tensor was real
            predictions_np = predictions_np.astype(np.complex64)

        # Create result
        timestamps = np.arange(self.history_length, self.history_length + n_steps)

        initial_np = initial_densities.squeeze(0).cpu().numpy()

        if verbose:
            print(f"Rollout complete. Output shape: {predictions_np.shape}")

        return RolloutResult(
            predictions=predictions_np,
            timestamps=timestamps,
            physics_metrics=physics_metrics,
            initial_densities=initial_np
        )

    def _compute_step_metrics(
        self,
        pred: torch.Tensor,
        overlap: torch.Tensor
    ) -> Dict[str, float]:
        """Compute physics metrics for a single prediction step."""
        metrics = {}

        # Hermiticity
        herm_error = (pred - pred.conj().transpose(-2, -1)).abs().mean()
        metrics['hermiticity_error'] = herm_error.item()

        # Trace
        trace = torch.einsum('bsij,bji->bs', pred, overlap.to(pred.dtype)).real
        metrics['trace_alpha'] = trace[:, 0].mean().item()
        if pred.shape[1] > 1:
            metrics['trace_beta'] = trace[:, 1].mean().item()

        # Frobenius norm
        frob = torch.sqrt((pred.abs() ** 2).sum(dim=(-2, -1))).mean()
        metrics['frobenius_norm'] = frob.item()

        return metrics

    @torch.no_grad()
    def predict_from_files(
        self,
        density_file: Union[str, Path],
        overlap_file: Union[str, Path],
        n_steps: int,
        field_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        start_idx: int = 0
    ) -> RolloutResult:
        """Convenience method to predict from numpy files.

        Args:
            density_file: Path to density_series.npy
            overlap_file: Path to overlap.npy
            n_steps: Number of steps to predict
            field_file: Path to field.npy (optional)
            output_dir: Directory to save results (optional)
            start_idx: Starting index in density series for bootstrap

        Returns:
            RolloutResult with predictions
        """
        # Load data
        density_series = np.load(density_file)
        overlap_matrix = np.load(overlap_file)

        field_sequence = None
        if field_file is not None:
            field_sequence = np.load(field_file)
            # Slice field to match rollout
            end_idx = start_idx + self.history_length + n_steps
            if end_idx <= len(field_sequence):
                field_sequence = field_sequence[start_idx:end_idx]
            else:
                # Pad with zeros if not enough field data
                pad_length = end_idx - len(field_sequence)
                field_sequence = np.concatenate([
                    field_sequence[start_idx:],
                    np.zeros((pad_length, 3), dtype=np.float32)
                ])

        # Extract initial densities
        initial_densities = density_series[start_idx:start_idx + self.history_length]

        # Run rollout
        result = self.rollout(
            initial_densities=initial_densities,
            overlap_matrix=overlap_matrix,
            n_steps=n_steps,
            field_sequence=field_sequence
        )

        # Save if output directory provided
        if output_dir is not None:
            result.save(output_dir)

        return result

    def compare_with_reference(
        self,
        result: RolloutResult,
        reference: np.ndarray,
        start_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Compare rollout predictions with reference trajectory.

        Args:
            result: RolloutResult from rollout prediction
            reference: Reference density matrices (n_steps, n_spin, nbf, nbf)
            start_idx: Starting index in reference for comparison

        Returns:
            Dictionary with error metrics per step
        """
        predictions = result.predictions
        n_steps = len(predictions)

        # Slice reference to match predictions
        ref_slice = reference[start_idx:start_idx + n_steps]

        if len(ref_slice) < n_steps:
            raise ValueError(
                f"Reference has {len(ref_slice)} steps but predictions have {n_steps}"
            )

        # Compute errors
        errors = {}

        # Frobenius error per step
        diff = predictions - ref_slice
        frob_errors = np.sqrt(np.sum(np.abs(diff) ** 2, axis=(-2, -1, -3)))
        errors['frobenius'] = frob_errors

        # Trace error per step (assuming overlap is identity for simplicity)
        pred_trace = np.trace(predictions, axis1=-2, axis2=-1)
        ref_trace = np.trace(ref_slice, axis1=-2, axis2=-1)
        errors['trace'] = np.abs(pred_trace - ref_trace)

        # Element-wise max error per step
        errors['max_element'] = np.max(np.abs(diff), axis=(-2, -1, -3))

        # Summary statistics
        errors['mean_frobenius'] = np.mean(frob_errors)
        errors['final_frobenius'] = frob_errors[-1]
        errors['cumulative_frobenius'] = np.cumsum(frob_errors)

        return errors
