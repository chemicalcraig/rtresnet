"""
Training loop and utilities for DensityResNet.

Provides:
- DensityTrainer: Main training class with full training loop
- TrainingHistory: Stores and visualizes training metrics
- Learning rate schedulers with warmup
- Checkpointing and early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time


@dataclass
class TrainingHistory:
    """Stores training history and metrics.

    Attributes:
        train_loss: Training loss per epoch
        val_loss: Validation loss per epoch
        learning_rates: Learning rate per epoch
        physics_metrics: Physics validation metrics per epoch
        epoch_times: Time per epoch in seconds
    """
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    physics_metrics: List[Dict[str, float]] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rates': self.learning_rates,
            'physics_metrics': self.physics_metrics,
            'epoch_times': self.epoch_times,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingHistory':
        """Create from dictionary."""
        history = cls()
        history.train_loss = d.get('train_loss', [])
        history.val_loss = d.get('val_loss', [])
        history.learning_rates = d.get('learning_rates', [])
        history.physics_metrics = d.get('physics_metrics', [])
        history.epoch_times = d.get('epoch_times', [])
        history.best_epoch = d.get('best_epoch', 0)
        history.best_val_loss = d.get('best_val_loss', float('inf'))
        return history

    def save(self, path: Union[str, Path]) -> None:
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate. Default: 0
        last_epoch: Last epoch index. Default: -1
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [self.min_lr + alpha * (base_lr - self.min_lr)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + cosine_decay * (base_lr - self.min_lr)
                    for base_lr in self.base_lrs]


class DensityTrainer:
    """Training class for DensityResNet.

    Handles the complete training loop including:
    - Forward/backward passes
    - Optimizer and scheduler steps
    - Validation
    - Checkpointing
    - Early stopping
    - Physics metrics logging

    Args:
        model: DensityResNet model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (e.g., PhysicsInformedLoss)
        config: Training configuration dictionary
        device: Device to train on. Default: auto-detect
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.training_config = config.get('training', {})

        # Device setup
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Move model to device
        self.model = model.to(device)
        self.criterion = criterion.to(device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history = TrainingHistory()

        # Early stopping
        self.early_stopping_counter = 0
        es_config = self.training_config.get('early_stopping', {})
        self.early_stopping_enabled = es_config.get('enabled', True)
        self.early_stopping_patience = es_config.get('patience', 50)
        self.early_stopping_min_delta = es_config.get('min_delta', 1e-6)

        # Checkpointing
        self.checkpoint_dir = Path(self.training_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Gradient clipping
        self.gradient_clip = self.training_config.get('gradient_clip', None)

        # Logging
        self.log_interval = self.training_config.get('log_interval', 10)
        self.save_interval = self.training_config.get('save_interval', 50)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        optimizer_name = self.training_config.get('optimizer', 'adamw').lower()
        lr = self.training_config.get('learning_rate', 1e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-5)

        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.training_config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')
        epochs = self.training_config.get('epochs', 500)

        if sched_type == 'cosine':
            return CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=sched_config.get('warmup_epochs', 10),
                total_epochs=epochs,
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 100),
                gamma=sched_config.get('gamma', 0.5)
            )
        elif sched_type == 'none' or sched_type is None:
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.train()
        total_loss = 0.0
        metrics = defaultdict(float)
        n_batches = 0

        for batch in self.train_loader:
            # Move data to device and ensure correct dtypes
            # Complex data needs cfloat (complex64), real data needs float32
            density_history = batch['density_history'].to(self.device)
            target = batch['target'].to(self.device)
            overlap = batch['overlap'].to(self.device).float()
            field = batch['field'].to(self.device).float()

            # Convert complex tensors to complex64 if they're complex128
            if density_history.is_complex() and density_history.dtype == torch.complex128:
                density_history = density_history.to(torch.complex64)
            if target.is_complex() and target.dtype == torch.complex128:
                target = target.to(torch.complex64)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(density_history, overlap, field)

            # Compute loss
            if hasattr(self.criterion, 'forward') and 'return_components' in \
               self.criterion.forward.__code__.co_varnames:
                loss, loss_dict = self.criterion(pred, target, overlap, return_components=True)
                for k, v in loss_dict.items():
                    metrics[k] += v.item()
            else:
                loss = self.criterion(pred, target, overlap)
                metrics['loss'] = loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metrics.items()}

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Run validation.

        Returns:
            Tuple of (average loss, loss metrics, physics metrics)
        """
        self.model.eval()
        total_loss = 0.0
        metrics = defaultdict(float)
        physics_metrics = defaultdict(float)
        n_batches = 0

        for batch in self.val_loader:
            # Move data to device and ensure correct dtypes
            density_history = batch['density_history'].to(self.device)
            target = batch['target'].to(self.device)
            overlap = batch['overlap'].to(self.device).float()
            field = batch['field'].to(self.device).float()

            # Convert complex tensors to complex64 if they're complex128
            if density_history.is_complex() and density_history.dtype == torch.complex128:
                density_history = density_history.to(torch.complex64)
            if target.is_complex() and target.dtype == torch.complex128:
                target = target.to(torch.complex64)

            # Forward pass (with projections in eval mode)
            pred = self.model(density_history, overlap, field)

            # Compute loss
            if hasattr(self.criterion, 'forward') and 'return_components' in \
               self.criterion.forward.__code__.co_varnames:
                loss, loss_dict = self.criterion(pred, target, overlap, return_components=True)
                for k, v in loss_dict.items():
                    metrics[k] += v.item()
            else:
                loss = self.criterion(pred, target, overlap)

            total_loss += loss.item()

            # Compute physics metrics
            phys = self._compute_physics_metrics(pred, target, overlap)
            for k, v in phys.items():
                physics_metrics[k] += v

            n_batches += 1

        # Average
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metrics.items()}
        avg_physics = {k: v / n_batches for k, v in physics_metrics.items()}

        return avg_loss, avg_metrics, avg_physics

    def _compute_physics_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        overlap: torch.Tensor
    ) -> Dict[str, float]:
        """Compute physics validation metrics."""
        metrics = {}

        # Hermiticity error
        herm_error = (pred - pred.conj().transpose(-2, -1)).abs()
        metrics['hermiticity_error'] = herm_error.mean().item()

        # Trace comparison
        pred_trace = torch.einsum('bsij,bji->bs', pred, overlap.to(pred.dtype)).real
        target_trace = torch.einsum('bsij,bji->bs', target, overlap.to(target.dtype)).real
        metrics['trace_mae'] = (pred_trace - target_trace).abs().mean().item()

        # Prediction error (Frobenius norm)
        error = pred - target
        metrics['frobenius_error'] = torch.sqrt((error.abs() ** 2).sum(dim=(-2, -1))).mean().item()

        # Eigenvalue statistics
        pred_herm = (pred + pred.conj().transpose(-2, -1)) / 2
        try:
            eigvals = torch.linalg.eigvalsh(pred_herm.reshape(-1, pred.shape[-1], pred.shape[-1]))
            metrics['min_eigenvalue'] = eigvals.min().item()
            metrics['max_eigenvalue'] = eigvals.max().item()
            metrics['negative_eigenvalues'] = (eigvals < -1e-10).sum().item()
        except Exception:
            pass

        return metrics

    def train(self, epochs: Optional[int] = None) -> TrainingHistory:
        """Run full training loop.

        Args:
            epochs: Number of epochs (overrides config if provided)

        Returns:
            TrainingHistory with all metrics
        """
        if epochs is None:
            epochs = self.training_config.get('epochs', 500)

        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics, physics_metrics = self.validate()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.learning_rates.append(current_lr)
            self.history.physics_metrics.append(physics_metrics)
            self.history.epoch_times.append(epoch_time)

            # Check for best model
            if val_loss < self.history.best_val_loss - self.early_stopping_min_delta:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self.early_stopping_counter = 0
                self._save_checkpoint('best_model.pt', is_best=True)
            else:
                self.early_stopping_counter += 1

            # Logging
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                self._log_epoch(epoch, train_loss, val_loss, physics_metrics, current_lr, epoch_time)

            # Periodic checkpoint
            if epoch % self.save_interval == 0 and epoch > 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Early stopping
            if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Save final checkpoint
        self._save_checkpoint('final_model.pt')
        self.history.save(self.checkpoint_dir / 'training_history.json')

        print("-" * 60)
        print(f"Training complete. Best val loss: {self.history.best_val_loss:.6f} at epoch {self.history.best_epoch}")

        return self.history

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        physics_metrics: Dict[str, float],
        lr: float,
        epoch_time: float
    ) -> None:
        """Log epoch results."""
        print(f"Epoch {epoch:4d} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        if physics_metrics:
            herm = physics_metrics.get('hermiticity_error', 0)
            trace = physics_metrics.get('trace_mae', 0)
            frob = physics_metrics.get('frobenius_error', 0)
            print(f"         Physics: Herm={herm:.2e}, Trace MAE={trace:.4f}, Frob={frob:.4f}")

    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save a checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history.to_dict(),
            'best_val_loss': self.history.best_val_loss,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            print(f"  -> Saved best model (val_loss: {self.history.best_val_loss:.6f})")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load a checkpoint to resume training.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = TrainingHistory.from_dict(checkpoint['history'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: Optional[torch.device] = None
    ) -> 'DensityTrainer':
        """Create trainer from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to use

        Returns:
            DensityTrainer instance with loaded state
        """
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        config = checkpoint['config']

        # Import here to avoid circular imports
        try:
            from models.density_resnet import create_model_from_config
        except ImportError:
            from ..models.density_resnet import create_model_from_config

        model = create_model_from_config(config)

        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            device=device
        )

        trainer.load_checkpoint(checkpoint_path)
        return trainer
