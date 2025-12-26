#!/usr/bin/env python
"""
Training script for DensityResNet.

Usage:
    python scripts/train.py --config configs/h2p_training.json
    python scripts/train.py --config configs/default.json --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np

from models import DensityResNet, create_model_from_config
from physics import PhysicsInformedLoss, create_loss_from_config
from data import create_dataloaders
from data.preprocessing import prepare_data_for_training
from training import DensityTrainer
from utils import load_config, save_config, validate_config, get_default_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DensityResNet for density matrix prediction'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu, overrides config)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for checkpoints (overrides config)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and validate configuration
    print("=" * 60)
    print("DensityResNet Training")
    print("=" * 60)

    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    config = validate_config(config)

    # Apply command-line overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.device is not None:
        config['device'] = args.device
    if args.seed is not None:
        config['seed'] = args.seed
    if args.output_dir is not None:
        config['training']['checkpoint_dir'] = args.output_dir

    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device setup
    device_str = config.get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load data
    print("\n" + "-" * 60)
    print("Loading data...")
    data_config = config['data']

    density_series, overlap_matrix, field_data, norm_stats = prepare_data_for_training(
        density_file=data_config['density_file'],
        overlap_file=data_config['overlap_file'],
        field_file=data_config.get('field_file'),
        normalize=data_config.get('normalize', True),
        normalization_mode=data_config.get('normalization_mode', 'frobenius'),
        validate=True
    )

    print(f"Density series shape: {density_series.shape}")
    print(f"Overlap matrix shape: {overlap_matrix.shape}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        density_series=density_series,
        overlap_matrix=overlap_matrix,
        field_data=field_data,
        history_length=config['model']['history_length'],
        batch_size=data_config['batch_size'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        num_workers=data_config.get('num_workers', 0),
        seed=seed
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")

    if args.checkpoint is not None:
        print(f"Loading from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = create_model_from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = create_model_from_config(config)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function
    print("\nCreating loss function...")
    criterion = create_loss_from_config(config)
    print(f"Loss weights: MSE={criterion.weight_mse}, Herm={criterion.weight_hermitian}, "
          f"Idemp={criterion.weight_idempotency}, Trace={criterion.weight_trace}")

    # Create trainer
    print("\n" + "-" * 60)
    print("Starting training...")

    trainer = DensityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device
    )

    # Resume from checkpoint if provided
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    history = trainer.train()

    # Save final config with normalization stats
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    config_save_path = checkpoint_dir / 'config.json'
    if norm_stats is not None:
        config['normalization'] = norm_stats.to_dict()
    save_config(config, config_save_path)
    print(f"\nConfig saved to: {config_save_path}")

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {history.best_val_loss:.6f} at epoch {history.best_epoch}")
    print(f"Final training loss: {history.train_loss[-1]:.6f}")
    print(f"Final validation loss: {history.val_loss[-1]:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
