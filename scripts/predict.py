#!/usr/bin/env python
"""
Prediction script for DensityResNet.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best_model.pt --density data/density_series.npy --overlap data/overlap.npy --n-steps 100
    python scripts/predict.py --checkpoint checkpoints/best_model.pt --config configs/h2p_training.json --n-steps 500
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np

from inference import DensityPredictor, RolloutResult


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run prediction/rollout with trained DensityResNet'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--density', type=str, default=None,
        help='Path to density_series.npy file'
    )
    parser.add_argument(
        '--overlap', type=str, default=None,
        help='Path to overlap.npy file'
    )
    parser.add_argument(
        '--field', type=str, default=None,
        help='Path to field.npy file (optional)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config JSON (uses data paths from config if --density/--overlap not provided)'
    )
    parser.add_argument(
        '--n-steps', type=int, required=True,
        help='Number of prediction steps'
    )
    parser.add_argument(
        '--start-idx', type=int, default=0,
        help='Starting index in density series for bootstrap (default: 0)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='predictions',
        help='Output directory for predictions (default: predictions/)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--apply-hermitian', action='store_true', default=True,
        help='Apply Hermitian projection (default: True)'
    )
    parser.add_argument(
        '--apply-mcweeney', action='store_true', default=False,
        help='Apply McWeeney purification (default: False)'
    )
    parser.add_argument(
        '--apply-trace-scaling', action='store_true', default=False,
        help='Apply trace scaling (default: False)'
    )
    parser.add_argument(
        '--n-electrons', type=float, nargs='+', default=None,
        help='Number of electrons per spin channel (e.g., --n-electrons 1.0 0.0)'
    )
    parser.add_argument(
        '--compare-reference', action='store_true', default=False,
        help='Compare predictions with reference trajectory'
    )
    parser.add_argument(
        '--no-save', action='store_true', default=False,
        help='Skip saving predictions to disk'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DensityResNet Prediction")
    print("=" * 60)

    # Determine data paths
    density_file = args.density
    overlap_file = args.overlap
    field_file = args.field

    if density_file is None or overlap_file is None:
        if args.config is None:
            raise ValueError(
                "Must provide either --density/--overlap or --config"
            )
        # Load paths from config
        from utils import load_config
        config = load_config(args.config)
        data_config = config['data']
        density_file = density_file or data_config.get('density_file')
        overlap_file = overlap_file or data_config.get('overlap_file')
        field_file = field_file or data_config.get('field_file')

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Density file: {density_file}")
    print(f"Overlap file: {overlap_file}")
    print(f"Field file: {field_file}")
    print(f"Prediction steps: {args.n_steps}")
    print(f"Start index: {args.start_idx}")

    # Device setup
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Physics projections
    print("\nPhysics projections:")
    print(f"  Hermitian: {args.apply_hermitian}")
    print(f"  McWeeney: {args.apply_mcweeney}")
    print(f"  Trace scaling: {args.apply_trace_scaling}")
    if args.n_electrons:
        print(f"  N_electrons: {args.n_electrons}")

    # Create predictor
    print("\n" + "-" * 60)
    print("Loading model...")

    predictor = DensityPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        apply_hermitian=args.apply_hermitian,
        apply_mcweeney=args.apply_mcweeney,
        apply_trace_scaling=args.apply_trace_scaling,
        n_electrons=args.n_electrons
    )

    print(f"Model loaded. History length: {predictor.history_length}")

    # Run prediction
    print("\n" + "-" * 60)
    print("Running prediction...")

    result = predictor.predict_from_files(
        density_file=density_file,
        overlap_file=overlap_file,
        n_steps=args.n_steps,
        field_file=field_file,
        output_dir=None if args.no_save else args.output_dir,
        start_idx=args.start_idx
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)
    print(f"Output shape: {result.predictions.shape}")
    print(f"Timestamps: {result.timestamps[0]} to {result.timestamps[-1]}")

    # Physics metrics summary
    if result.physics_metrics:
        print("\nPhysics metrics (final step):")
        final_metrics = result.physics_metrics[-1]
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")

        # Aggregate statistics
        print("\nPhysics metrics (mean over trajectory):")
        metric_keys = result.physics_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in result.physics_metrics]
            print(f"  {key}: {np.mean(values):.6f} ± {np.std(values):.6f}")

    # Compare with reference if requested
    if args.compare_reference:
        print("\n" + "-" * 60)
        print("Comparing with reference trajectory...")

        reference = np.load(density_file)
        ref_start = args.start_idx + predictor.history_length

        errors = predictor.compare_with_reference(
            result=result,
            reference=reference,
            start_idx=ref_start
        )

        print(f"\nError metrics:")
        print(f"  Mean Frobenius error: {errors['mean_frobenius']:.6f}")
        print(f"  Final Frobenius error: {errors['final_frobenius']:.6f}")

        # Save error metrics
        if not args.no_save:
            output_dir = Path(args.output_dir)
            np.save(output_dir / 'frobenius_errors.npy', errors['frobenius'])
            print(f"\nError metrics saved to: {output_dir}")

    # Save location
    if not args.no_save:
        print(f"\nPredictions saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
