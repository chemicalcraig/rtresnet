"""
Configuration management for density matrix prediction.

Handles loading, saving, validation, and merging of JSON configuration files.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from copy import deepcopy


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration dictionary.

    Returns:
        Default configuration with all supported options.
    """
    return {
        "model": {
            "type": "DensityResNet",
            "max_nbf": 16,
            "hidden_dim": 256,
            "num_resnet_blocks": 6,
            "history_length": 5,
            "n_spin": 2,
            "dropout": 0.1,
            "overlap_mode": "spectral",
            "field_dim": 3,
            "apply_hermitian_projection": True,
            "apply_mcweeney_projection": False,
            "apply_trace_scaling": False
        },
        "physics": {
            "n_electrons": None,  # e.g., [1.0, 0.0] for H2+
        },
        "loss": {
            "weight_mse": 1.0,
            "weight_hermitian": 0.1,
            "weight_idempotency": 0.05,
            "weight_trace": 0.1,
            "weight_positivity": 0.01
        },
        "data": {
            "density_file": "density_series.npy",
            "overlap_file": "overlap.npy",
            "field_file": None,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "normalize": True,
            "normalization_mode": "frobenius",
            "batch_size": 32,
            "num_workers": 0
        },
        "training": {
            "epochs": 500,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 10,
                "min_lr": 1e-6
            },
            "early_stopping": {
                "enabled": True,
                "patience": 50,
                "min_delta": 1e-6
            },
            "gradient_clip": 1.0,
            "mixed_precision": False,
            "checkpoint_dir": "checkpoints/",
            "log_interval": 10,
            "save_interval": 50
        },
        "prediction": {
            "mode": "single_step",  # or "rollout"
            "rollout_steps": 100,
            "bootstrap_steps": 5,
            "output_file": "predicted_densities.npy",
            "save_interval": 10
        },
        "device": "cuda",
        "seed": 42
    }


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        config = json.load(f)

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to a JSON file.

    Args:
        config: Configuration dictionary to save.
        path: Path to save the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Values in `override` take precedence over values in `base`.
    Nested dictionaries are merged recursively.

    Args:
        base: Base configuration dictionary.
        override: Configuration with values to override.

    Returns:
        Merged configuration dictionary.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill in missing values with defaults.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Validated configuration with defaults filled in.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    default = get_default_config()
    validated = merge_configs(default, config)

    # Validate required fields
    model_config = validated.get('model', {})

    if model_config.get('max_nbf', 0) <= 0:
        raise ValueError("model.max_nbf must be a positive integer")

    if model_config.get('hidden_dim', 0) <= 0:
        raise ValueError("model.hidden_dim must be a positive integer")

    if model_config.get('history_length', 0) <= 0:
        raise ValueError("model.history_length must be a positive integer")

    if model_config.get('n_spin') not in [1, 2]:
        raise ValueError("model.n_spin must be 1 or 2")

    # Validate training config
    training_config = validated.get('training', {})

    if training_config.get('epochs', 0) <= 0:
        raise ValueError("training.epochs must be a positive integer")

    if training_config.get('learning_rate', 0) <= 0:
        raise ValueError("training.learning_rate must be positive")

    # Validate data config
    data_config = validated.get('data', {})

    ratios = (
        data_config.get('train_ratio', 0) +
        data_config.get('val_ratio', 0) +
        data_config.get('test_ratio', 0)
    )
    if abs(ratios - 1.0) > 0.01:
        raise ValueError(f"Data split ratios must sum to 1.0, got {ratios}")

    # Validate physics config
    physics_config = validated.get('physics', {})
    n_electrons = physics_config.get('n_electrons')

    if n_electrons is not None:
        if isinstance(n_electrons, list):
            if len(n_electrons) != model_config.get('n_spin', 2):
                raise ValueError(
                    f"n_electrons list length {len(n_electrons)} != n_spin {model_config.get('n_spin')}"
                )
        elif not isinstance(n_electrons, (int, float)):
            raise ValueError("n_electrons must be a number or list of numbers")

    return validated


def create_config_for_molecule(
    nbf: int,
    n_spin: int,
    n_electrons: Union[float, list],
    data_dir: str,
    **kwargs
) -> Dict[str, Any]:
    """Create a configuration for a specific molecule.

    Convenience function to generate a config with molecule-specific settings.

    Args:
        nbf: Number of basis functions.
        n_spin: Number of spin channels (1 or 2).
        n_electrons: Number of electrons (total or per-spin list).
        data_dir: Directory containing the data files.
        **kwargs: Additional config overrides.

    Returns:
        Configuration dictionary.
    """
    config = get_default_config()

    # Set molecule-specific values
    config['model']['max_nbf'] = nbf
    config['model']['n_spin'] = n_spin
    config['physics']['n_electrons'] = n_electrons

    # Set data paths
    data_dir = Path(data_dir)
    config['data']['density_file'] = str(data_dir / "density_series.npy")
    config['data']['overlap_file'] = str(data_dir / "overlap.npy")

    field_file = data_dir / "field_synced.npy"
    if field_file.exists():
        config['data']['field_file'] = str(field_file)

    # Apply any additional overrides
    config = merge_configs(config, kwargs)

    return config
