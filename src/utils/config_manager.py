"""
Configuration Manager

Handles loading, saving, and merging configuration files.
Supports YAML format and hierarchical configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config if config is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {save_path}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        merged = _deep_update(merged, config)

    return merged


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dictionary.

    Args:
        base: Base dictionary
        update: Dictionary with updates

    Returns:
        Updated dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value

    return result


def resolve_config_references(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve variable references in config (e.g., ${seed}).

    Args:
        config: Configuration dictionary with potential references

    Returns:
        Configuration with resolved references
    """
    def resolve_value(value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Extract variable name
            var_name = value[2:-1]
            # Navigate nested keys (e.g., "system.n_qubits")
            keys = var_name.split('.')
            result = context
            for key in keys:
                result = result.get(key, value)
                if not isinstance(result, dict) and key != keys[-1]:
                    return value
            return result
        elif isinstance(value, dict):
            return {k: resolve_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item, context) for item in value]
        else:
            return value

    return resolve_value(config, config)


def load_experiment_config(
    experiment_name: str,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load experiment configuration with defaults and model configs.

    Args:
        experiment_name: Name of experiment config file (without .yaml)
        config_dir: Directory containing configs (default: project_root/config)
        overrides: Optional dictionary of config overrides

    Returns:
        Complete configuration dictionary
    """
    if config_dir is None:
        # Assume config is in project_root/config
        config_dir = Path(__file__).parent.parent.parent / 'config'

    config_dir = Path(config_dir)

    # Load defaults
    defaults_path = config_dir / 'defaults.yaml'
    config = load_config(defaults_path) if defaults_path.exists() else {}

    # Load experiment config
    experiment_path = config_dir / 'experiment' / f'{experiment_name}.yaml'
    if experiment_path.exists():
        experiment_config = load_config(experiment_path)
        config = merge_configs(config, experiment_config)

    # Load model-specific config
    model_type = config.get('model', {}).get('type', 'svm')
    model_path = config_dir / 'model' / f'{model_type}.yaml'
    if model_path.exists():
        model_config = load_config(model_path)
        config = merge_configs(config, model_config)

    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)

    # Resolve variable references
    config = resolve_config_references(config)

    return config
