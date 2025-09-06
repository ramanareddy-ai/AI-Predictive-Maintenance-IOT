"""
Configuration loader utility for the predictive maintenance system.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

    # Override with environment variables if they exist
    config = _override_with_env_vars(config)

    return config


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables."""
    # AWS configuration
    if 'AWS_REGION' in os.environ:
        config.setdefault('aws', {})['region'] = os.environ['AWS_REGION']

    if 'SAGEMAKER_ROLE' in os.environ:
        config.setdefault('aws', {}).setdefault('sagemaker', {})['role_arn'] = os.environ['SAGEMAKER_ROLE']

    if 'S3_BUCKET' in os.environ:
        config.setdefault('aws', {}).setdefault('s3', {})['bucket'] = os.environ['S3_BUCKET']

    # Model configuration
    if 'MODEL_SEQUENCE_LENGTH' in os.environ:
        config.setdefault('data', {}).setdefault('preprocessing', {})['sequence_length'] = int(os.environ['MODEL_SEQUENCE_LENGTH'])

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
