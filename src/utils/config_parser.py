"""Configuration parser and Windows path resolver for InspectAI."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

# Determine repository root by navigating two levels up from src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the absolute Path to the InspectAI project root."""
    return PROJECT_ROOT


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and parse the project YAML configuration file.

    Resolves config_path relative to current working directory first,
    and falls back to project root if not found.

    Args:
        config_path: Optional relative or absolute path to configuration YAML file.
            Defaults to 'configs/config.yaml'.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If YAML syntax is invalid.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        path = Path(config_path)
        if path.is_absolute():
            target_path = path
        elif path.is_file():
            target_path = path.resolve()
        else:
            # Fall back to resolving relative to PROJECT_ROOT
            target_path = (PROJECT_ROOT / path).resolve()

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file at {target_path} is empty or not a valid dictionary.")

    return config


def get_resolved_paths(config: Dict[str, Any], create_if_missing: bool = False) -> Dict[str, Path]:
    """Resolve all configured relative directory paths to absolute Windows Path objects.

    Args:
        config: Parsed configuration dictionary containing a 'paths' section.
        create_if_missing: Whether to create directories if they do not exist.

    Returns:
        Dict[str, Path]: Dictionary mapping path keys to absolute resolved Path objects.

    Raises:
        KeyError: If the 'paths' section is missing from the configuration.
    """
    if "paths" not in config or not isinstance(config["paths"], dict):
        raise KeyError("Configuration dictionary is missing a valid 'paths' section.")

    resolved = {}
    for key, rel_path in config["paths"].items():
        abs_path = (PROJECT_ROOT / rel_path).resolve()
        if create_if_missing:
            abs_path.mkdir(parents=True, exist_ok=True)
        resolved[key] = abs_path

    return resolved
