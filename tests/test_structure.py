"""Sanity, path resolution, and structural tests for InspectAI repository setup."""

import os
from pathlib import Path
import yaml
import pytest
from src.utils.config_parser import load_config, get_resolved_paths, get_project_root


def test_directory_structure_exists():
    """Verify that all essential directories exist."""
    required_dirs = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "docs",
        "models/checkpoints",
        "models/exported",
        "outputs/reports",
        "outputs/figures",
        "src/ingestion",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "src/inference",
        "src/streaming",
        "src/utils",
        "backend",
        "frontend",
        "tests",
    ]
    root = get_project_root()
    for d in required_dirs:
        p = root / d
        assert p.is_dir(), f"Expected directory missing: {p}"


def test_config_file_validity():
    """Verify that configs/config.yaml is valid YAML and has mandatory keys."""
    config = load_config()

    assert "project" in config
    assert "paths" in config
    assert "dataset" in config
    assert "preprocessing" in config
    assert "model" in config
    assert "training" in config
    assert "explainability" in config
    assert "streaming" in config
    assert config["project"]["device"] == "cpu"


def test_resolved_paths_on_windows():
    """Verify that all configured paths resolve to existing directories on Windows."""
    config = load_config()
    resolved = get_resolved_paths(config)

    expected_keys = [
        "raw_data_dir",
        "interim_data_dir",
        "processed_data_dir",
        "model_checkpoint_dir",
        "model_export_dir",
        "reports_dir",
        "figures_dir",
    ]

    for key in expected_keys:
        assert key in resolved, f"Missing path key in resolved config: {key}"
        path = resolved[key]
        assert isinstance(path, Path)
        assert path.is_absolute(), f"Path {path} is not absolute"
        assert path.exists(), f"Resolved directory does not exist on disk: {path}"


def test_load_config_from_different_working_directory(monkeypatch, tmp_path):
    """Verify load_config works regardless of the active working directory."""
    monkeypatch.chdir(tmp_path)
    # Even when cwd is changed to a temporary directory, load_config should resolve to PROJECT_ROOT
    config = load_config()
    assert config["project"]["name"] == "InspectAI"
