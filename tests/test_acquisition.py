"""Automated verification tests for downloaded CWRU raw benchmark files."""

import json
from pathlib import Path
import scipy.io
import pytest

from src.utils.config_parser import load_config, get_resolved_paths


def test_cwru_raw_files_exist():
    """Verify that all 13 required CWRU .mat files are present in data/raw/cwru."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"

    expected_files = [
        "97.mat", "98.mat", "99.mat", "100.mat",
        "118.mat", "185.mat", "222.mat",
        "105.mat", "169.mat", "209.mat",
        "130.mat", "197.mat", "234.mat",
    ]

    for fname in expected_files:
        fpath = raw_dir / fname
        assert fpath.is_file(), f"Expected CWRU raw file missing: {fpath}"
        assert fpath.stat().st_size > 100_000, f"File {fpath} is suspiciously small ({fpath.stat().st_size} bytes)"


def test_cwru_manifest_consistency():
    """Verify that download_manifest.json and download_manifest.csv exist and describe 13 valid files."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"

    manifest_json = raw_dir / "download_manifest.json"
    manifest_csv = raw_dir / "download_manifest.csv"

    assert manifest_json.is_file(), "download_manifest.json is missing"
    assert manifest_csv.is_file(), "download_manifest.csv is missing"

    with open(manifest_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 13, f"Expected 13 records in manifest, found {len(records)}"

    for record in records:
        assert record["status"] == "SUCCESS"
        assert record["is_valid_matlab"] is True
        assert record["file_size_bytes"] > 0
        assert len(record["sha256"]) == 64  # Valid SHA-256 hex string


def test_matlab_parsing_and_channels():
    """Verify that every file can be parsed with scipy.io.loadmat and contains DE_time."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"

    for mat_path in raw_dir.glob("*.mat"):
        mat_data = scipy.io.loadmat(mat_path)
        keys = [k for k in mat_data.keys() if not k.startswith("__")]
        de_keys = [k for k in keys if "DE_time" in k]
        assert len(de_keys) > 0, f"File {mat_path.name} missing Drive End (DE_time) channel"

