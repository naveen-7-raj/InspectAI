"""Unit and integration tests for Task 2.3 Vibration Waveform EDA artifacts."""

import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.config_parser import load_config, get_resolved_paths

# Dynamically import module with numeric prefix
eda_mod = importlib.import_module("notebooks.02_vibration_waveform_eda")
compute_signal_statistics = eda_mod.compute_signal_statistics


def test_compute_signal_statistics_math():
    """Verify statistical indicator calculations against known analytical signal."""
    # Sine wave with amplitude A = 2.0: RMS = 2 / sqrt(2) = sqrt(2) ~ 1.4142
    t = np.linspace(0, 1, 10000, endpoint=False)
    sine_wave = 2.0 * np.sin(2 * np.pi * 50 * t)

    stats = compute_signal_statistics(sine_wave)
    assert np.isclose(stats["mean"], 0.0, atol=1e-3)
    assert np.isclose(stats["rms"], np.sqrt(2), rtol=1e-2)
    assert np.isclose(stats["max"], 2.0, atol=1e-3)
    assert np.isclose(stats["min"], -2.0, atol=1e-3)
    assert np.isclose(stats["peak_to_peak"], 4.0, atol=1e-2)
    assert np.isclose(stats["crest_factor"], np.sqrt(2), rtol=1e-2)


def test_waveform_eda_summary_csv_exists_and_valid():
    """Verify that outputs/reports/waveform_eda_summary.csv contains 13 valid records."""
    config = load_config()
    reports_dir = get_resolved_paths(config)["reports_dir"]
    csv_path = reports_dir / "waveform_eda_summary.csv"

    assert csv_path.is_file(), f"Expected report missing: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 13, f"Expected 13 rows for 13 recordings, found {len(df)}"

    required_columns = [
        "file_name",
        "category",
        "defect_type",
        "defect_diameter_in",
        "motor_load_hp",
        "nominal_sampling_rate_hz",
        "total_samples",
        "duration_seconds",
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "peak_to_peak",
        "crest_factor",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing column in EDA CSV: {col}"

    # Verify no NaN or Inf in numeric columns
    numeric_cols = ["mean", "std", "min", "max", "rms", "peak_to_peak", "crest_factor"]
    for col in numeric_cols:
        assert not df[col].isna().any(), f"Column {col} contains NaN values"
        assert not np.isinf(df[col]).any(), f"Column {col} contains Inf values"

    # Physical sanity checks
    assert (df["rms"] > 0).all()
    assert (df["peak_to_peak"] > 0).all()
    assert (df["crest_factor"] >= 1.0).all()

    # Fault vibration energy should be markedly higher than normal baseline
    normal_rms = df[df["file_name"] == "97.mat"]["rms"].values[0]
    inner_race_rms = df[df["file_name"] == "105.mat"]["rms"].values[0]
    outer_race_rms = df[df["file_name"] == "130.mat"]["rms"].values[0]

    assert inner_race_rms > 2.0 * normal_rms, "Inner race fault RMS expected to be >2x healthy baseline"
    assert outer_race_rms > 3.0 * normal_rms, "Outer race fault RMS expected to be >3x healthy baseline"


def test_waveform_figures_exist_and_non_empty():
    """Verify that all generated visualization figures exist with valid content."""
    config = load_config()
    figures_dir = get_resolved_paths(config)["figures_dir"]

    expected_figures = [
        ("waveform_comparison_subplots.png", 50_000),
        ("waveform_comparison_overlay.png", 50_000),
        ("waveform_statistics_summary.png", 25_000),
    ]

    for fname, min_size in expected_figures:
        fig_path = figures_dir / fname
        assert fig_path.is_file(), f"Expected figure missing: {fig_path}"
        assert fig_path.stat().st_size > min_size, (
            f"Figure {fname} is smaller than expected ({fig_path.stat().st_size} bytes < {min_size} bytes)"
        )
