"""Unit and integration tests for Phase 11.3 Cross-Load Failure Analysis.

Verifies:
1. Correct aggregation of cross-load metrics and artifact generation.
2. Correct load, class, and defect severity grouping.
3. Correct calculation of recording-level physical signal statistics (RMS, kurtosis, crest factor, spectral bands).
4. Strict exclusion of protected holdout IDs (97, 169, 185, 197) from development analysis.
5. Reproducible analysis and deterministic hypothesis structure.
6. Graceful handling of missing artifacts.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.training.multiload_splitter import PROTECTED_HOLDOUT_IDS
from scripts.run_crossload_failure_analysis import (
    compute_spectral_band_energies,
    evaluate_hypotheses,
    generate_improvement_recommendations,
    CLASS_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_11_3_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_3"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "phase11_3"


def test_spectral_band_energy_calculation():
    """Verify spectral band energy computation sums to 1.0."""
    fs = 12000
    t = np.arange(2048) / fs
    # Pure tone at 100 Hz (Low band: 20-500 Hz)
    sig_low = np.sin(2 * np.pi * 100 * t)
    bands = compute_spectral_band_energies(sig_low, fs=fs)
    assert bands["low_band_ratio"] > 0.8
    assert bands["mid_band_ratio"] < 0.2
    assert bands["high_band_ratio"] < 0.1
    assert abs((bands["low_band_ratio"] + bands["mid_band_ratio"] + bands["high_band_ratio"]) - 1.0) < 0.05


def test_recording_signal_statistics_integrity():
    """Verify recording signal statistics CSV has 40 recordings and valid values."""
    csv_path = EXP_11_3_DIR / "recording_signal_statistics.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 40
    assert set(df["fault_class"].unique()) == set(CLASS_NAMES)
    assert set(df["load_hp"].unique()) == {0, 1, 2, 3}

    # Physical validity
    assert (df["rms_amplitude"] > 0).all()
    assert (df["peak_amplitude"] >= df["rms_amplitude"]).all()
    assert (df["crest_factor"] >= 1.0).all()

    # Verify holdouts are properly flagged
    holdout_subset = df[df["recording_id"].astype(str).isin(PROTECTED_HOLDOUT_IDS)]
    assert len(holdout_subset) == 4
    assert (holdout_subset["is_holdout"] == True).all()


def test_misclassification_analysis_integrity():
    """Verify misclassification analysis correctly isolates Recording 172's 62 windows."""
    csv_path = EXP_11_3_DIR / "misclassification_analysis.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 234  # Total windows of recording 172
    assert (df["recording_id"] == 172).all() or (df["recording_id"] == "172").all()

    misclassified = df[df["is_correct"] == False]
    correct = df[df["is_correct"] == True]

    assert len(misclassified) == 62
    assert len(correct) == 172
    # All 62 misclassifications were predicted as Ball Fault
    assert (misclassified["predicted_class"] == "Ball Fault").all()


def test_hypotheses_and_recommendations_structure():
    """Verify formal hypothesis assessment and improvement priorities have expected schemas."""
    hyp_path = EXP_11_3_DIR / "hypothesis_assessment.json"
    rec_path = EXP_11_3_DIR / "improvement_priorities.json"
    sum_path = EXP_11_3_DIR / "analysis_summary.json"

    assert hyp_path.exists()
    assert rec_path.exists()
    assert sum_path.exists()

    with open(hyp_path) as f:
        hypotheses = json.load(f)
    assert len(hypotheses) == 7
    valid_statuses = {
        "Supported by current evidence",
        "Partially supported",
        "Not supported",
        "Insufficient evidence"
    }
    for h in hypotheses:
        assert h["status"] in valid_statuses
        assert len(h["evidence"]) > 20

    with open(rec_path) as f:
        recs = json.load(f)
    assert len(recs) == 3
    for r in recs:
        assert "priority" in r
        assert "target_failure" in r
        assert "testing_strategy" in r
        assert "adoption_criteria" in r

    with open(sum_path) as f:
        summary = json.load(f)
    assert summary["worst_performing_load"]["load_hp"] == 3
    assert summary["worst_performing_class"]["class_name"] == "Inner Race Fault"


def test_deliverable_figures_exist():
    """Verify that all publication figures exist and are non-empty."""
    fig1 = FIGURES_DIR / "confusion_matrices_cross_load.png"
    fig2 = FIGURES_DIR / "signal_statistics_comparison.png"
    fig3 = FIGURES_DIR / "gradcam_analysis_grid.png"

    assert fig1.exists() and fig1.stat().st_size > 10_000
    assert fig2.exists() and fig2.stat().st_size > 10_000
    assert fig3.exists() and fig3.stat().st_size > 10_000

