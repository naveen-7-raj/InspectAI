"""Unit tests for Leave-One-Load-Out (LOLO) MultiLoadSplitter (Phase 11.1).

Verifies:
1. MultiLoadSplitter initializes correctly with the 40-recording manifest.
2. Protected holdout recordings (97, 169, 185, 197) are strictly excluded from all folds.
3. Exactly three folds are created with validation loads 1 HP, 2 HP, and 3 HP (no 0 HP val fold).
4. For every fold, training and validation recording sets are strictly disjoint (0 leakage).
5. All four classes (Normal, Inner Race, Ball, Outer Race) are present in train and validation.
6. All three defect sizes (0.007\", 0.014\", 0.021\") are present in train and validation.
7. Normalizer fits exclusively on training data without errors or NaNs.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.training.multiload_splitter import (
    MultiLoadSplitter,
    PROTECTED_HOLDOUT_IDS,
    EXPECTED_HOLDOUT_SHA256,
)

MANIFEST_PATH = Path("outputs/experiments/phase10_6/cwru_40_recordings_manifest.csv")
RAW_DIR = Path("data/raw/cwru")
OUTPUT_DIR = Path("outputs/experiments/phase11_1")


@pytest.fixture(scope="module")
def splitter():
    """Shared MultiLoadSplitter fixture."""
    return MultiLoadSplitter(manifest_path=MANIFEST_PATH, raw_data_dir=RAW_DIR)


def test_splitter_initialization(splitter):
    """Verify splitter loads exactly 40 recordings and identifies 36 development recordings."""
    assert len(splitter.df_manifest) == 40
    dev_recs = splitter.get_development_recordings()
    assert len(dev_recs) == 36
    holdout_recs = splitter.get_holdout_recordings()
    assert len(holdout_recs) == 4
    assert set(holdout_recs) == PROTECTED_HOLDOUT_IDS


def test_holdout_isolation(splitter):
    """Verify protected holdouts never appear in train or validation in any fold."""
    for fold_id in [1, 2, 3]:
        fold = splitter.get_fold(fold_id)
        train_set = set(fold.train_recording_ids)
        val_set = set(fold.val_recording_ids)

        assert train_set.isdisjoint(PROTECTED_HOLDOUT_IDS), f"Holdout found in Fold {fold_id} Train!"
        assert val_set.isdisjoint(PROTECTED_HOLDOUT_IDS), f"Holdout found in Fold {fold_id} Val!"


def test_three_folds_configuration(splitter):
    """Verify exactly 3 folds exist with correct val/train load mappings."""
    assert set(splitter.folds.keys()) == {1, 2, 3}

    fold1 = splitter.get_fold(1)
    assert fold1.val_load_hp == 1
    assert set(fold1.train_loads_hp) == {0, 2, 3}
    assert len(fold1.val_recording_ids) == 10
    assert len(fold1.train_recording_ids) == 26

    fold2 = splitter.get_fold(2)
    assert fold2.val_load_hp == 2
    assert set(fold2.train_loads_hp) == {0, 1, 3}
    assert len(fold2.val_recording_ids) == 10
    assert len(fold2.train_recording_ids) == 26

    fold3 = splitter.get_fold(3)
    assert fold3.val_load_hp == 3
    assert set(fold3.train_loads_hp) == {0, 1, 2}
    assert len(fold3.val_recording_ids) == 10
    assert len(fold3.train_recording_ids) == 26


def test_recording_disjointness_and_completeness(splitter):
    """Verify zero overlap between train and val, and union equals all 36 dev recordings."""
    all_dev = set(splitter.get_development_recordings())

    for fold_id in [1, 2, 3]:
        fold = splitter.get_fold(fold_id)
        train_set = set(fold.train_recording_ids)
        val_set = set(fold.val_recording_ids)

        # Zero overlap
        assert train_set.isdisjoint(val_set), f"Overlap in Fold {fold_id}: {train_set & val_set}"
        # Union equals all dev recordings
        assert (train_set | val_set) == all_dev, f"Incomplete dev coverage in Fold {fold_id}"


def test_class_presence_across_folds(splitter):
    """Verify all 4 classes exist in train and validation sets of every fold."""
    manifest = splitter.df_manifest.set_index("recording_id")

    for fold_id in [1, 2, 3]:
        fold = splitter.get_fold(fold_id)

        train_classes = set(manifest.loc[fold.train_recording_ids, "fault_class"])
        val_classes = set(manifest.loc[fold.val_recording_ids, "fault_class"])

        expected = {"Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"}
        assert train_classes == expected, f"Fold {fold_id} Train missing classes: {expected - train_classes}"
        assert val_classes == expected, f"Fold {fold_id} Val missing classes: {expected - val_classes}"


def test_defect_size_coverage(splitter):
    """Verify all 3 defect sizes (0.007, 0.014, 0.021) are present in both train and val."""
    manifest = splitter.df_manifest.set_index("recording_id")

    for fold_id in [1, 2, 3]:
        fold = splitter.get_fold(fold_id)

        train_sizes = set(manifest.loc[fold.train_recording_ids, "fault_size_inches"])
        val_sizes = set(manifest.loc[fold.val_recording_ids, "fault_size_inches"])

        for sz in [0.007, 0.014, 0.021]:
            assert sz in train_sizes, f"Defect size {sz} missing from Fold {fold_id} Train"
            assert sz in val_sizes, f"Defect size {sz} missing from Fold {fold_id} Val"


def test_normalizer_fitting_isolation(splitter):
    """Verify normalizer fits cleanly on mock training windows without NaNs."""
    np.random.seed(42)
    mock_train_windows = np.random.randn(500, 2048).astype(np.float32) * 0.5 + 0.1

    normalizer = splitter.fit_fold_normalizer(fold_id=1, training_windows=mock_train_windows)
    assert not np.isnan(normalizer.mean)
    assert not np.isnan(normalizer.std)
    assert normalizer.std > 0


def test_generated_artifacts_integrity():
    """Verify Phase 11.1 generated deliverables exist and report PASSED."""
    report_path = OUTPUT_DIR / "split_verification_report.json"
    splits_path = OUTPUT_DIR / "multiload_splits.json"
    summary_path = OUTPUT_DIR / "lolo_fold_summary.csv"
    dist_path = OUTPUT_DIR / "lolo_recording_distribution.csv"
    severity_path = OUTPUT_DIR / "lolo_defect_severity_analysis.json"

    assert report_path.exists(), "split_verification_report.json missing"
    assert splits_path.exists(), "multiload_splits.json missing"
    assert summary_path.exists(), "lolo_fold_summary.csv missing"
    assert dist_path.exists(), "lolo_recording_distribution.csv missing"
    assert severity_path.exists(), "lolo_defect_severity_analysis.json missing"

    with open(report_path) as f:
        rep = json.load(f)
    assert rep["status"] == "PASSED"
    assert rep["development_recordings_count"] == 36
    assert rep["number_of_folds"] == 3
