"""Unit and validation tests for DatasetSplitter (src/training/dataset_splitter.py)."""

from pathlib import Path
import pandas as pd
import pytest

from src.training.dataset_splitter import DatasetSplitter
from src.utils.config_parser import load_config, get_resolved_paths


@pytest.fixture
def splitter():
    return DatasetSplitter()


def test_no_recording_id_occurs_across_test_and_dev(splitter):
    """Test 1: Verify test and development recordings are strictly disjoint sets."""
    _, summary = splitter.create_split()

    test_recs = set(summary["test_recordings"])
    dev_recs = set(summary["dev_recordings"])

    assert len(test_recs) == 4
    assert len(dev_recs) == 9
    assert test_recs.isdisjoint(dev_recs), f"Leakage detected! Shared files: {test_recs & dev_recs}"


def test_all_windows_from_one_recording_stay_together(splitter):
    """Test 2: Verify that every window belonging to a recording has the exact same split assignment."""
    df, _ = splitter.create_split()

    for rec_id, group in df.groupby("source_file"):
        # Distinct splits for this file must be exactly 1
        unique_splits = group["split"].unique()
        assert len(unique_splits) == 1, f"Recording {rec_id} was split across: {unique_splits}"

        # If in dev, folds must also be identical across all windows of the recording
        if unique_splits[0] == "dev":
            assert len(group["fold_0"].unique()) == 1
            assert len(group["fold_1"].unique()) == 1


def test_every_window_assigned_exactly_once(splitter):
    """Test 3: Verify every window in the dataset appears in exactly one partition."""
    df, summary = splitter.create_split()

    assert len(df) == summary["total_windows"]
    assert summary["test_windows"] + summary["dev_windows"] == summary["total_windows"]
    assert set(df["split"].unique()) == {"test", "dev"}


def test_all_four_classes_represented_in_test_set(splitter):
    """Test 4: Verify that the final test holdout contains all 4 health classes."""
    _, summary = splitter.create_split()

    test_dist = summary["test_class_distribution"]
    expected_classes = {"Normal", "Ball_Fault", "Inner_Race_Fault", "Outer_Race_Fault"}

    assert set(test_dist.keys()) == expected_classes
    for cls_name, count in test_dist.items():
        assert count > 0, f"Class {cls_name} has 0 samples in test set"


def test_validation_folds_class_coverage(splitter):
    """Test 5: Verify that both 2-fold CV folds contain all 4 classes in both train and val."""
    _, summary = splitter.create_split()

    expected_classes = {"Normal", "Ball_Fault", "Inner_Race_Fault", "Outer_Race_Fault"}

    for fold_idx, finfo in summary["folds"].items():
        assert set(finfo["val_classes"]) == expected_classes, f"Fold {fold_idx} val missing classes"
        assert set(finfo["train_classes"]) == expected_classes, f"Fold {fold_idx} train missing classes"
        assert finfo["train_windows"] > 0
        assert finfo["val_windows"] > 0

        # Verify no recording leakage between train and val
        val_set = set(finfo["val_recordings"])
        train_set = set(finfo["train_recordings"])
        assert val_set.isdisjoint(train_set)


def test_deterministic_split_with_same_seed():
    """Test 6: Verify split repeatability when initialized with the same random seed."""
    s1 = DatasetSplitter()
    s2 = DatasetSplitter()

    df1, sum1 = s1.create_split()
    df2, sum2 = s2.create_split()

    pd.testing.assert_frame_equal(df1, df2)
    assert sum1 == sum2

