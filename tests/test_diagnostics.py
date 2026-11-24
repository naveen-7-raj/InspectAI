"""Unit tests for Model Diagnostics and Controlled Experiments (Phase 4, Task 4.3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.cnn_1d import VibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.training.trainer import ModelTrainer
from src.utils.config_parser import get_resolved_paths, load_config


class TestDiagnosticsAndExperiments:
    """Test suite verifying diagnostic experiments, isolation constraints, and metric pooling."""

    @pytest.fixture
    def config(self):
        return load_config()

    @pytest.fixture
    def split_metadata(self, config):
        paths = get_resolved_paths(config)
        return pd.read_csv(paths["processed_data_dir"] / "window_split_metadata.csv")

    def test_final_test_recordings_strict_exclusion(self, split_metadata):
        """Verify final test recordings (97, 185, 169, 197) are strictly excluded from dev pool."""
        expected_test = {"97", "185", "169", "197"}
        split_metadata["rec_str"] = split_metadata["recording_id"].astype(str)

        test_recs = set(split_metadata[split_metadata["split"] == "test"]["rec_str"].unique())
        dev_recs = set(split_metadata[split_metadata["split"] == "dev"]["rec_str"].unique())

        assert test_recs == expected_test
        assert len(test_recs.intersection(dev_recs)) == 0, "Test set recordings leaked into dev set!"

    def test_fold_recording_assignments(self, split_metadata):
        """Verify recording assignments for both folds."""
        split_metadata["rec_str"] = split_metadata["recording_id"].astype(str)

        f0_train = set(split_metadata[split_metadata["fold_0"] == "train"]["rec_str"].unique())
        f0_val = set(split_metadata[split_metadata["fold_0"] == "val"]["rec_str"].unique())
        f1_train = set(split_metadata[split_metadata["fold_1"] == "train"]["rec_str"].unique())
        f1_val = set(split_metadata[split_metadata["fold_1"] == "val"]["rec_str"].unique())

        assert f0_train == {"100", "105", "222", "234"}
        assert f0_val == {"98", "99", "118", "130", "209"}
        assert f1_train == {"98", "99", "118", "130", "209"}
        assert f1_val == {"100", "105", "222", "234"}

    def test_independent_fold_normalizers(self, config):
        """Verify normalizers fitted on Fold 0 vs Fold 1 have distinct statistics."""
        ds_f0 = CWRUVibrationDataset(split="train", fold=0, config=config)
        ds_f1 = CWRUVibrationDataset(split="train", fold=1, config=config)

        norm_f0 = VibrationNormalizer().fit(ds_f0.raw_windows[ds_f0.subset_indices])
        norm_f1 = VibrationNormalizer().fit(ds_f1.raw_windows[ds_f1.subset_indices])

        assert norm_f0.is_fitted and norm_f1.is_fitted
        assert norm_f0.fit_sample_count == 938
        assert norm_f1.fit_sample_count == 1172
        # Normalizer statistics must be distinct
        assert not np.isclose(norm_f0.std, norm_f1.std, atol=1e-3)

    def test_independent_model_initialization(self, config):
        """Verify models initialized with distinct seeds have different weights."""
        torch.manual_seed(42)
        m1 = VibrationCNN1D.from_config(config)

        torch.manual_seed(99)
        m2 = VibrationCNN1D.from_config(config)

        # Check weights are not identical
        assert not torch.allclose(m1.block1[0].weight, m2.block1[0].weight)

    def test_configurable_dropout_rate(self):
        """Verify CNN can be instantiated with different dropout rates (e.g. 0.2 vs 0.3)."""
        m_default = VibrationCNN1D(dropout_rate=0.3)
        m_low_drop = VibrationCNN1D(dropout_rate=0.2)

        assert m_default.dropout_rate == 0.3
        assert m_low_drop.dropout_rate == 0.2
        assert m_default.classifier[0].p == 0.3
        assert m_low_drop.classifier[0].p == 0.2

    def test_pooled_oof_sample_accounting(self, split_metadata):
        """Verify pooling Fold 0 val and Fold 1 val covers all 2,110 dev samples without duplicate."""
        f0_val_ids = split_metadata[split_metadata["fold_0"] == "val"]["window_id"].tolist()
        f1_val_ids = split_metadata[split_metadata["fold_1"] == "val"]["window_id"].tolist()

        assert len(f0_val_ids) == 1172
        assert len(f1_val_ids) == 938
        assert len(f0_val_ids) + len(f1_val_ids) == 2110

        # No duplicate window across validation folds
        overlap = set(f0_val_ids).intersection(set(f1_val_ids))
        assert len(overlap) == 0, f"Overlapping validation windows: {len(overlap)}"

    def test_baseline_checkpoints_preserved(self, config):
        """Verify Task 4.2 baseline checkpoints remain untouched on disk."""
        paths = get_resolved_paths(config)
        ckpt_dir = paths["model_checkpoint_dir"]

        f0_base = ckpt_dir / "fold_0_best.pth"
        f1_base = ckpt_dir / "fold_1_best.pth"

        assert f0_base.is_file(), "Baseline fold_0_best.pth missing!"
        assert f1_base.is_file(), "Baseline fold_1_best.pth missing!"

