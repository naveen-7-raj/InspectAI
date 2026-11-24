"""Unit tests for Phase 4, Task 4.4 Final Model Training and Unseen Test Evaluation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.cnn_1d import VibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import get_resolved_paths, load_config


class TestFinalModelEvaluation:
    """Test suite verifying final model training, checkpoint validity, and strict test isolation."""

    @pytest.fixture
    def config(self):
        return load_config()

    @pytest.fixture
    def resolved_paths(self, config):
        return get_resolved_paths(config)

    @pytest.fixture
    def split_metadata(self, resolved_paths):
        return pd.read_csv(resolved_paths["processed_data_dir"] / "window_split_metadata.csv")

    def test_development_and_test_partition_counts(self, split_metadata):
        """Verify exact sample counts and recording memberships for final evaluation."""
        split_metadata["rec_str"] = split_metadata["recording_id"].astype(str)

        dev_df = split_metadata[split_metadata["split"] == "dev"]
        test_df = split_metadata[split_metadata["split"] == "test"]

        assert len(dev_df) == 2110, f"Expected 2110 dev samples, got {len(dev_df)}"
        assert len(test_df) == 818, f"Expected 818 test samples, got {len(test_df)}"

        dev_recs = set(dev_df["rec_str"].unique())
        test_recs = set(test_df["rec_str"].unique())

        assert dev_recs == {"98", "99", "100", "105", "118", "130", "209", "222", "234"}
        assert test_recs == {"97", "169", "185", "197"}
        assert len(dev_recs.intersection(test_recs)) == 0

    def test_final_normalizer_provenance(self, resolved_paths):
        """Verify final normalizer was fitted exclusively on development recordings."""
        norm_path = resolved_paths["model_checkpoint_dir"] / "inspectai_final_normalizer.json"
        if not norm_path.is_file():
            pytest.skip("Final normalizer file not yet created.")

        with open(norm_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["is_fitted"] is True
        assert data["fit_sample_count"] == 2110
        # Check test recordings are NOT in provenance
        fit_recs = set(str(r) for r in data["fit_recording_ids"])
        test_recs = {"97", "169", "185", "197"}
        assert len(fit_recs.intersection(test_recs)) == 0, "Test recordings leaked into final normalizer!"

    def test_final_checkpoint_integrity_and_loading(self, resolved_paths, config):
        """Verify final checkpoint exists, contains exactly 4 epochs, and loads for inference."""
        ckpt_path = resolved_paths["model_checkpoint_dir"] / "inspectai_final_cnn.pth"
        if not ckpt_path.is_file():
            pytest.skip("Final checkpoint file not yet created.")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert ckpt["epoch"] == 4
        assert len(ckpt["training_history"]) == 4
        assert ckpt["hyperparameters"]["learning_rate"] == 0.0005
        assert ckpt["hyperparameters"]["dropout_rate"] == 0.3
        assert ckpt["training_samples_count"] == 2110

        # Verify model loads and executes forward pass
        model = VibrationCNN1D.from_config(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(2, 1, 2048)
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()

    def test_previous_checkpoints_preserved(self, resolved_paths):
        """Verify baseline and experiment checkpoints were not overwritten."""
        ckpt_dir = resolved_paths["model_checkpoint_dir"]
        assert (ckpt_dir / "fold_0_best.pth").is_file()
        assert (ckpt_dir / "fold_1_best.pth").is_file()
        assert (ckpt_dir / "normalizer_fold_0.json").is_file()
        assert (ckpt_dir / "normalizer_fold_1.json").is_file()
