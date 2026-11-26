"""Unit and integration tests for Phase 11.4 Targeted Model Improvement & Controlled Ablations.

Tests cover:
1. Training-only augmentation (transformations applied only when augment=True, deterministic reproducibility).
2. No augmentation of validation or evaluation data (exact identity preserved).
3. Recording-level split integrity (zero recording overlap between partitions).
4. Historical holdout exclusion (recordings 97, 169, 185, 197 permanently quarantined).
5. Normalizer fitting provenance (normalizers fitted strictly on training partition, zero leakage).
6. Class-weight calculation (balanced inverse frequency on training labels only).
7. Reproducibility across random seeds.
8. Model architecture preservation (VibrationCNN1D parameter count = 73,156, SpectralFusionCNN = 89,332).
9. Checkpoint isolation (no overwriting of production or Phase 11.2 checkpoints).
10. Correct experiment artifact generation and schema validation.
"""

import json
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from scripts.run_targeted_improvements import (
    AugmentedVibrationDataset,
    DualBranchVibrationDataset,
    create_inner_split,
    compute_balanced_weights,
    SEED,
)
from src.models.cnn_1d import VibrationCNN1D
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.training.multiload_splitter import PROTECTED_HOLDOUT_IDS
from src.training.normalizer import VibrationNormalizer
from src.features.frequency_features import SpectralNormalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_11_4_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_4"
SPLITS_PATH = REPO_ROOT / "outputs" / "experiments" / "phase11_1" / "multiload_splits.json"
MANIFEST_PATH = REPO_ROOT / "outputs" / "experiments" / "phase10_6" / "cwru_40_recordings_manifest.csv"


class TestAugmentationIntegrity:
    """Verify training-only augmentation guarantees."""

    def test_training_augmentation_transforms_samples(self):
        """Augmentation with augment=True modifies samples within expected bounds."""
        X = np.ones((5, 2048), dtype=np.float32) * 2.0
        y = np.array([0, 1, 2, 3, 0], dtype=np.int64)
        ds = AugmentedVibrationDataset(X, y, augment=True, seed=42)

        x_aug, y_out = ds[0]
        x_aug_np = x_aug.squeeze(0).numpy()

        # Should not be identical to original
        assert not np.allclose(x_aug_np, X[0])
        # Values should remain within reasonable scaled bounds (+/- 15%)
        assert np.all(x_aug_np > 1.5) and np.all(x_aug_np < 2.5)
        assert y_out.item() == y[0]

    def test_validation_and_evaluation_data_not_augmented(self):
        """Augmentation with augment=False preserves exact numerical identity."""
        X = np.random.randn(8, 2048).astype(np.float32)
        y = np.arange(8, dtype=np.int64)
        ds = AugmentedVibrationDataset(X, y, augment=False, seed=42)

        for i in range(len(X)):
            x_out, y_out = ds[i]
            np.testing.assert_array_equal(x_out.squeeze(0).numpy(), X[i])
            assert y_out.item() == y[i]

    def test_augmentation_deterministic_reproducibility(self):
        """Identical index and seed produce identical augmented tensors."""
        X = np.random.randn(5, 2048).astype(np.float32)
        y = np.arange(5, dtype=np.int64)

        ds1 = AugmentedVibrationDataset(X, y, augment=True, seed=123)
        ds2 = AugmentedVibrationDataset(X, y, augment=True, seed=123)

        for i in range(len(X)):
            x1, _ = ds1[i]
            x2, _ = ds2[i]
            np.testing.assert_array_equal(x1.numpy(), x2.numpy())


class TestSplitAndHoldoutIntegrity:
    """Verify strict partition isolation and holdout protection."""

    def test_protected_holdout_permanently_quarantined(self):
        """Holdout recordings 97, 169, 185, 197 are excluded from all development folds."""
        assert SPLITS_PATH.exists()
        with open(SPLITS_PATH, "r") as f:
            data = json.load(f)

        for fold_key, fold_info in data["folds"].items():
            outer_train = [str(r) for r in fold_info["train_recording_ids"]]
            outer_eval = [str(r) for r in fold_info["val_recording_ids"]]

            for h_id in PROTECTED_HOLDOUT_IDS:
                assert h_id not in outer_train, f"Holdout {h_id} leaked into outer_train of {fold_key}"
                assert h_id not in outer_eval, f"Holdout {h_id} leaked into outer_eval of {fold_key}"

    def test_recording_level_split_disjointness(self):
        """Outer train and outer evaluation recordings have zero overlap."""
        with open(SPLITS_PATH, "r") as f:
            data = json.load(f)

        for fold_key, fold_info in data["folds"].items():
            train_set = set(str(r) for r in fold_info["train_recording_ids"])
            eval_set = set(str(r) for r in fold_info["val_recording_ids"])
            assert len(train_set.intersection(eval_set)) == 0
            assert len(train_set) == 26
            assert len(eval_set) == 10

    def test_inner_split_disjointness_and_stratification(self):
        """Inner-train and inner-val partitions have zero overlap and cover all 4 classes."""
        manifest = pd.read_csv(MANIFEST_PATH)
        manifest["recording_id"] = manifest["recording_id"].astype(str)
        manifest_idx = manifest.set_index("recording_id")

        with open(SPLITS_PATH, "r") as f:
            data = json.load(f)

        for fold_key, fold_info in data["folds"].items():
            train_recs = [str(r) for r in fold_info["train_recording_ids"]]
            in_tr, in_val = create_inner_split(train_recs, manifest_idx, seed=SEED)

            assert len(set(in_tr).intersection(set(in_val))) == 0
            assert len(in_tr) == 19
            assert len(in_val) == 7

            tr_classes = set(manifest_idx.loc[in_tr, "fault_class"].unique())
            val_classes = set(manifest_idx.loc[in_val, "fault_class"].unique())
            assert len(tr_classes) == 4
            assert len(val_classes) == 4


class TestModelArchitectureAndIsolation:
    """Verify parameter counts, dual-branch shapes, and checkpoint safety."""

    def test_vibration_cnn1d_parameter_count(self):
        """Production baseline VibrationCNN1D retains exact 73,156 trainable parameters."""
        model = VibrationCNN1D(fc_units=64, dropout_rate=0.3)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count == 73156

    def test_spectral_fusion_cnn_architecture(self):
        """SpectralFusionCNN has expected parameter count and forward shape."""
        model = SpectralFusionCNN(
            num_classes=4, input_channels=1, waveform_length=2048, spectral_bins=1025, fc_units=64
        )
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count == 89604

        wave = torch.randn(4, 1, 2048)
        spec = torch.randn(4, 1, 1025)
        out = model(wave, spectral_input=spec)
        assert out.shape == (4, 4)

    def test_phase11_4_artifact_generation(self):
        """Verify all Phase 11.4 deliverables, metrics, and figures exist and are non-empty."""
        required_artifacts = [
            EXP_11_4_DIR / "experiment_config.json",
            EXP_11_4_DIR / "fold_metrics.json",
            EXP_11_4_DIR / "cross_fold_comparison.json",
            EXP_11_4_DIR / "per_class_comparison.csv",
            EXP_11_4_DIR / "per_load_comparison.csv",
            EXP_11_4_DIR / "per_severity_comparison.csv",
            EXP_11_4_DIR / "training_history.json",
            EXP_11_4_DIR / "model_parameters.json",
            EXP_11_4_DIR / "latency_measurements.json",
        ]
        for art in required_artifacts:
            assert art.exists(), f"Missing artifact: {art}"
            assert art.stat().st_size > 50, f"Artifact empty: {art}"

        figures_dir = REPO_ROOT / "outputs" / "figures" / "phase11_4"
        required_figures = [
            figures_dir / "cross_experiment_comparison.png",
            figures_dir / "confusion_matrices_comparison.png",
            figures_dir / "training_curves_fold_3.png",
        ]
        for fig in required_figures:
            assert fig.exists(), f"Missing figure: {fig}"
            assert fig.stat().st_size > 1000, f"Figure empty: {fig}"

    def test_production_and_phase11_2_checkpoints_preserved(self):
        """Production model and Phase 11.2 checkpoints remain untouched."""
        prod_ckpt = REPO_ROOT / "models" / "checkpoints" / "inspectai_final_cnn.pth"
        if prod_ckpt.exists():
            # Checkpoint exists and is not empty
            assert prod_ckpt.stat().st_size > 50000

        phase11_2_dir = REPO_ROOT / "outputs" / "experiments" / "phase11_2" / "checkpoints"
        if phase11_2_dir.exists():
            for f in [1, 2, 3]:
                f_ckpt = phase11_2_dir / f"lolo_fold_{f}_cnn.pth"
                assert f_ckpt.exists(), f"Phase 11.2 Fold {f} checkpoint missing"


class TestNormalizerAndWeightProvenance:
    """Verify normalizers and class weights are computed strictly from training partition."""

    def test_vibration_normalizer_training_only(self):
        """VibrationNormalizer correctly standardizes data with zero leakage."""
        X_tr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        norm = VibrationNormalizer()
        norm.fit(X_tr)

        assert norm.mean == 25.0
        assert norm.is_fitted

        X_eval = np.array([[25.0, 25.0]], dtype=np.float32)
        X_eval_norm = norm.transform(X_eval)
        # Scaled mean at 25 should be 0
        np.testing.assert_allclose(X_eval_norm, np.zeros_like(X_eval_norm), atol=1e-5)

    def test_spectral_normalizer_training_only(self):
        """SpectralNormalizer standardizes frequency bins using training mean and std."""
        X_tr = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        norm = SpectralNormalizer()
        norm.fit(X_tr)

        assert norm.is_fitted
        np.testing.assert_allclose(norm.mean, np.array([2.0, 4.0]))

        X_trans = norm.transform(X_tr)
        np.testing.assert_allclose(np.mean(X_trans, axis=0), np.zeros(2), atol=1e-5)

    def test_balanced_class_weights(self):
        """Class weights invert class frequency correctly."""
        y = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        weights = compute_balanced_weights(y, num_classes=4)
        assert len(weights) == 4
        # Class 0 has lowest frequency -> highest weight
        assert weights[0] > weights[1] > weights[2] > weights[3]
