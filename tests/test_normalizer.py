"""Unit tests for VibrationNormalizer (Phase 4, Task 4.1)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.training.normalizer import VibrationNormalizer


class TestVibrationNormalizer:
    """Test suite verifying training-only normalization behavior and safety constraints."""

    def test_fit_and_properties(self):
        """Verify parameter calculation on synthetic 2D array."""
        normalizer = VibrationNormalizer(epsilon=1e-8)
        assert not normalizer.is_fitted
        assert normalizer.mean is None
        assert normalizer.std is None

        # Data with known mean and std: mean=10.0, std=2.0
        data = np.array([[8.0, 10.0, 12.0], [10.0, 10.0, 10.0]], dtype=np.float32)
        normalizer.fit(data, source_recording_ids=["rec_1", "rec_2"])

        assert normalizer.is_fitted
        assert np.isclose(normalizer.mean, 10.0)
        assert np.isclose(normalizer.std, np.std(data))
        assert normalizer.fit_sample_count == 2
        assert normalizer.fit_recording_ids == ["rec_1", "rec_2"]

    def test_transform_and_inverse_transform(self):
        """Verify normalization z-score standardization and inverse reconstruction."""
        normalizer = VibrationNormalizer(epsilon=1e-8)
        raw_data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        normalizer.fit(raw_data)

        normalized = normalizer.transform(raw_data)
        assert normalized.dtype == np.float32
        assert normalized.shape == raw_data.shape
        # Global mean of normalized array must be ~0 and std must be ~1
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-5)

        # Inverse transform should reconstruct original physical values
        reconstructed = normalizer.inverse_transform(normalized)
        assert np.allclose(reconstructed, raw_data, atol=1e-5)

    def test_transform_tensor(self):
        """Verify PyTorch tensor transform produces expected shape, type, and values."""
        normalizer = VibrationNormalizer(epsilon=1e-8)
        train_data = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        normalizer.fit(train_data)

        tensor_in = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        tensor_out = normalizer.transform_tensor(tensor_in)

        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.dtype == torch.float32
        assert tensor_out.shape == tensor_in.shape
        assert np.allclose(tensor_out.numpy(), normalizer.transform(tensor_in.numpy()), atol=1e-6)

    def test_unfitted_raises_runtime_error(self):
        """Verify error is raised if transform or inverse_transform is called before fit."""
        normalizer = VibrationNormalizer()
        dummy = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="must be fitted before transforming"):
            normalizer.transform(dummy)

        with pytest.raises(RuntimeError, match="must be fitted before transforming tensor"):
            normalizer.transform_tensor(torch.tensor([1.0, 2.0]))

        with pytest.raises(RuntimeError, match="must be fitted before inverse transform"):
            normalizer.inverse_transform(dummy)

        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            normalizer.save("dummy_path.json")

    def test_zero_variance_epsilon_protection(self):
        """Verify division by zero does not produce NaN/Inf when variance is zero."""
        normalizer = VibrationNormalizer(epsilon=1e-8)
        constant_data = np.ones((5, 10), dtype=np.float32) * 42.0
        normalizer.fit(constant_data)

        assert normalizer.std == 0.0
        normalized = normalizer.transform(constant_data)

        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        assert np.all(normalized == 0.0)

    def test_invalid_data_rejection(self):
        """Verify corrupt values (NaN, Inf, empty, non-numeric) are rejected."""
        normalizer = VibrationNormalizer()

        # None input
        with pytest.raises(ValueError, match="cannot be None"):
            normalizer.fit(None)

        # Empty array
        with pytest.raises(ValueError, match="is empty"):
            normalizer.fit(np.array([]))

        # NaN values
        with pytest.raises(ValueError, match="contains corrupt values"):
            normalizer.fit(np.array([1.0, np.nan, 3.0]))

        # Inf values
        with pytest.raises(ValueError, match="contains corrupt values"):
            normalizer.fit(np.array([1.0, np.inf, 3.0]))

        # Non-numeric array
        with pytest.raises(TypeError, match="non-numeric"):
            normalizer.fit(np.array(["a", "b", "c"]))

    def test_save_and_load(self, tmp_path):
        """Verify serialization and round-trip parameter loading."""
        normalizer = VibrationNormalizer(epsilon=1e-7)
        data = np.random.RandomState(42).randn(10, 50).astype(np.float32)
        normalizer.fit(data, source_recording_ids=["rec_A", "rec_B"])

        save_file = tmp_path / "normalizer.json"
        saved_path = normalizer.save(save_file)
        assert saved_path.is_file()

        loaded_norm = VibrationNormalizer.load(save_file)
        assert loaded_norm.is_fitted
        assert np.isclose(loaded_norm.mean, normalizer.mean)
        assert np.isclose(loaded_norm.std, normalizer.std)
        assert loaded_norm.epsilon == normalizer.epsilon
        assert loaded_norm.fit_sample_count == normalizer.fit_sample_count
        assert loaded_norm.fit_recording_ids == normalizer.fit_recording_ids

        # Transformed values must match
        assert np.allclose(normalizer.transform(data), loaded_norm.transform(data))

