"""Unit tests for production DomainAdapter (CORAL) in src/inference/domain_adapter.py.

Verifies:
1. Initialization and source reference statistic calculations.
2. Calibration sample buffering and progress tracking.
3. Successful fit with sufficient samples (N >= min_samples).
4. Graceful fallback when calibration samples are insufficient.
5. Signal validation (flatline/dead sensor, NaN, invalid length).
6. Tikhonov regularization and stability under ill-conditioned/collinear features.
7. Clean reset and recalibration lifecycle.
8. Output telemetry schema and status flags.
9. Strictly label-free guarantees (zero target label inputs).
"""

import numpy as np
import pytest

from src.inference.domain_adapter import AdapterStatus, DomainAdapter


class TestDomainAdapterLifecycle:
    """Verify core DomainAdapter lifecycle methods."""

    @pytest.fixture
    def source_features(self):
        np.random.seed(42)
        n, d = 200, 32
        L = np.random.randn(d, d)
        C = L @ L.T
        return np.random.multivariate_normal(np.ones(d) * 2.0, C, size=n)

    def test_initialization_uninitialized(self):
        adapter = DomainAdapter(min_calibration_samples=50)
        assert adapter.status == AdapterStatus.UNINITIALIZED
        assert adapter.is_fitted is False
        assert adapter.buffer_count == 0
        assert adapter.calibration_progress == 0.0

    def test_set_source_statistics(self, source_features):
        adapter = DomainAdapter(min_calibration_samples=50)
        adapter.set_source_statistics(source_features)
        assert adapter.status == AdapterStatus.CALIBRATING
        assert adapter.source_mean is not None
        assert adapter.source_cov_sqrt is not None
        assert len(adapter.source_mean) == 32
        assert adapter.source_cov_sqrt.shape == (32, 32)

    def test_collect_calibration_samples(self, source_features):
        adapter = DomainAdapter(min_calibration_samples=10)
        adapter.set_source_statistics(source_features)

        for i in range(5):
            res = adapter.collect_calibration_sample(np.random.randn(32))
            assert res["collected_samples"] == i + 1
            assert res["is_ready_to_fit"] is False
            assert res["progress_ratio"] == (i + 1) / 10.0

        batch_res = adapter.collect_calibration_batch(np.random.randn(5, 32))
        assert batch_res["collected_samples"] == 10
        assert batch_res["is_ready_to_fit"] is True
        assert batch_res["progress_ratio"] == 1.0

    def test_fit_success_with_sufficient_samples(self, source_features):
        adapter = DomainAdapter(min_calibration_samples=20)
        adapter.set_source_statistics(source_features)

        target_data = np.random.randn(25, 32) * 2.0 - 1.0
        adapter.collect_calibration_batch(target_data)
        success = adapter.fit()

        assert success is True
        assert adapter.is_fitted is True
        assert adapter.status == AdapterStatus.READY
        assert adapter.A_coral is not None
        assert adapter.target_mean is not None
        assert adapter.condition_number is not None
        assert adapter.error_message is None

    def test_fit_insufficient_samples_triggers_fallback(self, source_features):
        adapter = DomainAdapter(min_calibration_samples=50)
        adapter.set_source_statistics(source_features)

        # Buffer only 15 samples (< 50)
        adapter.collect_calibration_batch(np.random.randn(15, 32))
        success = adapter.fit()

        assert success is False
        assert adapter.is_fitted is False
        assert adapter.status == AdapterStatus.FALLBACK
        assert "Insufficient calibration samples" in adapter.error_message

    def test_reset_and_recalibrate(self, source_features):
        adapter = DomainAdapter(min_calibration_samples=10)
        adapter.set_source_statistics(source_features)
        adapter.collect_calibration_batch(np.random.randn(15, 32))
        adapter.fit()
        assert adapter.is_fitted is True

        adapter.reset()
        assert adapter.is_fitted is False
        assert adapter.buffer_count == 0
        assert adapter.status == AdapterStatus.CALIBRATING

        adapter.collect_calibration_batch(np.random.randn(15, 32))
        success = adapter.fit()
        assert success is True
        assert adapter.is_fitted is True


class TestDomainAdapterSafeguards:
    """Verify numerical safeguards, validation, and label isolation."""

    @pytest.fixture
    def adapter_with_source(self):
        np.random.seed(123)
        X_s = np.random.randn(100, 16)
        adapter = DomainAdapter(min_calibration_samples=20, regularization=1e-3)
        adapter.set_source_statistics(X_s)
        return adapter

    def test_validate_window_signal_flatline_detection(self, adapter_with_source):
        flat_signal = np.zeros(2048, dtype=np.float32)
        valid, msg = adapter_with_source.validate_window_signal(flat_signal)
        assert valid is False
        assert "Dead/constant sensor signal" in msg

    def test_validate_window_signal_nan_detection(self, adapter_with_source):
        nan_signal = np.ones(2048, dtype=np.float32)
        nan_signal[50] = np.nan
        valid, msg = adapter_with_source.validate_window_signal(nan_signal)
        assert valid is False
        assert "NaN or infinite" in msg

    def test_validate_window_signal_invalid_length(self, adapter_with_source):
        short_signal = np.ones(1024, dtype=np.float32)
        valid, msg = adapter_with_source.validate_window_signal(short_signal)
        assert valid is False
        assert "Invalid window length" in msg

    def test_collinear_ill_conditioned_stability(self, adapter_with_source):
        # Collinear rank-1 features
        collinear = np.tile(np.random.randn(50, 1), (1, 16))
        success = adapter_with_source.fit(collinear)
        assert success is True
        assert adapter_with_source.is_fitted is True
        assert np.all(np.isfinite(adapter_with_source.A_coral))

    def test_label_free_signature_guarantee(self):
        import inspect
        sig_fit = inspect.signature(DomainAdapter.fit)
        sig_trans = inspect.signature(DomainAdapter.transform)
        sig_collect = inspect.signature(DomainAdapter.collect_calibration_sample)

        assert "label" not in sig_fit.parameters
        assert "target_labels" not in sig_fit.parameters
        assert "y" not in sig_trans.parameters
        assert "label" not in sig_collect.parameters

    def test_telemetry_schema(self, adapter_with_source):
        telem = adapter_with_source.get_telemetry()
        expected_keys = [
            "status",
            "is_fitted",
            "buffered_samples",
            "required_samples",
            "progress_percent",
            "condition_number",
            "regularization",
            "error_message",
        ]
        for k in expected_keys:
            assert k in telem
