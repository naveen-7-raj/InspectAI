"""Unit tests for frequency domain feature extraction and spectral normalization.

Tests verify:
1. FFT magnitude spectrum computation (shapes, scale, non-negativity).
2. Hilbert envelope spectrum computation.
3. Bearing characteristic frequency calculation.
4. SpectralNormalizer fitting, transforming, tensor support, and serialization.
5. Invariance, reproducibility, and edge case handling (zeros, constants).
"""

import numpy as np
import pytest
import torch

from src.features.frequency_features import (
    compute_fft_magnitude_spectrum,
    compute_hilbert_envelope_spectrum,
    get_bearing_characteristic_frequencies,
    SpectralNormalizer,
)


class TestFFTMagnitudeSpectrum:
    """Tests for compute_fft_magnitude_spectrum."""

    def test_single_window_shape_and_range(self):
        """Single 1D window of length 2048 produces 1025 non-negative frequency bins."""
        window = np.random.randn(2048).astype(np.float32)
        freqs, spectrum = compute_fft_magnitude_spectrum(window, sample_rate=12000)

        assert freqs.shape == (1025,)
        assert spectrum.shape == (1025,)
        assert spectrum.dtype == np.float32
        assert np.all(spectrum >= 0.0)
        assert not np.isnan(spectrum).any()
        assert not np.isinf(spectrum).any()

    def test_batch_2d_shape(self):
        """Batch of 2D windows (B, 2048) produces (B, 1025) spectra."""
        batch = np.random.randn(16, 2048).astype(np.float32)
        freqs, spectra = compute_fft_magnitude_spectrum(batch, sample_rate=12000)

        assert freqs.shape == (1025,)
        assert spectra.shape == (16, 1025)
        assert spectra.dtype == np.float32
        assert np.all(spectra >= 0.0)

    def test_single_tone_peak_identification(self):
        """A pure sinusoidal tone at 600 Hz should produce an energy peak at ~600 Hz."""
        fs = 12000
        n_samples = 2048
        t = np.arange(n_samples) / fs
        f_tone = 600.0  # Exactly 600 Hz
        window = np.sin(2 * np.pi * f_tone * t).astype(np.float32)

        freqs, spectrum = compute_fft_magnitude_spectrum(window, sample_rate=fs)

        peak_idx = np.argmax(spectrum[1:]) + 1  # exclude DC
        detected_freq = freqs[peak_idx]

        assert abs(detected_freq - f_tone) < (fs / n_samples)  # within frequency resolution (~5.86 Hz)

    def test_constant_signal_has_zero_ac_energy(self):
        """A DC constant signal should have energy at bin 0 and near zero elsewhere."""
        window = np.full(2048, 5.0, dtype=np.float32)
        freqs, spectrum = compute_fft_magnitude_spectrum(window, sample_rate=12000)

        # DC bin should be positive (scaled by 2/N)
        assert spectrum[0] > 0.0
        # All AC bins (index 1 onwards) should be zero
        assert np.allclose(spectrum[1:], 0.0, atol=1e-5)


class TestHilbertEnvelopeSpectrum:
    """Tests for compute_hilbert_envelope_spectrum."""

    def test_single_window_shape_and_range(self):
        """Single 1D window of length 2048 produces 1025 bins for envelope spectrum."""
        window = np.random.randn(2048).astype(np.float32)
        freqs, envelope_spec, env_time = compute_hilbert_envelope_spectrum(window, sample_rate=12000)

        assert freqs.shape == (1025,)
        assert envelope_spec.shape == (1025,)
        assert env_time.shape == (2048,)
        assert envelope_spec.dtype == np.float32
        assert np.all(envelope_spec >= 0.0)
        assert not np.isnan(envelope_spec).any()

    def test_batch_2d_shape(self):
        """Batch of 2D windows produces (B, 1025) envelope spectra."""
        batch = np.random.randn(8, 2048).astype(np.float32)
        freqs, envelope_spec, env_time = compute_hilbert_envelope_spectrum(batch, sample_rate=12000)

        assert freqs.shape == (1025,)
        assert envelope_spec.shape == (8, 1025)
        assert env_time.shape == (8, 2048)
        assert envelope_spec.dtype == np.float32
        assert np.all(envelope_spec >= 0.0)

    def test_amplitude_modulated_signal_envelope_detection(self):
        """An amplitude-modulated signal s(t) = [1 + m*cos(2*pi*f_mod*t)] * sin(2*pi*f_carrier*t)
        should have its modulation frequency extracted by Hilbert envelope analysis.
        """
        fs = 12000
        n_samples = 2048
        t = np.arange(n_samples) / fs
        f_carrier = 2500.0
        f_mod = 120.0  # representative of bearing impact rate

        carrier = np.sin(2 * np.pi * f_carrier * t)
        modulator = 1.0 + 0.8 * np.cos(2 * np.pi * f_mod * t)
        sig = (modulator * carrier).astype(np.float32)

        freqs, envelope_spec, _ = compute_hilbert_envelope_spectrum(sig, sample_rate=fs)

        # Peak in envelope spectrum should correspond to f_mod (120 Hz)
        peak_idx = np.argmax(envelope_spec[1:]) + 1  # exclude DC
        detected_mod_freq = freqs[peak_idx]

        assert abs(detected_mod_freq - f_mod) < 15.0  # within ~2-3 bins of 120 Hz


class TestBearingCharacteristicFrequencies:
    """Tests for theoretical bearing fault frequency calculation."""

    def test_skf6205_frequencies(self):
        """Verify kinematic frequency formulas for SKF 6205 bearing at ~1797 RPM (~29.95 Hz)."""
        freqs = get_bearing_characteristic_frequencies(shaft_rpm=1797.0)

        assert pytest.approx(freqs["fr"], rel=1e-2) == 29.95
        # For SKF 6205 at 1797 RPM: BPFO ~ 107.4 Hz, BPFI ~ 162.2 Hz, BSF ~ 70.6 Hz, FTF ~ 11.9 Hz
        assert pytest.approx(freqs["bpfo"], rel=1e-2) == 107.36
        assert pytest.approx(freqs["bpfi"], rel=1e-2) == 162.19
        assert pytest.approx(freqs["bsf"], rel=1e-2) == 70.585
        assert pytest.approx(freqs["ftf"], rel=1e-2) == 11.93


class TestSpectralNormalizer:
    """Tests for SpectralNormalizer training-fold-only standardizer."""

    def test_fit_and_transform_numpy(self):
        """Normalizer centers data to mean ~0 and unit variance."""
        rng = np.random.default_rng(42)
        # Synthetic spectra: 100 samples, 1025 bins
        train_spectra = (rng.normal(loc=5.0, scale=2.0, size=(100, 1025))).astype(np.float32)

        normalizer = SpectralNormalizer()
        normalizer.fit(train_spectra)

        assert normalizer.is_fitted
        assert normalizer.mean.shape == (1025,)
        assert normalizer.std.shape == (1025,)

        norm_spectra = normalizer.transform(train_spectra)
        assert norm_spectra.shape == (100, 1025)
        # Per-bin mean across samples should be close to 0, std close to 1
        assert np.allclose(norm_spectra.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(norm_spectra.std(axis=0), 1.0, atol=1e-4)

    def test_transform_torch_tensor(self):
        """Normalizer accepts and transforms PyTorch tensors seamlessly."""
        rng = np.random.default_rng(42)
        train_spectra = rng.normal(loc=10.0, scale=3.0, size=(50, 1025)).astype(np.float32)

        normalizer = SpectralNormalizer()
        normalizer.fit(train_spectra)

        tensor_in = torch.from_numpy(train_spectra)
        tensor_out = normalizer.transform(tensor_in)

        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape == (50, 1025)
        assert torch.allclose(tensor_out.mean(dim=0), torch.zeros(1025), atol=1e-3)

    def test_zero_variance_guard(self):
        """Constant spectral bins (std=0) should not produce division by zero or NaN."""
        train_spectra = np.zeros((20, 1025), dtype=np.float32)
        train_spectra[:, 10] = 5.0  # Constant column

        normalizer = SpectralNormalizer(epsilon=1e-8)
        normalizer.fit(train_spectra)

        norm_spectra = normalizer.transform(train_spectra)
        assert not np.isnan(norm_spectra).any()
        assert not np.isinf(norm_spectra).any()
        assert np.allclose(norm_spectra, 0.0)

    def test_serialization(self, tmp_path):
        """Normalizer serializes and deserializes accurately to JSON."""
        train_spectra = np.random.randn(30, 1025).astype(np.float32)
        normalizer = SpectralNormalizer()
        normalizer.fit(train_spectra)

        json_file = tmp_path / "spectral_normalizer.json"
        normalizer.save(json_file)

        loaded = SpectralNormalizer.load(json_file)
        assert loaded.is_fitted
        assert loaded.n_features == 1025
        assert np.allclose(loaded.mean, normalizer.mean)
        assert np.allclose(loaded.std, normalizer.std)

    def test_raises_when_not_fitted(self):
        """Calling transform on unfitted normalizer raises RuntimeError."""
        normalizer = SpectralNormalizer()
        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            normalizer.transform(np.zeros((5, 1025), dtype=np.float32))
