"""Unit and validation tests for SignalCleaner (src/preprocessing/signal_cleaner.py)."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import scipy.signal

from src.ingestion.loader import VibrationDataLoader
from src.preprocessing.signal_cleaner import ProcessedSignal, SignalCleaner
from src.utils.config_parser import load_config, get_resolved_paths


def test_resampling_48k_to_12k():
    """Verify 48 kHz to 12 kHz downsampling using anti-aliasing polyphase filtering."""
    orig_fs = 48000
    target_fs = 12000
    duration = 1.0  # 1 second
    n_samples = int(duration * orig_fs)

    t = np.linspace(0, duration, n_samples, endpoint=False)
    # 200 Hz tone (well within 12 kHz Nyquist of 6000 Hz)
    sig_48k = np.sin(2 * np.pi * 200 * t)

    resampled, applied = SignalCleaner.normalize_sampling_rate(sig_48k, orig_fs=orig_fs, target_fs=target_fs)

    assert applied is True
    # Exactly 1/4 of the samples
    assert len(resampled) == int(duration * target_fs)
    # Check that 200 Hz tone power is preserved without attenuation
    assert np.isclose(np.std(resampled), np.std(sig_48k), rtol=0.05)


def test_resampling_already_at_target_is_bypassed():
    """Verify that recordings already at 12 kHz are not resampled."""
    fs = 12000
    sig_12k = np.random.randn(6000)

    output, applied = SignalCleaner.normalize_sampling_rate(sig_12k, orig_fs=fs, target_fs=fs)

    assert applied is False
    assert len(output) == len(sig_12k)
    assert np.array_equal(output, sig_12k)


def test_dc_offset_removal():
    """Verify that DC bias is completely subtracted while dynamic oscillations are preserved."""
    dc_bias = 4.75
    oscillations = np.sin(np.linspace(0, 10 * np.pi, 2000))
    raw_signal = dc_bias + oscillations

    cleaned = SignalCleaner.remove_dc_offset(raw_signal)

    assert np.isclose(np.mean(cleaned), 0.0, atol=1e-12)
    assert np.isclose(np.std(cleaned), np.std(oscillations), atol=1e-12)
    assert np.isclose(np.max(cleaned) - np.min(cleaned), np.max(oscillations) - np.min(oscillations), atol=1e-12)


def test_butterworth_bandpass_filter_behavior():
    """Verify attenuation of sub-synchronous low drift and high-frequency noise."""
    fs = 12000
    t = np.linspace(0, 1.0, fs, endpoint=False)

    f_sub = 5.0      # Out of band: 5 Hz (should be attenuated by 20 Hz low cutoff)
    f_pass = 250.0   # In band: 250 Hz (should be preserved)
    f_noise = 5800.0 # Out of band: 5800 Hz (should be attenuated by 5000 Hz high cutoff)

    sig_inband = 1.0 * np.sin(2 * np.pi * f_pass * t)
    sig_low = 2.0 * np.sin(2 * np.pi * f_sub * t)
    sig_high = 2.0 * np.sin(2 * np.pi * f_noise * t)

    composite = sig_inband + sig_low + sig_high

    filtered = SignalCleaner.apply_filter(
        composite, fs=fs, filter_type="bandpass", low_cutoff=20.0, high_cutoff=5000.0, order=4
    )

    # In-band tone power must be preserved (relative error < 5%)
    # Compute FFT to inspect spectral bin amplitudes
    fft_comp = np.abs(np.fft.rfft(composite))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(fs, 1.0 / fs)

    idx_pass = np.argmin(np.abs(freqs - f_pass))
    idx_low = np.argmin(np.abs(freqs - f_sub))
    idx_high = np.argmin(np.abs(freqs - f_noise))

    # Passband energy maintained
    assert np.isclose(fft_filt[idx_pass], fft_comp[idx_pass], rtol=0.05)
    # Out of band low & high severely attenuated (at least 20 dB / factor of 10)
    assert fft_filt[idx_low] < 0.1 * fft_comp[idx_low]
    assert fft_filt[idx_high] < 0.1 * fft_comp[idx_high]


def test_invalid_input_validation():
    """Verify strict rejection of invalid array shapes, types, and empty inputs."""
    # Non-numeric string data
    with pytest.raises(TypeError, match="non-numeric"):
        SignalCleaner.validate_signal(np.array(["a", "b", "c"]))

    # None input
    with pytest.raises(ValueError, match="cannot be None"):
        SignalCleaner.validate_signal(None)

    # Empty array
    with pytest.raises(ValueError, match="is empty"):
        SignalCleaner.validate_signal(np.array([]))

    # 3D tensor
    with pytest.raises(ValueError, match="invalid dimensions"):
        SignalCleaner.validate_signal(np.ones((2, 2, 2)))

    # 2D matrix with multiple columns and rows
    with pytest.raises(ValueError, match="is a 2D matrix"):
        SignalCleaner.validate_signal(np.ones((10, 10)))

    # Invalid sampling rates
    with pytest.raises(ValueError, match="Original sampling rate must be positive"):
        SignalCleaner.normalize_sampling_rate(np.ones(100), orig_fs=-12000, target_fs=12000)

    with pytest.raises(ValueError, match="Target sampling rate must be positive"):
        SignalCleaner.normalize_sampling_rate(np.ones(100), orig_fs=12000, target_fs=0)


def test_nan_and_inf_strict_rejection():
    """Verify strict rejection of NaN and Inf without silent replacement."""
    nan_arr = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="contains corrupt values.*1 NaNs"):
        SignalCleaner.validate_signal(nan_arr)

    inf_arr = np.array([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(ValueError, match="contains corrupt values.*1 Infs"):
        SignalCleaner.validate_signal(inf_arr)


def test_raw_dataset_files_remain_untouched():
    """Verify that preprocessing never modifies the original raw .mat files."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"
    manifest_path = raw_dir / "download_manifest.json"

    assert manifest_path.is_file(), "Manifest file missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Check that SHA-256 digests of all 13 raw files match download_manifest.json exactly
    for rec in records:
        fpath = raw_dir / rec["filename"]
        assert fpath.is_file(), f"Raw file missing: {rec['filename']}"

        hasher = hashlib.sha256()
        with open(fpath, "rb") as f_in:
            for chunk in iter(lambda: f_in.read(65536), b""):
                hasher.update(chunk)

        current_sha256 = hasher.hexdigest()
        assert current_sha256 == rec["sha256"], (
            f"Integrity violation: {rec['filename']} was modified! Expected {rec['sha256']}, got {current_sha256}"
        )


def test_clean_signal_end_to_end_on_cwru_recordings():
    """Verify full clean_signal pipeline on authentic CWRU recordings."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"
    cleaner = SignalCleaner(target_sampling_rate=12000, remove_dc=True, filter_enabled=True)
    loader = VibrationDataLoader(raw_dir)

    # 1. Normal 48 kHz file (97.mat)
    rec_97 = loader.load_file("97.mat")
    proc_97 = cleaner.clean_signal(rec_97.channels["DE_time"], orig_fs=48000, metadata={"name": "97.mat"})

    assert isinstance(proc_97, ProcessedSignal)
    assert proc_97.applied_resampling is True
    assert proc_97.applied_dc_removal is True
    assert proc_97.applied_filtering is True
    assert proc_97.target_fs == 12000
    # Length reduced to 1/4
    assert np.isclose(len(proc_97.signal), len(rec_97.channels["DE_time"]) / 4, atol=2)
    assert np.isclose(np.mean(proc_97.signal), 0.0, atol=1e-3)

    # 2. Fault 12 kHz file (105.mat)
    rec_105 = loader.load_file("105.mat")
    proc_105 = cleaner.clean_signal(rec_105.channels["DE_time"], orig_fs=12000, metadata={"name": "105.mat"})

    assert proc_105.applied_resampling is False  # Already 12 kHz, no resampling
    assert proc_105.sample_count == len(rec_105.channels["DE_time"])  # Unchanged length
    assert proc_105.applied_dc_removal is True
    assert proc_105.applied_filtering is True
    assert np.isclose(np.mean(proc_105.signal), 0.0, atol=1e-3)

