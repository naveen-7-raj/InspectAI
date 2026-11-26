"""External signal preprocessing and windowing pipeline for zero-shot evaluation.

Ensures that external datasets (such as Paderborn University and MFPT) are strictly
conditioned to match InspectAI's 12 kHz, 2,048-sample dual-branch input contract:
1. Physical unit conversion (e.g. m/s^2 -> g).
2. Anti-aliased rational polyphase resampling to 12,000 Hz.
3. DC offset removal and 4th-order zero-phase Butterworth bandpass filtering (20 - 5000 Hz).
4. Window segmentation (2048 samples, 512 step, 75% overlap).
5. Dynamic Hilbert transform envelope spectrum extraction (1025 bins).
6. Standardization using frozen CWRU normalizers.
"""

import logging
from math import gcd
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import torch

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.ingestion.external_loader import ExternalRecording
from src.training.normalizer import VibrationNormalizer

logger = logging.getLogger("InspectAI.ExternalPreprocessor")

TARGET_FS = 12000
WINDOW_SIZE = 2048
STEP_SIZE = 512
LOW_CUTOFF = 20.0
HIGH_CUTOFF = 5000.0
G_ACCEL = 9.80665


class ExternalPreprocessor:
    """Preprocesses external vibration signals for zero-shot evaluation on InspectAI."""

    def __init__(
        self,
        target_fs: int = TARGET_FS,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        low_cutoff_hz: float = LOW_CUTOFF,
        high_cutoff_hz: float = HIGH_CUTOFF,
        filter_order: int = 4,
    ):
        """Initialize preprocessor parameters."""
        self.target_fs = target_fs
        self.window_size = window_size
        self.step_size = step_size
        self.low_cutoff_hz = low_cutoff_hz
        self.high_cutoff_hz = high_cutoff_hz
        self.filter_order = filter_order

    def resample_to_target(
        self, signal_arr: np.ndarray, orig_fs: float
    ) -> np.ndarray:
        """Resample signal to target_fs using anti-aliased polyphase rational resampling."""
        if abs(orig_fs - self.target_fs) < 1.0:
            return signal_arr.astype(np.float32)

        # Compute integer rational approximation up / down
        orig_int = int(round(orig_fs))
        common_divisor = gcd(self.target_fs, orig_int)
        up = self.target_fs // common_divisor
        down = orig_int // common_divisor

        logger.debug(f"Resampling from {orig_fs} Hz to {self.target_fs} Hz (up={up}, down={down})")
        resampled = signal.resample_poly(signal_arr, up, down)
        return resampled.astype(np.float32)

    def condition_continuous_signal(
        self, signal_arr: np.ndarray, orig_fs: float, units: str = "g"
    ) -> np.ndarray:
        """Apply unit conversion, resampling, DC removal, and bandpass filtering."""
        arr = np.asarray(signal_arr, dtype=np.float32)

        # 1. Unit conversion to gravitational g
        if units.lower() in ("m/s^2", "m/s2", "ms-2"):
            arr = arr / G_ACCEL

        # 2. Resample to target_fs
        resampled = self.resample_to_target(arr, orig_fs)

        # 3. DC bias subtraction
        resampled = resampled - float(np.mean(resampled))

        # 4. Zero-phase Butterworth bandpass filter
        nyquist = self.target_fs / 2.0
        low = self.low_cutoff_hz / nyquist
        high = min(self.high_cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(self.filter_order, [low, high], btype="bandpass")
        filtered = signal.filtfilt(b, a, resampled)

        return filtered.astype(np.float32)

    def segment_windows(
        self, conditioned_signal: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Slice conditioned signal into sliding windows with 75% overlap."""
        n_samples = len(conditioned_signal)
        if n_samples < self.window_size:
            raise ValueError(
                f"Signal length ({n_samples}) is shorter than required window size ({self.window_size})"
            )

        windows = []
        metadata = []
        start_idx = 0
        w_idx = 0

        while start_idx + self.window_size <= n_samples:
            w = conditioned_signal[start_idx : start_idx + self.window_size]
            windows.append(w)
            metadata.append(
                {
                    "window_index": w_idx,
                    "start_sample": start_idx,
                    "end_sample": start_idx + self.window_size,
                    "start_time_sec": start_idx / float(self.target_fs),
                    "end_time_sec": (start_idx + self.window_size) / float(self.target_fs),
                }
            )
            start_idx += self.step_size
            w_idx += 1

        return np.stack(windows, axis=0), metadata

    def process_recording(
        self,
        recording: ExternalRecording,
        wave_normalizer: Optional[VibrationNormalizer] = None,
        spec_normalizer: Optional[SpectralNormalizer] = None,
    ) -> Dict[str, Any]:
        """Complete pipeline from ExternalRecording to model-ready standardized tensors."""
        # 1. Condition continuous signal
        conditioned = self.condition_continuous_signal(
            recording.signal, recording.original_sampling_rate, units=recording.units
        )

        # 2. Extract sliding windows (N, 2048)
        windows, win_meta = self.segment_windows(conditioned)

        # 3. Extract Hilbert envelope spectrum (N, 1025)
        _, env_spectra, _ = compute_hilbert_envelope_spectrum(
            windows, sampling_rate_hz=self.target_fs
        )

        # 4. Standardize using frozen normalizers if provided
        windows_norm = windows.copy()
        spectra_norm = env_spectra.copy()

        if wave_normalizer is not None:
            windows_norm = wave_normalizer.transform(windows_norm)

        if spec_normalizer is not None:
            spectra_norm = spec_normalizer.transform(spectra_norm)

        # 5. Format into PyTorch tensors
        tensor_wave = torch.from_numpy(windows_norm).unsqueeze(1).float()  # (N, 1, 2048)
        tensor_spec = torch.from_numpy(spectra_norm).unsqueeze(1).float()  # (N, 1, 1025)

        return {
            "recording_id": recording.recording_id,
            "dataset_name": recording.dataset_name,
            "fault_class": recording.fault_class,
            "bearing_model": recording.bearing_model,
            "rotational_speed_rpm": recording.rotational_speed_rpm,
            "total_windows": len(windows),
            "raw_windows": windows,
            "raw_spectra": env_spectra,
            "tensor_wave": tensor_wave,
            "tensor_spec": tensor_spec,
            "window_metadata": win_meta,
        }
