"""Frequency-domain and Hilbert envelope spectrum feature extraction for vibration signals.

Provides:
- FFT Magnitude Spectrum (one-sided rfft).
- Hilbert Transform Envelope Spectrum (analytic signal magnitude -> zero-mean detrending -> one-sided rfft).
- Training-fold-only SpectralNormalizer with zero data leakage.
- Kinematic bearing frequency reference calculations based on physical geometry.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal
import torch

logger = logging.getLogger("InspectAI.FrequencyFeatures")


def compute_fft_magnitude_spectrum(
    signal: Union[np.ndarray, torch.Tensor],
    sampling_rate_hz: int = 12000,
    n_fft: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT magnitude spectrum for 1D or 2D signals.

    Args:
        signal: Array of shape (window_length,) or (N_windows, window_length).
        sampling_rate_hz: Signal sampling rate (default 12,000 Hz).
        n_fft: FFT length. Defaults to signal length along last dimension.
        sample_rate: Optional alias for sampling_rate_hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - freqs: 1D array of frequency bin centers in Hz (length n_fft // 2 + 1).
            - magnitudes: Array of one-sided magnitude spectrum with shape (..., n_fft // 2 + 1).
    """
    if sample_rate is not None:
        sampling_rate_hz = sample_rate
    if isinstance(signal, torch.Tensor):
        x = signal.detach().cpu().numpy()
    else:
        x = np.asarray(signal)

    if x.size == 0:
        raise ValueError("Cannot compute FFT on empty input array.")

    length = x.shape[-1]
    fft_len = n_fft or length
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sampling_rate_hz)

    # Compute one-sided real FFT
    complex_spec = np.fft.rfft(x, n=fft_len, axis=-1)
    # Scaled magnitude: 2/N for non-DC/Nyquist, 1/N for DC
    mag = np.abs(complex_spec) / float(length)
    if mag.ndim == 1:
        if len(mag) > 2:
            mag[1:-1] *= 2.0
    else:
        if mag.shape[-1] > 2:
            mag[..., 1:-1] *= 2.0

    return freqs.astype(np.float32), mag.astype(np.float32)


def compute_hilbert_envelope_spectrum(
    signal: Union[np.ndarray, torch.Tensor],
    sampling_rate_hz: int = 12000,
    n_fft: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Hilbert analytic envelope and its one-sided modulation spectrum.

    Methodology:
    1. Form analytic signal z[n] = x[n] + j*H(x[n]) via the Hilbert transform.
    2. Extract instantaneous envelope e[n] = |z[n]|.
    3. Subtract envelope mean (e[n] - mean(e)) to eliminate carrier DC dominance.
    4. Compute one-sided FFT of zero-mean envelope to expose defect impact modulation frequencies.

    Args:
        signal: Array of shape (window_length,) or (N_windows, window_length).
        sampling_rate_hz: Signal sampling rate (default 12,000 Hz).
        n_fft: FFT length for the envelope spectrum. Defaults to signal length.
        sample_rate: Optional alias for sampling_rate_hz.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - freqs: 1D array of frequency bin centers in Hz.
            - envelope_spectrum: One-sided magnitude spectrum of the envelope.
            - envelope_time: Raw instantaneous envelope in the time domain.
    """
    if sample_rate is not None:
        sampling_rate_hz = sample_rate
    if isinstance(signal, torch.Tensor):
        x = signal.detach().cpu().numpy()
    else:
        x = np.asarray(signal)

    if x.size == 0:
        raise ValueError("Cannot compute Hilbert envelope on empty input array.")

    # 1. Analytic signal via Hilbert transform
    analytic = scipy.signal.hilbert(x, axis=-1)
    envelope = np.abs(analytic)

    # 2. Subtract mean envelope to expose AC modulation components
    env_mean = np.mean(envelope, axis=-1, keepdims=True)
    zero_mean_envelope = envelope - env_mean

    # 3. FFT of zero-mean envelope
    freqs, env_mag = compute_fft_magnitude_spectrum(
        zero_mean_envelope,
        sampling_rate_hz=sampling_rate_hz,
        n_fft=n_fft,
    )

    return freqs.astype(np.float32), env_mag.astype(np.float32), envelope.astype(np.float32)


class SpectralNormalizer:
    """Standardizes frequency-domain feature vectors using statistics fitted strictly on the training fold.

    Guarantees zero data leakage across recording partitions.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}.")
        self.epsilon = float(epsilon)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.n_features: int = 0
        self.fit_sample_count: int = 0

    def fit(self, features: np.ndarray) -> "SpectralNormalizer":
        """Compute per-frequency-bin mean and standard deviation from training data only.

        Args:
            features: 2D array of shape (N_samples, n_freq_bins).

        Returns:
            self: Fitted normalizer.
        """
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (samples, features), got {arr.ndim}D array of shape {arr.shape}.")
        if arr.size == 0:
            raise ValueError("Training spectral features array is empty.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Training spectral features contain NaN or Inf values.")

        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        self.n_features = arr.shape[1]
        self.fit_sample_count = arr.shape[0]
        self.is_fitted = True

        logger.info(
            f"Fitted SpectralNormalizer on {self.fit_sample_count} samples x {self.n_features} frequency bins."
        )
        return self

    def transform(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply z-score standardization to spectral feature vectors (NumPy or PyTorch).

        Args:
            features: 1D or 2D array or tensor of spectral features.

        Returns:
            Union[np.ndarray, torch.Tensor]: Standardized float32 features.
        """
        if isinstance(features, torch.Tensor):
            return self.transform_tensor(features)

        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("SpectralNormalizer must be fitted before transform.")

        arr = np.asarray(features, dtype=np.float64)
        denom = np.maximum(self.std, self.epsilon)
        norm = (arr - self.mean) / denom
        return norm.astype(np.float32)

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply standardization to PyTorch tensor.

        Args:
            tensor: Float tensor of shape (..., n_features).

        Returns:
            torch.Tensor: Normalized float32 tensor.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("SpectralNormalizer must be fitted before transform_tensor.")

        t_mean = torch.from_numpy(self.mean).to(device=tensor.device, dtype=tensor.dtype)
        t_std = torch.from_numpy(self.std).to(device=tensor.device, dtype=tensor.dtype)
        denom = torch.clamp(t_std, min=self.epsilon)
        return (tensor - t_mean) / denom

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to dictionary."""
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "epsilon": self.epsilon,
            "is_fitted": self.is_fitted,
            "n_features": self.n_features,
            "fit_sample_count": self.fit_sample_count,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist fitted parameters to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted SpectralNormalizer.")

        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SpectralNormalizer":
        """Load parameters from JSON file."""
        path = Path(file_path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(epsilon=data.get("epsilon", 1e-8))
        instance.mean = np.array(data["mean"], dtype=np.float64) if data["mean"] is not None else None
        instance.std = np.array(data["std"], dtype=np.float64) if data["std"] is not None else None
        instance.is_fitted = bool(data["is_fitted"])
        instance.n_features = int(data.get("n_features", 0))
        instance.fit_sample_count = int(data.get("fit_sample_count", 0))
        return instance


def get_bearing_characteristic_frequencies(
    motor_rpm: float = 1797.0,
    pitch_diameter_mm: float = 39.04,
    ball_diameter_mm: float = 7.94,
    num_balls: int = 9,
    contact_angle_deg: float = 0.0,
    shaft_rpm: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate theoretical kinematic defect frequencies for deep groove ball bearing (SKF 6205-2RS).

    Formulas:
    - Shaft Speed fr = RPM / 60
    - BPFO = (N_b / 2) * fr * (1 - (d / D) * cos(alpha))
    - BPFI = (N_b / 2) * fr * (1 + (d / D) * cos(alpha))
    - BSF  = (D / (2 * d)) * fr * (1 - ((d / D) * cos(alpha))^2)
    - FTF  = (1 / 2) * fr * (1 - (d / D) * cos(alpha))

    Args:
        motor_rpm: Rotational speed in revolutions per minute.
        pitch_diameter_mm: Pitch diameter D (SKF 6205: 39.04 mm).
        ball_diameter_mm: Ball diameter d (SKF 6205: 7.94 mm).
        num_balls: Number of rolling elements Nb (SKF 6205: 9).
        contact_angle_deg: Contact angle alpha (0 degrees for radial bearing).
        shaft_rpm: Optional alias for motor_rpm.

    Returns:
        Dict[str, float]: Frequencies in Hz for fr, BPFO, BPFI, BSF, and FTF.
    """
    if shaft_rpm is not None:
        motor_rpm = shaft_rpm

    fr = float(motor_rpm) / 60.0
    alpha_rad = np.radians(contact_angle_deg)
    gamma = (ball_diameter_mm / pitch_diameter_mm) * np.cos(alpha_rad)

    bpfo = (num_balls / 2.0) * fr * (1.0 - gamma)
    bpfi = (num_balls / 2.0) * fr * (1.0 + gamma)
    bsf = (pitch_diameter_mm / (2.0 * ball_diameter_mm)) * fr * (1.0 - gamma**2)
    ftf = 0.5 * fr * (1.0 - gamma)

    return {
        "fr_hz": round(fr, 3),
        "bpfo_hz": round(bpfo, 3),
        "bpfi_hz": round(bpfi, 3),
        "bsf_hz": round(bsf, 3),
        "ftf_hz": round(ftf, 3),
        "fr": round(fr, 3),
        "bpfo": round(bpfo, 3),
        "bpfi": round(bpfi, 3),
        "bsf": round(bsf, 3),
        "ftf": round(ftf, 3),
    }
