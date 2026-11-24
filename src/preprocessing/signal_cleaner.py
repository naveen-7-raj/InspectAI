"""Digital signal preprocessing and sampling rate normalization for industrial vibration signals."""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal

logger = logging.getLogger("InspectAI.SignalCleaner")


@dataclass
class ProcessedSignal:
    """Container holding cleaned vibration signal and processing provenance metadata."""

    signal: np.ndarray
    original_fs: int
    target_fs: int
    applied_resampling: bool
    applied_dc_removal: bool
    applied_filtering: bool
    filter_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.signal)

    @property
    def duration_seconds(self) -> float:
        return len(self.signal) / self.target_fs if self.target_fs > 0 else 0.0


class SignalCleaner:
    """Preprocesses raw vibration signals: sampling-rate normalization, DC bias removal, and filtering."""

    def __init__(
        self,
        target_sampling_rate: int = 12000,
        resample_method: str = "resample_poly",
        remove_dc: bool = True,
        filter_enabled: bool = True,
        filter_type: str = "bandpass",
        low_cutoff_hz: float = 20.0,
        high_cutoff_hz: float = 5000.0,
        filter_order: int = 4,
    ) -> None:
        self.target_sampling_rate = int(target_sampling_rate)
        self.resample_method = resample_method
        self.remove_dc = remove_dc
        self.filter_enabled = filter_enabled
        self.filter_type = filter_type.lower()
        self.low_cutoff_hz = float(low_cutoff_hz)
        self.high_cutoff_hz = float(high_cutoff_hz)
        self.filter_order = int(filter_order)

    @staticmethod
    def validate_signal(signal: Any, name: str = "signal") -> np.ndarray:
        """Validate input array dimensions, numeric types, and absence of NaN/Inf.

        Args:
            signal: Input signal data.
            name: Identifier for error reporting.

        Returns:
            np.ndarray: Validated 1D float64 numpy array.

        Raises:
            TypeError: If input is not numeric.
            ValueError: If input is empty, has invalid dimensions, or contains NaN/Inf.
        """
        if signal is None:
            raise ValueError(f"Input '{name}' cannot be None.")

        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)

        if not np.issubdtype(signal.dtype, np.number):
            raise TypeError(f"Signal '{name}' has non-numeric dtype '{signal.dtype}'. Expected numeric data.")

        if signal.size == 0:
            raise ValueError(f"Signal '{name}' is empty (0 samples).")

        if signal.ndim > 2:
            raise ValueError(f"Signal '{name}' has invalid dimensions {signal.shape}. Expected 1D array.")

        if signal.ndim == 2:
            if signal.shape[0] != 1 and signal.shape[1] != 1:
                raise ValueError(
                    f"Signal '{name}' is a 2D matrix of shape {signal.shape}, expected single-channel vector."
                )

        # Flatten to 1D and ensure float64
        flat_signal = signal.reshape(-1).astype(np.float64, copy=True)

        nan_count = int(np.isnan(flat_signal).sum())
        inf_count = int(np.isinf(flat_signal).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"Signal '{name}' contains corrupt values ({nan_count} NaNs, {inf_count} Infs). "
                f"Silent replacement is strictly prohibited."
            )

        return flat_signal

    @classmethod
    def normalize_sampling_rate(
        cls,
        signal: np.ndarray,
        orig_fs: int,
        target_fs: int = 12000,
        method: str = "resample_poly",
    ) -> Tuple[np.ndarray, bool]:
        """Normalize sampling rate to target_fs using anti-aliasing polyphase filtering.

        Args:
            signal: 1D input signal array.
            orig_fs: Verified nominal sampling rate of the raw signal (Hz).
            target_fs: Desired target sampling rate (Hz). Default is 12000.
            method: Resampling method ('resample_poly').

        Returns:
            Tuple[np.ndarray, bool]: (Resampled signal, applied_resampling flag).

        Raises:
            ValueError: If orig_fs or target_fs are not positive integers.
        """
        signal = cls.validate_signal(signal, name="raw_signal")

        if orig_fs <= 0:
            raise ValueError(f"Original sampling rate must be positive, got {orig_fs}.")
        if target_fs <= 0:
            raise ValueError(f"Target sampling rate must be positive, got {target_fs}.")

        # If already at target sampling rate, do not resample
        if orig_fs == target_fs:
            logger.debug(f"Signal already at target sampling rate {target_fs} Hz. Resampling bypassed.")
            return signal.copy(), False

        # Calculate rational up/down factors using greatest common divisor
        gcd = math.gcd(orig_fs, target_fs)
        up = target_fs // gcd
        down = orig_fs // gcd

        logger.info(
            f"Resampling signal from {orig_fs} Hz to {target_fs} Hz "
            f"(polyphase factor up={up}, down={down}, ratio={target_fs/orig_fs:.4f})."
        )

        if method == "resample_poly":
            # Polyphase FIR filtering with automatic Kaiser-windowed anti-aliasing filter
            resampled = scipy.signal.resample_poly(signal, up=up, down=down)
        else:
            raise ValueError(f"Unsupported resampling method '{method}'. Supported: 'resample_poly'.")

        return resampled, True

    @staticmethod
    def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
        """Remove sensor zero-drift/DC bias by subtracting the signal arithmetic mean.

        Args:
            signal: 1D vibration array.

        Returns:
            np.ndarray: Zero-mean vibration signal.
        """
        mean_offset = float(np.mean(signal))
        logger.debug(f"Removing DC offset: {mean_offset:+.6e} g.")
        return signal - mean_offset

    @classmethod
    def apply_filter(
        cls,
        signal: np.ndarray,
        fs: int,
        filter_type: str = "bandpass",
        low_cutoff: float = 20.0,
        high_cutoff: float = 5000.0,
        order: int = 4,
    ) -> np.ndarray:
        """Apply zero-phase Butterworth filter to vibration signal.

        Args:
            signal: 1D vibration signal array.
            fs: Current sampling frequency of the signal (Hz).
            filter_type: "bandpass", "highpass", or "none".
            low_cutoff: Lower cutoff frequency (Hz).
            high_cutoff: Upper cutoff frequency (Hz).
            order: Filter order.

        Returns:
            np.ndarray: Filtered vibration signal.

        Raises:
            ValueError: If cutoff frequencies violate Nyquist criterion.
        """
        signal = cls.validate_signal(signal, name="filter_input")
        ftype = filter_type.lower()

        if ftype == "none":
            return signal.copy()

        nyquist = fs / 2.0

        if ftype == "bandpass":
            if not (0 < low_cutoff < high_cutoff < nyquist):
                raise ValueError(
                    f"Invalid bandpass cutoffs: 0 < {low_cutoff} < {high_cutoff} < {nyquist} (Nyquist) violated."
                )
            sos = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype="bandpass", fs=fs, output="sos")
        elif ftype == "highpass":
            if not (0 < low_cutoff < nyquist):
                raise ValueError(
                    f"Invalid highpass cutoff: 0 < {low_cutoff} < {nyquist} (Nyquist) violated."
                )
            sos = scipy.signal.butter(order, low_cutoff, btype="highpass", fs=fs, output="sos")
        else:
            raise ValueError(f"Unsupported filter type '{filter_type}'. Supported: 'bandpass', 'highpass', 'none'.")

        # Zero-phase filtering using second-order sections (sosfiltfilt avoids forward phase shifts)
        filtered = scipy.signal.sosfiltfilt(sos, signal)
        return filtered

    def clean_signal(
        self,
        signal: np.ndarray,
        orig_fs: int,
        target_fs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedSignal:
        """Execute full preprocessing pipeline: validation -> resampling -> DC removal -> filtering.

        Args:
            signal: Raw vibration signal.
            orig_fs: Nominal sampling frequency of the raw recording (Hz).
            target_fs: Desired sampling rate (Hz). Defaults to self.target_sampling_rate.
            metadata: Optional source-file provenance metadata dictionary.

        Returns:
            ProcessedSignal: Validated, normalized, cleaned signal container.
        """
        target_fs = target_fs if target_fs is not None else self.target_sampling_rate
        valid_signal = self.validate_signal(signal, name="raw_vibration")

        # 1. Sampling-Rate Normalization
        norm_signal, applied_resample = self.normalize_sampling_rate(
            valid_signal, orig_fs=orig_fs, target_fs=target_fs, method=self.resample_method
        )

        # 2. DC Offset Removal
        if self.remove_dc:
            norm_signal = self.remove_dc_offset(norm_signal)
            applied_dc = True
        else:
            applied_dc = False

        # 3. Configurable Filtering
        filter_meta: Dict[str, Any] = {"enabled": self.filter_enabled}
        if self.filter_enabled and self.filter_type != "none":
            norm_signal = self.apply_filter(
                norm_signal,
                fs=target_fs,
                filter_type=self.filter_type,
                low_cutoff=self.low_cutoff_hz,
                high_cutoff=self.high_cutoff_hz,
                order=self.filter_order,
            )
            applied_filter = True
            filter_meta.update(
                {
                    "type": self.filter_type,
                    "low_cutoff_hz": self.low_cutoff_hz,
                    "high_cutoff_hz": self.high_cutoff_hz,
                    "order": self.filter_order,
                }
            )
        else:
            applied_filter = False

        return ProcessedSignal(
            signal=norm_signal,
            original_fs=orig_fs,
            target_fs=target_fs,
            applied_resampling=applied_resample,
            applied_dc_removal=applied_dc,
            applied_filtering=applied_filter,
            filter_info=filter_meta,
            metadata=metadata or {},
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SignalCleaner":
        """Instantiate SignalCleaner using settings from central configuration.

        Args:
            config: Configuration dictionary loaded from config.yaml.

        Returns:
            SignalCleaner: Configured preprocessor instance.
        """
        p_cfg = config.get("preprocessing", {})
        filt_cfg = p_cfg.get("filtering", {})
        return cls(
            target_sampling_rate=int(p_cfg.get("target_sampling_rate_hz", 12000)),
            resample_method=str(p_cfg.get("resample_method", "resample_poly")),
            remove_dc=bool(p_cfg.get("remove_dc", True)),
            filter_enabled=bool(filt_cfg.get("enabled", True)),
            filter_type=str(filt_cfg.get("type", "bandpass")),
            low_cutoff_hz=float(filt_cfg.get("low_cutoff_hz", 20.0)),
            high_cutoff_hz=float(filt_cfg.get("high_cutoff_hz", 5000.0)),
            filter_order=int(filt_cfg.get("order", 4)),
        )
