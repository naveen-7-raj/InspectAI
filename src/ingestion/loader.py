"""Vibration data loader and validation engine for raw sensor recordings (.mat)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.io

logger = logging.getLogger("InspectAI.VibrationDataLoader")


@dataclass
class VibrationRecord:
    """Container holding validated vibration sensor data and extraction metadata."""

    file_path: Path
    file_name: str
    channels: Dict[str, np.ndarray]
    sample_counts: Dict[str, int]
    missing_channels: List[str]
    raw_mat_keys: Dict[str, str]
    rpm: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record summary to a dictionary (excluding full arrays for lightweight inspection)."""
        return {
            "file_name": self.file_name,
            "file_path": str(self.file_path),
            "channels_present": list(self.channels.keys()),
            "sample_counts": self.sample_counts,
            "missing_channels": self.missing_channels,
            "raw_mat_keys": self.raw_mat_keys,
            "rpm": self.rpm,
            "metadata": self.metadata,
        }


class BaseDataLoader(ABC):
    """Abstract base class for industrial sensor data loaders."""

    def __init__(self, raw_data_dir: Optional[Union[str, Path]] = None) -> None:
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else Path("data/raw")

    @abstractmethod
    def load_file(self, file_path: Union[str, Path]) -> VibrationRecord:
        """Load raw sensor signal file into validated VibrationRecord.

        Args:
            file_path: Relative or absolute path to the sensor recording.

        Returns:
            VibrationRecord: Container holding validated channel arrays and metadata.
        """
        pass


class VibrationDataLoader(BaseDataLoader):
    """Robust MATLAB .mat loader for benchmark industrial vibration datasets (CWRU)."""

    CANONICAL_CHANNELS = ["DE_time", "FE_time", "BA_time"]
    REQUIRED_CHANNELS = ["DE_time"]

    def __init__(self, raw_data_dir: Optional[Union[str, Path]] = None) -> None:
        super().__init__(raw_data_dir)
        self.raw_data_dir = self.raw_data_dir.resolve()

    def _resolve_file_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path against raw_data_dir or cwd."""
        p = Path(file_path)
        if p.is_file():
            return p.resolve()

        candidate = self.raw_data_dir / p
        if candidate.is_file():
            return candidate.resolve()

        # Check in cwru subfolder if not found directly
        cwru_candidate = self.raw_data_dir / "cwru" / p
        if cwru_candidate.is_file():
            return cwru_candidate.resolve()

        raise FileNotFoundError(f"Vibration data file not found: {file_path} (searched {p}, {candidate}, {cwru_candidate})")

    @staticmethod
    def _identify_channel_key(mat_keys: List[str], channel_name: str, file_stem: str) -> Optional[str]:
        """Dynamically identify the variable name in the MATLAB file corresponding to a channel.

        Args:
            mat_keys: List of all valid variable keys in the MAT file.
            channel_name: Target channel canonical identifier (e.g., 'DE_time', 'FE_time', 'BA_time').
            file_stem: Filename without extension (e.g. '97', '99', '105') used to resolve collisions.

        Returns:
            Optional[str]: Matched key name, or None if channel is absent.
        """
        # Find all keys matching the channel name (case-insensitive)
        candidates = [k for k in mat_keys if channel_name.lower() in k.lower()]
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Disambiguate if multiple candidates exist (e.g., 99.mat contains both X098_DE_time and X099_DE_time)
        digits = "".join(re.findall(r"\d+", file_stem))
        if digits:
            # Look for candidate containing X0*<digits>_
            pattern = re.compile(rf"X0*{digits}_", re.IGNORECASE)
            matched = [c for c in candidates if pattern.search(c)]
            if matched:
                logger.debug(f"Resolved collision for {channel_name} using stem '{file_stem}': {matched[0]}")
                return matched[0]

        # Fallback to the last candidate if prefix matching fails
        logger.warning(f"Multiple candidates found for {channel_name} in '{file_stem}': {candidates}. Selecting {candidates[-1]}")
        return candidates[-1]

    @staticmethod
    def _identify_rpm_key(mat_keys: List[str], file_stem: str) -> Optional[str]:
        """Identify tachometer / rotational speed variable in MATLAB keys."""
        rpm_candidates = [k for k in mat_keys if "rpm" in k.lower()]
        if not rpm_candidates:
            return None

        if len(rpm_candidates) == 1:
            return rpm_candidates[0]

        digits = "".join(re.findall(r"\d+", file_stem))
        if digits:
            pattern = re.compile(rf"0*{digits}", re.IGNORECASE)
            matched = [c for c in rpm_candidates if pattern.search(c)]
            if matched:
                return matched[0]

        return rpm_candidates[0]

    @staticmethod
    def _validate_channel_array(channel_name: str, raw_arr: Any, raw_key: str) -> np.ndarray:
        """Validate array dimensions, numeric types, non-emptiness, and absence of NaN/Inf.

        Args:
            channel_name: Canonical channel name.
            raw_arr: Raw object extracted from MATLAB dict.
            raw_key: Original MATLAB variable key.

        Returns:
            np.ndarray: Validated 1D numpy array preserving original values.

        Raises:
            TypeError: If array is not numeric.
            ValueError: If array is empty, has invalid dimensions, or contains NaN/Inf values.
        """
        if not isinstance(raw_arr, np.ndarray):
            raw_arr = np.asarray(raw_arr)

        if not np.issubdtype(raw_arr.dtype, np.number):
            raise TypeError(
                f"Channel '{channel_name}' (key '{raw_key}') has non-numeric dtype '{raw_arr.dtype}'."
            )

        if raw_arr.size == 0:
            raise ValueError(f"Channel '{channel_name}' (key '{raw_key}') is empty (0 samples).")

        # Check dimensions: must be 1D, or 2D with shape (N, 1) or (1, N)
        if raw_arr.ndim > 2:
            raise ValueError(
                f"Channel '{channel_name}' (key '{raw_key}') has invalid dimensions {raw_arr.shape}. Expected 1D or (N, 1)."
            )

        if raw_arr.ndim == 2:
            if raw_arr.shape[0] != 1 and raw_arr.shape[1] != 1:
                raise ValueError(
                    f"Channel '{channel_name}' (key '{raw_key}') is a 2D matrix {raw_arr.shape}, not a single sensor vector."
                )

        # Flatten to 1D preserving all sample values
        signal = raw_arr.reshape(-1)

        # Strict NaN and Infinity validation (no silent replacement)
        nan_count = int(np.isnan(signal).sum())
        inf_count = int(np.isinf(signal).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"Channel '{channel_name}' (key '{raw_key}') contains corrupt values: "
                f"{nan_count} NaNs, {inf_count} Infs. Silent replacement is forbidden."
            )

        return signal

    def load_file(self, file_path: Union[str, Path]) -> VibrationRecord:
        """Load and strictly validate a single raw MATLAB vibration file.

        Args:
            file_path: Relative or absolute path to the .mat recording.

        Returns:
            VibrationRecord: Strongly validated vibration data object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file cannot be parsed or validation fails.
            KeyError: If required primary channel ('DE_time') is missing.
        """
        resolved_path = self._resolve_file_path(file_path)
        logger.info(f"Loading vibration recording: {resolved_path.name}")

        try:
            mat_data = scipy.io.loadmat(resolved_path)
        except Exception as e:
            raise ValueError(f"Failed to read MATLAB file '{resolved_path.name}': {e}") from e

        all_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if not all_keys:
            raise ValueError(f"MATLAB file '{resolved_path.name}' contains no valid data variables.")

        file_stem = resolved_path.stem
        extracted_channels: Dict[str, np.ndarray] = {}
        sample_counts: Dict[str, int] = {}
        raw_mat_keys: Dict[str, str] = {}
        missing_channels: List[str] = []

        # Extract vibration channels
        for channel in self.CANONICAL_CHANNELS:
            key = self._identify_channel_key(all_keys, channel, file_stem)
            if key is not None:
                raw_arr = mat_data[key]
                validated_signal = self._validate_channel_array(channel, raw_arr, key)
                extracted_channels[channel] = validated_signal
                sample_counts[channel] = len(validated_signal)
                raw_mat_keys[channel] = key
                logger.debug(f"Extracted {channel} from '{key}' ({len(validated_signal)} samples, dtype: {validated_signal.dtype})")
            else:
                missing_channels.append(channel)
                logger.debug(f"Optional channel '{channel}' not found in '{resolved_path.name}'.")

        # Validate required channels (DE_time must be present)
        for req in self.REQUIRED_CHANNELS:
            if req not in extracted_channels:
                raise KeyError(
                    f"Required primary channel '{req}' not found in '{resolved_path.name}'. "
                    f"Available MATLAB keys: {all_keys}"
                )

        # Extract RPM if available
        rpm_value: Optional[float] = None
        rpm_key = self._identify_rpm_key(all_keys, file_stem)
        if rpm_key is not None:
            raw_rpm = mat_data[rpm_key]
            try:
                rpm_arr = np.asarray(raw_rpm).reshape(-1)
                if rpm_arr.size > 0 and np.issubdtype(rpm_arr.dtype, np.number):
                    rpm_value = float(rpm_arr[0])
                    raw_mat_keys["RPM"] = rpm_key
                    logger.debug(f"Extracted RPM from '{rpm_key}': {rpm_value}")
            except Exception as e:
                logger.warning(f"Could not parse RPM value from '{rpm_key}': {e}")

        # Assemble metadata
        metadata = {
            "file_size_bytes": resolved_path.stat().st_size,
            "all_mat_keys": all_keys,
            "channels_extracted_count": len(extracted_channels),
        }

        logger.info(
            f"Successfully parsed '{resolved_path.name}': "
            f"Channels={list(extracted_channels.keys())}, Missing={missing_channels}, RPM={rpm_value}"
        )

        return VibrationRecord(
            file_path=resolved_path,
            file_name=resolved_path.name,
            channels=extracted_channels,
            sample_counts=sample_counts,
            missing_channels=missing_channels,
            raw_mat_keys=raw_mat_keys,
            rpm=rpm_value,
            metadata=metadata,
        )
